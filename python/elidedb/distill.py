"""CHANNEL STUDENTS: an FDNN head per teacher, over vectors already computed.

The teachers - PE, SigLIP2 so400m, InternVideo2-1B, XCLIP, V-JEPA2 -
were built for accuracy and they are not deployable. Measured per
episode at write time: pe 1.17 s, sig2 0.70, iv2 1.14, xclip 0.49,
before V-JEPA2 runs twice more. That is the retrieval path costing an
order of magnitude more than the entire rest of the write, which makes
the teachers part of the product instead of part of the training.

A teacher's job is to be RIGHT. A student's job is to be THERE.

THE INPUT IS ALREADY PAID FOR. The write path streams every frame
through FDNN-V and stores a 1152-d vector per frame, at 0.020 s per
episode. Every channel output is a function of the same pixels, so
every channel can be a small head over those vectors instead of a
second pass over the video. The teacher is run ONCE, offline, to make
training pairs; after that it never touches an ingest again.

SKELETON, not a bag of layers. Each teacher pools a SPAN of frames into
one vector - a window for PE and SigLIP2, a whole clip for IV2, XCLIP
and V-JEPA2 - so the student mirrors that shape:

    attention pool over the span's frame vectors   (one learned query,
                                                    the teacher's own
                                                    aggregation step)
    pre-norm residual MLP -> the teacher's dimension
    L2 normalise                                   (every consumer
                                                    compares by cosine,
                                                    so the norm is not
                                                    part of the signal)

Mean pooling was the obvious first choice and it is wrong for the same
reason it is wrong in the teachers: a window's meaning is carried by a
few frames, and averaging buries them under the static ones.

Inference is deliberately NUMPY. The head is two small matmuls; running
them under a training framework costs more in dispatch than the
arithmetic, and the write path should not have to import one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# teacher output dimensions, by channel
DIMS = {"pe": 1024, "sig2": 1152, "iv2": 512, "xclip": 768,
        "vjepa": 1024, "act": 174}
FDNNV_D = 1152


def _l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


class Head:
    """A distilled channel. Loads from .npz, runs in numpy."""

    def __init__(self, w: dict):
        self.q = w["q"]                 # (d_in,)      pooling query
        self.W1, self.b1 = w["W1"], w["b1"]
        self.W2, self.b2 = w["W2"], w["b2"]
        self.Wp, self.bp = w["Wp"], w["bp"]
        self.g, self.beta = w["g"], w["beta"]

    @staticmethod
    def load(path):
        z = np.load(path)
        return Head({k: z[k] for k in z.files})

    def pool(self, X):
        """Attention-pool (n_frames, d) -> (d,) with one learned query."""
        if len(X) == 0:
            return np.zeros(self.q.shape[0], np.float32)
        s = X @ self.q
        s -= s.max()
        a = np.exp(s)
        a /= a.sum() + 1e-8
        return (a[:, None] * X).sum(0)

    def __call__(self, X):
        """(n_frames, 1152) frame vectors of one span -> teacher vector."""
        h = self.pool(np.asarray(X, np.float32))
        # pre-norm: the pooled vector's scale varies with span length
        h = self.g * (h - h.mean()) / (h.std() + 1e-6) + self.beta
        z = np.maximum(h @ self.W1 + self.b1, 0)          # ReLU
        h = h + z @ self.W2 + self.b2                     # residual
        return _l2(h @ self.Wp + self.bp)

    def batch(self, spans):
        """Many spans at once - the write path's actual call shape."""
        return np.stack([self(X) for X in spans]) if spans else \
            np.zeros((0, self.Wp.shape[1]), np.float32)


def pairs(store, channel, frames_table="frame_vectors"):
    """(list of frame-vector spans, teacher matrix) for one channel.

    A teacher row carries ts..t1; the student sees exactly the frame
    vectors inside that span, which is the same evidence the teacher
    had and no more. Spans with no frames are dropped rather than
    zero-filled - a student taught to map emptiness to a real vector
    learns to hallucinate.
    """
    import pyarrow.compute as pc
    tv = store.table(f"{channel}_vectors").scan()
    fv = store.table(frames_table).scan()
    fts = np.asarray(fv.column("ts").to_pylist(), np.int64)
    fst = np.asarray([str(s) for s in fv.column("stream").to_pylist()])
    F = np.asarray(fv.column("vector").to_pylist(), np.float32)

    order = np.argsort(fts, kind="stable")
    fts, fst, F = fts[order], fst[order], F[order]

    t0 = np.asarray(tv.column("ts").to_pylist(), np.int64)
    t1 = np.asarray(tv.column("t1").to_pylist(), np.int64)
    tst = np.asarray([str(s) for s in tv.column("stream").to_pylist()])
    T = _l2(np.asarray(tv.column("vector").to_pylist(), np.float32))

    X, Y, keys = [], [], []
    for i in range(len(t0)):
        lo = np.searchsorted(fts, t0[i], "left")
        hi = np.searchsorted(fts, t1[i], "right")
        if hi <= lo:
            continue
        sel = slice(lo, hi)
        m = fst[sel] == tst[i]
        if not m.any():
            continue
        X.append(F[sel][m])
        Y.append(T[i])
        keys.append((tst[i], int(t0[i])))
    return X, (np.stack(Y) if Y else np.zeros((0, 1), np.float32)), keys


def init(d_in=FDNNV_D, d_out=1024, hidden=512, seed=0):
    r = np.random.default_rng(seed)
    def n(*s):
        return (r.normal(size=s) / np.sqrt(s[0])).astype(np.float32)
    return {"q": n(d_in), "W1": n(d_in, hidden), "b1": np.zeros(hidden, np.float32),
            "W2": n(hidden, d_in), "b2": np.zeros(d_in, np.float32),
            "Wp": n(d_in, d_out), "bp": np.zeros(d_out, np.float32),
            "g": np.ones(d_in, np.float32), "beta": np.zeros(d_in, np.float32)}


def fidelity(head, X, Y):
    """Cosine to the teacher, ITS TRIVIAL BASELINE, and rank agreement.

    Cosine alone is not weak, it is actively misleading here, and the
    first PE student proved it: 0.9168 test cosine, which reads like a
    working student until you compute what a CONSTANT prediction of the
    corpus mean scores - 0.8629. The teacher's space is anisotropic
    (mean pairwise cosine 0.885), so almost all of that 0.92 is the
    shared mean and almost none of it is the episode. Nearest-neighbour
    agreement was 0.005.

    So every report carries `mean_baseline` next to `cosine`, and the
    number that decides whether a student ships is rank agreement -
    ranking is the only thing a retrieval channel is ever used for.
    """
    P = head.batch(X)
    cos = float(np.mean(np.sum(P * Y, 1)))
    mu = _l2(Y.mean(0))
    base = float(np.mean(Y @ mu))
    n = min(len(P), 400)
    Sp, St = P[:n] @ P[:n].T, Y[:n] @ Y[:n].T
    np.fill_diagonal(Sp, -9); np.fill_diagonal(St, -9)
    top1 = float(np.mean(Sp.argmax(1) == St.argmax(1)))
    k = min(10, n - 1)
    rp = np.argsort(-Sp, 1)[:, :k]
    rt = np.argsort(-St, 1)[:, :k]
    rec = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(rp, rt)]))
    return {"cosine": round(cos, 4), "mean_baseline": round(base, 4),
            "lift_over_mean": round(cos - base, 4),
            "nn_top1": round(top1, 4), "nn_recall@10": round(rec, 4)}
