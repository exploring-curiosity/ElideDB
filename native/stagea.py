"""Stage A of PROBLEM2.md: frozen V-JEPA 2 features per event window
+ codec-free optical flow channel. Saves per corpus:
  pooled (N,1024)   clip embedding (temporal+spatial mean)
  spat   (N,256,1024) per-spatial-cell temporal mean  (late interaction)
  delta  (N,256)      per-cell temporal-change magnitude (motion gate)
  flow   (N,17)       Farneback direction histogram (8 bins x mag) +
                      early/mid/late net (dx,dy,mag) profile
No labels anywhere; events = the given retrieval units."""
import sys, json, subprocess
sys.path.insert(0, 'native')
import numpy as np, torch, cv2
from pathlib import Path

MID = "facebook/vjepa2-vitl-fpc64-256"
OUT = Path('data/cache/stagea_prim_v2.npz')


def frames(mp4):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 480, 640, 3)


def flow_feat(F, a, b):
    T = len(F)
    idx = np.clip(np.linspace(a - 2, b + 4, 11).astype(int), 0, T - 1)
    hist = np.zeros(8)
    prof = []
    for k in range(10):
        f0 = cv2.cvtColor(cv2.resize(F[idx[k]], (320, 240)),
                          cv2.COLOR_RGB2GRAY)
        f1 = cv2.cvtColor(cv2.resize(F[idx[k + 1]], (320, 240)),
                          cv2.COLOR_RGB2GRAY)
        fl = cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 3, 15,
                                          3, 5, 1.2, 0)
        mag = np.hypot(fl[..., 0], fl[..., 1])
        mov = mag > 1.0
        if mov.sum() > 20:
            ang = np.arctan2(fl[..., 1][mov], fl[..., 0][mov])
            h, _ = np.histogram(ang, 8, (-np.pi, np.pi),
                                weights=mag[mov])
            hist += h
            prof.append([float(fl[..., 0][mov].mean()),
                         float(fl[..., 1][mov].mean()),
                         float(mag[mov].mean())])
        else:
            prof.append([0.0, 0.0, 0.0])
    hist = hist / max(hist.sum(), 1e-6)
    P = np.asarray(prof)
    thirds = [P[:3].mean(0), P[3:7].mean(0), P[7:].mean(0)]
    return np.concatenate([hist, np.concatenate(thirds)])


def main():
    from transformers import AutoVideoProcessor, AutoModel
    from tqdm import tqdm
    proc = AutoVideoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(
        MID, dtype=torch.float16).to('mps').eval()
    evs, pooled, spat, delta, flow = [], [], [], [], []
    eps = sorted(Path('data/prim_actions_v2').glob('ep*'))
    for ep in tqdm(eps, unit='ep', desc='stageA encode (~12 min)'):
        meta = json.loads((ep / 'meta.json').read_text())
        F = frames(sorted(ep.glob('cam*.mp4'))[0])
        for e in meta['events']:
            if not e['ok']:
                continue
            a, b = int(e['t0'] * 10), int(e['t1'] * 10)
            if b - a < 4:
                continue
            idx = np.clip(np.linspace(a - 2, b + 4, 32).astype(int),
                          0, len(F) - 1)
            inputs = proc(list(F[idx]), return_tensors='pt').to('mps')
            with torch.no_grad():
                out = model(**{k: (v.half() if v.dtype == torch.float32
                                   else v) for k, v in inputs.items()})
            hs = out.last_hidden_state[0].float().cpu().numpy()
            G = hs.reshape(16, 256, 1024)        # (t, cell, d)
            pooled.append(G.mean((0, 1)).astype(np.float16))
            spat.append(G.mean(0).astype(np.float16))
            delta.append(np.abs(np.diff(G, axis=0)).mean((0, 2))
                         .astype(np.float16))
            flow.append(flow_feat(F, a, b).astype(np.float32))
            evs.append((e['prim'], ep.name, meta['arm']))
    np.savez_compressed(
        OUT, pooled=np.stack(pooled), spat=np.stack(spat),
        delta=np.stack(delta), flow=np.stack(flow),
        prim=np.array([e[0] for e in evs]),
        ep=np.array([e[1] for e in evs]),
        arm=np.array([e[2] for e in evs]))
    print('saved', OUT, len(evs), 'events')


if __name__ == '__main__':
    main()
