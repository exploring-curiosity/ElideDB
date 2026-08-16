"""Encode pass 2: what Stage B actually needs -
  h1,h2  (N,256,1024) temporal-HALF token means (h2-h1 = direction)
  pooled_r (N,1024)   REVERSED-clip embedding (arrow-of-time negative)
  h1r,h2r (N,256,1024) reversed halves
No labels; same event units."""
import sys, json, subprocess
sys.path.insert(0, 'native')
import numpy as np, torch
from pathlib import Path

MID = "facebook/vjepa2-vitl-fpc64-256"
OUT = Path('data/cache/stagea2_prim_v2.npz')


def frames(mp4):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 480, 640, 3)


def main():
    from transformers import AutoVideoProcessor, AutoModel
    from tqdm import tqdm
    proc = AutoVideoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(
        MID, dtype=torch.float16).to('mps').eval()

    def enc(clip):
        inputs = proc(list(clip), return_tensors='pt').to('mps')
        with torch.no_grad():
            out = model(**{k: (v.half() if v.dtype == torch.float32
                               else v) for k, v in inputs.items()})
        G = out.last_hidden_state[0].float().cpu().numpy() \
            .reshape(16, 256, 1024)
        return (G[:8].mean(0).astype(np.float16),
                G[8:].mean(0).astype(np.float16))

    H1, H2, H1R, H2R = [], [], [], []
    eps = sorted(Path('data/prim_actions_v2').glob('ep*'))
    for ep in tqdm(eps, unit='ep', desc='stageA2 halves+rev (~25 min)'):
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
            h1, h2 = enc(F[idx])
            r1, r2 = enc(F[idx[::-1]])
            H1.append(h1); H2.append(h2)
            H1R.append(r1); H2R.append(r2)
    np.savez_compressed(OUT, h1=np.stack(H1), h2=np.stack(H2),
                        h1r=np.stack(H1R), h2r=np.stack(H2R))
    print('saved', OUT, len(H1), 'events')


if __name__ == '__main__':
    main()
