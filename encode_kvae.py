"""Encode preprocessed datasets with KVAE-Audio (48 kHz continuous latents, 64-d @ 50 Hz).

Reuses each dataset's manifest (labels, speakers, context ids): for every item the source audio is
loaded exactly as preprocess.py did (same trim), brought to the same speech level (-20 dB +
limiter), resampled to 48 kHz and encoded to the latent mean. Writes <dir>/latents/<id>.npy
(float16, (T, 64)), updates "frames" to latent frames, and stats.pt (per-dim mean / std).
Resumable: existing latent files are kept.

  python encode_kvae.py data/kvae_norm=data/clips.txt data/kvae_ja=data/clips_ja.txt ...
"""
import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dacts.infer import limit, speech_level_db
from preprocess import load_audio, read_metadata

KVAE_SR = 48000


class Clips(Dataset):
    def __init__(self, rows, level_db):
        self.rows, self.level_db = rows, level_db

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        uid, path, text, spk, start, end, ch = self.rows[i]
        try:
            w = load_audio(path, start, end, ch).numpy()
            w = limit(w * 10 ** ((self.level_db - speech_level_db(w)) / 20))
            return uid, AF.resample(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), 44100, KVAE_SR)
        except Exception as e:  # noqa: BLE001
            print(f"skip {uid}: {e}", file=sys.stderr)
            return uid, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="dataset_dir=clip_list")
    ap.add_argument("--level_db", type=float, default=-20.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from kvae_audio import KVAEAudio

    model = KVAEAudio.load(hf_hub_download("kandinskylab/KVAE-Audio", "kvae-audio.pt"), map_location="cpu").cuda().eval()
    if os.environ.get("KVAE_COMPILE") == "1":  # torch.compile (triton-windows; run inside the VS 2022 env, CC = its
        # cl.exe): 1.5-2.6x faster; latents differ by 0.0008 on average, decoded audio by 0.017 dB (eager vs compiled)
        model.encoder = torch.compile(model.encoder, dynamic=True)
    for pair in args.pairs:
        d, lst = pair.split("=")
        d = Path(d)
        (d / "latents").mkdir(exist_ok=True)
        items = [json.loads(l) for l in open(d / "manifest.jsonl", encoding="utf-8")]
        want = {it["id"] for it in items}
        rows = [r for r in read_metadata(lst, "list", "") if r[0] in want and not (d / "latents" / f"{r[0]}.npy").exists()]
        print(f"{d}: {len(items)} items, {len(rows)} to encode", flush=True)
        dl = DataLoader(Clips(rows, args.level_db), batch_size=None, num_workers=args.workers, prefetch_factor=8)
        with torch.no_grad():
            for uid, w in tqdm(dl, desc=f"kvae {d.name}", mininterval=5):
                if w is None or len(w) < KVAE_SR // 10:
                    continue
                try:
                    z, *_ = model.encode(w[None, None].cuda(), sample_rate=KVAE_SR)  # (1, 64, T) mean latent
                except torch.OutOfMemoryError:  # one abnormally long upload must not stop the whole set
                    z = None
                if z is None:
                    torch.cuda.empty_cache()  # (outside the except block, which still holds the failed tensors)
                    print(f"skip {uid}: out of GPU memory ({len(w) / KVAE_SR:.0f} s)", file=sys.stderr, flush=True)
                    continue
                np.save(d / "latents" / f"{uid}.npy", z[0].T.half().cpu().numpy())
        # frames, stats, manifest
        kept, s1, s2, n = [], torch.zeros(64, dtype=torch.float64), torch.zeros(64, dtype=torch.float64), 0
        for it in items:
            f = d / "latents" / f"{it['id']}.npy"
            if not f.exists():
                continue
            z = torch.from_numpy(np.load(f).astype(np.float32)).double()
            it["frames"] = int(z.shape[0])
            s1 += z.sum(0); s2 += (z ** 2).sum(0); n += z.shape[0]
            kept.append(it)
        mean = s1 / n
        std = (s2 / n - mean ** 2).clamp(min=1e-8).sqrt()
        torch.save({"mean": mean.float(), "std": std.float(), "frames": n, "codec": "kvae"}, d / "stats.pt")
        (d / "manifest.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in kept), encoding="utf-8")
        print(f"{d}: {len(kept)}/{len(items)} encoded, {n / 50 / 3600:.1f} h of latents", flush=True)


if __name__ == "__main__":
    main()
