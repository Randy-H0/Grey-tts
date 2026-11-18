"""GPU version of label_clean.py: same measures (hf_noise.measures) and mapping, computed with
torch on the GPU while DataLoader workers decode the audio.

  python label_clean_gpu.py data/kvae_norm=data/clips.txt data/kvae_ja=data/clips_ja.txt ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from label_clean import clean_label
from preprocess import load_audio, read_metadata

SR, NFFT, HOP = 44100, 2048, 512


class Clips(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        uid, path, text, spk, a, b, ch = self.rows[i]
        try:
            return uid, load_audio(path, a, b, ch)
        except Exception:  # noqa: BLE001
            return uid, None


@torch.no_grad()
def measures_gpu(w: torch.Tensor):
    """Identical to hf_noise.measures, in torch."""
    n = (len(w) - NFFT) // HOP
    if n < 20:
        return None
    frames = w.unfold(0, NFFT, HOP)[:n] * torch.hann_window(NFFT, periodic=False, device=w.device)
    S = torch.fft.rfft(frames, dim=1).abs() ** 2
    f = torch.fft.rfftfreq(NFFT, 1 / SR, device=w.device)
    speech = 10 * torch.log10(S[:, (f > 100) & (f < 4000)].sum(1) + 1e-12)
    hf = 10 * torch.log10(S[:, (f > 8000) & (f < 20000)].sum(1) + 1e-12)
    loud = speech > torch.quantile(speech, 0.7)
    hf_floor = float(torch.quantile(hf, 0.1) - speech[loud].median())
    p20 = 10 * torch.log10(torch.quantile(S[:, f > 10000], 0.2, dim=0) + 1e-14)
    pad = torch.nn.functional.pad(p20[None, None], (8, 8), mode="replicate")[0, 0]
    neigh = torch.stack([pad[i:i + len(p20)] for i in range(17) if i != 8]).median(0).values
    return hf_floor, float((p20 - neigh).max())


def main():
    for pair in sys.argv[1:]:
        d, lst = pair.split("=")
        d = Path(d)
        items = [json.loads(l) for l in open(d / "manifest.jsonl", encoding="utf-8")]
        want = {it["id"] for it in items}
        rows = [r for r in read_metadata(lst, "list", "") if r[0] in want]
        res = {}
        dl = DataLoader(Clips(rows), batch_size=None, num_workers=12, prefetch_factor=8)
        for k, (uid, w) in enumerate(dl):
            res[uid] = measures_gpu(w.cuda()) if w is not None else None
            if k % 5000 == 0:
                print(f"{d.name}: {k}/{len(rows)}", flush=True)
        for it in items:
            m = res.get(it["id"])
            if m is None:
                it["clean"] = None
                continue
            it["dirt"] = {"hf_floor_db": round(m[0], 2), "whine_db": round(m[1], 2)}
            it["clean"] = clean_label(*m)
        (d / "manifest.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        c = np.array([i["clean"] for i in items if i.get("clean") is not None])
        print(f"{d.name}: clean mean {c.mean():.2f}, <0.5: {(c < 0.5).mean() * 100:.1f}% of clips", flush=True)


if __name__ == "__main__":
    main()
