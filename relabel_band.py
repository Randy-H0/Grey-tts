"""Band labels from each clip's measured bandwidth instead of "full band unless the dataset says otherwise".

Common Voice clips are lossy uploads: all of CV-JA and about half of CV-ES stop at ~11-16 kHz, and a third of the
Speech Accent Archive is band-limited too. Unlabelled, they all counted as full band (band = 1), so the model
learned that "full band" Spanish / Japanese can sound muffled, and band = 1 at inference couldn't fix it.

band = clip((bandwidth_kHz - 11) / 9, 0, 1): ~11 kHz (24 kHz-style / low-bitrate mp3) -> 0, >= 20 kHz -> 1.
Bandwidth = highest frequency within 55 dB of the speech band's level. Manifests are backed up first
(manifest.jsonl.bak_band); `band_khz` keeps the measurement. Decoding + measuring on 10 CPU processes, which send back
one number per clip (2026-10-21: sending the decoded audio back -- ~400k large pickles for Emilia -- garbled one
message, killed the pool's result thread and hung the run).

  python relabel_band.py data/kvae_cves=data/clips_cves.txt data/kvae_cvja=data/clips_cvja.txt ...
"""
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def measure(job):
    """CPU worker: file -> (id, bandwidth in kHz or None), from at most 20 s of the clip."""
    uid, path, a, b = job
    try:
        from dacts.audio_io import read_any
        w, sr = read_any(str(path))
        w = w.mean(1)
        if a is not None and b is not None:
            w = w[int(a * sr): int(b * sr)]
        return uid, bandwidth(np.ascontiguousarray(w[: 20 * sr], dtype=np.float32), sr)
    except Exception:  # noqa: BLE001
        return uid, None


_win = {}


def bandwidth(w, sr):
    """Highest frequency within 55 dB of the speech band's level (kHz). (Same maths as the earlier GPU version.)"""
    n = 4096
    if w is None or len(w) < n * 2:
        return None
    if "h" not in _win:
        _win["h"] = np.hanning(n).astype(np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(w, n)[:: n // 2] * _win["h"]
    P = 10 * np.log10((np.abs(np.fft.rfft(fr, axis=1)) ** 2).mean(0) + 1e-12)
    f = np.fft.rfftfreq(n, 1 / sr)
    ref = np.median(P[(f > 300) & (f < 3000)])
    above = f[P > ref - 55]
    return float(above.max()) / 1000 if len(above) else None


def low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass


def main():
    from preprocess import read_metadata
    for arg in sys.argv[1:]:
        d, lst = (Path(x) for x in arg.split("="))
        rows = {r[0]: (r[1], r[4], r[5]) for r in read_metadata(str(lst), "list", "")}
        mf = d / "manifest.jsonl"
        bak = d / "manifest.jsonl.bak_band"
        items = [json.loads(l) for l in open(bak if bak.exists() else mf, encoding="utf-8")]
        if bak.exists() and all("band_khz" in json.loads(l) for l in open(mf, encoding="utf-8")):
            print(f"{d}: already measured", flush=True)  # e.g. ahead of its turn, next to the pipeline
            continue
        jobs = [(it["id"], *rows[it["id"]]) for it in items if it["id"] in rows]
        bw = {}
        with Pool(int(os.environ.get("BAND_WORKERS", "10")), initializer=low_priority) as pool:  # decode (mp3) + measure on the CPU; only numbers come back
            for uid, k in pool.imap_unordered(measure, jobs, chunksize=32):
                bw[uid] = k
        n0 = n1 = nm = 0
        for it in items:
            k = bw.get(it["id"])
            if k is None:
                nm += 1
                continue
            it["band_khz"] = round(k, 1)
            it["band"] = round(float(np.clip((k - 11.0) / 9.0, 0.0, 1.0)), 2)
            n0 += it["band"] < 0.5
            n1 += it["band"] >= 0.5
        if not bak.exists():
            shutil.copy(mf, bak)
        mf.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        ks = [i["band_khz"] for i in items if "band_khz" in i]
        print(f"{d}: {n1} full-band, {n0} band-limited, {nm} unmeasured; median {np.median(ks):.1f} kHz", flush=True)


if __name__ == "__main__":
    main()
