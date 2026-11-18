"""Full-band 'dirty audio' measures (the 16 kHz quality models can't hear above 8 kHz).

Per clip, at 44.1 kHz:
  hf_floor_db : 8-20 kHz band level in the quietest 10% of frames, relative to the speech level
                (hiss / broadband HF noise that never goes away; clean studio audio goes dark)
  whine_db    : strongest steady narrow line above 10 kHz (bin power that stays up all the time,
                relative to its neighbours) - e.g. a constant 20 kHz tone
Writes data/speaker_hf_noise.json {speaker: {"hf_floor_db", "whine_db", "n"}} (medians over 30 clips).

  python hf_noise.py
"""
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from preprocess import load_audio, read_metadata

LISTS = ["data/clips.txt", "data/clips_ja.txt"] + [f"data/clips_seamless_{k}.txt" for k in range(3)]
SR, NFFT, HOP = 44100, 2048, 512


def measures(w: np.ndarray):
    n = (len(w) - NFFT) // HOP
    if n < 20:
        return None
    idx = np.arange(NFFT)[None] + HOP * np.arange(n)[:, None]
    S = np.abs(np.fft.rfft(w[idx] * np.hanning(NFFT), axis=1)) ** 2  # (frames, bins)
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    speech = 10 * np.log10(S[:, (f > 100) & (f < 4000)].sum(1) + 1e-12)
    hf = 10 * np.log10(S[:, (f > 8000) & (f < 20000)].sum(1) + 1e-12)
    loud = speech > np.percentile(speech, 70)
    hf_floor = float(np.percentile(hf, 10) - np.median(speech[loud]))
    # steady narrow lines: per-bin 20th-percentile power vs. the median of +-8 neighbouring bins
    hi = f > 10000
    p20 = 10 * np.log10(np.percentile(S[:, hi], 20, axis=0) + 1e-14)
    pad = np.pad(p20, 8, mode="edge")
    neigh = np.median(np.stack([pad[i:i + len(p20)] for i in range(17) if i != 8]), axis=0)
    whine = float((p20 - neigh).max())
    return hf_floor, whine


def run(r):
    uid, path, text, spk, a, b, ch = r
    try:
        m = measures(load_audio(path, a, b, ch).numpy())
        return spk, m
    except Exception:  # noqa: BLE001
        return spk, None


def main():
    rng = random.Random(0)
    by = defaultdict(list)
    for lst in LISTS:
        for r in read_metadata(lst, "list", ""):
            by[r[3]].append(r)
    pick = [r for s, rs in by.items() for r in rng.sample(rs, min(30, len(rs)))]
    res = defaultdict(list)
    with ProcessPoolExecutor(10) as ex:
        for spk, m in ex.map(run, pick, chunksize=20):
            if m is not None:
                res[spk].append(m)
    out = {s: {"hf_floor_db": round(float(np.median([x[0] for x in v])), 2),
               "whine_db": round(float(np.median([x[1] for x in v])), 2), "n": len(v)} for s, v in res.items()}
    json.dump(out, open("data/speaker_hf_noise.json", "w"), indent=1)
    grp = lambda s: ("Seamless" if s.startswith("si_") else "EARS" if s.startswith("ears") else
                     "Expresso" if s.startswith("expresso") else "Japanese")
    g = defaultdict(list)
    for s, v in out.items():
        g[grp(s)].append((v["hf_floor_db"], v["whine_db"]))
    for k, v in g.items():
        a = np.array(v)
        print(f"{k:9s} {len(a):3d} spk | HF floor median {np.median(a[:, 0]):6.1f} dB (worst {a[:, 0].max():6.1f}) | "
              f"whine median {np.median(a[:, 1]):5.1f} dB (worst {a[:, 1].max():5.1f})")
    print("noisiest HF floor:", ", ".join(f"{s} {v['hf_floor_db']:.0f}" for s, v in sorted(out.items(), key=lambda x: -x[1]["hf_floor_db"])[:10]))
    print("strongest whine  :", ", ".join(f"{s} {v['whine_db']:.0f}" for s, v in sorted(out.items(), key=lambda x: -x[1]["whine_db"])[:10]))
    print("si_00_0394:", out.get("si_00_0394"))


if __name__ == "__main__":
    main()
