"""Per-clip 'clean' label from full-band dirt measures (hf_noise.measures):
  hf_floor_db : 8-20 kHz hiss that never goes away (relative to speech)
  whine_db    : steady narrow tone above 10 kHz
dirt = max(hiss score, whine score), each mapped 0 at studio-typical .. 1 at clearly dirty;
clean = 1 - dirt (absolute scale: studio audio ~1). Runs on CPU; writes the label into every
manifest given (and the raw values under "dirt").

  python label_clean.py data/kvae_norm=data/clips.txt data/kvae_ja=data/clips_ja.txt ...
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from hf_noise import measures
from preprocess import load_audio, read_metadata

HISS = (-62.0, -48.0)   # dB: studio-typical .. clearly hissy
WHINE = (4.0, 18.0)     # dB above neighbours: none .. strong steady tone


def run(r):
    uid, path, text, spk, a, b, ch = r
    try:
        return uid, measures(load_audio(path, a, b, ch).numpy())
    except Exception:  # noqa: BLE001
        return uid, None


def clean_label(hf, wh):
    s1 = (hf - HISS[0]) / (HISS[1] - HISS[0])
    s2 = (wh - WHINE[0]) / (WHINE[1] - WHINE[0])
    return round(float(1 - np.clip(max(s1, s2), 0, 1)), 4)


def main():
    for pair in sys.argv[1:]:
        d, lst = pair.split("=")
        d = Path(d)
        items = [json.loads(l) for l in open(d / "manifest.jsonl", encoding="utf-8")]
        want = {it["id"] for it in items}
        rows = [r for r in read_metadata(lst, "list", "") if r[0] in want]
        res = {}
        with ProcessPoolExecutor(10) as ex:
            for k, (uid, m) in enumerate(ex.map(run, rows, chunksize=32)):
                res[uid] = m
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
