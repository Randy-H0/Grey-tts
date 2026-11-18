"""Adds mood / dominance / whisper labels to preprocessed datasets (no audio needed).

  mood      : valence from the emotion model (sad / negative .. happy / positive)
  dominance : dominance from the emotion model (meek .. assertive)
  whisper   : 0 = normal voicing .. 1 = fully whispered, from the share of voiced frames
              relative to the speaker's usual share
mood / dominance are speaker-relative like speed: 0.5 = that speaker's median, 0 / 1 at the corpus
5th / 95th percentile of deviations. Whispered clips (whisper >= 0.5) lose their volume label:
whisper is its own control now, and volume means quiet .. shouting of the normal voice.

  python add_labels.py data/dac_norm data/dac_ja data/dac_seamless_0 data/dac_seamless_1 data/dac_seamless_2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def relative(items, get, out_key):
    by_spk = defaultdict(list)
    for it in items:
        v = get(it)
        if v is not None:
            by_spk[it["spk_name"]].append(v)
    if not by_spk:  # (no emotion model ran: Emilia since 2026-10-30 gets Energy from Kestrel-SL instead) -> no label
        for it in items:
            it[out_key] = None
        return None, None
    med = {s: float(np.median(v)) for s, v in by_spk.items()}
    rel = np.array([get(it) - med[it["spk_name"]] for it in items if get(it) is not None])
    lo, hi = np.percentile(rel, [5, 95])
    for it in items:
        v = get(it)
        if v is None:
            it[out_key] = None
            continue
        r = v - med[it["spk_name"]]
        it[out_key] = round(float(np.clip(0.5 + 0.5 * (r / hi if r >= 0 else r / -lo), 0, 1)), 4)
    return round(float(lo), 4), round(float(hi), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    data = {d: [json.loads(l) for l in open(Path(d) / "manifest.jsonl", encoding="utf-8")] for d in args.dirs}
    items = [it for v in data.values() for it in v]
    emo = lambda k: (lambda it: (it.get("emotion") or {}).get(k))
    m = relative(items, emo("valence"), "mood")
    d = relative(items, emo("dominance"), "dominance")
    # whisper: voiced share vs. the speaker's typical (median of clips that are clearly voiced)
    vf = defaultdict(list)
    for it in items:
        v = it["prosody"].get("voiced_frac")
        if v is not None and v >= 0.1:
            vf[it["spk_name"]].append(v)
    typ = {s: float(np.median(v)) for s, v in vf.items()}
    n_w = 0
    for it in items:
        v = it["prosody"].get("voiced_frac", 1.0)
        t = typ.get(it["spk_name"], 0.3)
        it["whisper"] = round(float(np.clip((1 - v / t) / 0.8, 0, 1)) if v < 0.1 else 0.0, 4)
        if it["whisper"] >= 0.5:
            it["volume"] = None  # whisper has its own control; volume = normal-voice loudness
            n_w += 1
    for dname, its in data.items():
        (Path(dname) / "manifest.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in its),
                                                    encoding="utf-8")
        c = json.loads((Path(dname) / "conditioning.json").read_text())
        c.update({"mood_map": m, "dominance_map": d})
        (Path(dname) / "conditioning.json").write_text(json.dumps(c, indent=1))
    mo = np.array([i["mood"] for i in items if i["mood"] is not None] or [float("nan")])
    print(f"{len(items)} clips: mood dev p5/p95 {m}, dominance {d}; {n_w} whispered clips "
          f"({sum(i['whisper'] > 0 for i in items)} with any whisper); mood mean {mo.mean():.2f}")


if __name__ == "__main__":
    main()
