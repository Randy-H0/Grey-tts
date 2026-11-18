"""Whisper labels from the source instead of the voicing detector.

add_labels.py called a clip "whispered" when the pitch tracker found almost no voiced frames. On real
whisper recordings that's right, but it also fired on ~9,500 ordinary clips (quiet, creaky, noisy or
mumbled speech in Seamless / VCTK / EARS emotion sets): two thirds of all "whisper" examples were normal
speech, so the label barely meant whisper and only the few voices with real whisper data learned it.

Now: whisper = 1 only for recordings that ARE whispers (Expresso + EARS "whisper" sessions, JVS
whisper10); detector-only whispers elsewhere become "unknown" (None) and get their clean label back.
Manifests are backed up to manifest.jsonl.bak_whisper first.

  python fix_whisper_labels.py data/kvae_norm data/kvae_jvs ...
"""
import json
import shutil
import sys
from pathlib import Path

from preprocess import read_metadata


def real_whisper_ids():
    ids = set()
    for lst in Path("data").glob("clips*.txt"):
        if "jvs" in lst.name:
            ids |= {r[0] for r in read_metadata(str(lst), "list", "") if "whisper10" in str(r[1]).replace("\\", "/")}
    return ids


def main():
    jvs = real_whisper_ids()
    for d in map(Path, sys.argv[1:]):
        mf = d / "manifest.jsonl"
        bak = d / "manifest.jsonl.bak_whisper"
        items = [json.loads(l) for l in open(bak if bak.exists() else mf, encoding="utf-8")]  # always from the original
        real = fake = missed = 0
        for it in items:
            is_real = it["id"] in jvs or ("_whisper_" in it["id"] and str(it.get("spk_name", "")).startswith(("expresso", "ears")))
            w = it.get("whisper")
            if is_real:
                missed += (w or 0) < 0.5
                it["whisper"], it["volume"] = 1.0, None
                if it.get("clean") is not None:  # quiet != dirty: whispers carry no clean label
                    it["clean_measured"], it["clean"] = it["clean"], None
                real += 1
            elif w is not None and w > 0:
                it["whisper_measured"] = w
                if w < 0.5:  # a partial detector hit on ordinary speech
                    it["whisper"] = 0.0
                    continue
                fake += 1
                it["whisper"] = None
                if it.get("clean") is None and it.get("clean_measured") is not None:
                    it["clean"] = it.pop("clean_measured")
        if not bak.exists():
            shutil.copy(mf, bak)
        mf.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        print(f"{d}: {real} real whispers ({missed} the detector had missed), {fake} false whispers -> unknown", flush=True)


if __name__ == "__main__":
    main()
