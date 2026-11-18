"""Cut long solo recordings into utterances and transcribe them (faster-whisper large-v3).
Runs in the separate ASR env:  .venv-asr/Scripts/python.exe transcribe.py ...

Writes a clip list (path|text|speaker|start|end|) that preprocess.py reads. Give --lang when the
recordings are in one known language (Whisper's language detection can mislabel mixed files).

Clips are cut only inside real pauses measured in the audio (quiet for >= 0.12 s), never at
Whisper's word times: Whisper tends to end a word before its last sound (final consonants, breathy
releases) has faded. Neighbouring clips never overlap; each keeps a short pad of the pause.

  .venv-asr/Scripts/python.exe transcribe.py --speaker a tester --lang en --out data/clips_voice.txt rec1.flac rec2.wav
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MIN_S, MAX_S, MAX_GAP = 1.5, 12.0, 0.6
HOP = 0.01                      # energy frame (s)
ABOVE_FLOOR_DB = 8.0            # "still sound" = this far above the noise floor
MAX_EXTEND, PAD, GUARD = 1.0, 0.12, 0.04  # how far a boundary may move, extra pad, gap kept to the neighbour


def energy_db(audio, sr=16000):
    n = int(HOP * sr)
    fr = audio[: len(audio) // n * n].reshape(-1, n)
    return 10 * np.log10((fr ** 2).mean(1) + 1e-12)


def silences(db, floor, lo, hi, min_len=0.12):
    """Longest quiet run (s) inside [lo, hi]: (start, end) or None."""
    q = db < floor + ABOVE_FLOOR_DB
    a, b = max(0, int(lo / HOP)), min(len(q), int(hi / HOP))
    best, run_start = None, None
    for k in range(a, b + 1):
        if k < b and q[k]:
            run_start = k if run_start is None else run_start
        elif run_start is not None:
            if best is None or k - run_start > best[1] - best[0]:
                best = (run_start, k)
            run_start = None
    if best and (best[1] - best[0]) * HOP >= min_len:
        return best[0] * HOP, best[1] * HOP
    return None


def load_16k(path):
    """Any length / rate / channel count -> 16 kHz mono float32, read in blocks (multi-hour files fit in memory)."""
    from math import gcd

    sr = sf.info(str(path)).samplerate
    g = gcd(16000, sr)
    out = []
    for blk in sf.blocks(str(path), blocksize=sr * 30, dtype="float32", always_2d=True):
        out.append(resample_poly(blk.mean(1), 16000 // g, sr // g).astype("float32"))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def _bad(seg) -> bool:
    """Whisper unsure or hallucinating (repetitions, non-speech): drop the sentences it is in."""
    return seg.avg_logprob < -1.0 or seg.compression_ratio > 2.4 or seg.no_speech_prob > 0.6


def utterances(model, path, lang, recheck_lang=True, batched=None, batch_size=16):
    """Sentences cut only inside real pauses (measured in the audio), so no word loses its ending:
    Whisper's word times end early (final consonants, breaths), so they only suggest where to look.
    Each sentence gets its own language (a file may switch language) and a 'bad' flag (unreliable text)."""
    audio = load_16k(path)
    dur = len(audio) / 16000
    if dur < 1.0:
        return
    if batched is not None:  # many VAD chunks per GPU pass: ~9x faster on long recordings, same words
        segs, info = batched.transcribe(audio, language=lang, word_timestamps=True, beam_size=5,
                                        condition_on_previous_text=False, batch_size=batch_size)
    else:
        segs, info = model.transcribe(audio, language=lang, vad_filter=True, word_timestamps=True, beam_size=5,
                                      condition_on_previous_text=False)
    words = [(w, _bad(s)) for s in segs for w in (s.words or [])]
    if not words:
        return
    db = energy_db(audio)
    floor = float(np.percentile(db, 10))
    # pause after each word (searched from its start: Whisper's end can sit before the real end)
    gaps = [silences(db, floor, words[k][0].start + 0.05, words[k + 1][0].start + 0.05) for k in range(len(words) - 1)] + [None]
    cur = None
    head = silences(db, floor, max(0.0, words[0][0].start - MAX_EXTEND), words[0][0].start + 0.02)
    cut_start = max(0.0, (head[1] if head else words[0][0].start) - PAD)
    for k, (w, bad) in enumerate(words):
        if cur is None:
            cur = {"start": cut_start, "text": w.word.strip(), "first": w.start, "lang": info.language, "bad": bad}
        else:
            cur["text"] += w.word
            cur["bad"] = cur["bad"] or bad
        g = gaps[k]
        length = w.end - cur["first"]
        sentence_end = w.word.rstrip().endswith((".", "?", "!", "。", "？", "！"))
        nxt_len = (words[k + 1][0].end - cur["first"]) if k + 1 < len(words) else 0
        long_pause = g is not None and g[1] - g[0] >= MAX_GAP
        if k + 1 == len(words) or (g is not None and (long_pause or (sentence_end and length >= 3.0) or nxt_len > MAX_S)):
            if g is None:  # last word: extend through its tail
                tail = silences(db, floor, w.start + 0.05, min(dur, w.end + MAX_EXTEND))
                end = min(dur, (tail[0] if tail else min(dur, w.end + 0.3)) + PAD)
            else:
                mid = (g[0] + g[1]) / 2
                end = min(g[0] + PAD, mid)
                cut_start = max(g[1] - PAD, mid)
            cur["cut"] = (cur["start"], end)
            if lang is None and recheck_lang and end - cur["start"] >= 1.0:
                # the file's language was guessed from its start; check this sentence on its own
                chunk = audio[int(cur["start"] * 16000): int(end * 16000)]
                try:
                    l2, p2, _ = model.detect_language(chunk)
                except Exception:  # noqa: BLE001
                    l2, p2 = cur["lang"], 0.0
                if l2 != cur["lang"] and p2 >= 0.8:
                    seg2, _ = model.transcribe(chunk, language=l2, beam_size=5, condition_on_previous_text=False)
                    seg2 = list(seg2)
                    cur["text"] = " ".join(s.text.strip() for s in seg2)
                    cur["lang"] = l2
                    cur["bad"] = any(_bad(s) for s in seg2) or not cur["text"]
            yield cur
            cur = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--lang", default=None, help="en / ja / es ... (None = detect per file)")
    ap.add_argument("--out", required=True, help="clip list; with {lang} in it: one list per language found")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="large-v3", help="large-v3-turbo: ~4x faster, ~96%% the same words")
    ap.add_argument("--batch", type=int, default=0, help="> 0: batched inference (long recordings)")
    a = ap.parse_args()
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # (only needed for transcribing)

    model = WhisperModel(a.model, device=a.device, compute_type="float16" if a.device == "cuda" else "int8")
    batched = BatchedInferencePipeline(model) if a.batch > 0 else None
    rows, total = [], 0.0
    for f in a.files:
        for u in utterances(model, f, a.lang, batched=batched, batch_size=a.batch):
            if u["cut"][1] - u["cut"][0] >= MIN_S and u["text"].strip() and not u["bad"]:
                s, e = u["cut"]
                rows.append((a.lang or u["lang"], f"{Path(f).resolve()}|{u['text'].strip()}|{a.speaker}|{s:.2f}|{e:.2f}|"))
                total += e - s
        print(f"{f}: {len(rows)} clips so far", flush=True)
    # --out with {lang} in it (Grey Studio): one clip list per language, each sentence by its own detected language
    outs = sorted({lg for lg, _ in rows}) if "{lang}" in a.out else [None]
    for lg in outs:
        part = [r for l2, r in rows if lg is None or l2 == lg]
        Path(a.out.replace("{lang}", lg or "")).write_text("\n".join(part) + "\n", encoding="utf-8")
    print(f"{len(rows)} clips, {total / 60:.1f} min -> {a.out}" + (f" ({', '.join(outs)})" if outs != [None] else ""))


if __name__ == "__main__":
    main()
