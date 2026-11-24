"""Encode a speech corpus to DAC codes and score each utterance's style.

Input formats (--format):
  lj    LJSpeech metadata.csv: `id|raw text|normalized text`, audio at <wav_dir>/<id>.wav
  list  `path|text|speaker[|start|end[|channel]]` (seconds; paths relative to --wav_dir or
        absolute). This is what prepare_data.py writes.

Output (--out): codes/<id>.npy (int16, [9, frames]), manifest.jsonl (incl. per-utterance
style score 0 = narration .. 1 = acting, and volume 0..0.1 whisper .. 1 shouting), speakers.json, speaker_styles.tsv (speakers sorted
by mean style), stats.pt (normalized DAC codebooks + per-dim latent mean/std).

Style and volume are measured on the natural recording (a whisper is quiet, a yell loud); the
audio is then brought to --encode_level_db before DAC encoding, because DAC reconstructs quiet
input noticeably worse. Loudness is restored at inference from the volume setting. Resumable: rerun to continue.
"""
import argparse
import hashlib
import os
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

import dacts  # noqa: F401
from dacts.codec import SAMPLE_RATE, LatentSpace, load_dac, normalized_codebooks
from dacts.infer import limit, speech_level_db
from dacts.style import assign_scores, assign_speed, assign_volume, prosody_features


def read_metadata(path, fmt, wav_dir):
    """Returns rows: (uid, path, text, speaker, start, end, channel)."""
    rows = []
    content = Path(path).read_text(encoding="utf-8")
    lines = content.splitlines()
    if lines and not content.endswith("\n"):
        lines = lines[:-1]  # last line may still be being written by prepare_data.py
    for line in lines:
        cols = line.strip().split("|")
        if len(cols) < 2:
            continue
        if fmt == "lj":
            text = cols[2] if len(cols) > 2 and cols[2].strip() else cols[1]
            rows.append((cols[0], Path(wav_dir) / f"{cols[0]}.wav", text, "0", None, None, None))
            continue
        p = Path(wav_dir) / cols[0]
        spk = cols[2] if len(cols) > 2 else "0"
        start = float(cols[3]) if len(cols) > 4 else None
        end = float(cols[4]) if len(cols) > 4 else None
        ch = int(cols[5]) if len(cols) > 5 and cols[5] != "" else None
        key = f"{cols[0]}|{start}|{end}|{ch}"
        uid = f"{spk}_{Path(cols[0]).stem}_{hashlib.md5(key.encode()).hexdigest()[:8]}"
        rows.append((uid, p, cols[1], spk, start, end, ch))
    seen = set()  # an interrupted prepare_data run can repeat a file's clips
    return [r for r in rows if not (r[0] in seen or seen.add(r[0]))]


def load_audio(path, start=None, end=None, channel=None, trim_db=40.0, pad_s=0.05):
    sr = sf.info(str(path)).samplerate
    a = int(start * sr) if start is not None else 0
    b = int(end * sr) if end is not None else -1
    wav, sr = sf.read(str(path), start=a, stop=b if b > 0 else None, dtype="float32", always_2d=True)
    return finish_audio(wav, sr, channel, trim_db, pad_s)


def finish_audio(wav, sr, channel=None, trim_db=40.0, pad_s=0.05):
    """load_audio after the read: (N, channels) float32 as sf.read returns it -> channel / mean, 44.1 kHz, trimmed,
    peak-limited. (11-22: split out so tools/process_onepass.py can read a clip once and reuse the same maths.)"""
    wav = torch.from_numpy(wav[:, channel] if channel is not None else wav.mean(1))
    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, sr, SAMPLE_RATE)
    # trim leading/trailing silence -- except singing clips (NO_TRIM=1): they are whole bars, downbeat to downbeat,
    # and trimming would shift them off the beat grid
    frames = wav[: len(wav) // 512 * 512].view(-1, 512)
    if len(frames) and os.environ.get("NO_TRIM") != "1":
        db = 10 * torch.log10(frames.pow(2).mean(1) + 1e-10)
        # keep soft word endings (final s / t, breaths): anything audibly above the room's noise floor
        # counts as sound, even when it is more than trim_db below the loudest frame
        floor = torch.quantile(db, 0.1) if len(db) >= 10 else db.min()
        loud = torch.nonzero(db > torch.minimum(db.max() - trim_db, floor + 8.0)).squeeze(1)
        pad = int(pad_s * SAMPLE_RATE)
        wav = wav[max(0, int(loud[0]) * 512 - pad): min(len(wav), (int(loud[-1]) + 1) * 512 + pad)]
    peak = wav.abs().max()
    return wav * (0.99 / peak) if peak > 0.99 else wav  # keep natural level; only prevent clipping


class _LabelClips(torch.utils.data.Dataset):
    """Worker side of fast labelling: load + trim a clip, measure its prosody on the CPU, resample to 16 kHz."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        uid, path, text, spk, start, end, ch = self.rows[i]
        try:
            w = load_audio(path, start, end, ch)
            return uid, text, spk, len(w), prosody_features(w, SAMPLE_RATE), AF.resample(w, SAMPLE_RATE, 16000)
        except Exception as e:  # noqa: BLE001
            return uid, None, str(e), 0, None, None


def _one_thread(_):
    torch.set_num_threads(1)  # many workers: one CPU thread each


def _emo_batch(emo, wavs):
    """emo.batch, split in halves on GPU out-of-memory (long clips, e.g. VoxPopuli speeches, next to other jobs);
    a single clip that still does not fit is scored on its first 20 s."""
    try:
        return emo.batch(wavs, 16000)
    except torch.OutOfMemoryError:
        pass
    torch.cuda.empty_cache()  # (outside the except block: the exception keeps the failed batch's tensors alive)
    if len(wavs) == 1:
        return emo.batch([wavs[0][: 20 * 16000]], 16000)
    h = len(wavs) // 2
    return _emo_batch(emo, wavs[:h]) + _emo_batch(emo, wavs[h:])


def label_fast(rows, done, emo, lang, pf, workers=8, emo_ids=None):
    """Labels without codes (KVAE era), fast: workers load audio + measure prosody in parallel on the CPU,
    the GPU scores emotion in length-sorted batches. Same item format as the one-by-one path."""
    todo = [r for r in rows if r[0] not in done]
    dl = torch.utils.data.DataLoader(_LabelClips(todo), batch_size=None, num_workers=workers, prefetch_factor=8,
                                     worker_init_fn=_one_thread)
    buf = []

    def flush():
        buf.sort(key=lambda x: x[3])
        parts, cur, secs = [], [], 0.0  # up to 16 clips and ~2 min of audio per batch: GPU memory follows length
        for x in buf:
            if cur and (len(cur) == 16 or secs + len(x[5]) / 16000 > 120):
                parts.append(cur)
                cur, secs = [], 0.0
            cur.append(x)
            secs += len(x[5]) / 16000
        if cur:
            parts.append(cur)
        for part in parts:
            want = [emo is not None and (emo_ids is None or x[0] in emo_ids) for x in part]
            got = iter(_emo_batch(emo, [x[5] for x, w in zip(part, want) if w]) if any(want) else [])
            emos = [next(got) if w else None for w in want]
            for (uid, text, spk, n, pros, _), e in zip(part, emos):
                it = {"id": uid, "text": text, "frames": n // 512, "spk_name": spk, "lang": lang, "prosody": pros}
                if e is not None:
                    it["emotion"] = e
                pf.write(json.dumps(it) + "\n")
                done[uid] = it
        pf.flush()
        buf.clear()

    for x in tqdm(dl, total=len(todo), desc="encode+score"):
        if x[1] is None:
            print(f"skip {x[0]}: {x[2]}")
            continue
        buf.append(x)
        if len(buf) >= 128:
            flush()
    if buf:
        flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--wav_dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["lj", "list"], default="list")
    ap.add_argument("--no_emotion", action="store_true", help="style score from prosody only")
    ap.add_argument("--workers", type=int, default=8, help="CPU workers for loading + prosody (fast labelling path)")
    ap.add_argument("--emotion_every", type=int, default=1, help="emotion models on every Nth clip of each voice only "
                    "(the rest: style / mood / dominance unknown = dropped labels); 3 for the huge Emilia sets")
    ap.add_argument("--no_acting", action="store_true", help="skip emotion2vec (v14: Style = Energy only; its "
                    "'acting' verdict was mostly noise -- owner 2026-10-27)")
    ap.add_argument("--emotion_max", type=int, default=0, help="and at most this many clips per voice (0 = no cap; "
                    "owner 2026-10-27: labelling 1.1M clips took ~15 h, the emotion models were the bottleneck)")
    ap.add_argument("--lang", default="en", help="language of this corpus (en, ja, es)")
    ap.add_argument("--no_codes", action="store_true", help="labels only, no DAC codes (KVAE: run encode_kvae.py after)")
    ap.add_argument("--encode_level_db", type=float, default=-20.0,
                    help="bring each clip to this speech level before DAC (DAC is less accurate on quiet input); "
                         "style/volume are still measured on the natural level")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "codes").mkdir(parents=True, exist_ok=True)
    rows = read_metadata(args.metadata, args.format, args.wav_dir)
    # stable speaker ids: existing ones never change, new speakers are appended
    spk_path = out / "speakers.json"
    speakers = json.loads(spk_path.read_text()) if spk_path.exists() else {}
    for s in sorted({r[3] for r in rows}):
        speakers.setdefault(s, len(speakers))

    partial = out / "items.partial.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            it = json.loads(line)
            done[it["id"]] = it

    model = None if args.no_codes else load_dac("cuda")
    cb = None if args.no_codes else normalized_codebooks(model)
    emo = None
    if not args.no_emotion:
        from dacts.emotion import EmotionScorer
        emo = EmotionScorer("cuda", acting=not args.no_acting)

    if args.no_codes:  # KVAE era: fast parallel labelling (no DAC codes to make)
        with open(partial, "a", encoding="utf-8") as pf:
            emo_ids = None
            if args.emotion_every > 1 or args.emotion_max:
                by = {}
                for r in rows:
                    by.setdefault(r[3], []).append(r[0])
                cap = args.emotion_max or 10**9
                emo_ids = {u for us in by.values() for k, u in enumerate(us)
                           if k % args.emotion_every == 0 and k // args.emotion_every < cap}
            label_fast(rows, done, emo, args.lang, pf, workers=args.workers, emo_ids=emo_ids)
    with open(partial, "a", encoding="utf-8") as pf:
        for uid, path, text, spk, start, end, ch in tqdm([] if args.no_codes else rows, desc="encode+score"):
            if uid in done:
                if "level_db" not in done[uid]["prosody"]:  # backfill features added later
                    try:
                        done[uid]["prosody"] = prosody_features(load_audio(path, start, end, ch).cuda(), SAMPLE_RATE)
                    except Exception as e:
                        print(f"skip backfill {uid}: {e}")
                        continue
                    pf.write(json.dumps(done[uid]) + "\n")  # later lines override earlier ones
                    pf.flush()
                continue
            try:
                wav = load_audio(path, start, end, ch)
                if args.no_codes:  # frames are set by encode_kvae.py
                    it = {"id": uid, "text": text, "frames": len(wav) // 512, "spk_name": spk, "lang": args.lang,
                          "prosody": prosody_features(wav.cuda(), SAMPLE_RATE)}
                    if emo is not None:
                        it["emotion"] = emo(wav, SAMPLE_RATE)
                    pf.write(json.dumps(it) + "\n")
                    pf.flush()
                    done[uid] = it
                    continue
                with torch.no_grad():
                    w_enc = limit(wav.numpy() * 10 ** ((args.encode_level_db - speech_level_db(wav.numpy())) / 20))
                    x = model.preprocess(torch.from_numpy(w_enc)[None, None].cuda(), SAMPLE_RATE)
                    _, codes, _, _, _ = model.encode(x)
                np.save(out / "codes" / f"{uid}.npy", codes[0].cpu().numpy().astype(np.int16))
                it = {"id": uid, "text": text, "frames": codes.shape[-1], "spk_name": spk, "lang": args.lang,
                      "prosody": prosody_features(wav.cuda(), SAMPLE_RATE)}
                if emo is not None:
                    it["emotion"] = emo(wav, SAMPLE_RATE)
            except Exception as e:
                print(f"skip {uid} ({path}): {e}")
                continue
            pf.write(json.dumps(it) + "\n")
            pf.flush()
            done[uid] = it

    items = [done[r[0]] for r in rows if r[0] in done]
    for it in items:
        it["speaker"] = speakers[it["spk_name"]]

    # style/volume are corpus-wide relative scores, so they are assigned after all files are measured
    assign_scores(items)
    cond = assign_volume(items)
    cond.update(assign_speed(items))
    cond["level_db_median"] = round(float(np.median([it["prosody"]["level_db"] for it in items])), 2)
    cond["encode_level_db"] = args.encode_level_db
    (out / "conditioning.json").write_text(json.dumps(cond, indent=1))
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for it in items:
            mf.write(json.dumps(it) + "\n")
    by_spk = {}
    for it in items:
        by_spk.setdefault(it["spk_name"], []).append(it["style"])
    # --emotion_every: unmeasured clips have style None (not scored), so the mean is over the measured ones
    table = sorted((float(np.mean(m)) if (m := [x for x in v if x is not None]) else float("nan"), nm, len(v))
                   for nm, v in by_spk.items())
    lines = ["speaker\tmean_style\tutterances"] + [f"{nm}\t{m:.3f}\t{c}" for m, nm, c in table]
    (out / "speaker_styles.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("speakers by style (0 = narration, 1 = acting):")
    for m, nm, c in (table if len(table) <= 10 else table[:5] + [(float("nan"), "...", 0)] + table[-5:]):
        print(f"  {m:.3f}  {nm}  ({c} utts)")

    if args.no_codes:
        (out / "speakers.json").write_text(json.dumps(speakers, indent=1))
        print(f"{len(items)} utterances, {len(speakers)} speakers -> {out} (no codes: run encode_kvae.py)")
        return
    # latent normalization stats over all encoded frames
    lat = LatentSpace(cb, torch.zeros(cb.shape[0] * 8), torch.ones(cb.shape[0] * 8)).cuda()
    s1 = torch.zeros(cb.shape[0] * 8, dtype=torch.float64, device="cuda")
    s2 = torch.zeros_like(s1)
    n = 0
    for it in tqdm(items, desc="latent stats"):
        codes = torch.from_numpy(np.load(out / "codes" / f"{it['id']}.npy").astype(np.int64)).cuda()
        z = lat.encode(codes[None])[0].double()
        s1 += z.sum(0)
        s2 += (z**2).sum(0)
        n += z.shape[0]
    mean = s1 / n
    std = (s2 / n - mean**2).clamp(min=1e-8).sqrt()
    torch.save({"codebooks": cb, "mean": mean.float().cpu(), "std": std.float().cpu()}, out / "stats.pt")
    (out / "speakers.json").write_text(json.dumps(speakers, indent=1))
    print(f"{len(items)} utterances, {n * 512 / SAMPLE_RATE / 3600:.2f} h, {len(speakers)} speakers -> {out}")


if __name__ == "__main__":
    main()
