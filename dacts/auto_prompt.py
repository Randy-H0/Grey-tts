"""Any reference recording -> a prompt for prompted cloning, with no typing.

Finds 4-10 s stretches that start and end in real pauses (no cut words), keeps the one that sounds cleanest
(UTMOS naturalness: background music, sound effects, clipping or mangled delivery in the prompt carry straight
into every generated line, so this matters more than anything else), transcribes it with Whisper (transformers, large-v3-turbo; downloaded once), and encodes it like the
training data (Synthesizer.make_prompt).
"""
import numpy as np
import torch
import torchaudio.functional as AF

ASR_MODEL = "openai/whisper-large-v3-turbo"
_asr = {}


def pick_segment(wav44: torch.Tensor, lo: float = 4.0, hi: float = 10.0):
    """(start, end) seconds of the best 4-10 s stretch bounded by pauses, or None."""
    c = candidates(wav44, lo, hi, 1)
    return c[0] if c else None


def candidates(wav44: torch.Tensor, lo: float = 4.0, hi: float = 10.0, k: int = 8):
    """Up to k non-overlapping 4-10 s stretches bounded by pauses, best (most speech) first."""
    import transcribe as tr  # pause helpers (energy / silences)

    x = AF.resample(wav44.float(), 44100, 16000).numpy()
    db = tr.energy_db(x)
    floor = float(np.percentile(db, 10))
    loud = db > floor + tr.ABOVE_FLOOR_DB
    quiet = ~loud
    cuts, run = [0.0], None  # cut points: middles of quiet runs >= 0.15 s (+ the ends)
    for k, q in enumerate(np.append(quiet, False)):
        if q and run is None:
            run = k
        elif not q and run is not None:
            if (k - run) * tr.HOP >= 0.15:
                cuts.append((run + k) / 2 * tr.HOP)
            run = None
    cuts.append(len(x) / 16000)
    segs = []
    for i, s in enumerate(cuts):
        for e in cuts[i + 1:]:
            if e - s > hi:
                break
            if e - s >= lo:
                a, b = int(s / tr.HOP), int(e / tr.HOP)
                segs.append((loud[a:b].mean() + 0.02 * (e - s), s, e))  # mostly speech, longer is better
    out = []
    for _, s, e in sorted(segs, reverse=True):
        if len(out) < k and all(e <= s2 or s >= e2 for s2, e2 in out):
            out.append((s, e))
    if not out and len(x) / 16000 >= 2.0:  # no pause structure: just take up to 10 s
        out = [(0.0, min(hi, len(x) / 16000))]
    return out


@torch.no_grad()
def naturalness(clip44: torch.Tensor, device: str = "cpu") -> float:
    """UTMOS (1-5) of a clip; None if the scorer can't be loaded (offline first run)."""
    if "mos" not in _asr:
        try:
            _asr["mos"] = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True,
                                         verbose=False).eval().to(device)
        except Exception:  # noqa: BLE001
            _asr["mos"] = None
    m = _asr["mos"]
    if m is None:
        return None
    x = AF.resample(clip44.float(), 44100, 16000)[None].to(next(m.parameters()).device)
    return float(m(x, 16000))


def _pipe(device):
    if "pipe" not in _asr:
        from transformers import pipeline

        _asr["pipe"] = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device,
                                torch_dtype=torch.float32)  # fp32 + beam search: the prompt text must match its audio
                                                            # exactly (fp16 greedy heard "a beast would" for "a bee should")
    return _asr["pipe"]


def _fw(device):
    """(Grey Studio, 11-17) faster-whisper large-v3-turbo -- the copy the Train tab already ships -- instead of the
    transformers copy of the same model: one 1.6 GB recogniser in the download, not two. fp32 + beam 5 as below."""
    if "fw" not in _asr:
        from dacts.cuda_dlls import expose_torch_dlls
        expose_torch_dlls()  # (11-18: cuBLAS / cuDNN from PyTorch's own folder: PCs without the CUDA toolkit)
        from faster_whisper import WhisperModel
        gpu = str(device).startswith("cuda") and torch.cuda.is_available()
        small = gpu and torch.cuda.get_device_properties(0).total_memory < 10 * 2 ** 30  # (11-17: 4-8 GB cards)
        _asr["fw"] = WhisperModel("large-v3-turbo", device="cuda" if gpu else "cpu",
                                  compute_type=("int8_float16" if small else "float32") if gpu else "int8")
    return _asr["fw"]


def _has_fw() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:  # (the repo's main venv: faster-whisper lives in .venv-asr; the built app has it)
        return False


USE_FW = bool(__import__("os").environ.get("GREY_APP")) and _has_fw()  # (the app: faster-whisper; the website keeps transformers)


@torch.no_grad()
def spoken_language(wav44: torch.Tensor, device: str = "cuda"):
    """Whisper's language id over (up to) the first 30 s of the whole reference, e.g. "nl"; None if unsure.
    A 4-10 s prompt stretch alone is too short: owner's Dutch reference with a few English words was taken for
    English and the prompt text came out as an English TRANSLATION of what he said."""
    try:
        if USE_FW:
            x = AF.resample(wav44[: 30 * 44100].float(), 44100, 16000).numpy()
            lang, prob, _ = _fw(device).detect_language(x)
            return lang or None
        pipe = _pipe(device)
        x = AF.resample(wav44[: 30 * 44100].float(), 44100, 16000).numpy()
        feats = pipe.feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(pipe.model.device, dtype=pipe.model.dtype)
        tok = pipe.tokenizer.decode(pipe.model.detect_language(feats)[0])  # "<|nl|>"
        return tok.strip("<|>") or None
    except Exception:  # noqa: BLE001
        return None


@torch.no_grad()
def transcribe(wav44: torch.Tensor, device: str = "cuda", language=None) -> str:
    x = AF.resample(wav44.float(), 44100, 16000).numpy()
    if USE_FW:
        segs, _ = _fw(device).transcribe(x, language=language, beam_size=5, task="transcribe", condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()
    kw = {"task": "transcribe", "num_beams": 5, **({"language": language} if language else {})}
    return _pipe(device)({"raw": x, "sampling_rate": 16000}, generate_kwargs=kw)["text"].strip()


# Longest prompt (owner 2026-10-29: "do everything you can to make the voice clone rate go higher"): v14XL, 65 voices
# whose prompt could grow from one clip (5.7 s) to ~13 s of the same voice, scored against a recording neither prompt
# held: SIM 0.451 -> 0.545 (+0.095 +- 0.013, better on 57/65). Training saw prompts up to 15 s (cross_max_sec); was 10.
PROMPT_MAX_S = 15.0


def auto_prompt(synth, wav44: torch.Tensor, device: str = "cuda"):
    """Reference recording (44.1 kHz mono) -> {"latents", "text", "lang"} for Synthesizer(prompt=...), or None."""
    from . import text as T

    segs = candidates(wav44, hi=PROMPT_MAX_S)
    if not segs:
        return None
    longest = max(e - s for s, e in segs)  # (the cleanest of the LONG stretches: length is what buys voice match)
    segs = [(s, e) for s, e in segs if e - s >= 0.8 * longest]
    seg, best = segs[0], None
    if len(segs) > 1:  # the cleanest-sounding stretch, not just the one with the most speech
        for s, e in segs:
            q = naturalness(wav44[int(s * 44100): int(e * 44100)], device)
            if q is None:
                break
            if best is None or q > best:
                seg, best = (s, e), q
    clip = wav44[int(seg[0] * 44100): int(seg[1] * 44100)]
    # (you 2026-11-05: clones "always start with an eu oh ah noise"): candidates are cut in the MIDDLE of pauses (and the
    # first one at the file's start), so the reference kept up to a second of silence / room noise its words don't cover,
    # and every take began with ~1 s of sound before its first word (measured: 0.8-1.6 s on 9 of 9 takes; none from a
    # trimmed clip). The model continues its reference: no silence in it but a little padding.
    t0, t1 = _trim(clip)
    clip = clip[int(t0 * 44100): int(t1 * 44100)]
    seg = (seg[0] + t0, seg[0] + t0 + len(clip) / 44100)
    spoken = spoken_language(wav44, device)  # the whole reference's language, forced for the prompt stretch
    words = transcribe(clip, device, language=spoken)
    if not words:
        return None
    lang = spoken if spoken in T.langs_of(getattr(synth.model.cfg, 'n_langs', 8)) else T.detect_lang(words, T.langs_of(getattr(synth.model.cfg, 'n_langs', 8)))  # (only languages this model reads)
    p = synth.make_prompt(clip, words, lang=lang, max_seconds=PROMPT_MAX_S + 0.5)
    return {**p, "latents": p["latents"].cpu(), "seconds": round(seg[1] - seg[0], 2),
            "naturalness": None if best is None else round(best, 2)}


def _trim(wav44: torch.Tensor, pad: float = 0.12):
    """(start, end) seconds of the sound in a clip, without the silence before/after it (+ a little padding)."""
    import transcribe as tr

    x = AF.resample(wav44.float(), 44100, 16000).numpy()
    db = tr.energy_db(x)
    loud = np.flatnonzero(db > float(np.percentile(db, 10)) + tr.ABOVE_FLOOR_DB)
    if not len(loud):
        return 0.0, len(wav44) / 44100
    return max(0.0, loud[0] * tr.HOP - pad), min(len(wav44) / 44100, (loud[-1] + 1) * tr.HOP + pad)


JOIN_GAP = 0.35  # s of pause kept between two read lines joined into one reference


def _said(clip44: torch.Tensor, text: str, lang, device: str = "cuda") -> str:
    """The words of a read line as they were really said (you 2026-11-05: a clone's test clips came out as "hi (gibberish)
    this is how I sound (gibberish)": the reference's text and audio didn't match -- the site trusted the sentence it had
    asked for, and anything else said in the recording became mumbling). Whisper hears the line; if its words clearly
    differ from the script's (match < 80 %), they are used instead."""
    import difflib
    import re
    try:
        heard = transcribe(clip44, device, language=lang or None)
    except Exception:  # noqa: BLE001
        return text
    if lang in ("ja", "zh"):
        norm = lambda t: [c for c in t if c.isalnum()]  # noqa: E731
    else:
        norm = lambda t: re.findall(r"\w+", t.lower())  # noqa: E731
    a, b = norm(text), norm(heard)
    if not b:
        return text
    return text if difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.8 else heard


@torch.no_grad()
def script_prompt(synth, wav44: torch.Tensor, script, device: str = "cuda", lo: float = 4.0, hi: float = PROMPT_MAX_S):
    """Recordings made by reading known sentences -> prompt, with no speech recognition at all.
    script: [{"start", "end", "text", "lang", "style"?}] (seconds into wav44), one per recorded sentence, in order.
    Candidates are runs of consecutive sentences spanning lo-hi s; the cleanest-sounding (UTMOS) wins.
    style (the site asks for each line to be read a certain way): the prompt sets how every take is delivered, so it
    comes from normal / story / calm lines; else any line but a whispered one (a whispered prompt whispers every take;
    the model never trained on whispered prompts). A run never spans a line it may not use."""
    from . import text as T

    segs = []
    for s in script:
        a, b = float(s["start"]), float(s["end"])
        t0, t1 = _trim(wav44[int(a * 44100): int(b * 44100)])
        if t1 - t0 >= 0.5 and str(s.get("text", "")).strip():
            segs.append((a + t0, a + t1, str(s["text"]).strip(), s.get("lang"), str(s.get("style") or "")))
    if not segs:
        return None

    def runs_for(ok):
        out = []
        for i in range(len(segs)):
            if not ok(segs[i]):
                continue
            for j in range(i, len(segs)):
                if j > i and (not ok(segs[j]) or segs[j][0] - segs[j - 1][1] > 2.0):  # (not adjacent audio)
                    break
                span = segs[j][1] - segs[i][0]
                if span > hi:
                    break
                if span >= lo:
                    out.append((i, j))
        return out
    neutral = lambda s: s[4] in ("", "normal", "narrated", "calm")  # noqa: E731
    usable = lambda s: s[4] != "whisper"  # noqa: E731
    runs = runs_for(neutral) or runs_for(usable)
    if not runs:  # short sentences only: the longest single usable one
        cand = [k for k in range(len(segs)) if neutral(segs[k])] or [k for k in range(len(segs)) if usable(segs[k])]
        if not cand:
            return None
        k = max(cand, key=lambda k: min(hi, segs[k][1] - segs[k][0]))
        runs = [(k, k)]
    span = lambda r: segs[r[1]][1] - segs[r[0]][0]  # noqa: E731
    longest = max(map(span, runs))  # (the cleanest of the LONG runs: length is what buys voice match, PROMPT_MAX_S)
    runs = [r for r in runs if span(r) >= 0.8 * longest]
    best, pick = None, runs[0]
    if len(runs) > 1:
        for i, j in runs[:12]:
            q = naturalness(wav44[int(segs[i][0] * 44100): int(segs[j][1] * 44100)], device)
            if q is None:
                break
            if best is None or q > best:
                best, pick = q, (i, j)
    i, j = pick
    lang = segs[i][3] if segs[i][3] in T.LANGS else None
    sep = "" if lang in ("ja", "zh") else " "
    # (you 2026-11-05: "the test clips say hi (???) this is how I sound (??)"): the lines are joined with at most JOIN_GAP s
    # of the room's own sound between them (was: the whole pause, up to 2 s, which no word covers), and each line's words
    # are what was really said (_said: the script's unless the recording clearly differs from it)
    parts, texts = [], []
    for k in range(i, j + 1):
        a, b = segs[k][0], segs[k][1]
        if k > i:
            e = segs[k - 1][1]
            parts.append(wav44[int(e * 44100): int((e + min(JOIN_GAP, max(0.0, a - e))) * 44100)])
        parts.append(wav44[int(a * 44100): int(b * 44100)])
        texts.append(_said(parts[-1], segs[k][2], lang, device))
    words = sep.join(texts)
    lang = lang or T.detect_lang(words, T.langs_of(getattr(synth.model.cfg, 'n_langs', 8)))
    clip = torch.cat(parts)
    p = synth.make_prompt(clip, words, lang=lang, max_seconds=max(hi, PROMPT_MAX_S) + 0.5)  # (hi: the 30 s long prompt)
    return {**p, "latents": p["latents"].cpu(), "seconds": round(len(clip) / 44100, 2),
            "naturalness": None if best is None else round(best, 2), "from_script": True}


# the deliveries the site asks for ("Read this: excited", ...; dashboard/server.py READ_STYLES)
EXPRESSIONS = ("normal", "narrated", "calm", "happy", "excited", "surprised", "sad", "worried", "annoyed", "whisper")


@torch.no_grad()
def expression_prompts(synth, wav44: torch.Tensor, script, hi: float = 7.0, gap: float = 0.3):
    """Recordings made by reading known sentences, each asked in a delivery -> ({delivery: prompt}, {delivery: s}).
    Per delivery: its lines (silence trimmed) joined in reading order with `gap` s between them, up to `hi` s, as one
    reference clip (latents in float16: a voice file carries up to ten of them). The asked delivery is trusted.
    seconds: every line read in that delivery, whole, as the recording screen counts them (what the site shows as how
    complete the voice is; 2026-10-24: counting only the silence-trimmed speech, a voice recorded to 100% showed 98%)."""
    from . import text as T

    by, read = {}, {}
    for s in script or []:
        st = s.get("style") or "normal"
        if st not in EXPRESSIONS or not str(s.get("text", "")).strip():
            continue
        a, b = float(s["start"]), float(s["end"])
        read[st] = read.get(st, 0.0) + max(0.0, b - a)
        t0, t1 = _trim(wav44[int(a * 44100): int(b * 44100)])
        if t1 - t0 >= 0.6:
            by.setdefault(st, []).append((a + t0, a + t1, str(s["text"]).strip(), s.get("lang")))
    prompts, secs = {}, {st: round(v, 1) for st, v in read.items() if v > 0}
    for st, segs in by.items():
        pick, used = [], 0.0
        for seg in sorted(segs, key=lambda x: x[1] - x[0]) if len(segs) > 1 else segs:  # (short lines first: more fit)
            d = seg[1] - seg[0]
            if pick and used + gap + d > hi:
                continue
            pick.append(seg)
            used += d + (gap if len(pick) > 1 else 0)
        pick.sort()  # reading order
        lang = pick[0][3] if pick[0][3] in T.LANGS else None
        sep = "" if lang in ("ja", "zh") else " "
        dev = str(getattr(synth, "device", "cuda"))
        words = sep.join(_said(wav44[int(p[0] * 44100): int(p[1] * 44100)], p[2], lang, dev) for p in pick)
        lang = lang or T.detect_lang(words, T.langs_of(getattr(synth.model.cfg, 'n_langs', 8)))
        sil = torch.zeros(int(gap * 44100), dtype=wav44.dtype)
        parts = []
        for k, (a, b, *_) in enumerate(pick):
            parts += ([sil] if k else []) + [wav44[int(a * 44100): int(b * 44100)]]
        clip = torch.cat(parts)
        p = synth.make_prompt(clip, words, lang=lang, max_seconds=hi + 3.5)
        prompts[st] = {"latents": p["latents"].detach().cpu().half(), "text": words, "lang": lang,
                       "seconds": round(len(clip) / 44100, 2)}
    return prompts, secs

