"""Speech editing: change words in a real recording; the rest of it stays the recording itself.

    from dacts.infer import Synthesizer
    from dacts.edit import edit_speech
    synth = Synthesizer.from_checkpoint("models/v16plus.pt")
    out, info = edit_speech(synth, "talk.wav", "She regularly did yoga to maintain her strength.")

How (worked out on 8 unheard voices with v18, 11-24; tools/edit_infill_demo.py + tools/edit_splice_ctc.py were the bench):
- Transcribe (Whisper) unless the old transcript is given; the new text is compared word by word, each changed stretch is
  one edit (edits closer than 2 words merge).
- Word times from a CTC forced aligner (letters, 20 ms frames; hubert-large-ls960-ft, English letters; Whisper's own word
  times were 100-300 ms off).
- Grey regenerates a PHRASE, never a lone word (one word between fixed audio: 3/8 right vs 8/8): the changed word(s) + 1-2
  neighbours each side (the first always, more only up to a pause) + one SPARE word each side that is said again and
  thrown away (the word right next to the fixed audio comes out damaged).
- The text goes in as ONE sentence (the words before + a space + the phrase): glued without the space it is training's
  "a new clip starts here" layout and Grey paused, ran late, rushed and dropped words (Solver 11-24).
- The recording is encoded once and sliced at the cuts (both sides from one encode, as in training); the audio before is
  pinned, the audio after is pinned and given as context; each phrase word gets its REAL length from the recording, only
  the changed word a new one (Grey's rhythm model x this speaker's pace).
- Joins: in the take, the phrase is found by aligning it too; the real side keeps its whole word + half the recording's
  gap there, Grey's side starts half a gap early (>= 30 ms each: no "sudden start"); crossfades centred on the cut
  (25 ms in words, 120 ms in quiet); comfort noise where Grey's take is quieter than the recording's own background, never
  above it (Grey's pauses are dead silent; strung-together scraps of room noise clicked).
English only for now (the aligner's letters). Outputs should be marked as AI-edited (dacts.marking) wherever they are saved.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

SR = 44100
SPF = 882  # output samples per latent frame (44.1 kHz / 50 Hz)
ALIGNER = "facebook/hubert-large-ls960-ft"  # ("facebook/wav2vec2-base-960h": 360 MB, 90 % of boundaries within 20 ms, worst 80 ms)
_al: dict = {}


def _norm(s: str) -> str:
    return re.sub(r"[^\w']", "", s.lower())


# ---------------------------------------------------------------------------------------------------- alignment


class Aligner:
    """CTC forced alignment of known words (English letters). align() -> [(word, start s, end s)], the "|" gap between
    each pair of words (start s, end s) and each word's mean log-probability (a word that is there reads ~ -0.1, one that
    is not ~ -10)."""

    def __init__(self, name: str = ALIGNER, device: str = "cpu"):
        from transformers import AutoModelForCTC, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForCTC.from_pretrained(name).eval().to(device)
        self.device = device
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id

    @torch.no_grad()
    def align(self, wav, sr: int, words: list[str]):
        w16 = AF.resample(torch.as_tensor(np.ascontiguousarray(wav), dtype=torch.float32), sr, 16000)
        letters = [re.sub(r"[^A-Z']", "", t.upper()) or "A" for t in words]
        toks, owner = [], []
        for i, l in enumerate(letters):
            if i:
                toks.append(self.vocab["|"]); owner.append(-1)
            for c in l:
                toks.append(self.vocab[c]); owner.append(i)
        x = ((w16 - w16.mean()) / (w16.std() + 1e-7))[None].to(self.device)
        lp = self.model(x).logits.log_softmax(-1).float().cpu()
        if lp.shape[1] < len(toks):
            raise ValueError("the recording is too short for its words")
        path = AF.forced_align(lp, torch.tensor([toks]), blank=self.blank)[0][0].tolist()
        spf = len(w16) / 16000 / lp.shape[1]
        k, first, last, sep, sc, prev = -1, {}, {}, {}, {}, self.blank
        for f, p in enumerate(path):
            if p != self.blank and (p != prev or k < 0):
                k += 1
            if p != self.blank and owner[k] >= 0:
                first.setdefault(owner[k], f)
                last[owner[k]] = f
                sc.setdefault(owner[k], []).append(float(lp[0, f, p]))
            elif p != self.blank:
                sep.setdefault(k, [f, f])[1] = f
            prev = p
        out = [(words[i], first[i] * spf, (last[i] + 1) * spf) for i in range(len(words))]
        seps = [(f0 * spf, (f1 + 1) * spf) for _, (f0, f1) in sorted(sep.items())]
        scores = [sum(sc[i]) / len(sc[i]) for i in range(len(words))]
        return out, seps, scores


def aligner(device: str = "cpu", name: str = ALIGNER) -> Aligner:
    if name not in _al:
        _al[name] = Aligner(name, device)
    return _al[name]


# ---------------------------------------------------------------------------------------------------- helpers


def _quiet_point(w, lo, hi, win=0.01):
    h = int(win * SR)
    a, b = max(h, int(lo * SR)), min(len(w) - h, int(hi * SR))
    if b <= a:
        return int((lo + hi) / 2 * SR)
    e = [float(np.mean(w[i - h // 2:i + h // 2] ** 2)) for i in range(a, b, max(1, h // 4))]
    return a + int(np.argmin(e)) * max(1, h // 4)


def _cut_before(w, prev_end, start):
    lo, hi = prev_end + 0.02, start - 0.03
    return _quiet_point(w, lo, hi) if hi - lo >= 0.01 else _quiet_point(w, min(prev_end, start), max(prev_end, start) + 0.005)


def _window(ws, i, j, k):
    """the regenerated phrase around ws[i..j]: 1..k neighbours each side (the first always, more only up to a >= 0.1 s
    pause), >= 1 real word kept each side"""
    L, R = i, j
    for n in range(k):
        if L > 1 and (n == 0 or ws[L][1] - ws[L - 1][2] < 0.1):
            L -= 1
    for n in range(k):
        if R + 2 < len(ws) and (n == 0 or ws[R + 1][1] - ws[R][2] < 0.1):
            R += 1
    return L, R


def diff_words(old: list[str], new: list[str]):
    """[(i, j, new_words)]: old[i..j] (inclusive; j = i - 1 for an insertion) becomes new_words, merged when < 2 words apart"""
    sm = difflib.SequenceMatcher(a=[_norm(t) for t in old], b=[_norm(t) for t in new], autojunk=False)
    ed = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ed.append([i1, i2 - 1, j1, j2 - 1])
    merged = []
    for e in ed:
        if merged and e[0] - merged[-1][1] <= 2:
            merged[-1][1], merged[-1][3] = e[1], e[3]
        else:
            merged.append(e)
    return [(i1, i2, new[j1:j2 + 1]) for i1, i2, j1, j2 in merged]


# ---------------------------------------------------------------------------------------------------- one edit


def _edit_once(synth, w, ws, i, j, new_words, lang="en", ctx_words=2, spare=1, seed=0, al=None, log=print, soft=0.25):
    """Replace ws[i..j] (word times of the recording w, 44.1 kHz) by new_words. Returns the edited recording."""
    from . import text as T
    if not new_words:
        raise ValueError("removing words is not supported yet: type the replacement")
    if i < 1 or j > len(ws) - 3:
        raise ValueError("an edit needs at least one word before it and two after it in the recording")
    blank = synth.model.cfg.add_blank
    label = " ".join(new_words)
    oi, oj = i, j
    L, R = _window(ws, i, j, ctx_words)
    R, L = min(R + spare, len(ws) - 2), max(1, L - spare)
    phrase = [t[0] for t in ws[L:i]] + list(new_words) + [t[0] for t in ws[j + 1:R + 1]]
    new = " ".join(phrase)
    i, j = L, R
    before = " ".join(t[0] for t in ws[:i])
    after = " ".join(t[0] for t in ws[j + 1:])
    # cuts in the recording (snapped to latent frames: both sides come from one encode, sliced)
    a = _cut_before(w, ws[i - 1][2], ws[i][1])
    nxt = ws[j + 1][1]
    b = _quiet_point(w, ws[j][2] - 0.01, max(ws[j][2] + 0.06, nxt + 0.02))
    af, bf = round(a / SPF), round(b / SPF)
    a, b = af * SPF, bf * SPF
    voice = "__edit__"
    synth.add_voice(voice, torch.from_numpy(w.copy()), SR)
    whole = synth.make_prompt(torch.from_numpy(w.copy()), " ".join(t[0] for t in ws), lang=lang, max_seconds=600.0, sr=SR)
    prompt = {**whole, "latents": whole["latents"][:af], "text": before, "one_sentence": True}
    suf = whole["latents"][bf:]
    line = f"{new} {after}".strip()
    pb = T.encode(before, blank, lang)
    E = lambda t: T.encode(before + " " + t, blank, lang)[len(pb):]  # noqa: E731  (token ids in the one-sentence layout)
    full = E(line)
    part = E(new + " ")
    n_new = 0
    while n_new < min(len(full), len(part)) and full[n_new] == part[n_new]:
        n_new += 1

    def word_ms(sent):
        d = synth(sent, speaker=voice, lang=lang, durations_only=True)
        return [x["ms"] for x in d if "word" in x]
    # this speaker's pace: their real word times / the rhythm model's, for the original sentence
    sent_old = " ".join(t[0] for t in ws)
    pred_old, real_old = word_ms(sent_old), [(t[2] - t[1]) * 1000 for t in ws]
    n_ = min(len(pred_old), len(real_old))
    pace = sum(real_old[:n_]) / max(1.0, sum(pred_old[:n_]))
    pred_new = word_ms(" ".join(x for x in (before, new, after) if x))
    nb = len(before.split()) if before else 0
    eo, n_lab, n_old = oi - i, len(new_words), oj - oi + 1
    new_ms = sum(pred_new[nb + eo:nb + eo + n_lab]) * pace
    # (11-24, you: "mine speeds up a lot") the rhythm model runs short for a word on its own ("general" 0.26 s for
    # "president"'s 0.46): at least the OLD word's real length in this recording, scaled by syllables
    syl = lambda t: max(1, len(re.findall(r"[aeiouy]+", t.lower().rstrip("e")) or [1]))  # noqa: E731
    old_ms = (ws[oj][2] - ws[oi][1]) * 1000
    new_ms = max(new_ms, old_ms * (sum(syl(t) for t in new_words) / max(1, sum(syl(t[0]) for t in ws[oi:oj + 1]))) ** 0.8)
    # per-word frames: neighbours = their real span (word start -> next start; the first from a, the last to b)
    tb = [0]
    for m in range(1, len(phrase) + 1):
        pre = E(" ".join(phrase[:m]) + " ")
        q = 0
        while q < min(len(full), len(pre)) and full[q] == pre[q]:
            q += 1
        tb.append(q)
    tb[-1] = n_new
    n_win = j - i + 1  # words of the ORIGINAL window

    def span(m):
        return (ws[i + m + 1][1] if m + 1 < n_win else b / SR) - (ws[i + m][1] if m else a / SR)
    segs = []
    for wi in range(len(phrase)):
        if eo <= wi < eo + n_lab:
            if wi == eo:
                old_span = sum(span(q) for q in range(eo, eo + n_old))
                fr = (old_span - (ws[oj][2] - ws[oi][1])) * 50 + new_ms / 20
                segs.append((tb[wi], tb[wi + n_lab], max(4, round(fr))))
            continue
        ow = wi if wi < eo else wi - n_lab + n_old
        segs.append((tb[wi], tb[wi + 1], max(3, round(span(ow) * 50))))
    inf = {"latents": suf, "n_new": n_new, "ctx": True, "mid_frames": 0, "pin_prompt": True, "lead_frames": 0,
           "seg_frames": segs}
    if soft > 0:  # (11-24, you: "mine speeds up a lot") the unchanged phrase words held to the recording's own latents,
        # frame for frame (their segments are their real spans), while t < soft: they keep this speaker's timing and
        # rhythm; the new word's frames are free from the start; after `soft` everything is free to blend
        ed_i = sum(1 for wi in range(eo))
        Lf = sum(fr for _, _, fr in segs[:ed_i])
        Ef = segs[ed_i][2]
        Of = round(sum(span(q) for q in range(eo, eo + n_old)) * 50)
        Tot = sum(fr for _, _, fr in segs)
        lat = whole["latents"]
        fo, fs = [], []
        for f in range(Tot):
            src = af + f if f < Lf else (af + Lf + Of + (f - Lf - Ef) if f >= Lf + Ef else None)
            if src is not None and 0 <= src < lat.shape[0]:
                fo.append(af + f)
                fs.append(src)
        if fo:
            inf["soft"] = {"frames": fo, "lat": lat[fs], "until": soft}
    torch.manual_seed(seed)
    y = np.asarray(synth(line, speaker=voice, lang=lang, prompt=prompt, prompt_guidance=2.0, infill=inf, mastering=False)[0],
                   np.float32)
    mid = int(inf.get("mid_frames_used", 0)) * SPF
    post = w[b:]
    n_cmp = min(len(y) - mid, len(post))
    if n_cmp > SR // 4:  # level: the take's copy of the words after vs the real ones
        y = y * (np.sqrt(np.mean(post[:n_cmp] ** 2) + 1e-12) / max(np.sqrt(np.mean(y[mid:mid + n_cmp] ** 2) + 1e-12), 1e-6))
    y = _trim_pauses(y, ws, i, j, oi, oj, phrase, after.split(), al, log)
    return _splice(w, y, ws, i, j, oi, oj, phrase, after.split(), a, spare, al, log)


def _trim_pauses(y, ws, i, j, oi, oj, phrase, after, al, log):
    """(11-24, you: a tester's "has a seam") Grey can put a pause where the recording had none ("is ... a chapter": 0.22 s for
    the recording's 0.06): every gap between the phrase's words in the take longer than the recording's own gap there
    (+60 ms) loses the extra from its quiet middle (Grey's pauses are dead silent: an inaudible cut)."""
    yw, _, _ = al.align(y, SR, phrase + after)
    eo = oi - i
    n_new = len(phrase) - (oi - i) - (j - oj)  # the new words (phrase = ws[i:oi] + new + ws[oj+1:j+1])
    cuts = []
    for k in range(len(phrase) - 1):  # gap after phrase word k
        if k < eo:  # before the edited words: original word i+k -> i+k+1 (k = eo-1: the gap before the old words)
            tgt = ws[i + k + 1][1] - ws[i + k][2]
        elif k < eo + n_new - 1:  # inside the new words: no recording to copy, a normal word gap at most 0.1 s
            tgt = 0.1
        else:  # from the last new word on: the gap after the old words, then the recording's own gaps
            q = oj + (k - (eo + n_new - 1))
            if q + 1 >= len(ws):
                continue
            tgt = ws[q + 1][1] - ws[q][2]
        g0, g1 = yw[k][2], yw[k + 1][1]
        extra = (g1 - g0) - max(tgt, 0.0)
        if extra > 0.06:
            mid = (g0 + g1) / 2
            seg = y[int((mid - extra / 2) * SR):int((mid + extra / 2) * SR)]
            pk = max(float(np.mean(y[q:q + 2205] ** 2)) for q in range(0, max(1, len(y) - 2205), 1100)) + 1e-12
            if len(seg) and float(np.mean(seg ** 2)) < pk * 1e-4:  # (only real silence, < -40 dB from the take's peak)
                cuts.append((int((mid - extra / 2) * SR), int((mid + extra / 2) * SR)))
                log(f"edit: pause after '{phrase[k]}' {g1 - g0:.2f} s -> {max(tgt, 0.0):.2f} s (the recording's)")
    for c0, c1 in sorted(cuts, reverse=True):  # cut with a 5 ms fade on both sides
        f = int(0.005 * SR)
        r = np.linspace(1, 0, f, dtype=np.float32)
        y = np.concatenate([y[:c0 - f], y[c0 - f:c0] * r + y[c1:c1 + f] * (1 - r), y[c1 + f:]])
    return y


def _splice(w, y, ws, i, j, oi, oj, phrase, after, a, spare, al, log):
    """The take y (Grey's audio from cut a on) into the recording w at the phrase's real word gaps."""
    yw, ysep, _ = al.align(y, SR, phrase + after)
    _, osep, _ = al.align(w, SR, [t[0] for t in ws])
    nw = len(phrase)
    n_sp = min(spare, j - oj)
    e_, je = nw - 1 - n_sp, j - n_sp  # the last regenerated word kept (the spare after it: the recording's copy)
    new_st, new_en = yw[0][1], yw[e_][2]
    st = max(0, int((new_st - max(0.04, ws[i][1] - a / SR)) * SR))
    nx_st, nxt = yw[e_ + 1][1], ws[je + 1][1]
    old_en, gap = ws[je][2], nxt - ws[je][2]
    if gap > 0.12:  # a real pause after: the recording's pause length
        c = _quiet_point(y, new_en, max(new_en + 0.03, min(new_en + 0.15, nx_st - 0.01)))
        bb = int(min(nxt - 0.03, old_en + max(c / SR - new_en, 0.08)) * SR)
    else:  # words run together: keep the recording's gap at the join, >= 30 ms a side
        half2 = max(0.03, gap / 2)
        c = int(min(new_en + half2, nx_st - 0.01) * SR)
        bb = int(max(nxt - half2, old_en + 0.01) * SR)
    a_out = a
    if spare and i < oi:  # in after the left spare word (the recording's copy of it); a stray filler/repeat in the take
        # is passed over by aligning with the spare word written twice (only when its first copy reads as real)
        yw2, _, sc2 = al.align(y, SR, [phrase[0]] + phrase + after)
        rep = sc2[0] > -6.0
        half = max(0.03, (ws[i + 1][1] - ws[i][2]) / 2)
        a_out = int(min(ws[i][2] + half, ws[i + 1][1] - 0.01) * SR)
        nx_take, sp_end = (yw2[2][1], yw2[1][2]) if rep else (yw[1][1], yw[0][2])
        st = int(max(nx_take - half, sp_end + 0.01) * SR)
    # comfort noise: only what Grey's take lacks below the recording's background (per-bin 10th percentile)
    NF, HOP = 1024, 256
    win = torch.hann_window(NF)
    Xo = torch.stft(torch.from_numpy(w.astype(np.float32)), NF, HOP, window=win, return_complex=True)
    Po = torch.quantile(Xo.abs() ** 2, 0.10, dim=1)
    Y = torch.stft(torch.from_numpy(y.astype(np.float32)), NF, HOP, window=win, return_complex=True)
    fill = torch.nn.functional.avg_pool1d((Po[:, None] - Y.abs() ** 2).clamp(min=0)[None], 5, 1, 2, count_include_pad=False)[0]
    y2 = y.copy()
    if float(fill.sum()) > 0:
        ph = torch.rand(fill.shape, generator=torch.Generator().manual_seed(0)) * 2 * np.pi
        y2 = y2 + torch.istft(torch.polar(fill.sqrt().contiguous(), ph), NF, HOP, window=win, length=len(y2)).numpy().astype(np.float32)
    pk = 10 * np.log10(max(float(np.mean(w[q:q + 2205] ** 2)) for q in range(0, max(1, len(w) - 2205), 1100)) + 1e-10)

    def xf(sa, ia, sb, ib):
        h = int(0.03 * SR)
        lv = lambda s_, q: 10 * np.log10(np.mean(s_[max(0, q - h):q + h] ** 2) + 1e-10)  # noqa: E731
        return 120 if max(lv(sa, ia), lv(sb, ib)) < pk - 30 else 25
    fades = [xf(w, a_out, y2, st), xf(y2, c, w, bb)]
    out = _centred([(w, 0, a_out), (y2, st, c), (w, bb, len(w))], fades)
    log(f"edit: '{' '.join(t[0] for t in ws[oi:oj + 1])}' -> phrase '{' '.join(phrase)}', joins at {a_out / SR:.2f} s / {bb / SR:.2f} s")
    return out


def _centred(segs, ms):
    sig, s0, e0 = segs[0]
    out = sig[s0:e0].copy()
    for (sg, st, en), m in zip(segs[1:], ms):
        hh = int(m / 2000 * SR)
        ext = sig[e0:e0 + hh]
        hh = min(hh, len(ext), st)
        out = np.concatenate([out, ext[:hh]])
        n = 2 * hh
        if n > 0 and len(out) >= n:
            ramp = np.sin(np.linspace(0, np.pi / 2, n, dtype=np.float32)) ** 2
            out[-n:] = out[-n:] * (1 - ramp) + sg[st - hh:st + hh] * ramp
            out = np.concatenate([out, sg[st + hh:en]])
        else:
            out = np.concatenate([out, sg[st:en]])
        sig, e0 = sg, en
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------------- public


def load_audio(audio, sr: int = None) -> np.ndarray:
    """path / array (+ sr) -> mono float32 at 44.1 kHz"""
    if isinstance(audio, (str, Path)):
        import soundfile as sf
        x, sr = sf.read(str(audio), dtype="float32", always_2d=True)
        x = x.mean(1)
    else:
        x = np.asarray(audio, np.float32)
        x = x.mean(1) if x.ndim == 2 else x
    if sr and sr != SR:
        x = AF.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, SR).numpy()
    return np.ascontiguousarray(x, np.float32)


def edit_speech(synth, audio, new_text: str, sr: int = None, old_text: str = None, lang: str = "en", seed: int = 0,
                ctx_words: int = 2, spare: int = 1, device: str = None, log=print, soft: float = 0.25):
    """Change a recording's words to `new_text`. audio: path or array at `sr` (default 44.1 kHz). old_text: what the
    recording says (None: Whisper transcribes it). Returns (edited audio at 44.1 kHz, info dict)."""
    if lang not in (None, "en"):
        raise ValueError("speech editing is English-only for now")
    lang = "en"
    w = load_audio(audio, sr)
    dev = device or ("cuda" if next(synth.model.parameters()).is_cuda else "cpu")
    if not old_text:
        from .auto_prompt import transcribe
        old_text = transcribe(torch.from_numpy(w), dev, language="en")
    al = aligner("cpu")
    old_words, new_words = old_text.split(), new_text.split()
    edits = diff_words(old_words, new_words)
    if not edits:
        return w, {"old_text": old_text, "new_text": new_text, "edits": []}
    done, cur_words = [], list(old_words)
    shift = 0
    for i1, i2, nw in edits:  # left to right; each edit re-aligns the audio the previous one made
        i1s, i2s = i1 + shift, i2 + shift
        ws, _, _ = al.align(w, SR, cur_words)
        if i2s < i1s:  # an insertion: treat it as replacing the word before + the new ones
            i2s = i1s - 1
            i1s -= 1
            nw = [cur_words[i1s]] + list(nw)
        w = _edit_once(synth, w, ws, i1s, i2s, nw, lang, ctx_words, spare, seed, al, log, soft)
        done.append({"old": " ".join(cur_words[i1s:i2s + 1]), "new": " ".join(nw)})
        cur_words = cur_words[:i1s] + list(nw) + cur_words[i2s + 1:]
        shift += len(nw) - (i2s - i1s + 1)
    return w, {"old_text": old_text, "new_text": " ".join(cur_words), "edits": done}
