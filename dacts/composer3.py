"""Composer v3: chords first, then the melody over them; everything in key and on the 8th grid (owner 2026-10-19).
  ChordModel: half-bar chord tokens (key-relative: root + 12*minor, 24 = no chord) given mode / genre / tempo.
  Composer3: like v2 (dacts/composer.py) + the chord sequence in its memory (each note can attend to the chord under
             it) + the chord at the previous note as a step input. Trained on clean data (build_composer_v3.py).
  Generation is constrained: pitches only from the key's scale, onsets only on 8th positions, lengths whole 8ths."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .composer import (BPM_BUCKETS, CHARS, C_SEP, GENRES, LANGS, MAX_LEN, MAX_ON, N_CHAR, N_PITCH, P_HI, P_LO,
                       bpm_bucket, encode_words)

N_CHORD = 26  # 0..23 key-relative triads, 24 no chord, 25 unknown
MAJ, MIN = (0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10)


def genre_vec(genre):
    v = [0.0] * len(GENRES)
    for g in genre or []:
        g = {"rap": "hiphop", "poprock": "pop"}.get(g, g)
        if g in GENRES:
            v[GENRES.index(g)] = 1.0
    return v


class Cond(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mode, self.bpm = nn.Embedding(3, d), nn.Embedding(BPM_BUCKETS + 1, d)
        self.genre, self.bars, self.lang = nn.Linear(len(GENRES), d, bias=False), nn.Embedding(33, d), nn.Embedding(len(LANGS), d)

    def forward(self, mode, bpm, genre, bars, lang):
        return self.mode(mode) + self.bpm(bpm) + self.genre(genre) + self.bars(bars) + self.lang(lang)


class ChordModel(nn.Module):
    def __init__(self, d=192, layers=3, heads=4, dropout=0.2):
        super().__init__()
        self.cond = Cond(d)
        self.tok = nn.Embedding(N_CHORD + 1, d)  # + start
        self.pos = nn.Embedding(64, d)
        l = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.net = nn.TransformerEncoder(l, layers)
        self.out = nn.Linear(d, 25)  # predicts 0..24 (never "unknown")

    def forward(self, chords_in, mode, bpm, genre):
        """chords_in (B, H): start token then the previous chords. Returns logits (B, H, 25)."""
        z = torch.zeros_like(mode)
        x = self.tok(chords_in) + self.pos(torch.arange(chords_in.shape[1], device=chords_in.device))[None]
        x = x + self.cond(mode, bpm, genre, z, z)[:, None]
        h = chords_in.shape[1]
        causal = torch.triu(torch.ones(h, h, dtype=torch.bool, device=x.device), 1)
        return self.out(self.net(x, mask=causal))

    @torch.no_grad()
    def generate(self, halves, mode=None, bpm=None, genre=None, temperature=1.0, max_run=4):
        """77% of half-bars repeat the chord before them, so step-by-step sampling tends to sit on one chord: a chord may
        last at most `max_run` half-bars (2 bars; the data's mean is 2.2), and the piece prefers to start on the tonic."""
        dev = next(self.parameters()).device
        L = lambda v: torch.tensor([v], device=dev)
        seq, run = [N_CHORD], 0
        for i in range(halves):
            lo = self(torch.tensor([seq], device=dev), L({"major": 1, "minor": 2}.get(mode, 0)),
                      L(bpm_bucket(bpm) if bpm else BPM_BUCKETS), torch.tensor([genre_vec(genre)], device=dev))[0, -1].float()
            lo[24] -= 3.0  # no-chord stretches are for instrumental gaps, not for writing songs
            if i == 0:
                lo[12 if mode == "minor" else 0] += 2.0  # start at home
            if run >= max_run:
                lo[seq[-1]] = -1e9
            c = int(torch.multinomial(F.softmax(lo / temperature, -1), 1))
            run = run + 1 if c == seq[-1] else 1
            seq.append(c)
        return seq[1:]


NAMES = "C C# D D# E F F# G G# A A# B".split()
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}


def parse_chords(text, tonic, bars):
    """'C G Am F' (one chord per bar; '|' optional; 7ths / sus / add reduce to their triad) -> key-relative half-bar
    tokens for `bars` bars (the progression repeats to fill). Returns None for empty / unparseable input."""
    import re
    out = []
    for tok in re.split(r"[\s|,]+", (text or "").strip()):
        m = re.match(r"^([A-Ga-g])([#b]?)(m(?!aj))?", tok)
        if not m:
            continue
        name = m.group(1).upper() + m.group(2)
        name = FLAT.get(name, name)
        if name not in NAMES:
            continue
        out.append((NAMES.index(name) - tonic) % 12 + (12 if m.group(3) else 0))
    if not out:
        return None
    halves = [c for c in out for _ in (0, 1)]
    return [halves[i % len(halves)] for i in range(2 * bars)]


def chord_names(tokens, tonic):
    """key-relative half-bar tokens -> one name per bar (the bar's first chord; '-' = no chord)."""
    nm = lambda c: "-" if c >= 24 else NAMES[(tonic + c) % 12] + ("m" if c >= 12 else "")
    return [nm(tokens[i]) for i in range(0, len(tokens), 2)]


def syllables(word, cap=8):
    """Rough syllable count: vowel groups (Latin, pinyin) or kana morae (Japanese; small kana, marks and the long
    mark don't count). 1..cap."""
    import re
    n = len(re.findall(r"[aeiouyáéíóúàèìòùäëïöüâêîôûąęó]+", word.lower()))
    n += len(re.findall(r"[ぁ-んァ-ン]", word)) - len(re.findall(r"[ぁぃぅぇぉゃゅょゎっァィゥェォャュョヮッ]", word))
    return max(1, min(cap, n))


def song_plan(words, bpm, rate=2.8, max_per_bar=6.0, max_bars=16):
    """Pace a lyric line: (composer bpm, bars at that bpm, seconds per composer 16th). Aims at ~`rate` syllables a
    second (a flat syllables-per-bar made slow songs crawl at ~1/s: "reading a book very slowly", owner 2026-10-19).
    Below 95 bpm the melody is written in double time (half-time feel): its 8th grid lands on the song's 16ths, so it
    stays on the song's grid and can move at a sung pace."""
    cb = bpm * 2 if bpm < 95 else bpm
    syl = sum(syllables(w) for w in words) if words else 0
    per_bar = min(max_per_bar, rate * 240.0 / cb)
    bars = max(2, min(max_bars, -(-syl // max(1, round(per_bar))) if syl else 4))
    if cb != bpm:
        bars += bars % 2  # double time: an even count, so the song gets whole bars
    return cb, bars, 60.0 / cb / 4


def compose(chord_model, melody_model, words=None, key="C", mode="major", bpm=120, genre=None, bars=8, lang="en",
            chords=None, temperature=0.8):
    """Lyrics (+ key / tempo / genre / bars, optional chords) -> notes and chords.
    chords: e.g. 'C G Am F' -- the melody is written over them; None -> the chord model writes a progression, which
    is returned too. Returns {"notes": [(onset16, len16, MIDI, word or -1)], "chords": [one name per bar], "tonic", ...}"""
    tonic = NAMES.index(FLAT.get(key, key))
    given = parse_chords(chords, tonic, bars) if chords else None
    toks = given or chord_model.generate(2 * bars, mode, bpm, genre)
    notes = melody_model.generate(words, toks, mode, bpm, genre, bars=bars, lang=lang, temperature=temperature)
    root = 60 + tonic - (12 if tonic > 4 else 0)  # middle register; the voice's own range is applied by the caller
    return {"notes": [(on, ln, root + p, w) for on, ln, p, w, _ in notes], "chords": chord_names(toks, tonic),
            "chords_given": given is not None, "key": f"{NAMES[tonic]} {mode}", "bpm": bpm, "bars": bars}


class Composer3(nn.Module):
    def __init__(self, d=256, enc_layers=3, dec_layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.cond = Cond(d)
        self.char, self.pos = nn.Embedding(N_CHAR, d), nn.Embedding(1024, d)
        self.chord, self.halfpos = nn.Embedding(N_CHORD, d), nn.Embedding(64, d)
        self.is_chord = nn.Parameter(torch.zeros(d))
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc_layers)
        self.no_word = nn.Parameter(torch.zeros(d))
        self.e_on, self.e_len = nn.Embedding(MAX_ON, d), nn.Embedding(MAX_LEN + 1, d)
        self.e_pitch, self.e_nw = nn.Embedding(N_PITCH + 1, d), nn.Embedding(2, d)
        self.e_beat, self.e_bar, self.e_pch = nn.Embedding(16, d), nn.Embedding(33, d), nn.Embedding(N_CHORD, d)
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.norm = nn.LayerNorm(d)
        self.h_on, self.h_len, self.h_pitch, self.h_nw = (nn.Linear(d, n) for n in (MAX_ON, MAX_LEN + 1, N_PITCH, 2))

    def encode(self, chars, cmask, chords, hmask, mode, bpm, genre, bars, lang):
        g = self.cond(mode, bpm, genre, bars, lang)
        x = self.char(chars) + self.pos(torch.arange(chars.shape[1], device=chars.device))[None]
        c = self.chord(chords) + self.halfpos(torch.arange(chords.shape[1], device=chords.device))[None] + self.is_chord
        mem = torch.cat([g[:, None], x, c], 1)
        m = torch.cat([torch.ones_like(cmask[:, :1]), cmask, hmask], 1)
        return self.enc(mem, src_key_padding_mask=~m), m

    def word_vec(self, mem, idx, has):
        wv = torch.gather(mem, 1, idx[..., None].expand(-1, -1, mem.shape[-1]))
        return torch.where(has[..., None], wv, self.no_word.expand_as(wv))

    def step_inputs(self, mem, on, ln, pitch, nw, abs_on, prev_chord, cur_c, has_c, nxt_c, has_n):
        return (self.e_on(on) + self.e_len(ln) + self.e_pitch(pitch) + self.e_nw(nw) + self.e_pch(prev_chord)
                + self.e_beat(abs_on % 16) + self.e_bar((abs_on // 16).clamp(max=32))
                + self.word_vec(mem, cur_c, has_c) + 0.5 * self.word_vec(mem, nxt_c, has_n))

    def forward(self, mem, mmask, x):
        n = x.shape[1]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
        h = self.norm(self.dec(x, mem, tgt_mask=causal, memory_key_padding_mask=~mmask))
        return self.h_on(h), self.h_len(h), self.h_pitch(h), self.h_nw(h)

    @torch.no_grad()
    def generate(self, words=None, chords=None, mode="major", bpm=None, genre=None, bars=8, lang=None, temperature=0.8,
                 max_notes=256):
        """chords: key-relative half-bar tokens (len >= 2*bars), e.g. from ChordModel.generate. Returns
        [(onset16, len16, rel_pitch, word or -1, chord)] -- every pitch in the scale, every onset on an 8th."""
        dev = next(self.parameters()).device
        chars, starts = encode_words(words) if words else ([C_SEP], [])
        chords = list(chords or [25] * (2 * bars))
        L = lambda v: torch.tensor([v], device=dev)
        mem, mm = self.encode(torch.tensor([chars], device=dev), torch.ones(1, len(chars), dtype=torch.bool, device=dev),
                              torch.tensor([chords], device=dev), torch.ones(1, len(chords), dtype=torch.bool, device=dev),
                              L({"major": 1, "minor": 2}.get(mode, 0)), L(bpm_bucket(bpm) if bpm else BPM_BUCKETS),
                              torch.tensor([genre_vec(genre)], device=dev), L(min(32, bars) if bars else 0),
                              L(LANGS.index(lang) if lang in LANGS else (4 if words else 0)))
        nwd = len(words) if words else 0
        # a word gets at least a note per syllable, so long words aren't crammed into one short note (the voice then
        # swallows syllables: "skipping words", owner 2026-10-19)
        syl = [syllables(w) for w in (words or [])]
        wc = lambda k: starts[k] + 1 if (words and 0 <= k < nwd) else 0
        scale = MAJ if mode == "major" else MIN
        # in the scale, inside the range real melodies use (-8..+19 around the tonic: 99% of our data; a sampled -24 put
        # notes at C2) and no leap beyond an octave (0.4% of real leaps)
        pmask = torch.tensor([((p + P_LO) % 12) in scale and -8 <= p + P_LO <= 19 for p in range(N_PITCH)], device=dev)
        seq = {k: [v] for k, v in (("on", 0), ("len", 0), ("p", N_PITCH), ("nw", 0), ("abs", 0), ("pc", 25), ("cur", -1),
                                   ("nxt", 0 if words else -1))}
        out, abs_on, wi = [], 0, -1
        for _ in range(max_notes):
            T_ = lambda k: torch.tensor([seq[k]], device=dev)
            x = self.step_inputs(mem, T_("on"), T_("len"), T_("p"), T_("nw"), T_("abs"), T_("pc"),
                                 torch.tensor([[wc(k) for k in seq["cur"]]], device=dev), torch.tensor([[k >= 0 for k in seq["cur"]]], device=dev),
                                 torch.tensor([[wc(k) for k in seq["nxt"]]], device=dev), torch.tensor([[k >= 0 for k in seq["nxt"]]], device=dev))
            lo_on, lo_len, lo_p, lo_nw = (l[0, -1].float() for l in self(mem, mm, x))
            # on the 8th grid: onset class c moves by c-1 sixteenths -> keep only even absolute positions
            ok_on = torch.tensor([c == 0 or (abs_on + c - 1) % 2 == 0 for c in range(MAX_ON)], device=dev)
            if words and wi < nwd - 1:
                ok_on[0] = False  # words left: keep going
            if not out:
                ok_on[1:] &= torch.tensor([(c - 1) < 16 for c in range(1, MAX_ON)], device=dev)  # start in bar 1
            lo_on = lo_on.masked_fill(~ok_on, -1e9)
            lo_len = lo_len.masked_fill(torch.tensor([bool(l < 2 or l % 2) for l in range(MAX_LEN + 1)], device=dev), -1e9)
            pm = pmask
            if out:
                pm = pmask & (torch.arange(N_PITCH, device=dev) + P_LO - out[-1][2]).abs().le(12)
            lo_p = lo_p.masked_fill(~pm, -1e9)
            samp = lambda l: int(torch.multinomial(F.softmax(l / temperature, -1), 1))
            if words and wi >= nwd - 1 and sum(1 for n in out if n[3] == wi) >= max(3, syl[-1]):
                break  # the last word: at most 3 notes, then the line ends (no endless run on the final syllable)
            on = samp(lo_on)
            if on == 0:
                break
            abs_on += on - 1
            if bars and abs_on >= bars * 16 and not (words and wi < nwd - 1):
                break  # past the asked bars: stop -- unless words are left, which keep going (chords repeat below)
            if words and 0 <= wi < nwd - 1:
                have = sum(1 for n in out if n[3] == wi)
                if have < syl[wi]:
                    lo_nw[1] = -1e9  # stay on this word: it needs a note per syllable
                elif have >= syl[wi] + 3:
                    lo_nw[0] = -1e9  # but no endless melisma either
            new = (samp(lo_nw) if words and wi < nwd - 1 else 0) if wi >= 0 else (1 if words else 0)
            ln, p = samp(lo_len), samp(lo_p)
            wi += new
            ch = chords[(abs_on // 8) % len(chords)] if chords else 25
            out.append((abs_on, ln, p + P_LO, wi if words else -1, ch))
            for k, v in (("on", on), ("len", ln), ("p", p), ("nw", new), ("abs", abs_on), ("pc", ch), ("cur", wi),
                         ("nxt", wi + 1 if words and wi + 1 < nwd else -1)):
                seq[k].append(v)
        return out
