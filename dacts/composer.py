"""Composer: lyrics + key mode + tempo + genre + bars -> a note sequence on the 16th-note grid (owner 2026-10-19:
"a small composer model"). The voice model then sings the notes; to the user it is still lyrics + key + BPM in.

Notes are written one at a time; per note four factors: onset (16ths after the previous note's onset; class 0 = END),
length (16ths), pitch (semitones above the key's tonic, octave chosen by register), new_word (does it start the next
word or slur on the current one). A word pointer advances on new_word; each note step sees its word's encoding.
Lyrics are optional (melody-only clips train the same decoder with no words)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = "abcdefghijklmnopqrstuvwxyz'"
C_PAD, C_SEP = 0, 1  # char ids: 0 pad, 1 word separator, 2.. letters
N_CHAR = len(CHARS) + 2
MAX_ON, MAX_LEN = 48, 32          # onset delta 1..47 (0 = END), length 1..32 sixteenths
P_LO, P_HI = -24, 36              # pitch relative to the tonic (register-normalized), 60 classes
N_PITCH = P_HI - P_LO
BPM_BUCKETS = 16
GENRES = ("pop", "rock", "folk", "hiphop", "indie", "electronic", "alternative", "singersongwriter", "world", "dance",
          "rnb", "soul", "funk", "latin", "blues", "country", "reggae", "chanson", "jazz", "easylistening", "metal", "punk")
LANGS = ("none", "en", "es", "pl", "other")


def bpm_bucket(bpm):
    return int(min(BPM_BUCKETS - 1, max(0, (math.log(bpm / 60) / math.log(200 / 60)) * BPM_BUCKETS)))


def encode_words(words):
    """words -> char ids with separators, and the char position where each word starts."""
    ids, starts = [C_SEP], []
    for w in words:
        w = "".join(c for c in w.lower() if c in CHARS)
        starts.append(len(ids))
        ids += [CHARS.index(c) + 2 for c in w] or [C_SEP]
        ids.append(C_SEP)
    return ids, starts


class Composer(nn.Module):
    def __init__(self, d=256, enc_layers=3, dec_layers=4, heads=4, dropout=0.2):  # small: 54k clips overfit a 22M model
        super().__init__()
        self.d = d
        self.char = nn.Embedding(N_CHAR, d)
        self.pos = nn.Embedding(1024, d)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc_layers)
        # global conditions (each dropped in training -> "unknown"): mode, tempo, genre, bars, language
        self.mode = nn.Embedding(3, d)       # 0 unknown, 1 major, 2 minor
        self.bpm = nn.Embedding(BPM_BUCKETS + 1, d)  # last = unknown
        self.genre = nn.Linear(len(GENRES), d, bias=False)
        self.bars = nn.Embedding(33, d)      # 0 unknown
        self.lang = nn.Embedding(len(LANGS), d)
        self.no_word = nn.Parameter(torch.zeros(d))
        # note-step inputs
        self.e_on = nn.Embedding(MAX_ON, d)
        self.e_len = nn.Embedding(MAX_LEN + 1, d)
        self.e_pitch = nn.Embedding(N_PITCH + 1, d)  # last = start token
        self.e_nw = nn.Embedding(2, d)
        self.e_beat = nn.Embedding(16, d)    # position of the onset in the bar (16ths)
        self.e_bar = nn.Embedding(33, d)     # which bar
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.norm = nn.LayerNorm(d)
        self.h_on, self.h_len, self.h_pitch, self.h_nw = (nn.Linear(d, n) for n in (MAX_ON, MAX_LEN + 1, N_PITCH, 2))

    def encode(self, chars, cmask, mode, bpm, genre, bars, lang):
        """chars (B, Lc); cmask (B, Lc) True = real; globals (B,) / genre (B, 22). Returns memory (B, 1+Lc, d), mask."""
        g = self.mode(mode) + self.bpm(bpm) + self.genre(genre) + self.bars(bars) + self.lang(lang)
        x = self.char(chars) + self.pos(torch.arange(chars.shape[1], device=chars.device))[None]
        mem = torch.cat([g[:, None], x], 1)
        m = torch.cat([torch.ones_like(cmask[:, :1]), cmask], 1)
        return self.enc(mem, src_key_padding_mask=~m), m

    def word_vec(self, mem, word_char, has):
        wv = torch.gather(mem, 1, word_char[..., None].expand(-1, -1, mem.shape[-1]))
        return torch.where(has[..., None], wv, self.no_word.expand_as(wv))

    def step_inputs(self, mem, on, ln, pitch, nw, abs_on, cur_char, has_cur, nxt_char, has_nxt):
        """Step i's input = the previous note (teacher forcing; step 0 = a start token): on / ln / pitch / nw (B, N)
        classes, abs_on its absolute 16th (bar + beat position), plus the word the pointer is on and the next word
        (char positions in `mem`; has_* False -> no word: melody-only, or past the last word)."""
        return (self.e_on(on) + self.e_len(ln) + self.e_pitch(pitch) + self.e_nw(nw)
                + self.e_beat(abs_on % 16) + self.e_bar((abs_on // 16).clamp(max=32))
                + self.word_vec(mem, cur_char, has_cur) + 0.5 * self.word_vec(mem, nxt_char, has_nxt))

    @torch.no_grad()
    def generate(self, words=None, mode=None, bpm=None, genre=None, bars=None, lang=None, temperature=0.9, max_notes=256):
        """Write notes. words: list[str] or None (melody only); mode "major"/"minor"/None; bpm float/None;
        genre list[str]/None; bars int/None (the piece stops at that many bars). Returns [(onset16, len16, rel_pitch,
        word_index or -1)], pitch relative to the tonic in the middle register (add the tonic + octave to sing it)."""
        dev = next(self.parameters()).device
        chars, starts = encode_words(words) if words else ([C_SEP], [])
        gv = [1.0 if g in (genre or []) else 0.0 for g in GENRES]
        L = lambda v: torch.tensor([v], device=dev)
        mem, mm = self.encode(torch.tensor([chars], device=dev), torch.ones(1, len(chars), dtype=torch.bool, device=dev),
                              L({"major": 1, "minor": 2}.get(mode, 0)), L(bpm_bucket(bpm) if bpm else BPM_BUCKETS),
                              torch.tensor([gv], device=dev), L(min(32, bars) if bars else 0),
                              L(LANGS.index(lang) if lang in LANGS else (4 if words else 0)))
        nw_words = len(words) if words else 0
        wc = lambda k: starts[k] + 1 if (words and 0 <= k < nw_words) else 0
        seq = {"on": [0], "len": [0], "p": [N_PITCH], "nw": [0], "abs": [0], "cur": [-1], "nxt": [0 if words else -1]}
        out, abs_on, wi = [], 0, -1
        for _ in range(max_notes):
            T_ = lambda k, dt=torch.long: torch.tensor([seq[k]], dtype=dt, device=dev)
            x = self.step_inputs(mem, T_("on"), T_("len"), T_("p"), T_("nw"), T_("abs"),
                                 torch.tensor([[wc(k) for k in seq["cur"]]], device=dev), torch.tensor([[k >= 0 for k in seq["cur"]]], device=dev),
                                 torch.tensor([[wc(k) for k in seq["nxt"]]], device=dev), torch.tensor([[k >= 0 for k in seq["nxt"]]], device=dev))
            lo_on, lo_len, lo_p, lo_nw = (l[0, -1].float() for l in self(mem, mm, x))
            samp = lambda l: int(torch.multinomial(F.softmax(l / temperature, -1), 1))
            if words and wi < nw_words - 1:
                lo_on[0] = -1e9  # words left: don't stop yet
            new = (samp(lo_nw) if words and wi < nw_words - 1 else 0) if wi >= 0 else (1 if words else 0)
            on = samp(lo_on)
            if on == 0:
                break
            d = on - 1; ln = max(1, samp(lo_len[1:]) + 1); p = samp(lo_p)
            abs_on += d
            if bars and abs_on >= bars * 16:
                break
            if new:
                wi += 1
            out.append((abs_on, ln, p + P_LO, wi if words else -1))
            for k, v in (("on", on), ("len", min(MAX_LEN, ln)), ("p", p), ("nw", new), ("abs", abs_on), ("cur", wi),
                         ("nxt", wi + 1 if words and wi + 1 < nw_words else -1)):
                seq[k].append(v)
            if words and wi >= nw_words - 1 and abs_on >= 16 * max(1, (bars or 0)) - 16 and not bars:
                pass
        return out

    def forward(self, mem, mmask, x):
        n = x.shape[1]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
        h = self.norm(self.dec(x, mem, tgt_mask=causal, memory_key_padding_mask=~mmask))
        return self.h_on(h), self.h_len(h), self.h_pitch(h), self.h_nw(h)
