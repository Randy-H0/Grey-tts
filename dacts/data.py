import json
import array
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from . import text as T
from .codec import frame_rate


def style_tag(it) -> str:
    """Which recording session / delivery a clip belongs to, where the data says so: Expresso styles (happy,
    whisper, laughing ...), EARS emotions and reading styles; whispers everywhere; otherwise 'default'."""
    s = it.get("spk_name") or ""
    if (it.get("whisper") or 0) >= 0.5:
        return "whisper"
    p = it["id"][len(s) + 1:].split("_")
    if s.startswith("expresso") and len(p) > 2:
        return p[1]
    if s.startswith("ears") and p:
        return "_".join(p[:2]) if p[0] in ("emo", "rainbow", "sentences") and len(p) > 1 else p[0]
    return "default"


def load_manifest(root):
    with open(Path(root) / "manifest.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class CodesDataset(Dataset):
    def __init__(self, root, items, cfg):
        self.root, self.cfg = Path(root), cfg  # items may carry their own "root" (multi-dataset runs)
        max_frames = int(cfg.max_seconds * frame_rate(cfg))
        self.kvae = getattr(cfg, "codec", "dac") == "kvae"
        self.items = []
        for it in items:
            # untranscribed audio (manifest "notext": true): one placeholder token; the model learns to continue
            # the voice from a given start, with no text (cfg.notext), see DacTTS.forward
            tokens = [T.BLANK] if it.get("notext") else T.encode(it["text"], cfg.add_blank, it.get("lang", "en"))
            patches = it["frames"] // cfg.patch
            # MAS needs at least one decoder token per text token
            if 0 < len(tokens) <= patches and it["frames"] <= max_frames:
                # in place + a compact uint16 array (ids up to 276 since Ukrainian, 2026-11-16; was bytes, ids < 256): every
                # DataLoader worker holds all items; oversampled copies share one dict
                it["tokens"], it["patches"] = array.array("H", tokens), patches
                self.items.append(it)
        # delivery references: OTHER clips of the same speaker (their latents, a 3-8 s window)
        self.dlv = getattr(cfg, "use_dlv", False)
        self.vre = getattr(cfg, "use_vre", False) or getattr(cfg, "use_ref_seg", False)  # (v14 reference segment: same sampling)
        if self.dlv or self.vre:
            self.dlv_pool = {}
            for it in self.items:  # (not whispered clips: a whisper would leak into "how this voice sounds")
                if not (it.get("whisper") or 0) >= 0.5:
                    self.dlv_pool.setdefault(it.get("spk_name"), []).append((str(it.get("root", self.root)), it["id"]))
        # cross-clip prompts: OTHER clips of the same speaker, preferably the same session / delivery
        self.cross = getattr(cfg, "cross_prob", 0) > 0
        if self.cross:
            self.x_by_tag, self.x_by_spk = {}, {}
            for k, it in enumerate(self.items):
                if (it.get("whisper") or 0) >= 0.5:
                    continue  # whispers are never the prompt: "here is the normal voice, now whisper" (whisper rows)
                self.x_by_tag.setdefault((it.get("spk_name"), style_tag(it)), []).append(k)
                self.x_by_spk.setdefault(it.get("spk_name"), []).append(k)
            self.x_wh = {}  # whispered clips per speaker: whispered prompts (cfg.cross_whisper_prob / _any)
            for k, it in enumerate(self.items):
                if (it.get("whisper") or 0) >= 0.5:
                    self.x_wh.setdefault(it.get("spk_name"), []).append(k)
        # notes per sung clip (cfg.use_notes, data/singing/build_note_store.py): per decoder frame [pitch, sung, onset, known]
        self.use_notes = bool(getattr(cfg, "use_notes", False))
        self.note_store = {}
        if self.use_notes:
            import pickle
            with open(getattr(cfg, "note_store", "data/singing/note_store.pkl"), "rb") as f:
                self.note_store = pickle.load(f)
        # song continuation (owner 2026-10-18): clips cut back-to-back from one song (manifest "song", "seq", "t0", "t1")
        self.song_cont = getattr(cfg, "song_cont_prob", 0) > 0
        if self.song_cont:
            self.by_song = {}
            for k, it in enumerate(self.items):
                if it.get("song") is not None and it.get("seq") is not None:
                    self.by_song[(it["song"], int(it["seq"]))] = k
        # zero-shot references: x-vectors of OTHER clips of the same speaker (rows of <root>/spkemb.npy)
        self.zs = getattr(cfg, "zs_weight", 0) > 0
        self._mm = {}
        if self.zs:
            hold = set(filter(None, getattr(cfg, "zs_holdout", "").split(",")))
            self.zs_row, self.zs_by_spk = {}, {}
            # (sorted: a set's order changes per process -- if an id were in two dirs, workers could disagree; the later
            # dir in sorted order wins, as dacts/fastdata.py does; code review 2026-10-29)
            for root in sorted({str(it.get("root", self.root)) for it in self.items}):
                f = Path(root) / "spkemb_ids.json"
                if f.exists():
                    for r, uid in enumerate(json.loads(f.read_text(encoding="utf-8"))):
                        self.zs_row[uid] = (root, r)
            for it in self.items:
                if it["id"] in self.zs_row and it.get("spk_name") not in hold and not (it.get("whisper") or 0) >= 0.5:
                    self.zs_by_spk.setdefault(it.get("spk_name"), []).append(self.zs_row[it["id"]])

    def __len__(self):
        return len(self.items)

    def _up_clean(self, root):
        """Clean label of a dataset's upsampled copies when they were Preen-cleaned (owner 2026-10-31: Dutch + Polish redone
        like dream22: shards/latents_upsampled_preen.json), else None: the clip's own label."""
        key = (str(root), "_up_clean")
        if key not in self._mm:
            f = Path(root) / "shards" / "latents_upsampled_preen.json"
            self._mm[key] = float(json.loads(f.read_text(encoding="utf-8"))["clean"]) if f.exists() else None
        return self._mm[key]

    def _read(self, root, kind, uid):
        """A clip's (T, D) array: from <root>/shards/<kind>.npy (one memory-mapped file per dataset, see
        pack_shards.py) when packed, else from the per-clip file <root>/<kind>/<id>.npy. None if missing."""
        key = (str(root), kind)
        sh = self._mm.get(key)
        if sh is None:
            d = Path(root) / "shards"
            if (d / f"{kind}.npy").exists():
                sh = (np.load(d / f"{kind}.npy", mmap_mode="r"),
                      json.loads((d / f"{kind}_index.json").read_text(encoding="utf-8")))
            else:
                sh = False
            self._mm[key] = sh
        if sh:
            loc = sh[1].get(uid)
            return None if loc is None else np.asarray(sh[0][loc[0]:loc[0] + loc[1]])
        p = Path(root) / kind / f"{uid}.npy"
        return np.load(p) if p.exists() else None

    def _spkemb(self, root):
        if root not in self._mm:  # opened lazily in each worker (memory-mapped, never copied)
            self._mm[root] = np.load(Path(root) / "spkemb.npy", mmap_mode="r")
        return self._mm[root]

    def _note_features(self, segs):
        """(clip id, latent frames) segments -> (decoder frames, 4) float16: [(MIDI - 60) / 24, sung 0/1, note onset 0/1,
        known 0/1]. Clips without notes (speech, other voices' prompt clips) are 'unknown' (all zero)."""
        from .codec import KVAE_HOP, KVAE_SR
        p = self.cfg.patch
        spf = p * KVAE_HOP / KVAE_SR  # seconds per decoder frame
        parts = []
        for cid, frames in segs:
            n = frames // p
            f = np.zeros((n, 4), np.float16)
            notes = self.note_store.get(cid) if cid is not None else None
            if notes is not None:
                f[:, 3] = 1.0
                for on, du, midi in notes:
                    a, b = int(round(on / spf)), int(round((on + du) / spf))
                    a, b = max(0, a), min(n, max(b, a + 1))
                    if a >= n:
                        continue
                    f[a:b, 0] = (midi - 60.0) / 24.0
                    f[a:b, 1] = 1.0
                    f[a, 2] = 1.0
            parts.append(f)
        return np.concatenate(parts, 0)

    def _prompt_labels(self, used):
        """cfg.use_prompt_labels: the prompt clips' labels (as __getitem__ gives a row's own), each averaged over the
        clips by length; None where no clip has it. used: [(item, patches)]."""
        out = {}
        for k in ("whisper", "style", "mood", "dominance", "volume", "speed", "clean", "band"):
            s_, w_ = 0.0, 0
            for o, n in used:
                v = o.get(k, 0.5) if k in ("style", "volume", "speed") else o.get(k)
                if k == "band" and v is None:
                    v = 1.0
                if k == "clean" and v is not None and getattr(self.cfg, "clean_band_rule", False):
                    b_ = o.get("band")
                    v = float(v) * (1.0 if b_ is None else float(b_))
                if v is None or v != v:
                    continue
                s_, w_ = s_ + float(v) * n, w_ + n
            out[k] = s_ / w_ if w_ else None
        return out

    def _clip_labels(self, o):
        """cfg.use_clip_labels: one reference clip's labels in PROMPT_LABELS order (as __getitem__ gives a row's own),
        NaN where unknown; each hidden now and then (cfg.label_drop), so a partly labelled reference is normal too."""
        out = []
        for k in ("whisper", "style", "mood", "dominance", "volume", "speed", "clean", "band"):
            v = o.get(k, 0.5) if k in ("style", "volume", "speed") else o.get(k)
            if k == "band" and v is None:
                v = 1.0
            if k == "clean" and v is not None and getattr(self.cfg, "clean_band_rule", False):
                b_ = o.get("band")
                v = float(v) * (1.0 if b_ is None else float(b_))
            if v is None or v != v or random.random() < getattr(self.cfg, "label_drop", 0.0):
                v = float("nan")
            out.append(float(v))
        return out

    def _clip_label_rows(self, used):
        """Per reference patch and per reference word: the labels of the clip it belongs to. -> (patches (P, L),
        words (T, L)) float32 arrays, in the order the clips are joined."""
        if not used:
            return None, None
        labs = [self._clip_labels(o) for o, _ in used]
        pat = np.concatenate([np.tile(np.asarray(l, np.float32), (n, 1)) for l, (_, n) in zip(labs, used)])
        tok = np.concatenate([np.tile(np.asarray(l, np.float32), (len(o["tokens"]), 1)) for l, (o, _) in zip(labs, used)])
        return pat, tok

    def _ref_pool(self, it):
        """The speaker's OTHER clips (root, id) for delivery / voice references, in pool order (FastDataset: lazily)."""
        return [p for p in self.dlv_pool.get(it.get("spk_name"), []) if p[1] != it["id"]]

    def _zs_others(self, it):
        """x-vector rows (root, row) of the speaker's other clips (all of them if this is its only one)."""
        pool = self.zs_by_spk.get(it.get("spk_name"), [])
        own = self.zs_row.get(it["id"])
        return [p for p in pool if p != own] or pool

    def plan_cross(self, i, rnd):
        """Prompt clips for row i chosen up front (PlannedBatchSampler): the same rules and odds as the random choice
        in __getitem__ -- same speaker, same session / delivery 80% of the time, whole clips, at most cross_max_sec
        and no longer than the row -- so the sampler knows the row's real length. -> (clip indices, patches) or None."""
        if not self.cross or rnd.random() >= self.cfg.cross_prob:
            return None
        it = self.items[i]
        own = it["id"]
        budget = int(min(self.cfg.cross_max_sec * 50, max(150, it["patches"] * self.cfg.patch) * getattr(self.cfg, "cross_len_mult", 1.0)))

        def others(pool):  # does the pool hold another clip than this one? (oversampling repeats a clip <= ~5x)
            return len(pool) > 64 or any(self.items[k]["id"] != own for k in pool)
        wh = (it.get("whisper") or 0) >= 0.5
        mixed = False
        p_wh = getattr(self.cfg, "cross_whisper_prob" if wh else "cross_whisper_any", 0.0)
        wpool = self.x_wh.get(it.get("spk_name"), []) if p_wh > 0 else []
        if wpool and others(wpool) and rnd.random() < p_wh:
            pool = wpool  # a whispered prompt (its labels say so: cfg.use_prompt_labels)
        elif getattr(self.cfg, "cross_mix_prob", 0.0) > 0 and rnd.random() < self.cfg.cross_mix_prob:
            # a mixed reference: clips from anywhere in the speaker's recordings, whispers included (cfg.use_clip_labels)
            pool = list(self.x_by_spk.get(it.get("spk_name"), [])) + list(self.x_wh.get(it.get("spk_name"), []))
            mixed = True
        else:
            tag = [] if wh else self.x_by_tag.get((it.get("spk_name"), style_tag(it)), [])
            pool = tag if tag and others(tag) and rnd.random() < 0.8 else self.x_by_spk.get(it.get("spk_name"), [])
        want = rnd.randint(min(100, budget), budget)
        if mixed:  # a mixed reference uses the full reference length (several clips, something to choose between)
            budget = want = int(self.cfg.cross_max_sec * 50)
        seen, cand = {own}, []  # (11-14) one row per clip: oversampled copies of a clip are not two prompts
        for k in rnd.sample(pool, min(len(pool), 16)):
            h = self.items[k]["id"]
            if h not in seen:
                seen.add(h)
                cand.append(k)
        cand = cand[:10]
        got, chosen = 0, []
        for k in cand:
            p = self.items[k]["patches"]
            if (got + p) * self.cfg.patch > budget:  # (budget / want / 100 are FRAMES, got is patches: code review 2026-10-29, matters at patch 2/4)
                continue
            chosen.append(k)
            got += p
            if got * self.cfg.patch >= want:
                break
        return (tuple(chosen), got) if got * self.cfg.patch >= 100 else None

    def __getitem__(self, i):
        plan, planned = None, isinstance(i, tuple)  # (row, its planned prompt) from PlannedBatchSampler
        if planned:
            i, plan = i
        it = self.items[i]
        root = Path(it.get("root", self.root))
        if self.kvae:  # continuous latents (T, 64)
            z = self._read(root, "latents", it["id"])
            codes = z[: it["patches"] * self.cfg.patch].astype(np.float32).T
        else:
            codes = np.load(root / "codes" / f"{it['id']}.npy")[: self.cfg.n_q, : it["patches"] * self.cfg.patch]
        dirty = False  # target = this clip's damaged copy, labelled dirty (cfg.dirty_target_prob)
        if self.kvae and getattr(self.cfg, "dirty_target_prob", 0) > 0 and random.random() < self.cfg.dirty_target_prob:
            zd = self._read(root, "latents_corrupt", it["id"])
            if zd is not None and len(zd) >= codes.shape[1]:
                codes = np.asarray(zd[: codes.shape[1]], dtype=np.float32).T
                dirty = True
        lp_band = None  # target = a low-passed copy of this full-band clip, labelled band-limited (cfg.lowpass_prob)
        # only copies of clips MEASURED full band: a copy of an already-muffled clip sounds the same as its
        # original and teaches nothing about bandwidth
        if self.kvae and not dirty and getattr(self.cfg, "lowpass_prob", 0) > 0 and (it.get("band") or 0) >= 0.95 \
                and random.random() < self.cfg.lowpass_prob:
            zl = self._read(root, "latents_lowpass", it["id"])
            if zl is not None and len(zl) >= codes.shape[1]:
                loc = self._mm[(str(root), "latents_lowpass")][1][it["id"]]
                codes = np.asarray(zl[: codes.shape[1]], dtype=np.float32).T
                lp_band = float(np.clip(((loc[2] if len(loc) > 2 else 12.0) - 11.0) / 9.0, 0.0, 1.0))
        up_band = None  # target = an AudioSR-upsampled 48 kHz copy of this band-limited clip (cfg.upsampled_prob)
        up_clean = None  # ... or a Preen-cleaned + Plume copy (tools/latent_upscale_set.py --preen): its own clean label
        if self.kvae and not dirty and lp_band is None and getattr(self.cfg, "upsampled_prob", 0) > 0 \
                and random.random() < self.cfg.upsampled_prob:
            zu = self._read(root, "latents_upsampled", it["id"])
            if zu is not None and len(zu) >= 50:
                codes = np.asarray(zu[: max(1, len(zu) // self.cfg.patch) * self.cfg.patch], dtype=np.float32).T
                up_band = getattr(self.cfg, "upsampled_band", 0.9)  # near full band, but its own value
                up_clean = self._up_clean(root)
        # previous line's text (context), hidden part of the time so the model also works without it
        prev_text = None
        if it.get("prev_text") and random.random() >= getattr(self.cfg, "ctx_drop", 0.0):  # (torch seeds `random` per worker)
            prev_text = it["prev_text"]
        tokens, known_len = list(it["tokens"]), 0
        notext = bool(it.get("notext"))
        if not notext and getattr(self.cfg, "text_hide", 0) > 0 and random.random() < self.cfg.text_hide:
            # owner 2026-10-17: speech from "blankness" -- hide the words of some ordinary clips (one BLANK token,
            # no text losses, see DacTTS.forward), so the model can talk with no text given (voice + labels only)
            tokens, notext, prev_text = [T.BLANK], True, None
        song_done = False
        used = []  # the other clips given as this row's prompt (cfg.use_prompt_labels)
        segs = [(it["id"], codes.shape[1])]  # (clip id, latent frames) in order: the note features follow them
        if self.song_cont and it.get("song") is not None and random.random() < self.cfg.song_cont_prob:
            # "the bars before, then these bars": the clips right before this one in the same song (back-to-back, no gap)
            # are given; the model continues the melody / phrase / groove. With lyrics, the prompt clips need lyrics
            # too (their words are aligned to their audio); a no-lyrics clip continues no-lyrics (one BLANK token)
            budget = int(min(self.cfg.cross_max_sec * 50, max(150, codes.shape[1])))
            want = random.randint(min(100, budget), budget)
            za, ta, txt, got, seq, t0, pids = [], [], [], 0, int(it["seq"]), it.get("t0"), []
            while got < want:
                k = self.by_song.get((it["song"], seq - 1))
                if k is None:
                    break
                o = self.items[k]
                if t0 is None or o.get("t1") is None or abs(float(o["t1"]) - float(t0)) > 0.05:
                    break  # not back-to-back: a gap (skipped bars) breaks the continuation
                if (not notext and o.get("notext")) or got + o["patches"] > budget:
                    break
                z = self._read(Path(o.get("root", self.root)), "latents", o["id"])
                if z is None:
                    break
                za.insert(0, np.asarray(z[: o["patches"] * self.cfg.patch], dtype=np.float32).T)
                pids.insert(0, (o["id"], o["patches"] * self.cfg.patch))
                if not notext:
                    ta = list(o["tokens"]) + ta; txt.insert(0, o["text"])
                got += o["patches"]; seq -= 1; t0 = o.get("t0")
            if got >= 50:  # at least 1 s (a bar or two) of the song before
                codes = np.concatenate(za + [codes], axis=1)
                segs = pids + segs
                known_len = got * self.cfg.patch
                if notext:
                    tokens = [T.BLANK]  # no-lyrics continuation: one placeholder for prompt + new bars
                else:
                    tokens, prev_text = ta + tokens, " ".join(txt)
                song_done = True
        if planned:  # the prompt the sampler planned (and budgeted for); notext rows go without, as below
            if plan is not None and self.cross and not notext and not song_done:
                za, ta, txt, got, pids = [], [], [], 0, []
                for k in plan[0]:
                    o = self.items[k]
                    z = self._read(Path(o.get("root", self.root)), "latents", o["id"])
                    if z is None:
                        continue
                    used.append((o, o["patches"]))
                    za.append(np.asarray(z[: o["patches"] * self.cfg.patch], dtype=np.float32).T)
                    pids.append((None, o["patches"] * self.cfg.patch))
                    ta += o["tokens"]
                    txt.append(o["text"])
                    got += o["patches"]
                if got * self.cfg.patch < 100:  # (got: patches; 100 frames = 2 s)
                    used = []
                if got * self.cfg.patch >= 100:
                    codes = np.concatenate(za + [codes], axis=1)
                    segs = pids + segs
                    tokens, known_len = ta + tokens, got * self.cfg.patch
                    prev_text = " ".join(txt)
        elif self.cross and not notext and not song_done and random.random() < self.cfg.cross_prob:
            # "clip(s) A, then clip B", exactly like cloning: A's words + audio come first and are given; whole clips
            # only (their text must match their audio), together at most cross_max_sec and no longer than B
            budget = int(min(self.cfg.cross_max_sec * 50, max(150, codes.shape[1])))
            same = [k for k in self.x_by_tag.get((it.get("spk_name"), style_tag(it)), []) if self.items[k]["id"] != it["id"]]
            wh = (it.get("whisper") or 0) >= 0.5
            if wh:
                same = []  # whisper row: prompt = the same person talking normally (or whispering: cross_whisper_prob)
            pool = same if same and random.random() < 0.8 else \
                [k for k in self.x_by_spk.get(it.get("spk_name"), []) if self.items[k]["id"] != it["id"]]
            p_wh = getattr(self.cfg, "cross_whisper_prob" if wh else "cross_whisper_any", 0.0)
            wpool = [k for k in self.x_wh.get(it.get("spk_name"), []) if self.items[k]["id"] != it["id"]] if p_wh > 0 else []
            if wpool and random.random() < p_wh:
                pool = wpool
            elif getattr(self.cfg, "cross_mix_prob", 0.0) > 0 and random.random() < self.cfg.cross_mix_prob:
                pool = [k for k in list(self.x_by_spk.get(it.get("spk_name"), [])) + list(self.x_wh.get(it.get("spk_name"), []))
                        if self.items[k]["id"] != it["id"]]
            want = random.randint(min(100, budget), budget)
            za, ta, txt, got, pids = [], [], [], 0, []
            for k in random.sample(pool, min(len(pool), 10)):
                o = self.items[k]
                if got + o["patches"] > budget:
                    continue
                z = self._read(Path(o.get("root", self.root)), "latents", o["id"])
                if z is None:
                    continue
                used.append((o, o["patches"]))
                za.append(np.asarray(z[: o["patches"] * self.cfg.patch], dtype=np.float32).T)
                pids.append((None, o["patches"] * self.cfg.patch))  # another clip of the voice: its notes unknown
                ta += o["tokens"]
                txt.append(o["text"])
                got += o["patches"]
                if got * self.cfg.patch >= want:  # (got: patches; want and 100: frames)
                    break
            if got * self.cfg.patch < 100:
                used = []
            if got * self.cfg.patch >= 100:  # at least 2 s of prompt
                codes = np.concatenate(za + [codes], axis=1)
                segs = pids + segs
                tokens, known_len = ta + tokens, got * self.cfg.patch
                prev_text = " ".join(txt)  # the prompt's words are the "previous line", as when cloning
        repa = None
        if getattr(self.cfg, "repa_weight", 0) > 0 and not getattr(self.cfg, "_repa_off", False):  # SSL targets (T', repa_dim), padded/trimmed to the latents
            n = codes.shape[1]
            repa = np.zeros((n, self.cfg.repa_dim), np.float16)
            f = self._read(root, "repa", it["id"])  # None: no REPA target for this clip (loss masks it)
            if f is not None:
                f = f[:n]
                repa[: len(f)] = f
                if 0 < len(f) < n:  # SSL framing drops ~1 frame at the end
                    repa[len(f):] = f[-1]
        ref = None
        if self.zs:
            ref = np.full(self.cfg.zs_in, np.nan, np.float32)
            others = self._zs_others(it)
            if others:
                k = random.randint(1, min(self.cfg.zs_max_refs, len(others)))
                vs = [self._spkemb(r)[i] for r, i in random.sample(others, k)]
                v = np.mean(np.stack(vs).astype(np.float32), 0)
                ref = v / max(float(np.linalg.norm(v)), 1e-6)
                if getattr(self.cfg, "zs_noise", 0) > 0:
                    n = np.random.default_rng(random.getrandbits(32)).standard_normal(len(ref)).astype(np.float32)  # (numpy is not reseeded per worker)
                    ref = ref + n / np.linalg.norm(n) * random.uniform(0, self.cfg.zs_noise)
                    ref = ref / max(float(np.linalg.norm(ref)), 1e-6)
        dlv = None
        ref_pool = self._ref_pool(it) if self.dlv or self.vre else []  # (one pool for both: the same list twice)
        if self.dlv:
            pool = ref_pool
            if pool:
                r_root, r_id = random.choice(pool)
                z = self._read(r_root, "latents", r_id)
                if z is not None and len(z) >= 50:
                    n = min(len(z), random.randint(150, 400))  # 3-8 s
                    s = random.randint(0, len(z) - n)
                    dlv = np.asarray(z[s:s + n], dtype=np.float32)
        vre = None
        if self.vre:  # voice reference: 1-4 OTHER clips of the same speaker, concatenated (3-20 s of their latents)
            pool = ref_pool
            want = random.randint(150, int(self.cfg.vre_max_sec * 50))
            parts, got = [], 0
            for r_root, r_id in random.sample(pool, min(len(pool), 4)):
                z = self._read(r_root, "latents", r_id)
                if z is None or len(z) < 50:
                    continue
                parts.append(np.asarray(z[: want - got], dtype=np.float32))
                got += len(parts[-1])
                if got >= want:
                    break
            if got >= 100:
                vre = np.concatenate(parts)
        lm_tok = None
        if getattr(self.cfg, "lm_name", ""):  # tokenized here (worker) instead of in the training loop
            if getattr(self, "_tok", None) is None:
                from transformers import AutoTokenizer
                from .lm import TextLM
                self._tok, self._tokenize = AutoTokenizer.from_pretrained(self.cfg.lm_name), TextLM.tokenize
            lm_tok = self._tokenize(self._tok, "" if notext else it["text"], prev_text)
        zc = None
        if getattr(self.cfg, "corrupt_prob", 0) > 0 and not known_len and not dirty:  # damaged copy of this clip (make_corrupt.py), if made
            f = self._read(root, "latents_corrupt", it["id"])
            if f is not None:
                zc = np.asarray(f[: codes.shape[1]], dtype=np.float32)
        nan = float("nan")  # missing label (e.g. volume of a whispered clip)
        vol = it.get("volume", 0.5)
        band = it.get("band", 1.0) if it.get("band") is not None else 1.0
        band = lp_band if lp_band is not None else up_band if up_band is not None else band
        clean = 0.0 if dirty else up_clean if up_clean is not None else it.get("clean")
        if clean is not None and clean != clean:  # NaN (noise floor unmeasurable, e.g. near-silent upload): no label
            clean = None
        if getattr(self.cfg, "clean_band_rule", False) and clean is not None and band < 1.0:
            # "clean" = no noise AND full band: a muffled recording counts as (partly) dirty, so clean is learned only
            # from real (or upsampled) full-band audio and asking for clean pulls every voice to full band
            clean = float(clean) * float(band)
        return (tokens, codes, it.get("speaker", 0), nan if it.get("style", 0.5) is None else it.get("style", 0.5), nan if vol is None else vol,
                it.get("speed", 0.5), T.LANGS.get(it.get("lang", "en"), 0), "" if notext else it["text"], prev_text,
                {**{k: it.get(k) for k in ("mood", "dominance", "whisper", "sing", "bpm", "bpb", "key", "genre")},
                 "clean": clean, "band": band, "_alt": 1.0 if (dirty or lp_band is not None) else 2.0 if up_band is not None else 0.0},  # 1 damaged / low-passed, 2 upsampled
                repa, ref, lm_tok, dlv, zc, known_len, vre, notext,  # band: full-band unless the manifest says otherwise
                self._note_features(segs) if self.use_notes else None,
                self._prompt_labels(used) if used and getattr(self.cfg, "use_prompt_labels", False) else None,
                *(self._clip_label_rows(used) if getattr(self.cfg, "use_clip_labels", False) else (None, None)))


def collate(batch):
    b = len(batch)
    x_len = torch.tensor([len(b_[0]) for b_ in batch])
    y_len = torch.tensor([b_[1].shape[1] for b_ in batch])
    tokens = torch.zeros(b, int(x_len.max()), dtype=torch.long)
    cont = batch[0][1].dtype.kind == "f"  # KVAE latents (D, T) float vs DAC codes (n_q, T) int
    codes = torch.zeros(b, batch[0][1].shape[0], int(y_len.max()), dtype=torch.float32 if cont else torch.long)
    for i, (t, c, *_) in enumerate(batch):
        tokens[i, : len(t)] = torch.tensor(t)
        codes[i, :, : c.shape[1]] = torch.from_numpy(c if cont else c.astype(np.int64))
    if cont:
        codes = codes.transpose(1, 2)  # (B, T, D) like LatentSpace.encode output
    spk = torch.tensor([b_[2] for b_ in batch])
    style = torch.tensor([b_[3] for b_ in batch], dtype=torch.float32)
    volume = torch.tensor([b_[4] for b_ in batch], dtype=torch.float32)
    speed = torch.tensor([b_[5] for b_ in batch], dtype=torch.float32)
    lang = torch.tensor([b_[6] for b_ in batch])
    ctx = {"texts": [b_[7] for b_ in batch], "prev_texts": [b_[8] for b_ in batch],
           "labels": {k: torch.tensor([float("nan") if b_[9].get(k) is None else b_[9][k] for b_ in batch])
                      for k in ("mood", "dominance", "whisper", "clean", "band")}}
    # (11-14) rows whose TARGET is a damaged / low-passed / upsampled copy (the prompt is the original): kept out of the critic
    ctx["alt_tgt"] = torch.tensor([float(b_[9].get("_alt", 0.0)) for b_ in batch])  # 0 original, 1 damaged / low-passed, 2 upsampled
    # singing labels (dacts/music.py; manifest values already 0..1): only when the batch has any; missing = NaN = dropped
    for k in ("sing", "bpm", "bpb", "key"):
        if any(b_[9].get(k) is not None for b_ in batch):
            ctx["labels"][k] = torch.tensor([float("nan") if b_[9].get(k) is None else float(b_[9][k]) for b_ in batch])
    if any(b_[9].get("genre") is not None for b_ in batch):  # multi-hot vector; a row of NaN = no genre
        n_g = len(next(b_[9]["genre"] for b_ in batch if b_[9].get("genre") is not None))
        ctx["labels"]["genre"] = torch.tensor([[float("nan")] * n_g if b_[9].get("genre") is None else list(b_[9]["genre"])
                                               for b_ in batch], dtype=torch.float32)
    wid = torch.full(tokens.shape, -1, dtype=torch.long)  # word index per token (emphasis control)
    for i, (t, *_) in enumerate(batch):
        wid[i, : len(t)] = torch.tensor(T.word_ids(t))
    ctx["word_ids"] = wid
    if len(batch[0]) > 13 and any(b_[13] is not None for b_ in batch):  # delivery reference latents
        n = max(len(b_[13]) for b_ in batch if b_[13] is not None)
        lat = torch.zeros(b, n, batch[0][1].shape[0])
        m = torch.zeros(b, n, dtype=torch.bool)
        for i, b_ in enumerate(batch):
            if b_[13] is not None:
                lat[i, : len(b_[13])], m[i, : len(b_[13])] = torch.from_numpy(b_[13]), True
        ctx["dlv"] = (lat, m)
    if len(batch[0]) > 16 and any(b_[16] is not None for b_ in batch):  # voice reference latents (cfg.use_vre)
        n = max(len(b_[16]) for b_ in batch if b_[16] is not None)
        lat = torch.zeros(b, n, batch[0][1].shape[0])
        m = torch.zeros(b, n, dtype=torch.bool)
        for i, b_ in enumerate(batch):
            if b_[16] is not None:
                lat[i, : len(b_[16])], m[i, : len(b_[16])] = torch.from_numpy(b_[16]), True
        ctx["vre"] = (lat, m)
    if len(batch[0]) > 18 and any(b_[18] is not None for b_ in batch):  # per-frame notes (cfg.use_notes)
        T_ = max(len(b_[18]) for b_ in batch if b_[18] is not None)
        nt_ = torch.zeros(b, T_, 4)
        for i, b_ in enumerate(batch):
            if b_[18] is not None:
                nt_[i, : len(b_[18])] = torch.from_numpy(b_[18].astype(np.float32))
        ctx["notes"] = nt_
    if len(batch[0]) > 19 and any(b_[19] is not None for b_ in batch):  # how the cross-clip prompts were delivered
        nan = float("nan")
        ctx["prompt_labels"] = {k: torch.tensor([nan if b_[19] is None or b_[19].get(k) is None else float(b_[19][k])
                                                 for b_ in batch], dtype=torch.float32)
                                for k in ("whisper", "style", "mood", "dominance", "volume", "speed", "clean", "band")}
    if len(batch[0]) > 20 and any(b_[20] is not None for b_ in batch):  # per-clip reference labels (patches / words)
        n_l = next(b_[20] for b_ in batch if b_[20] is not None).shape[1]
        pp = torch.full((b, max(len(b_[20]) for b_ in batch if b_[20] is not None), n_l), float("nan"))
        pt = torch.full((b, tokens.shape[1], n_l), float("nan"))
        for i, b_ in enumerate(batch):
            if b_[20] is not None:
                pp[i, : len(b_[20])] = torch.from_numpy(b_[20])
                pt[i, : len(b_[21])] = torch.from_numpy(b_[21][: tokens.shape[1]])
        ctx["prompt_patch_labels"], ctx["prompt_token_labels"] = pp, pt
    if len(batch[0]) > 17 and any(b_[17] for b_ in batch):  # untranscribed rows
        ctx["notext"] = torch.tensor([bool(b_[17]) for b_ in batch])
    if len(batch[0]) > 15 and any(b_[15] for b_ in batch):  # cross-clip rows: frames given as the prompt (clip A)
        ctx["known_len"] = torch.tensor([b_[15] for b_ in batch])
    if len(batch[0]) > 14 and any(b_[14] is not None for b_ in batch):  # damaged copies (prompt robustness)
        zc = torch.zeros(b, int(y_len.max()), batch[0][1].shape[0])
        has = torch.zeros(b, dtype=torch.bool)
        for i, b_ in enumerate(batch):
            if b_[14] is not None:
                zc[i, : len(b_[14])], has[i] = torch.from_numpy(b_[14]), True
        ctx["corrupt"] = (zc, has)
    if len(batch[0]) > 12 and batch[0][12] is not None:  # LM tokens (padded here, run on the GPU later)
        n = max(len(b_[12][0]) for b_ in batch)
        ids = torch.ones(b, n, dtype=torch.long)  # 1 = XLM-R <pad>
        mask = torch.zeros(b, n, dtype=torch.bool)
        seg = torch.zeros(b, n, dtype=torch.long)
        for i, b_ in enumerate(batch):
            k = len(b_[12][0])
            ids[i, :k], mask[i, :k], seg[i, :k] = torch.tensor(b_[12][0]), True, torch.tensor(b_[12][1])
        ctx["lm_tokens"] = (ids, mask, seg)
    if len(batch[0]) > 11 and batch[0][11] is not None:
        ctx["ref"] = torch.from_numpy(np.stack([b_[11] for b_ in batch]))
    if batch[0][10] is not None:
        r = torch.zeros(b, int(y_len.max()), batch[0][10].shape[1], dtype=torch.float16)
        for i, b_ in enumerate(batch):
            r[i, : len(b_[10])] = torch.from_numpy(b_[10])
        ctx["repa"] = r
    return tokens, x_len, codes, y_len, spk, style, volume, speed, lang, ctx


class PlannedBatchSampler(Sampler):
    """BucketBatchSampler for cross-clip training: each epoch the prompts are planned first (ds.plan_cross), then rows
    are bucketed by their REAL decoder length (row + planned prompt) and packed up to max_tokens padded tokens.
    Yields lists of (row, plan) for CodesDataset.__getitem__."""

    def __init__(self, ds, max_tokens, shuffle=True, margin=2, longest_first=False):
        self.ds, self.max_tokens, self.shuffle, self.margin = ds, max_tokens, shuffle, margin
        self.longest_first = longest_first
        self.rnd, self._up_idx = random.Random(), {}
        self.batches = self._make()

    def _make(self):
        n, items, rnd = len(self.ds.items), self.ds.items, self.rnd
        plans = [self.ds.plan_cross(i, rnd) for i in range(n)]
        # the target can be the clip's AudioSR copy (cfg.upsampled_prob), which is LONGER where preprocessing trimmed the
        # clip's silence but AudioSR got the untrimmed file (Common Voice: up to +1.4 s): budget the longer of the two
        tgt = [max(items[i]["patches"], self._up_len(items[i])) for i in range(n)]
        lengths = [tgt[i] + (plans[i][1] if plans[i] else 0) + self.margin for i in range(n)]
        jitter = (lambda: rnd.random()) if self.shuffle else (lambda: 0.0)
        order = sorted(range(n), key=lambda i: lengths[i] + jitter())
        batches, cur, longest = [], [], 0
        for i in order:
            l = max(longest, lengths[i])
            if cur and l * (len(cur) + 1) > self.max_tokens:
                batches.append(cur)
                cur, l = [], lengths[i]
            cur.append((i, plans[i]))
            longest = l
        if cur:
            batches.append(cur)
        if self.shuffle:
            rnd.shuffle(batches)
        if self.longest_first:  # memory test: the biggest padded batches (longest rows) first
            batches.sort(key=lambda b: -max(lengths[i] for i, _ in b))
        return batches

    def _up_len(self, it):
        """Patches of the clip's AudioSR copy (0 if none / not used): from its dataset's upsampled index."""
        if getattr(self.ds.cfg, "upsampled_prob", 0) <= 0:
            return 0
        root = str(it.get("root", self.ds.root))
        if root not in self._up_idx:
            f = Path(root) / "shards" / "latents_upsampled_index.json"
            self._up_idx[root] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        loc = self._up_idx[root].get(it["id"])
        return 0 if loc is None else loc[1] // self.ds.cfg.patch

    def __iter__(self):
        yield from self.batches
        self.batches = self._make()

    def __len__(self):
        return len(self.batches)


class BucketBatchSampler(Sampler):
    """Batches of similar length, capped by total padded decoder tokens."""

    def __init__(self, lengths, max_tokens, shuffle=True):
        self.lengths, self.max_tokens, self.shuffle = lengths, max_tokens, shuffle
        self.batches = self._make()

    def _make(self):
        jitter = (lambda: random.random()) if self.shuffle else (lambda: 0.0)
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i] + jitter())
        batches, cur, longest = [], [], 0
        for i in order:
            l = max(longest, self.lengths[i])
            if cur and l * (len(cur) + 1) > self.max_tokens:
                batches.append(cur)
                cur, l = [], self.lengths[i]
            cur.append(i)
            longest = l
        if cur:
            batches.append(cur)
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self.batches
        self.batches = self._make()

    def __len__(self):
        return len(self.batches)


TRAIN_UNUSED = ("prosody", "emotion", "dirt", "band_khz", "band_src")  # measurement details training never reads


class Item:
    """A manifest item for training, ~4x smaller than its dict (every DataLoader worker holds all ~2.4M of them).
    Behaves like the dict where training uses it: it[k], it[k] = v, it.get(k, d), k in it, it.pop(k, d)."""
    _F = ("id", "text", "frames", "spk_name", "lang", "speaker", "root", "style", "volume", "speed", "mood",
          "dominance", "whisper", "clean", "band", "tokens", "patches", "has_corrupt", "prev_text", "notext",
          "sing", "bpm", "bpb", "key", "genre", "no_trim", "song", "seq", "t0", "t1")
    __slots__ = _F + ("_x",)
    _MISSING = object()

    def __init__(self, d):
        x = None
        for k in Item._F:
            object.__setattr__(self, k, Item._MISSING)
        for k, v in d.items():
            if k in Item.__slots__ and k != "_x":
                object.__setattr__(self, k, sys.intern(v) if k in ("spk_name", "lang", "root") and isinstance(v, str) else v)
            else:
                x = x or {}
                x[k] = v
        object.__setattr__(self, "_x", x)

    def get(self, k, default=None):
        if k in Item._F:
            v = getattr(self, k)
            return default if v is Item._MISSING else v
        return (self._x or {}).get(k, default)

    def __getitem__(self, k):
        v = self.get(k, Item._MISSING)
        if v is Item._MISSING:
            raise KeyError(k)
        return v

    def __setitem__(self, k, v):
        if k in Item._F:
            object.__setattr__(self, k, v)
        else:
            if self._x is None:
                object.__setattr__(self, "_x", {})
            self._x[k] = v

    def __contains__(self, k):
        return self.get(k, Item._MISSING) is not Item._MISSING

    def pop(self, k, default=None):
        v = self.get(k, default)
        if k in Item._F:
            object.__setattr__(self, k, Item._MISSING)
        elif self._x:
            self._x.pop(k, None)
        return v

    def __getstate__(self):  # (pickled into every DataLoader worker) present fields only: the _MISSING sentinel
        # must not travel, an unpickled copy would be a different object and no longer read as "missing"
        return {k: getattr(self, k) for k in Item._F if getattr(self, k) is not Item._MISSING}, self._x

    def __setstate__(self, st):
        fields, x = st
        for k in Item._F:
            object.__setattr__(self, k, fields.get(k, Item._MISSING))
        object.__setattr__(self, "_x", x)


def load_datasets(paths, base_speakers=None, slim=False):
    """Merge several preprocessed dataset dirs (e.g. English + Japanese) for one training run.

    Speaker ids of the first dir are kept as-is; speakers of later dirs are appended. Latent
    normalization stats are combined weighted by frames. Conditioning maps come from the first dir.
    Returns items (each with its own "root"), speakers {name: id}, stats dict, conditioning dict.
    slim: drop TRAIN_UNUSED from every item (training): every DataLoader worker holds its own copy of all items, and
    with ~2.4M clips those nested dicts made each worker ~13 GB.
    """
    import torch

    # base_speakers: the base model's {name: id}; new voices are appended (fine-tunes without the base data)
    speakers, items, parts, conds = dict(base_speakers or {}), [], [], []
    for p in map(Path, paths):
        spk = json.loads((p / "speakers.json").read_text())
        for name in sorted(spk, key=spk.get):
            speakers.setdefault(name, len(speakers))
        inv = {v: k for k, v in spk.items()}
        dir_items = load_manifest(p)
        root = str(p)
        for it in dir_items:
            it["speaker"] = speakers[it.get("spk_name") or inv[it["speaker"]]]
            it["root"] = root  # one shared string per dir
            if slim:
                for k in TRAIN_UNUSED:
                    it.pop(k, None)
        if slim:
            dir_items = [Item(it) for it in dir_items]
        items += dir_items
        parts.append((torch.load(p / "stats.pt"), sum(it["frames"] for it in dir_items)))
        cpath = p / "conditioning.json"
        conds.append(json.loads(cpath.read_text()) if cpath.exists() else {})
    total = sum(w for _, w in parts)
    mean = sum(st["mean"].double() * w for st, w in parts) / total
    second = sum((st["std"].double() ** 2 + st["mean"].double() ** 2) * w for st, w in parts) / total
    stats = {"mean": mean.float(), "std": (second - mean**2).clamp(min=1e-8).sqrt().float()}
    if "codebooks" in parts[0][0]:
        stats["codebooks"] = parts[0][0]["codebooks"]
    else:
        stats["codec"] = "kvae"
    conditioning = {}
    for c in reversed(conds):
        conditioning.update(c)  # first dir wins
    return items, speakers, stats, conditioning
