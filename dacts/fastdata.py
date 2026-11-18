"""Prebuilt, shared training data (owner 2026-10-22: "skip this prebuilding and go straight to training").

train.py rebuilt the whole data index at every start (v13xl: ~16 min for 3.5M manifest rows, mostly turning every
text into tokens -- Japanese ~3 ms a line), and each DataLoader worker got its own pickled copy of it (~7.5 GB a
worker: Windows has no fork). Here the same index is built ONCE into flat arrays under data/_cache/train_<key>/ and
opened memory-mapped: a restart reads it in seconds, and all workers share one copy through the OS file cache.

The semantics are exactly train.py's + dacts.data's (load_datasets, the val split, corrupt flags, voice caps,
oversampling, CodesDataset's filter and prompt pools, PlannedBatchSampler); tests/test_fastdata.py compares the two
on real data. The cache key covers the data dirs (file sizes / times), the config fields that shape the data, the
base model's speakers and this code, so a stale cache is never used.

Prebuild for a launcher (low priority, ~2-3 GB RAM):  python -m dacts.fastdata data/_train_v13xl.cmd
"""
import hashlib
import json
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import text as T
from .codec import frame_rate
from .data import TRAIN_UNUSED, CodesDataset, PlannedBatchSampler, style_tag

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "_cache"
VERSION = 3  # (3, 2026-11-16: tokens 2 bytes each ('<u2'): Ukrainian letters are ids 244-276, past one byte)
# bump when the cache's layout / meaning changes (the code fingerprint covers the build itself)
TRI = ("style", "volume", "speed", "mood", "dominance", "whisper", "clean", "band")  # labels: missing / None / value
UNSUPPORTED = ("sing", "bpm", "bpb", "key", "genre", "song", "seq", "t0", "t1")       # singing (gone since v12)
KINDS = ("latents", "latents_corrupt", "latents_lowpass", "latents_upsampled")         # shard indexes kept as arrays
CFG_KEYS = ("max_seconds", "patch", "add_blank", "codec", "val_split", "n_val", "corrupt_prob", "voice_cap",
            "oversample", "only_speakers", "cross_prob", "use_dlv", "use_vre", "zs_weight", "zs_holdout",
            "use_notes", "song_cont_prob", "lm_name", "upsampled_prob", "exclude_speakers")
CODE = ("dacts/text.py",)  # whole files in the key; plus the build's own functions (_code_fingerprint)


class Unsupported(Exception):
    """Data this fast path doesn't model (singing fields, only_speakers): train.py then uses the normal path."""


def h64(s) -> int:
    """Stable 64-bit hash (Python's own hash() changes per process)."""
    return int.from_bytes(hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest(), "little")


def _spk_key(spk):  # x_by_spk / dlv / zs pools are keyed by spk_name, which can be None
    return h64("S\x00" + ("\x01None" if spk is None else spk))


def _tag_key(spk, tag):
    return h64("T\x00" + ("\x01None" if spk is None else spk) + "\x00" + tag)


# ------------------------------------------------------------------------------------------------- cache key
def _code_fingerprint() -> str:
    """What the cache's CONTENTS depend on: the build and everything it calls (tokenizing, speaker / session keys,
    pools, the frame rate) plus dacts/text.py. Run-time code -- how training picks references, labels, the batch
    sampler -- isn't in it, so changing that needs no rebuild (it used to: any edit to data.py or fastdata.py did)."""
    import inspect
    from . import codec as C
    from . import data as D
    funcs = (build, _tok_job, _Strings, _csr, h64, _spk_key, _tag_key, D.style_tag, C.frame_rate)
    parts = [inspect.getsource(f) for f in funcs] + [(ROOT / f).read_text(encoding="utf-8") for f in CODE]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def cache_key(paths, cfg, base_speakers, context) -> str:
    def stamp(p):
        p = Path(p)
        if not p.exists():
            return [p.name, None]
        s = p.stat()
        return [p.name, s.st_size, int(s.st_mtime)]
    files = ["manifest.jsonl", "speakers.json", "stats.pt", "conditioning.json", "spkemb_ids.json"] + \
            [f"shards/{k}_index.json" for k in KINDS]
    blob = {"v": VERSION, "dirs": [[str(Path(d)), [stamp(Path(d) / f) for f in files]] for d in paths],
            "cfg": {k: getattr(cfg, k, None) for k in CFG_KEYS},
            "base": hashlib.sha256(json.dumps(base_speakers or {}, sort_keys=True).encode()).hexdigest(),
            "context": stamp(context) if getattr(cfg, "lm_name", "") else None,
            "code": _code_fingerprint()}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------- build
def _tok_job(job):
    add_blank, rows = job
    return [np.asarray(T.encode(t, add_blank, lang), "<u2").tobytes() for t, lang in rows]


class _Strings:
    """A string column written as one UTF-8 blob + offsets (no Python objects kept)."""

    def __init__(self):
        self.parts, self.lens = [], []

    def add(self, s):
        b = (s or "").encode("utf-8")
        self.parts.append(b)
        self.lens.append(len(b))

    def save(self, d, name):
        off = np.zeros(len(self.lens) + 1, np.int64)
        np.cumsum(np.asarray(self.lens, np.int64), out=off[1:])
        np.save(d / f"{name}_off.npy", off)
        np.save(d / f"{name}_blob.npy", np.frombuffer(b"".join(self.parts), np.uint8))


def _csr(keys, values):
    """keys (uint64, one per value) -> (sorted unique keys, offsets, values grouped by key, original order kept)."""
    keys = np.asarray(keys, np.uint64)
    values = np.asarray(values, np.int64)
    order = np.argsort(keys, kind="stable")
    k, v = keys[order], values[order]
    uk, start = np.unique(k, return_index=True)
    off = np.append(start, len(k)).astype(np.int64)
    return uk, off, v


def build(paths, cfg, base_speakers, context, out, workers=3, log=print):
    import torch
    from multiprocessing import Pool

    t_start = time.time()
    out = Path(out)
    tmp = out.with_name(out.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if getattr(cfg, "only_speakers", ""):
        raise Unsupported("only_speakers")
    if getattr(cfg, "use_notes", False) or getattr(cfg, "song_cont_prob", 0) > 0:
        raise Unsupported("singing features")
    add_blank = cfg.add_blank
    ctx = {}
    if getattr(cfg, "lm_name", "") and context and Path(context).exists():  # previous line of each clip (train.py)
        for line in open(context, encoding="utf-8"):
            c = json.loads(line)
            ctx[c["id"]] = c.get("prev_text")
    speakers = dict(base_speakers or {})
    roots, langs, tags, spk_names = [], [], [], []
    lang_ix, tag_ix, spkname_ix = {}, {}, {}
    ids, texts, prevs, spk_tab = _Strings(), _Strings(), _Strings(), _Strings()
    num = {k: [] for k in ("root", "speaker", "frames", "lang", "spk", "spk_s", "tag", "notext_s", "prev_s")}
    tri = {k: ([], []) for k in TRI}
    locs = {k: ([], [], []) for k in KINDS}
    tokens, parts, conds, langs_by_spk = [], [], [], {}
    pool = Pool(workers) if workers > 1 else None
    for p in map(Path, paths):
        t0 = time.time()
        spk = json.loads((p / "speakers.json").read_text())
        for name in sorted(spk, key=spk.get):
            speakers.setdefault(name, len(speakers))
        inv = {v: k for k, v in spk.items()}
        root = str(p)
        rid = len(roots)
        roots.append(root)
        with open(p / "manifest.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]  # one dir at a time (<= ~372k rows)
        n = len(rows)
        a = {k: np.zeros(n, np.int32) for k in num}
        tv = {k: (np.zeros(n, np.float32), np.zeros(n, np.uint8)) for k in TRI}
        tok_rows, frames_sum = [], 0
        for j, it in enumerate(rows):
            for k in TRAIN_UNUSED:
                it.pop(k, None)
            for k in UNSUPPORTED:
                if it.get(k) is not None:
                    raise Unsupported(f"{p.name}: {k}")
            a["speaker"][j] = speakers[it.get("spk_name") or inv[it["speaker"]]]
            a["root"][j] = rid
            a["frames"][j] = it["frames"]
            frames_sum += it["frames"]
            lg = it.get("lang", Ellipsis)
            if lg is Ellipsis or lg is None:
                a["lang"][j] = -1 if lg is Ellipsis else -2
            else:
                a["lang"][j] = lang_ix.setdefault(lg, len(lang_ix))
            if "spk_name" not in it:
                a["spk_s"][j] = 0
            elif it["spk_name"] is None:
                a["spk_s"][j] = 1
            else:
                a["spk_s"][j] = 2
                s = it["spk_name"]
                if s not in spkname_ix:
                    spkname_ix[s] = len(spkname_ix)
                    spk_tab.add(s)
                a["spk"][j] = spkname_ix[s]
            a["tag"][j] = tag_ix.setdefault(style_tag(it), len(tag_ix))
            nt = it.get("notext", Ellipsis)
            a["notext_s"][j] = 0 if nt is Ellipsis else 1 if nt is None else 2 + bool(nt)
            for k in TRI:
                v = it.get(k, Ellipsis)
                if v is Ellipsis:
                    continue
                if v is None:
                    tv[k][1][j] = 1
                else:
                    tv[k][0][j], tv[k][1][j] = float(v), 2
            pt = it.get("prev_text", Ellipsis)  # manifest prev_text, then the context file (train.py: "if c")
            if it["id"] in ctx:
                pt = ctx[it["id"]]
            a["prev_s"][j] = 0 if pt is Ellipsis else 1 if pt is None else 2
            prevs.add(pt if isinstance(pt, str) else "")
            ids.add(it["id"])
            texts.add(it["text"])
            sname = it.get("spk_name")
            langs_by_spk.setdefault(sname, Counter())[it.get("lang", "en")] += 1  # train.py: speaker_langs
            tok_rows.append(None if nt is not Ellipsis and nt else (it["text"], it.get("lang", "en")))
        # tokens (the slow part: Japanese ~3 ms a line), in worker processes
        todo = [r for r in tok_rows if r is not None]
        chunks = [(add_blank, todo[i:i + 2000]) for i in range(0, len(todo), 2000)]
        done = [b for part in (pool.imap(_tok_job, chunks) if pool else map(_tok_job, chunks)) for b in part]
        it_done = iter(done)
        tokens += [np.asarray([T.BLANK], "<u2").tobytes() if r is None else next(it_done) for r in tok_rows]
        for k in num:
            num[k].append(a[k])
        for k in TRI:
            tri[k][0].append(tv[k][0])
            tri[k][1].append(tv[k][1])
        for kind in KINDS:  # where each clip's latents sit in the dir's shard files
            st_, ln_, ex_ = np.full(n, -1, np.int64), np.zeros(n, np.int64), np.full(n, np.nan, np.float32)
            f = p / "shards" / f"{kind}_index.json"
            if f.exists():
                idx = json.loads(f.read_text(encoding="utf-8"))
                for j, it in enumerate(rows):
                    loc = idx.get(it["id"])
                    if loc is not None:
                        st_[j], ln_[j] = loc[0], loc[1]
                        if len(loc) > 2:
                            ex_[j] = loc[2]
                del idx
            for arr, x in zip(locs[kind], (st_, ln_, ex_)):
                arr.append(x)
        parts.append((torch.load(p / "stats.pt"), frames_sum))
        cpath = p / "conditioning.json"
        conds.append(json.loads(cpath.read_text()) if cpath.exists() else {})
        log(f"  {p.name}: {n} clips ({time.time() - t0:.0f} s)")
        del rows, tok_rows, done
    if pool:
        pool.close()
    # ---- columns
    col = {k: np.concatenate(v) for k, v in num.items()}
    N = len(col["frames"])
    for k in TRI:
        col[k], col[k + "_s"] = np.concatenate(tri[k][0]), np.concatenate(tri[k][1])
    for kind in KINDS:
        col[kind + "_start"], col[kind + "_len"], col[kind + "_x"] = (np.concatenate(x) for x in locs[kind])
    tok_off = np.zeros(N + 1, np.int64)
    np.cumsum([len(b) // 2 for b in tokens], out=tok_off[1:])  # (in tokens, not bytes)
    np.save(tmp / "tok_off.npy", tok_off)
    np.save(tmp / "tok_blob.npy", np.frombuffer(b"".join(tokens), "<u2"))
    tok_len = np.diff(tok_off)
    del tokens
    for s, name in ((ids, "id"), (texts, "text"), (prevs, "prev"), (spk_tab, "spkname")):
        s.save(tmp, name)
    id_list = [bytes(x).decode("utf-8") for x in np.split(np.load(tmp / "id_blob.npy"), np.load(tmp / "id_off.npy")[1:-1])]
    lang_tab = list(lang_ix)
    spk_tab_list = list(spkname_ix)
    tag_tab = list(tag_ix)
    patch = cfg.patch
    patches = col["frames"] // patch
    whisper = np.where(col["whisper_s"] == 2, col["whisper"], 0.0) >= 0.5  # (it.get("whisper") or 0) >= 0.5

    def lang_of(c):
        v = col["lang"][c]
        return "en" if v == -1 else None if v == -2 else lang_tab[v]

    def spk_of(c, missing=None):
        s = col["spk_s"][c]
        return missing if s == 0 else None if s == 1 else spk_tab_list[col["spk"][c]]

    # ---- train.py: shuffle, val split, corrupt flags, voice caps, oversampling (same order of random calls)
    items = list(range(N))
    random.Random(0).shuffle(items)
    split = str(getattr(cfg, "val_split", "auto") or "auto").strip().lower()
    both = split == "both" or (split == "auto" and N < 20)
    n_val = max(1, round(N * float(split[:-1]) / 100)) if split.endswith("%") else cfg.n_val
    n_val = max(0, min(n_val, N - (0 if both else 1)))
    val_items, train_items = items[:n_val], items[n_val:]
    for lang in sorted({lang_of(i) for i in items} - {lang_of(i) for i in val_items}, key=str):
        extra = next((i for i in train_items if lang_of(i) == lang), None)
        if extra is not None and (both or len(train_items) > 1):
            train_items.remove(extra)
            val_items.append(extra)
    if both:
        train_items = list(items)
    excl = set(filter(None, getattr(cfg, "exclude_speakers", "").split(",")))
    if excl:  # voices that must never be trained on again (dashboard/access.py REMOVED_VOICES, the legal pages' promise):
        # out of training, validation and every prompt / zero-shot pool (all built from these lists below)
        n0 = len(train_items) + len(val_items)
        train_items = [i for i in train_items if spk_of(i) not in excl]
        val_items = [i for i in val_items if spk_of(i) not in excl]
        log(f"excluded voices {sorted(excl)}: {n0 - len(train_items) - len(val_items)} clips left out")
    has_corrupt = (col["latents_corrupt_start"] >= 0) if getattr(cfg, "corrupt_prob", 0) > 0 else np.zeros(N, bool)
    rootname = [Path(r).name for r in roots]
    for spec in filter(None, getattr(cfg, "voice_cap", "").split(",")):
        key, cap = spec.split(":")
        rnd, per, keep = random.Random(0), {}, []
        mine = [i for i in train_items if rootname[col["root"][i]] == key]
        rnd.shuffle(mine)
        for i in mine:
            s = spk_of(i)
            per[s] = per.get(s, 0) + 1
            if per[s] <= int(cap):
                keep.append(i)
        drop = set(mine) - set(keep)
        train_items = [i for i in train_items if i not in drop]
        log(f"voice cap {key}: {len(mine)} -> {len(keep)} clips ({len(per)} voices, <= {cap} each)")
    for spec in filter(None, cfg.oversample.split(",")):
        key, k = spec.rsplit(":", 1)
        if key.startswith("@"):  # (11-15) a file of voice names, one per line, e.g. @data/female_voices.txt:2
            names = {ln.strip() for ln in open(key[1:], encoding="utf-8") if ln.strip()}
            hit = lambda i, names=names: str(spk_of(i, "")) in names  # noqa: E731
            log(f"oversample {key}: {len(names)} voices x{k}")
        elif key.endswith("*"):
            hit = lambda i, pre=key[:-1]: str(spk_of(i, "")).startswith(pre)  # noqa: E731
        elif key == "whisper":
            hit = lambda i: bool(whisper[i])  # noqa: E731
        elif key == "corrupt":
            hit = lambda i: bool(has_corrupt[i])  # noqa: E731
        else:
            hit = lambda i, key=key: key in (lang_of(i), spk_of(i), rootname[col["root"][i]])  # noqa: E731
        train_items += [i for i in train_items if hit(i)] * (int(k) - 1)
    # ---- CodesDataset: the filter, then its pools (over rows, oversampled copies included)
    max_frames = int(cfg.max_seconds * frame_rate(cfg))
    keep = (tok_len > 0) & (tok_len <= patches) & (col["frames"] <= max_frames)
    rows = np.asarray([i for i in train_items if keep[i]], np.int32)
    R = len(rows)
    rw = whisper[rows]
    spk_key = np.asarray([_spk_key(spk_of(c)) for c in range(N)], np.uint64)
    tag_key = np.asarray([_tag_key(spk_of(c), tag_tab[col["tag"][c]]) for c in range(N)], np.uint64)
    nw = np.flatnonzero(~rw)
    wr = np.flatnonzero(rw)  # whispered rows: x_wh (whispered prompts, cfg.cross_whisper_*)
    for name, keys, sel in (("dlv", spk_key, nw), ("xspk", spk_key, nw), ("xtag", tag_key, nw), ("xwh", spk_key, wr)):
        uk, off, v = _csr(keys[rows[sel]], sel)
        np.save(tmp / f"{name}_keys.npy", uk)
        np.save(tmp / f"{name}_off.npy", off)
        np.save(tmp / f"{name}_rows.npy", v.astype(np.int32))
    # zero-shot refs: uid -> (root, row) of <root>/spkemb_ids.json (later roots win, as the dict in CodesDataset)
    zh, zr, zw = [], [], []
    for rid, root in sorted(enumerate(roots), key=lambda x: x[1]):  # (sorted by path, last wins: as dacts/data.py)
        f = Path(root) / "spkemb_ids.json"
        if f.exists():
            u = json.loads(f.read_text(encoding="utf-8"))
            zh.append(np.asarray([h64(x) for x in u], np.uint64))
            zr.append(np.full(len(u), rid, np.int32))
            zw.append(np.arange(len(u), dtype=np.int64))
    idh = np.asarray([h64(x) for x in id_list], np.uint64)
    zs_root, zs_row = np.full(N, -1, np.int32), np.full(N, -1, np.int64)
    if zh:
        zh, zr, zw = np.concatenate(zh), np.concatenate(zr), np.concatenate(zw)
        uk, last = np.unique(zh[::-1], return_index=True)  # keep the LAST occurrence
        last = len(zh) - 1 - last
        pos = np.searchsorted(uk, idh)
        ok = (pos < len(uk)) & (uk[np.minimum(pos, len(uk) - 1)] == idh)
        zs_root[ok], zs_row[ok] = zr[last[pos[ok]]], zw[last[pos[ok]]]
        np.save(tmp / "zsu_keys.npy", uk)
        np.save(tmp / "zsu_root.npy", zr[last])
        np.save(tmp / "zsu_row.npy", zw[last])
    hold = set(filter(None, getattr(cfg, "zs_holdout", "").split(",")))
    zsel = np.asarray([k for k in range(R) if zs_root[rows[k]] >= 0 and spk_of(rows[k]) not in hold and not rw[k]], np.int64)
    uk, off, v = _csr(spk_key[rows[zsel]] if len(zsel) else [], zsel)
    np.save(tmp / "zs_keys.npy", uk)
    np.save(tmp / "zs_off.npy", off)
    np.save(tmp / "zs_rows.npy", v.astype(np.int32))
    # (root, uid) -> clip: the shard lookups
    rk = np.asarray([h64(f"{col['root'][c]}\x00{id_list[c]}") for c in range(N)], np.uint64)
    order = np.argsort(rk, kind="stable")
    np.save(tmp / "ru_keys.npy", rk[order])
    np.save(tmp / "ru_clip.npy", order.astype(np.int32))
    # ---- per clip / per row arrays
    up = np.where(col["latents_upsampled_start"] >= 0, col["latents_upsampled_len"] // patch, 0) \
        if getattr(cfg, "upsampled_prob", 0) > 0 else np.zeros(N, np.int64)
    for k, v in col.items():
        np.save(tmp / f"{k}.npy", v)
    for k, v in (("idh", idh), ("spk_key", spk_key), ("tag_key", tag_key), ("zs_root", zs_root), ("zs_row", zs_row),
                 ("has_corrupt", has_corrupt), ("rows", rows), ("up", up.astype(np.int32))):
        np.save(tmp / f"{k}.npy", v)
    # ---- what train.py prints / keeps
    fr = frame_rate(cfg)
    hours = float(col["frames"][rows].sum()) / fr / 3600
    by_lang = {}
    for c in rows:
        lg = lang_of(c) or "en"
        by_lang[lg] = by_lang.get(lg, 0) + col["frames"][c] / fr / 3600
    rand_pool, rng0 = {}, random.Random(0)  # train.py's random sample set (cfg.sample_random): pools from train_items
    for i in train_items:
        t = texts.parts[i].decode("utf-8")
        if (col["notext_s"][i] == 3) or not (20 <= len(t) <= 160):
            continue
        pl = rand_pool.setdefault(f"{lang_of(i) or 'en'}|talk", [])
        if len(pl) < 3000:
            pl.append((t, spk_of(i)))
        elif rng0.random() < 0.01:
            pl[rng0.randrange(3000)] = (t, spk_of(i))
    total = sum(w for _, w in parts)
    mean = sum(st["mean"].double() * w for st, w in parts) / total
    second = sum((st["std"].double() ** 2 + st["mean"].double() ** 2) * w for st, w in parts) / total
    stats = {"mean": mean.float(), "std": (second - mean ** 2).clamp(min=1e-8).sqrt().float()}
    if "codebooks" in parts[0][0]:
        stats["codebooks"] = parts[0][0]["codebooks"]
    else:
        stats["codec"] = "kvae"
    torch.save(stats, tmp / "stats.pt")
    conditioning = {}
    for c in reversed(conds):
        conditioning.update(c)
    val = [{"id": id_list[c], "text": texts.parts[c].decode("utf-8"), "speaker": int(col["speaker"][c]),
            "lang": lang_of(c) or "en", "spk_name": spk_of(c)} for c in val_items]
    small = {"roots": roots, "langs": lang_tab, "tags": tag_tab, "patch": patch, "clips": N, "rows": R}
    main = {"speakers": speakers, "conditioning": conditioning, "val": val, "rand_pool": rand_pool, "hours": hours,
            "by_lang": by_lang, "train_items": len(train_items),
            "speaker_langs": {("" if s is None else s): c.most_common(1)[0][0] for s, c in langs_by_spk.items() if s},
            "built": time.strftime("%Y-%m-%d %H:%M"), "build_seconds": round(time.time() - t_start)}
    (tmp / "main.json").write_text(json.dumps(main), encoding="utf-8")
    (tmp / "meta.json").write_text(json.dumps(small), encoding="utf-8")  # written last: marks a complete cache
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    log(f"fast data cache: {N} clips, {R} training rows, {hours:.0f} h -> {out} ({time.time() - t_start:.0f} s)")
    return out


# ------------------------------------------------------------------------------------------------- runtime
class Store:
    """An opened cache: arrays memory-mapped on first use (each worker opens its own maps; the OS shares the pages)."""

    def __init__(self, d):
        self.dir = Path(d)
        m = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.roots, self.langs, self.tags, self.patch = m["roots"], m["langs"], m["tags"], m["patch"]
        self.root_ix = {r: i for i, r in enumerate(self.roots)}
        self._a = {}

    def a(self, name):
        x = self._a.get(name)
        if x is None:
            x = self._a[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return x

    def s(self, name, i):
        off = self.a(name + "_off")
        return bytes(self.a(name + "_blob")[off[i]:off[i + 1]]).decode("utf-8")

    def tokens(self, c):
        off = self.a("tok_off")
        return self.a("tok_blob")[off[c]:off[c + 1]].tolist()  # (uint16 ids since VERSION 3)

    def find(self, keys_name, h):
        keys = self.a(keys_name)
        j = int(np.searchsorted(keys, np.uint64(h)))
        return j if j < len(keys) and keys[j] == np.uint64(h) else -1

    def clip_of(self, rid, uid):
        j = self.find("ru_keys", h64(f"{rid}\x00{uid}"))
        return -1 if j < 0 else int(self.a("ru_clip")[j])


_MISS = object()


class ItemView:
    """One clip, read from the cache; answers .get / [] / in like the manifest item it was built from."""
    __slots__ = ("st", "c")

    def __init__(self, st, c):
        self.st, self.c = st, int(c)

    def get(self, k, default=None):
        st, c = self.st, self.c
        if k == "id":
            return st.s("id", c)
        if k == "text":
            return st.s("text", c)
        if k == "root":
            return st.roots[st.a("root")[c]]
        if k == "speaker":
            return int(st.a("speaker")[c])
        if k == "frames":
            return int(st.a("frames")[c])
        if k == "patches":
            return int(st.a("frames")[c]) // st.patch
        if k == "tokens":
            return st.tokens(c)
        if k == "spk_name":
            s = st.a("spk_s")[c]
            return default if s == 0 else None if s == 1 else st.s("spkname", int(st.a("spk")[c]))
        if k == "lang":
            v = int(st.a("lang")[c])
            return default if v == -1 else None if v == -2 else st.langs[v]
        if k in TRI:
            s = st.a(k + "_s")[c]
            return default if s == 0 else None if s == 1 else float(st.a(k)[c])
        if k == "notext":
            s = st.a("notext_s")[c]
            return default if s == 0 else None if s == 1 else bool(s == 3)
        if k == "prev_text":
            s = st.a("prev_s")[c]
            return default if s == 0 else None if s == 1 else st.s("prev", c)
        if k == "has_corrupt":
            return bool(st.a("has_corrupt")[c])
        return default

    def __getitem__(self, k):
        v = self.get(k, _MISS)
        if v is _MISS:
            raise KeyError(k)
        return v

    def __contains__(self, k):
        return self.get(k, _MISS) is not _MISS


class RowsView:
    """The training rows (oversampled copies included) as a sequence of ItemViews."""

    def __init__(self, st):
        self.st, self.rows = st, st.a("rows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return ItemView(self.st, self.rows[i])

    def __iter__(self):
        return (ItemView(self.st, c) for c in self.rows)


class RowPool:
    """CodesDataset's dict-of-lists pools (x_by_tag, x_by_spk, dlv_pool, zs_by_spk) as sorted keys + offsets."""

    def __init__(self, st, name, key, make):
        self.st, self.name, self.key, self.make = st, name, key, make

    def rows(self, h):
        j = self.st.find(self.name + "_keys", h)
        if j < 0:
            return None
        off = self.st.a(self.name + "_off")
        return self.st.a(self.name + "_rows")[off[j]:off[j + 1]]

    def get(self, k, default=None):
        r = self.rows(self.key(k))
        return default if r is None else self.make(r)


class LazyPool(Sequence):
    """A pool as rows of the cache, each element made only when random.choice / random.sample reads it. The pools as
    lists cost a worker ~70% of its time (2026-10-30 profile: a big voice's ~3500 ids decoded twice per row, plus the
    row's own id once per pool entry); same length, order and elements, so the same random calls pick the same."""
    __slots__ = ("rows", "make")

    def __init__(self, rows, make):
        self.rows, self.make = rows, make

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        if isinstance(j, slice):
            return [self.make(r) for r in self.rows[j]]
        return self.make(self.rows[j])


class LocMap:
    """A dir's shard index (<root>/shards/<kind>_index.json: uid -> [start, length(, extra)]) from the cache."""

    def __init__(self, st, rid, kind):
        self.st, self.rid, self.kind = st, rid, kind

    def get(self, uid, default=None):
        c = self.st.clip_of(self.rid, uid)
        if c < 0:
            return default
        s = int(self.st.a(self.kind + "_start")[c])
        if s < 0:
            return default
        n, x = int(self.st.a(self.kind + "_len")[c]), float(self.st.a(self.kind + "_x")[c])
        return (s, n) if x != x else (s, n, x)

    def __getitem__(self, uid):
        v = self.get(uid)
        if v is None:
            raise KeyError(uid)
        return v


class ZsRowMap:
    def __init__(self, st):
        self.st = st

    def get(self, uid, default=None):
        j = self.st.find("zsu_keys", h64(uid))
        if j < 0:
            return default
        return (self.st.roots[int(self.st.a("zsu_root")[j])], int(self.st.a("zsu_row")[j]))


class FastDataset(CodesDataset):
    """CodesDataset over the cache: same __getitem__ (inherited), same pools; nothing per worker but file maps."""

    def __init__(self, cache_dir, cfg):  # (no CodesDataset.__init__: everything was built into the cache)
        self.cache_dir, self.cfg = str(cache_dir), cfg
        self._open()

    def _open(self):
        st = self.st = Store(self.cache_dir)
        cfg = self.cfg
        self.root = Path(st.roots[0])
        self.kvae = getattr(cfg, "codec", "dac") == "kvae"
        self.items = RowsView(st)
        self.dlv = getattr(cfg, "use_dlv", False)
        self.vre = getattr(cfg, "use_vre", False) or getattr(cfg, "use_ref_seg", False)  # (v14 reference segment: same pool)
        rows = st.a("rows")
        roots_of = lambda r: [(st.roots[st.a("root")[c]], st.s("id", c)) for c in rows[r]]  # noqa: E731
        self.dlv_pool = RowPool(st, "dlv", _spk_key, roots_of)
        self.cross = getattr(cfg, "cross_prob", 0) > 0
        self.x_by_tag = RowPool(st, "xtag", lambda k: _tag_key(*k), lambda r: r.tolist())
        self.x_by_spk = RowPool(st, "xspk", _spk_key, lambda r: r.tolist())
        self.x_wh = RowPool(st, "xwh", _spk_key, lambda r: r.tolist())
        self.use_notes, self.note_store, self.song_cont, self.by_song = False, {}, False, {}
        self.zs = getattr(cfg, "zs_weight", 0) > 0
        self._mm = {}
        self.zs_row = ZsRowMap(st)
        self.zs_by_spk = RowPool(st, "zs", _spk_key, lambda r: [
            (st.roots[int(st.a("zs_root")[c])], int(st.a("zs_row")[c])) for c in rows[r]])

    def __getstate__(self):  # pickled into each DataLoader worker: just where the cache is
        return {"cache_dir": self.cache_dir, "cfg": self.cfg}

    def __setstate__(self, s):
        self.cache_dir, self.cfg = s["cache_dir"], s["cfg"]
        self._open()

    def _ref_pool(self, it):
        """CodesDataset._ref_pool without building the list: the voice's rows minus those with this clip's id (ids
        compared by their stored hash; an equal hash is checked on the strings, so a collision changes nothing)."""
        st = self.st
        r = self.dlv_pool.rows(_spk_key(it.get("spk_name")))
        if r is None:
            return []
        rows, idh = st.a("rows"), st.a("idh")
        own_c = it.c
        keep = idh[rows[r]] != idh[own_c]
        if not keep.all():
            own = st.s("id", own_c)
            for j in np.flatnonzero(~keep):
                keep[j] = st.s("id", int(rows[r[j]])) != own
            r = r[keep]
        return LazyPool(r, lambda x: (st.roots[st.a("root")[rows[x]]], st.s("id", rows[x])))

    def _zs_others(self, it):
        """CodesDataset._zs_others without building the list (same elements, order and fallback)."""
        st = self.st
        r = self.zs_by_spk.rows(_spk_key(it.get("spk_name")))
        if r is None:
            return []
        rows, zr, zw = st.a("rows"), st.a("zs_root"), st.a("zs_row")
        make = lambda x: (st.roots[int(zr[rows[x]])], int(zw[rows[x]]))  # noqa: E731  (as zs_by_spk's elements)
        own = self.zs_row.get(it["id"])
        if own is not None:
            cl = rows[r]
            same_root = np.asarray([x == own[0] for x in st.roots])[zr[cl]]
            keep = ~(same_root & (zw[cl] == own[1]))
            if keep.any() and not keep.all():  # ([...] or pool: nothing left -> the whole pool)
                r = r[keep]
        return LazyPool(r, make)

    def _read(self, root, kind, uid):
        key = (str(root), kind)
        if key not in self._mm and kind in KINDS:
            rid = self.st.root_ix.get(str(root))
            d = Path(root) / "shards"
            if rid is not None and (d / f"{kind}.npy").exists():
                self._mm[key] = (np.load(d / f"{kind}.npy", mmap_mode="r"), LocMap(self.st, rid, kind))
        return super()._read(root, kind, uid)

    def plan_cross(self, i, rnd):
        """CodesDataset.plan_cross on the arrays (same rules, same random calls in the same order)."""
        if not self.cross or rnd.random() >= self.cfg.cross_prob:
            return None
        st = self.st
        rows, idh, frames = st.a("rows"), st.a("idh"), st.a("frames")
        c = int(rows[i])
        own = idh[c]
        patch = st.patch
        budget = int(min(self.cfg.cross_max_sec * 50, max(150, int(frames[c]) // patch * self.cfg.patch) * getattr(self.cfg, "cross_len_mult", 1.0)))

        def others(pool):
            return len(pool) > 64 or any(idh[rows[k]] != own for k in pool)
        wh = st.a("whisper_s")[c] == 2 and st.a("whisper")[c] >= 0.5
        mixed = False
        p_wh = getattr(self.cfg, "cross_whisper_prob" if wh else "cross_whisper_any", 0.0)
        wpool = self.x_wh.rows(st.a("spk_key")[c]) if p_wh > 0 else None
        if wpool is not None and len(wpool) and others(wpool) and rnd.random() < p_wh:
            pool = wpool
        elif getattr(self.cfg, "cross_mix_prob", 0.0) > 0 and rnd.random() < self.cfg.cross_mix_prob:
            spk_ = self.x_by_spk.rows(st.a("spk_key")[c])
            wsp_ = self.x_wh.rows(st.a("spk_key")[c])
            pool = ([] if spk_ is None else spk_.tolist()) + ([] if wsp_ is None else wsp_.tolist())
            mixed = True
        else:
            tag = [] if wh else (self.x_by_tag.rows(st.a("tag_key")[c]))
            tag = [] if tag is None else tag
            spk = self.x_by_spk.rows(st.a("spk_key")[c])
            pool = tag if len(tag) and others(tag) and rnd.random() < 0.8 else ([] if spk is None else spk)
        pool = pool.tolist() if hasattr(pool, "tolist") else pool
        want = rnd.randint(min(100, budget), budget)
        if mixed:
            budget = want = int(self.cfg.cross_max_sec * 50)
        seen, cand = {own}, []  # (11-14) one row per clip: oversampled copies of a clip are not two prompts
        for k in rnd.sample(pool, min(len(pool), 16)):
            h = idh[rows[k]]
            if h not in seen:
                seen.add(h)
                cand.append(k)
        cand = cand[:10]
        got, chosen = 0, []
        for k in cand:
            p = int(frames[rows[k]]) // patch
            if (got + p) * patch > budget:  # (budget / want / 100 are FRAMES, got is patches: code review 2026-10-29, matters at patch 2/4)
                continue
            chosen.append(k)
            got += p
            if got * patch >= want:
                break
        return (tuple(chosen), got) if got * patch >= 100 else None


def _plan_chunk(job):
    """(dataset, first row, end row, seed) -> plan_cross of those rows (a planner process; the dataset reopens its cache)."""
    ds, a, b, seed = job
    rnd = random.Random(seed)
    return [ds.plan_cross(i, rnd) for i in range(a, b)]


def parallel_plans(ds, n, rnd, workers=None):
    """plan_cross for every row, split over processes (2026-11-07: serially 7.7M rows took 395 s of every start). Each
    chunk has its own random stream from `rnd`: the same rules and distribution, other draws (the plan is new at every
    start anyway). FAST_PLAN_WORKERS=0 -> serial."""
    import os
    if workers is None:
        workers = int(os.environ.get("FAST_PLAN_WORKERS", max(1, min(12, (os.cpu_count() or 4) - 4))))
    if workers <= 1 or n < 200_000:
        return [ds.plan_cross(i, rnd) for i in range(n)]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    k = workers * 4
    edges = [n * j // k for j in range(k + 1)]
    jobs = [(ds, edges[j], edges[j + 1], rnd.getrandbits(64)) for j in range(k)]
    t = time.time()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        plans = [p for part in ex.map(_plan_chunk, jobs) for p in part]
    print(f"fast data: planned {n:,} rows in {time.time() - t:.0f} s on {workers} processes", flush=True)
    return plans


class FastPlannedBatchSampler(PlannedBatchSampler):
    """PlannedBatchSampler with the row lengths read from the cache."""

    def _make(self):
        ds, rnd = self.ds, self.rnd
        n = len(ds.items)
        plans = parallel_plans(ds, n, rnd)
        st = ds.st
        rows = st.a("rows")
        tgt = np.maximum(st.a("frames")[rows] // st.patch, st.a("up")[rows]).tolist()
        lengths = [tgt[i] + (plans[i][1] if plans[i] else 0) + self.margin for i in range(n)]
        # (sorted by length + a uniform [0, 1) jitter, as before, in numpy: the Python sort of 7.7M keys was ~1 min per start)
        key = np.asarray(lengths, dtype=np.float64)
        if self.shuffle:
            key = key + np.random.default_rng(rnd.getrandbits(64)).random(n)
        order = np.argsort(key, kind="stable").tolist()
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
        if self.longest_first:
            batches.sort(key=lambda b: -max(lengths[i] for i, _ in b))
        return batches


class Prepared:
    """What train.py needs from the data, from the cache."""

    def __init__(self, d, cfg):
        import torch
        self.dir = Path(d)
        m = json.loads((self.dir / "main.json").read_text(encoding="utf-8"))
        self.speakers, self.conditioning, self.val_items = m["speakers"], m["conditioning"], m["val"]
        self.speaker_langs, self.hours, self.by_lang = m["speaker_langs"], m["hours"], m["by_lang"]
        self.rand_pool = {tuple(k.split("|")): [tuple(x) for x in v] for k, v in m["rand_pool"].items()}
        self.stats = torch.load(self.dir / "stats.pt")
        self.train_ds = FastDataset(self.dir, cfg)


def prepare(paths, cfg, base_speakers, context, build_if_missing=True, log=print):
    """train.py: the cache for this data + config (built now if missing), or None if this data needs the normal path."""
    d = CACHE / f"train_{cache_key(paths, cfg, base_speakers, context)}"
    if not (d / "meta.json").exists():
        if not build_if_missing:
            return None
        log(f"fast data: building the cache once -> {d}")
        try:
            build(paths, cfg, base_speakers, context, d, log=log)
        except Unsupported as e:
            log(f"fast data: not for this data ({e}); using the normal loader")
            return None
    else:
        log(f"fast data: cache {d.name}")
    return Prepared(d, cfg)


def _from_launcher(cmd_file):
    """The data / config / base speakers a launcher .cmd would train with (same rules as train.py)."""
    import argparse
    import shlex

    import torch

    from .config import Config
    lines = Path(cmd_file).read_text(encoding="utf-8").splitlines()
    line = next(l for l in lines if "train.py" in l)
    import re as _re  # (11-17) the launchers pass `train.py %D% ... %A% ...`: expand their `set X=...` variables
    sets = {m.group(1): m.group(2) for m in (_re.match(r"\s*set ([A-Za-z_]\w*)=(.*)$", l) for l in lines) if m}
    line = _re.sub(r"%(\w+)%", lambda m: sets.get(m.group(1), m.group(0)), line)
    argv = shlex.split(line.split("train.py", 1)[1].split(">>")[0], posix=False)
    ap = argparse.ArgumentParser()
    for a in ("--out", "--init", "--ckpt_dir"):
        ap.add_argument(a, default=None)
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--context", default="data/context.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--base_speakers", action="store_true")
    args, _ = ap.parse_known_args(argv)
    ck_path = Path(args.ckpt_dir or args.out) / "last.pt"
    src = ck_path if args.resume and ck_path.exists() else (Path(args.init) if args.init else None)
    ck = torch.load(src, map_location="cpu", weights_only=False, mmap=True) if src else None
    cfg = Config.from_dict(ck["cfg"]) if ck else Config()
    cfg.override(args.set)
    set_f = Path(args.out) / "SET"  # (11-17) train.py applies runs/<run>/SET after --set: so must the hand build, or its
    if set_f.exists():  # cache key (oversample / exclude_speakers / voice_cap live in SET) is not the one training asks for
        for ln in set_f.read_text(encoding="utf-8-sig").splitlines():
            ln = ln.split("#", 1)[0].strip()
            if ln:
                try:
                    cfg.override([ln])
                except (KeyError, ValueError):
                    pass
    base = ck["speakers"] if (args.base_speakers and ck) else None
    return args.data, cfg, base, args.context


if __name__ == "__main__":
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # next to a training run: let it go first
    paths, cfg, base, context = _from_launcher(sys.argv[1])
    d = CACHE / f"train_{cache_key(paths, cfg, base, context)}"
    if (d / "meta.json").exists():
        print(f"already built: {d}")
    else:
        build(paths, cfg, base, context, d)
    print(f"FASTDATA_READY {d}")
