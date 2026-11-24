"""The voice critic's row pool (11-24 train-loop rewrite, notes_solver/TRAIN_LOOP_REWRITE.md 3.2; cfg.critic_pool).

v17's critic rolled out a fixed 4 rows every 4th step, but only 2.52 of them were real on average (the rest repeated
real rows to keep the compiled shape), and on word-gap steps 1 of 4. The pool keeps the real windows that
critic_windows / critic_edit_windows cut on harvest steps and hands the critic exactly `rows` real rows per call:
one pool for normal windows, one for word-gap windows (a full word-gap pool goes first). Windows are detached, about
1 MB a row; the conditioning they carry is a few steps old when they are judged.

Shapes of one window batch (n rows, w decoder steps): x1 / cond / ctx (n, w, c), mask / gen / known (n, w),
g (n, d), ref (memory (n, Tr, d), mask (n, Tr), on (n,)) or None. Tr differs between batches: the memories are
padded with zeros and a False mask (a row without memory keeps its one True slot and on = 0, as the model makes it).
"""
import torch

_ROW_KEYS = ("x1", "cond", "mask", "ctx", "g", "gen", "known")


def _trim(ci, k):
    out = {key: (None if ci.get(key) is None else ci[key][:k]) for key in _ROW_KEYS}
    out["ref"] = None if ci.get("ref") is None else tuple(r[:k] for r in ci["ref"])
    return out


def _rows(ci):
    return ci["x1"].shape[0]


class CriticPool:
    def __init__(self, rows: int):
        self.rows = rows
        self.items = {False: [], True: []}  # edit flag -> list of trimmed window batches

    def size(self, edit: bool = False) -> int:
        return sum(_rows(c) for c in self.items[edit])

    def short(self) -> bool:
        """Harvest again? (the normal pool is below a call's worth; word-gap windows come along on harvest steps)"""
        return self.size(False) < self.rows

    def add(self, ci) -> int:
        """ci from critic_windows / critic_edit_windows (n_real real rows first) -> rows kept."""
        if ci is None:
            return 0
        k = int(ci.get("n_real", _rows(ci)))
        if k <= 0 or ci.get("ctx") is None:
            return 0
        self.items[bool(ci.get("edit"))].append(_trim(ci, k))
        return k

    def ready(self):
        """-> the pool to fire (True = word gaps, False = normal) or None."""
        for edit in (True, False):
            if self.size(edit) >= self.rows:
                return edit
        return None

    def take(self, edit: bool):
        """Exactly `rows` real rows from the pool (oldest first) as one critic batch; the rest stays."""
        got, need, keep = [], self.rows, []
        for c in self.items[edit]:
            n = _rows(c)
            if need <= 0:
                keep.append(c)
            elif n <= need:
                got.append(c)
                need -= n
            else:
                got.append(_trim(c, need))
                rest = {key: (None if c.get(key) is None else c[key][need:]) for key in _ROW_KEYS}
                rest["ref"] = None if c.get("ref") is None else tuple(r[need:] for r in c["ref"])
                keep.append(rest)
                need = 0
        self.items[edit] = keep
        out = {key: torch.cat([c[key] for c in got]) for key in _ROW_KEYS if got[0].get(key) is not None}
        out["ref"] = _cat_refs([c.get("ref") for c in got], [_rows(c) for c in got], got[0]["x1"].device)
        out["n_real"] = self.rows
        if edit:
            out["edit"] = True
        return out


def _cat_refs(refs, ns, dev):
    """Reference memories of several batches -> one, padded to the longest (zeros, mask False)."""
    real = [r for r in refs if r is not None]
    if not real:
        return None
    d = real[0][0].shape[-1]
    dt_m, dt_k = real[0][0].dtype, real[0][1].dtype
    tr = max(r[0].shape[1] for r in real)
    mems, masks, ons = [], [], []
    for r, n in zip(refs, ns):
        if r is None:  # a batch without reference segments: one empty slot per row, switched off (as the model does)
            m = torch.zeros(n, tr, d, device=dev, dtype=dt_m)
            k = torch.zeros(n, tr, device=dev, dtype=dt_k)
            k[:, 0] = 1
            mems.append(m), masks.append(k), ons.append(torch.zeros(n, device=dev, dtype=real[0][2].dtype))
            continue
        pad = tr - r[0].shape[1]
        mems.append(torch.nn.functional.pad(r[0], (0, 0, 0, pad)))
        masks.append(torch.nn.functional.pad(r[1], (0, pad)))
        ons.append(r[2])
    return torch.cat(mems), torch.cat(masks), torch.cat(ons)
