"""Adam for the voice table, rows only (11-24 v18, you: "Do 2"; notes_solver/TRAIN_LOOP_REWRITE.md 3.4; cfg.row_adam_table).

The table is 569,344 x 256 = 146M of the model's 495M weights, but a step touches ~6 rows (the batch's voices). The fused
AdamW kept 1.17 GB of state for all of it and read / wrote every row every step. Here only the rows seen recently keep
state, in a ring of `slots` (16,384 rows, ~2,700 steps of history), and every live row gets the normal AdamW update every
step -- its own gradient when its voice is in the batch, zero otherwise -- exactly as the dense optimizer moves it. A row
older than the ring has momentum 0.9^2700 and second moment 0.98^2700 of its last value: dropping it changes nothing a
float holds. NOT torch.optim.SparseAdam: that skips the momentum steps between visits (a visited row moves ~11x less).

The gradient stays dense (the embedding's ordinary backward, so the clip and every norm see it as before); this reads the
batch's rows out of it and the caller frees it. Row ids come from the batch on the CPU, so there are no GPU waits.
Weight decay 0 (the table's group had none).
"""
import torch


class RowAdam:
    def __init__(self, weight: torch.nn.Parameter, betas=(0.9, 0.98), eps=1e-8, slots: int = 16384):
        self.w, (self.b1, self.b2), self.eps, self.S = weight, betas, eps, int(slots)
        dev, d = weight.device, weight.shape[1]
        self.M = torch.zeros(self.S, d, device=dev, dtype=torch.float32)
        self.V = torch.zeros(self.S, d, device=dev, dtype=torch.float32)
        self.row_of = [-1] * self.S  # (CPU) slot -> row
        self.slot_of = {}  # (CPU) row -> slot
        self.ptr, self.t = 0, 0
        # (Solver 11-24, row_adam_check.py: first-insertion eviction lost revisited voices' momentum, and a new row could
        # take a live batch row's slot) a slot whose row was seen in the last `hold` steps, or is in this batch, is skipped
        self.hold, self.last = 1024, [-10 ** 9] * self.S
        self._rows_dev = None  # live slots' rows on the GPU (rebuilt when the ring changes)

    def _slots(self, rows):
        now, out, changed = self.t + 1, [], False
        for r in rows:  # first mark the batch's rows already in the ring (renewed: they can't be evicted below)
            s = self.slot_of.get(r)
            if s is not None:
                self.last[s] = now
        for r in rows:
            s = self.slot_of.get(r)
            if s is None:  # a new row: the next slot not seen in the last `hold` steps (its old row's state is dropped)
                n_ = 0
                while now - self.last[self.ptr] < self.hold and n_ < self.S:
                    self.ptr = (self.ptr + 1) % self.S
                    n_ += 1
                s = self.ptr
                self.ptr = (self.ptr + 1) % self.S
                old = self.row_of[s]
                if old >= 0:
                    self.slot_of.pop(old, None)
                self.row_of[s], self.slot_of[r], self.last[s] = r, s, now
                self.M[s].zero_()
                self.V[s].zero_()
                changed = True
            out.append(s)
        if changed or self._rows_dev is None:
            live = [(s, r) for s, r in enumerate(self.row_of) if r >= 0]
            self._live_slots = torch.tensor([s for s, _ in live], dtype=torch.long).to(self.w.device, non_blocking=True)
            self._rows_dev = torch.tensor([r for _, r in live], dtype=torch.long).to(self.w.device, non_blocking=True)
        return out

    @torch.no_grad()
    def step(self, rows, lr: float, bad=None):
        """rows: the batch's voice ids (CPU ints, unique); lr: this step's rate for the table; bad: 0-dim GPU tensor,
        1 = skip the update (the NaN guard's flag, as the fused AdamW's found_inf)."""
        g = self.w.grad
        rows = [int(r) for r in rows]
        sl = self._slots(rows)
        self.t += 1
        G = torch.zeros_like(self.M)
        if rows and g is not None:
            idx_s = torch.tensor(sl, dtype=torch.long).to(self.w.device, non_blocking=True)
            idx_r = torch.tensor(rows, dtype=torch.long).to(self.w.device, non_blocking=True)
            G.index_copy_(0, idx_s, g.index_select(0, idx_r).float())
        b1, b2 = self.b1, self.b2
        M_new = self.M * b1 + G * (1 - b1)
        V_new = self.V * b2 + G * G * (1 - b2)
        if bad is not None:
            keep = bad > 0
            M_new = torch.where(keep, self.M, M_new)
            V_new = torch.where(keep, self.V, V_new)
        self.M.copy_(M_new)
        self.V.copy_(V_new)
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        ls = self._live_slots
        if not len(ls):
            return
        upd = (self.M[ls] / ((self.V[ls] / bc2).sqrt() + self.eps)) * (lr / bc1)
        if bad is not None:
            upd = upd * (1 - bad.float())
        self.w.data.index_add_(0, self._rows_dev, (-upd).to(self.w.dtype))

    def state_dict(self):
        return {"M": self.M.cpu(), "V": self.V.cpu(), "row_of": list(self.row_of), "ptr": self.ptr, "t": self.t, "S": self.S,
                "last": list(self.last)}

    def load_state_dict(self, sd):
        assert int(sd["S"]) == self.S, (sd["S"], self.S)
        self.M.copy_(sd["M"].to(self.M.device))
        self.V.copy_(sd["V"].to(self.V.device))
        self.row_of = [int(r) for r in sd["row_of"]]
        self.slot_of = {r: s for s, r in enumerate(self.row_of) if r >= 0}
        self.ptr, self.t, self._rows_dev = int(sd["ptr"]), int(sd["t"]), None
        self.last = [int(x) for x in sd.get("last", [-10 ** 9] * self.S)]
        self._slots([])

    @torch.no_grad()
    def from_dense(self, exp_avg, exp_avg_sq, step: int):
        """The dense AdamW's state of the table -> the ring: the S rows with the largest momentum (the recently visited
        ones; the rest have decayed below anything a float holds), weakest first so they leave the ring first."""
        m, v = exp_avg.float(), exp_avg_sq.float()
        score = m.abs().amax(1)
        k = int(min(self.S, int((score > 0).sum())))
        top = torch.topk(score, k).indices.flip(0) if k else torch.zeros(0, dtype=torch.long, device=score.device)
        rows = top.tolist()
        for s, r in enumerate(rows):
            self.row_of[s], self.slot_of[r] = r, s
        if rows:
            ti = top.to(m.device)
            self.M[:k].copy_(m.index_select(0, ti).to(self.M.device))
            self.V[:k].copy_(v.index_select(0, ti).to(self.V.device))
        self.ptr, self.t, self._rows_dev = k % self.S, int(step), None
        self._slots([])
        return k
