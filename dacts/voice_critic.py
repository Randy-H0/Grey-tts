"""Voice critic in the loop (owner 2026-10-30: "a similarity model that directly is able to communicate and connect with
the tts to tell it how to steer closer to our voice, for training").

Why the old speaker-similarity loss (dacts/sim_loss.py, v8b 2026-10-13) never moved: it judged the ONE-JUMP clean
estimate of a single noisy step against the same clip. The identity-vs-t probe (evals/identity_t_probe.py, 2026-10-30)
showed the voice is decided across the noisy half of the path (t < 0.5), where a one-jump estimate is a blur that
sounds like no one -- the verifier's gradient there is noise -- and comparing against the same clip only rewards
copying that recording, which the flow loss already does.

Here, for a few prompted rows, the model really clones: a short sway-scheduled Euler run from pure noise (as the site
samples, conditional pass only) over a window of the row (the end of its prompt + the start of what it generates),
with gradient through every step (each decoder call checkpointed: memory is one step's activations). A crop of the
finished take is decoded (frozen KVAE) and fingerprinted (frozen WavLM-SV); the target is the fingerprint of the
window's KNOWN frames -- the prompt: another recording of the speaker on cross rows, another stretch of the clip on
same-clip rows. loss = 1 - cosine. Only the decoder learns from it: its inputs come detached from the main step, which
train.py has already backpropagated (and freed) before the critic runs. Rows are batched and it runs every few steps:
training is bound by kernel launches, so a 4-step rollout costs about what a whole training step does, whatever its size.

Judge with a DIFFERENT verifier (Seed-TTS-eval's WavLM-large ECAPA) and by ear: a gain on WavLM-SV alone is the model
learning to please its critic.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .codec import KVAE_SR, load_kvae


def critic_windows(x1, cond, mask, ctx, g, ref, gen, known, ok, rows: int, win: int, k_min: int, g_min: int):
    """Pick up to `rows` rows (ok & room for a window of `win` steps holding >= k_min known and >= g_min generated
    steps) and cut that window out of each. -> the critic's batch (detached), or None. All lengths in decoder steps."""
    T = mask.shape[1]
    if T < win:
        return None
    ck = F.pad(known.float().cumsum(1), (1, 0))
    cg = F.pad(gen.float().cumsum(1), (1, 0))
    K, G = ck[:, win:] - ck[:, :-win], cg[:, win:] - cg[:, :-win]  # (B, T - win + 1) per window start
    valid = (K >= k_min) & (G >= g_min) & ok[:, None]
    pick = valid.any(1).nonzero().flatten()
    if not len(pick):
        return None
    pick = pick[torch.randperm(len(pick), device=pick.device)[:rows]]
    n_real = len(pick)
    if n_real < rows:  # (2026-11-07) always `rows` rows: the compiled rollout is built for one fixed shape, and each new
        # row count (1..rows) cost another ~4 min compile at every start. The filler rows repeat real ones; the critic
        # scores only the first n_real, so they get no gradient and the loss is the same.
        pick = pick[torch.arange(rows, device=pick.device) % n_real]
    start = (valid[pick].float() * torch.rand(valid[pick].shape, device=valid.device)).argmax(1)  # a random valid start
    fr = start[:, None] + torch.arange(win, device=start.device)[None]  # (n, win)

    def cut(a):
        if a is None:
            return None
        a = a[pick]
        i = fr if a.dim() == 2 else fr[..., None].expand(-1, -1, a.shape[-1])
        return torch.gather(a, 1, i).detach()
    sub = lambda a: None if a is None else a[pick].detach()  # noqa: E731  (per-row, not per-step)
    return {"x1": cut(x1.float()), "cond": cut(cond), "mask": cut(mask), "ctx": cut(ctx), "g": sub(g),
            "ref": None if ref is None else tuple(sub(r) for r in ref), "gen": cut(gen), "known": cut(known),
            "n_real": n_real}


def critic_edit_windows(x1, cond, mask, ctx, g, ref, gen, known, ok, rows: int, win: int, gap_min: int, gap_max: int):
    """(11-22, you: "add the possibility to perform voice match on the filled in words as loss") Word-gap rows
    (cfg.edit_span_prob): the generated frames are ONE short span with known (real) frames on both sides. A window of
    `win` steps centred on that gap; VoiceCritic then puts only the gap of its own take into the real recording and
    fingerprints that against the real recording of the same stretch -- the filled-in words must sound like this person
    in this room. Same shapes / keys as critic_windows (+ "edit": True), so the compiled rollout is reused."""
    B, T = mask.shape
    if T < win:
        return None
    m = mask.bool()
    gn, kn = gen & m, known & m
    n_gen = gn.sum(1)
    idx = torch.arange(T, device=mask.device)[None]
    first = torch.where(gn, idx, T).min(1).values
    last = torch.where(gn, idx, -1).max(1).values
    rb = torch.arange(B, device=mask.device)
    inner = (first > 0) & (last < m.sum(1) - 1) & kn[rb, (first - 1).clamp(min=0)] & kn[rb, (last + 1).clamp(max=T - 1)]
    cand = ok & inner & ((last - first + 1) == n_gen) & (n_gen >= gap_min) & (n_gen <= gap_max) & (m.sum(1) >= win)
    pick = cand.nonzero().flatten()
    if not len(pick):
        return None
    pick = pick[torch.randperm(len(pick), device=pick.device)[:rows]]
    n_real = len(pick)
    if n_real < rows:  # (fixed row count: the compiled rollout's shape, as critic_windows)
        pick = pick[torch.arange(rows, device=pick.device) % n_real]
    centre = (first[pick] + last[pick]) // 2
    start = torch.minimum((centre - win // 2).clamp(min=0), (m[pick].sum(1) - win).clamp(min=0))
    fr = start[:, None] + torch.arange(win, device=start.device)[None]

    def cut(a):
        if a is None:
            return None
        a = a[pick]
        i = fr if a.dim() == 2 else fr[..., None].expand(-1, -1, a.shape[-1])
        return torch.gather(a, 1, i).detach()
    sub = lambda a: None if a is None else a[pick].detach()  # noqa: E731
    return {"x1": cut(x1.float()), "cond": cut(cond), "mask": cut(mask), "ctx": cut(ctx), "g": sub(g),
            "ref": None if ref is None else tuple(sub(r) for r in ref), "gen": cut(gen), "known": cut(known),
            "n_real": n_real, "edit": True}


def contrast_loss(eh, er, bank, m: float = 0.2, tau: float = 0.05, same_thr: float = 0.55):
    """(Solver's reference, notes_solver/contrast_ref.py) eh (G, D) take fingerprints (unit, with grad); er (G, D) their
    prompts' (unit, no grad); bank (N, D) earlier prompts (unit, no grad). loss = -log(e^((own-m)/tau) / (that + sum
    e^(neg/tau))), bank entries with cos(prompt, entry) > same_thr (same voice) are not negatives. fp32.
    -> loss (G,), wrong (G,) weight on wrong prompts (detached), margin (G,) own minus hardest other (detached)."""
    eh, er, bank = eh.float(), er.float().detach(), bank.float().detach()
    own = (eh * er).sum(-1)
    neg = (eh @ bank.T).masked_fill((er @ bank.T) > same_thr, float("-inf"))
    zo = (own - m) / tau
    lse = torch.logsumexp(torch.cat([zo[:, None], neg / tau], 1), 1)
    loss = lse - zo
    with torch.no_grad():
        wrong = 1 - (zo - lse).exp()
        hard = neg.max(1).values
        margin = torch.where(torch.isfinite(hard), own - hard, torch.full_like(own, float("nan")))
    return loss, wrong, margin


class PromptBank:
    """FIFO of the last `size` prompt fingerprints (detached, fp32); pushed AFTER the step's loss. Not in the checkpoint."""

    def __init__(self, size: int = 256):
        self.size, self.e = size, None

    def __len__(self):
        return 0 if self.e is None else len(self.e)

    def push(self, er):
        er = er.detach().float()
        self.e = er if self.e is None else torch.cat([self.e, er])[-self.size:]


class VoiceCritic(nn.Module):
    def __init__(self, latent, device="cuda", crop: int = 150, steps: int = 4, sway: float = -1.0, compile_mode: str = "",
                 sv_model: str = "microsoft/wavlm-base-plus-sv", asr_model: str = "", asr_weight: float = 0.0,
                 contrast_weight: float = 0.0, contrast_m: float = 0.2, contrast_tau: float = 0.05, contrast_bank: int = 256,
                 sv2_model: str = "", sv_weight: float = 1.0, sv2_weight: float = 0.0):
        """compile_mode (2026-11-01, "only 35% of gpu is used"): "" eager; "default": the rollout's checkpointed decoder
        step and the KVAE decode compiled (fixed shapes: the window, the crop and the 32 voice tokens never change, only
        the row count); "graphs": the rollout step also as CUDA graphs (one replay instead of thousands of launches)."""
        super().__init__()
        import torchaudio.transforms as AT
        from transformers import AutoModelForAudioXVector

        self.latent = latent
        self.kvae = load_kvae(device).requires_grad_(False)
        self.campp = sv_model == "campp"  # cfg.critic_sv (11-14): "campp" = CAM++ VoxCeleb (dacts/campp.py), else a HF x-vector model
        if self.campp:
            from .campp import CamppVerifier
            self.sv = CamppVerifier().to(device)
        else:
            self.sv = AutoModelForAudioXVector.from_pretrained(sv_model).to(device).eval().requires_grad_(False)
        # word critic (11-14, you "we can add another critic"): a frozen CTC recognizer (NOT the gate's Whisper) hears the
        # take's crop and the REAL clip's same frames (the take is generated on the real clip's frame-aligned text, so the
        # sounds line up); loss = KL(real's letter posteriors || take's), per frame. No transcripts needed.
        self.asr, self.asr_weight, self.last_word = None, float(asr_weight), None
        if asr_model and asr_weight > 0:
            from transformers import AutoModelForCTC
            self.asr = AutoModelForCTC.from_pretrained(asr_model).to(device).eval().requires_grad_(False)
        # contrast term (11-14, you "do sound not like other voices", Solver notes_solver/contrast_ref.py): InfoNCE of the
        # take vs its own prompt against a FIFO bank of earlier critic steps' prompt fingerprints (other voices)
        self.contrast_w, self.contrast_m, self.contrast_tau = float(contrast_weight), float(contrast_m), float(contrast_tau)
        self.bank, self.last_contrast = PromptBank(int(contrast_bank)), None
        # second verifier (11-14, cfg.critic_sv2): only in the grouped path; the voice term becomes
        # w1 * cos(sv) + w2 * cos(sv2) (the seed carries -critic_weight / n), last_cos2 logged as critic2
        self.sv2, self.sv_w, self.sv2_w, self.last_cos2 = None, float(sv_weight), float(sv2_weight), None
        if sv2_model and sv2_weight > 0:
            assert sv2_model == "eres2", sv2_model
            from .eres2 import Eres2Verifier
            self.sv2 = Eres2Verifier().to(device)
        self.rs = AT.Resample(KVAE_SR, 16000).to(device)  # (AF.resample rebuilt this filter every call: ~13% of the critic's CPU)
        self.crop, self.steps, self.sway = crop, steps, sway
        self.group = 1  # rows fingerprinted per call (cfg.critic_group; 1 = one at a time, the least memory)
        self.bf16 = False  # (2026-11-09, cfg.critic_bf16) decode + fingerprint under bf16 autocast instead of fp32
        assert compile_mode in ("", "default", "graphs"), compile_mode
        self.compile_mode = compile_mode
        self._step, self._decode = None, self.kvae.decode
        self._edit_gen = None  # (word-gap critic windows: which frames of the take are generated)
        if compile_mode and str(device).startswith("cuda"):
            self._decode = torch.compile(self.kvae.decode, dynamic=False)

    def _compiled_step(self, decoder):
        """One rollout step, checkpointed INSIDE the compiled region (a compiled module inside an eager checkpoint
        recompiled during the recomputation, 2026-10-30): one decoder call, recomputed in the backward."""
        if self._step is None:
            def step(x, t, cond, mask, g, ctx, ref):
                def run(x_):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return decoder.forward(x_, t, cond, mask, g, ctx=ctx, ref=ref).float()
                return checkpoint(run, x, use_reentrant=False)
            self._step = torch.compile(step, dynamic=False, mode="reduce-overhead" if self.compile_mode == "graphs" else None)
        return self._step

    def _prec(self):
        """the decode + fingerprint precision: fp32 (autocast off, as before) or bf16 autocast (cfg.critic_bf16)."""
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.bf16 else torch.autocast(device_type="cuda", enabled=False)

    def wav16(self, z):
        """normalized latents (B, T, 64) -> 16 kHz waves (B, S), differentiable."""
        return self.rs(self._decode(self.latent.denormalize(z.float()).transpose(1, 2))[:, 0])

    def letters(self, a):
        """16 kHz waves -> the word critic's per-frame letter log-probabilities (B, F, V), fp32."""
        self.asr.eval()  # (dropout / layer drop off even if a parent .train() reached it)
        a = (a - a.mean(-1, keepdim=True)) / (a.var(-1, keepdim=True) + 1e-7).sqrt()
        return F.log_softmax(self.asr(input_values=a).logits.float(), -1)

    def _word_ctc(self, lr_, lh):
        """CTC loss per letter of the take's letter log-probs lh (B, F, V) against the greedy letters of the real crop's
        lr_ (repeats collapsed, blanks dropped). -> (B,) losses, (B,) bool: rows with >= 3 letters (else weight 0)."""
        blank = int(self.asr.config.pad_token_id)
        labs = []
        for ids in lr_.argmax(-1):
            ids = torch.unique_consecutive(ids)
            labs.append(ids[ids != blank])
        n = torch.tensor([len(x) for x in labs], device=lh.device)
        tgt = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=blank) if int(n.max()) else lh.new_zeros(len(labs), 1, dtype=torch.long)
        il = torch.full((lh.shape[0],), lh.shape[1], dtype=torch.long, device=lh.device)
        loss = F.ctc_loss(lh.transpose(0, 1), tgt, il, n.clamp(min=1), blank=blank, reduction="none", zero_infinity=True)
        return loss / n.clamp(min=1), n >= 3

    def embed(self, z, a=None):
        """normalized latents (B, T, 64) -> unit x-vectors (B, 512), differentiable. a: the crop's 16 kHz wave if made."""
        a = self.wav16(z) if a is None else a
        if self.campp:  # raw 16 kHz wave in; fbank + per-clip mean removal inside, unit vectors out
            return self.sv(a)
        a = (a - a.mean(-1, keepdim=True)) / (a.var(-1, keepdim=True) + 1e-7).sqrt()  # the feature extractor's normalisation
        return F.normalize(self.sv(input_values=a).embeddings.float(), dim=-1)

    def rollout(self, decoder, ci, fast: bool = False):
        """The model's own clone of the windows in ci (patched, normalized latents), gradient through every step.
        fast: the compiled decoder, every step's activations kept (no checkpoint recomputation): it runs after the
        main step's graph is freed, and 4 steps x a few 6 s windows weigh less than one training batch."""
        n = self.steps
        u = torch.linspace(0, 1, n + 1)
        ts_f = (u + self.sway * (torch.cos(math.pi / 2 * u) - 1 + u)).tolist()  # (as synthesize: steps crowd the start)
        x = torch.randn_like(ci["x1"])
        b = x.shape[0]
        if self.compile_mode and x.is_cuda and not fast:
            f = self._compiled_step(decoder)
            if self.compile_mode == "graphs":
                torch.compiler.cudagraph_mark_step_begin()
            for k in range(n):
                tk = torch.full((b,), ts_f[k], device=x.device)
                x = x + (ts_f[k + 1] - ts_f[k]) * f(x, tk, ci["cond"], ci["mask"], ci["g"], ci["ctx"], ci["ref"])
            return x
        for k in range(n):
            tk = torch.full((b,), ts_f[k], device=x.device)

            def step(x_, tk_=tk):  # (.forward: the plain decoder -- train.py's compiled one recompiles inside the
                # checkpoint's recomputation and no longer matches what the forward pass saved, 2026-10-30)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x_.is_cuda):
                    return decoder.forward(x_, tk_, ci["cond"], ci["mask"], ci["g"], ctx=ci["ctx"], ref=ci["ref"]).float()
            if fast:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                    v = decoder(x, tk, ci["cond"], ci["mask"], ci["g"], ctx=ci["ctx"], ref=ci["ref"]).float()
                x = x + (ts_f[k + 1] - ts_f[k]) * v
            else:
                x = x + (ts_f[k + 1] - ts_f[k]) * checkpoint(step, x, use_reentrant=False)
        return x

    def forward(self, decoder, ci, patch: int = 1, fast: bool = False, weight: float | None = None):
        """-> (loss, mean cosine) or None when no window has room for a crop of the take and of its prompt.
        weight: backpropagate weight * loss right here and return the loss detached. Same gradients, far less memory:
        each row's decode + fingerprint graph (~2.5 GB in fp32, speedups/scratch/critic_mem.py) is backpropagated into
        the take and freed before the next row, then the rollout once -- holding all rows' graphs for one backward ran
        the 15 GB card out of memory on most critic steps (v14xl-vc1: 242 of its 251 skipped batches, 2026-10-30)."""
        from .model import unpatchify
        rw = getattr(self, "ref_weight", None)  # (11-24) the word critic follows the call's weight (see _grouped)
        self._wscale = float(weight) / rw if (weight is not None and rw) else 1.0
        # where the crops can go, copied to the CPU while the rollout is queued: no GPU wait per row below
        gk = torch.stack([ci["gen"], ci["known"]]).repeat_interleave(patch, dim=2)
        ev = None
        if gk.is_cuda:
            gk, ev = gk.to("cpu", non_blocking=True), torch.cuda.Event()
            ev.record()
        xh = unpatchify(self.rollout(decoder, ci, fast), patch)
        x1 = unpatchify(ci["x1"], patch)
        if ev is not None:
            ev.synchronize()
        gen, known = gk[0], gk[1]
        nr = ci.get("n_real", gen.shape[0])  # (filler rows past n_real: rolled out for the fixed shape, never scored)
        gen, known = gen[:nr], known[:nr]
        L, cos = self.crop, []
        self._edit_gen = None
        if ci.get("edit"):  # (word-gap rows) one crop per row, centred on the gap, the SAME frames as the real target
            ok_g, ok_k = [], []
            for i in range(gen.shape[0]):
                gi = gen[i].nonzero().flatten()
                if not len(gi) or gen.shape[1] < L:
                    ok_g.append(gen.new_zeros(0)); ok_k.append((gen.new_zeros(0), L))
                    continue
                s_ = int(min(max(int((gi[0] + gi[-1]) // 2) - L // 2, 0), gen.shape[1] - L))
                ok_g.append(torch.tensor([s_])); ok_k.append((torch.tensor([s_]), L))
            self._edit_gen = gen.to(xh.device)
        else:
            ok_g = [gen[i].unfold(0, L, 1).all(-1).nonzero().flatten() if gen.shape[1] >= L else gen.new_zeros(0)
                    for i in range(gen.shape[0])]
            ok_k = []
            for i in range(gen.shape[0]):
                for kl in (L, 100):  # an all-prompt stretch: 3 s, else 2 s (same-clip prompts can sit on both sides)
                    k_ok = known[i].unfold(0, kl, 1).all(-1).nonzero().flatten() if known.shape[1] >= kl else known.new_zeros(0)
                    if len(k_ok):
                        break
                ok_k.append((k_ok, kl))
        n = sum(1 for g_ok, (k_ok, _) in zip(ok_g, ok_k) if len(g_ok) and len(k_ok))  # rows in the mean (no random draws)
        take = xh if weight is None else xh.detach().requires_grad_()
        # d(weight * (1 - mean cos)) / d cos_i as the one backward computed it: (-weight) / n, in fp32
        seed = torch.full((1,), -weight, device=xh.device) / n if weight is not None and n else None
        if self.group > 1:
            return self._grouped(xh, x1, take, seed, ok_g, ok_k, nr)
        for i in range(nr):
            g_ok, (k_ok, kl) = ok_g[i], ok_k[i]
            if not len(g_ok) or not len(k_ok):
                continue
            # (the draws stay between the rows' fingerprints: WavLM's layer drop draws from the same CPU generator)
            s = int(g_ok[torch.randint(len(g_ok), (1,))])
            ks = int(k_ok[torch.randint(len(k_ok), (1,))])
            with self._prec():
                with torch.no_grad():
                    er = self.embed(x1[i:i + 1, ks:ks + kl])
                eh = self.embed(self._crop(take, x1, i, s, L)[None])
            c_i = (eh * er).sum(-1)
            if seed is not None:
                c_i.backward(seed)  # into take.grad; this row's graph is freed here
                c_i = c_i.detach()
            cos.append(c_i)
        if not cos:
            return None
        c = torch.cat(cos).mean()
        if seed is not None:
            xh.backward(take.grad)  # the rollout: gradients into the decoder, as (weight * loss).backward() did
            return (1 - c).detach(), c.detach()
        return 1 - c, c.detach()

    def _crop(self, take, x1, i, s, L):
        """the take's crop of row i; on word-gap windows only its generated frames, the rest the real recording
        (gradient flows through the filled-in words only)"""
        c = take[i, s:s + L]
        if self._edit_gen is None:
            return c
        return torch.where(self._edit_gen[i, s:s + L, None], c, x1[i, s:s + L].to(c.dtype))

    def _grouped(self, xh, x1, take, seed, ok_g, ok_k, nr):
        """forward()'s per-row loop with the fingerprints in groups of self.group rows (2026-11-07, you: "make the jobs
        bigger"): one decode + WavLM call per group instead of per row. Shapes stay fixed -- every group is padded to
        `group` rows by repeating its first row, the padding's cosines get a zero seed and are left out of the mean --
        so the compiled KVAE decode sees (group, crop) and at most two prompt lengths, never a new shape. The crops are
        all drawn first (other dice than the per-row loop, the same distribution); the maths per row is unchanged."""
        L, G, dev = self.crop, self.group, xh.device
        draws = {}
        for i in range(nr):
            g_ok, (k_ok, kl) = ok_g[i], ok_k[i]
            if len(g_ok) and len(k_ok):
                draws[i] = (int(g_ok[torch.randint(len(g_ok), (1,))]), int(k_ok[torch.randint(len(k_ok), (1,))]), kl)
        if not draws:
            return None
        rows = list(draws)
        chunks = lambda ids: [ids[c:c + G] for c in range(0, len(ids), G)]  # noqa: E731
        padded = lambda ch: ch + [ch[0]] * (G - len(ch))  # noqa: E731
        er, er2 = {}, {}
        with self._prec(), torch.no_grad():
            for kl in sorted({d[2] for d in draws.values()}):
                for ch in chunks([i for i in rows if draws[i][2] == kl]):
                    a_ = self.wav16(torch.stack([x1[i, draws[i][1]:draws[i][1] + kl] for i in padded(ch)]))
                    e = self.embed(None, a_)
                    er.update({i: e[q] for q, i in enumerate(ch)})
                    if self.sv2 is not None:
                        e2 = self.sv2(a_)
                        er2.update({i: e2[q] for q, i in enumerate(ch)})
        cos, cos2, words, cst = [], [], [], []
        use_c = self.contrast_w > 0 and len(self.bank) >= 32
        for ch in chunks(rows):
            pc = padded(ch)
            with self._prec():
                ah = self.wav16(torch.stack([self._crop(take, x1, i, draws[i][0], L) for i in pc]))
                eh = self.embed(None, ah)
            c_g = (eh * torch.stack([er[i] for i in pc])).sum(-1)  # (G,)
            c_v = c_g
            if self.sv2 is not None:  # both verifiers, each against its own fingerprint of the prompt crop
                c2 = (self.sv2(ah) * torch.stack([er2[i] for i in pc])).sum(-1)
                c_v = self.sv_w * c_g + self.sv2_w * c2
                cos2.append(c2[:len(ch)].detach())
            obj = None
            if seed is not None:
                obj = (c_v * torch.cat([seed.expand(len(ch)), seed.new_zeros(G - len(ch))])).sum()
            if use_c:  # the take must beat the nearest OTHER voice by m (padding rows weigh nothing)
                l_c, wrong, margin = contrast_loss(eh, torch.stack([er[i] for i in pc]), self.bank.e, self.contrast_m, self.contrast_tau)
                cst.append(torch.stack([l_c[:len(ch)].detach(), wrong[:len(ch)], margin[:len(ch)]], 1))
                if seed is not None:
                    real = torch.cat([seed.new_ones(len(ch)), seed.new_zeros(G - len(ch))])
                    obj = obj + (-seed) * self.contrast_w * (l_c * real).sum()  # (-seed = weight / n)
            if self.asr is not None:  # word critic on the same crops: the real clip's same frames as the target
                with self._prec():
                    with torch.no_grad():
                        lr_ = self.letters(self.wav16(torch.stack([x1[i, draws[i][0]:draws[i][0] + L] for i in pc])))
                    lh = self.letters(ah)
                # (Solver 11-14: a per-frame KL mostly scores spike TIMING -- 10 ms shift 1.9, other words 7.8; CTC against
                # the real crop's own greedy letters separates them 40-70x: shift 0.14, other words 10.2)
                kl, ok = self._word_ctc(lr_, lh)  # (G,) per-letter CTC loss, (G,) rows with >= 3 letters
                ok = ok & torch.cat([ok.new_ones(len(ch)), ok.new_zeros(G - len(ch))])
                if ok.any():
                    words.append(kl[ok].detach())
                if seed is not None:  # weight * mean over the scored rows (padding / near-silent rows weigh nothing)
                    # (11-24 Solver) scaled with the call's weight like the voice terms (_wscale = weight / cfg.critic_weight:
                    # 1.0 exactly as v17 ran; the pooled critic's own weight moved only the voice part, and nc ran away)
                    obj = obj + self.asr_weight * getattr(self, "_wscale", 1.0) * (kl * ok).sum() / len(rows)
            if obj is not None:
                obj.backward()
                c_g = c_g.detach()
            cos.append(c_g[:len(ch)])
        self.last_word = torch.cat(words).mean() if words else None
        self.last_cos2 = torch.cat(cos2).mean() if cos2 else None
        self.last_contrast = torch.nanmean(torch.cat(cst), 0) if cst else None  # (loss, weight on wrong prompts, margin)
        if self.contrast_w > 0:
            self.bank.push(torch.stack([er[i] for i in rows]))
        c = torch.cat(cos).mean()
        if seed is not None:
            xh.backward(take.grad)
            return (1 - c).detach(), c.detach()
        return 1 - c, c.detach()
