"""Code picker: chooses the actual DAC codes for each frame, codebook by codebook.

The flow decoder outputs continuous codebook vectors. Where several codes are plausible (the fine,
noise-like detail), the flow learns their *average*, and snapping an average to the nearest code
gives detail that is too regular (e.g. too-tonal high frequencies). This head instead samples
codebook 1, then 2 given 1, ... then 9 given 1..8 (DAC's codes are residual corrections of each
other), from the flow's final output plus the aligned text features and some neighbouring frames.

It is trained on detached inputs with its own optimizer, so it never changes the flow model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, d, k=7):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):  # x (B, F, d)
        h = self.dw((x * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        return x + self.ff(self.norm(h))


class _DepthLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):  # x (N, K, d), causal over the codebook axis
        n, k, d = x.shape
        q, kk, v = self.qkv(self.n1(x)).view(n, k, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, kk, v, is_causal=True).transpose(1, 2).reshape(n, k, d)
        x = x + self.o(a)
        return x + self.ff(self.n2(x))


class CodeHead(nn.Module):
    def __init__(self, cfg, d: int = 384, n_conv: int = 3, n_depth: int = 3, heads: int = 6, codebook_size: int = 1024):
        super().__init__()
        self.cfg, self.k, self.size = cfg, cfg.n_q, codebook_size
        self.inp = nn.Linear(cfg.latent_dim + cfg.token_dim + cfg.enc_dim, d)  # frame latent + its patch's [mu, h]
        self.convs = nn.ModuleList([_ConvBlock(d) for _ in range(n_conv)])
        self.code_emb = nn.Embedding(self.k * codebook_size, d)  # previous codebook's code (per-codebook tables)
        self.pos = nn.Parameter(torch.zeros(self.k, d))
        self.depth = nn.ModuleList([_DepthLayer(d, heads) for _ in range(n_depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Parameter(torch.randn(self.k, d, codebook_size) * d**-0.5)  # one classifier per codebook
        self.out_b = nn.Parameter(torch.zeros(self.k, codebook_size))

    def frame_features(self, x_frames, cond, mask):
        """x_frames (B, F, latent_dim) normalized flow output; cond (B, T, token_dim + enc_dim) per patch;
        mask (B, F) -> (B, F, d) per-frame context."""
        cond_f = cond.repeat_interleave(self.cfg.patch, dim=1)[:, : x_frames.shape[1]]
        h = self.inp(torch.cat([x_frames, cond_f.to(x_frames.dtype)], dim=-1))
        for c in self.convs:
            h = c(h, mask)
        return h

    def _logits(self, ctx, prev_codes):
        """ctx (N, d); prev_codes (N, j) codes of codebooks 0..j-1 -> logits (N, j+1, size)."""
        j = prev_codes.shape[1]
        x = ctx[:, None].expand(-1, j + 1, -1) + self.pos[: j + 1]
        if j:
            offs = torch.arange(j, device=prev_codes.device) * self.size
            x = x + F.pad(self.code_emb(prev_codes + offs), (0, 0, 1, 0))  # shift right: position i sees code i-1
        for layer in self.depth:
            x = layer(x)
        x = self.norm(x)
        return torch.einsum("nkd,kdc->nkc", x, self.out[: j + 1]) + self.out_b[: j + 1]

    def loss(self, x_frames, cond, codes, mask, max_frames: int = 4096):
        """codes (B, n_q, F) true DAC codes; teacher-forced cross-entropy on a random subset of frames."""
        ctx = self.frame_features(x_frames, cond, mask)
        sel = mask.flatten().nonzero().squeeze(1)
        if sel.numel() > max_frames:
            sel = sel[torch.randperm(sel.numel(), device=sel.device)[:max_frames]]
        ctx = ctx.flatten(0, 1)[sel]
        tgt = codes[:, : self.k].transpose(1, 2).flatten(0, 1)[sel]  # (N, k)
        logits = self._logits(ctx, tgt[:, :-1]).float()
        ce = F.cross_entropy(logits.flatten(0, 1), tgt.flatten())
        acc = (logits.argmax(-1) == tgt).float().mean(0)  # per codebook
        return ce, acc.detach()

    @torch.no_grad()
    def sample(self, x_frames, cond, mask, temperature: float = 1.0, top_p: float = 0.95):
        """-> codes (B, n_q, F)."""
        b, f, _ = x_frames.shape
        ctx = self.frame_features(x_frames, cond, mask).flatten(0, 1)
        codes = torch.zeros(ctx.shape[0], 0, dtype=torch.long, device=ctx.device)
        for _ in range(self.k):
            logits = self._logits(ctx, codes)[:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                p = torch.softmax(logits / temperature, -1)
                if top_p < 1:  # nucleus: drop the unlikely tail
                    sp, si = p.sort(-1, descending=True)
                    drop = sp.cumsum(-1) - sp > top_p
                    p = p.scatter(-1, si, sp.masked_fill(drop, 0))
                nxt = torch.multinomial(p, 1).squeeze(1)
            codes = torch.cat([codes, nxt[:, None]], dim=1)
        return codes.view(b, f, self.k).transpose(1, 2)
