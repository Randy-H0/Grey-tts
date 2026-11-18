"""Voice enhancer (you 2026-11-11: "a model whose goal truly is to push generated latents forward into the same voice";
"let's build the enhancer, from 15xl"). Replaces the two-pass Enhance voice (the finished take through voice-to-voice
again: the whole decoder twice) with ONE pass of a small network.

In: Grey's finished take (normalized KVAE latents, (B, T, 64)) + the reference recording's latents ((B, Tr, 64)).
Out: the same take -- same words, same timing, frame for frame -- with the voice moved toward the reference.

Shape: the take's frames go through a stack of transformer blocks (the main model's Block: self-attention with rotary
positions + conv feed-forward); after every 2nd block they read the reference through cross-attention. The reference is
read FRAME BY FRAME (its own small encoder, not pooled to a few tokens): how this voice says each sound is in the
frames. The output layer starts at zero and the network adds to its input, so an untrained enhancer returns the take
unchanged. No time input, no guidance, no sampling: one call.

Size (default): 384 wide, 10 blocks + 3 reference blocks, ~38M. Cost: one pass of a network a tenth the decoder's size,
against the decoder's 16+ batched passes per take.

Two kinds (cfg "flow"):
  flow=False  one call, y = take + correction. Learned by plain regression: where the target can't be told from the
              input it answers with the average, which can dull fine detail.
  flow=True   a few small steps (default 4) along a learned path that STARTS AT THE TAKE (plus a little noise) and ends
              at the real voice: the network gets the step time t as well and returns a velocity. Each step sees what
              the earlier ones committed to, so it lands on one sharp answer instead of the average. Still tiny: 4
              passes of a 38M network.

    enh = VoiceEnhancer(); y = enh.enhance(z, z_mask, ref, ref_mask)        VoiceEnhancer.load(path, device)
"""
import math
from pathlib import Path

import torch
import torch.nn as nn

from .model import Block, CrossBlock, rope


class VoiceEnhancer(nn.Module):
    def __init__(self, c=64, d=384, layers=10, heads=6, ref_layers=3, cross_every=2, mult=4, dropout=0.1, flow=False,
                 noise=0.25):
        super().__init__()
        self.cfg = dict(c=c, d=d, layers=layers, heads=heads, ref_layers=ref_layers, cross_every=cross_every, mult=mult,
                        dropout=dropout, flow=flow, noise=noise)
        self.flow, self.noise = flow, noise
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d)) if flow else None
        self.heads = heads
        self.inp = nn.Linear(c, d)
        self.ref_inp = nn.Linear(c, d)
        self.ref_blocks = nn.ModuleList([Block(d, heads, 2, dropout) for _ in range(ref_layers)])
        self.ref_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([Block(d, heads, mult, dropout) for _ in range(layers)])
        self.cross = nn.ModuleList([CrossBlock(d, heads) for _ in range(layers // cross_every)])
        self.cross_every = cross_every
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, c)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def memory(self, ref, ref_mask):
        """the reference read once: (B, Tr, d) for every step's cross-attention."""
        hd = self.cfg["d"] // self.heads
        r = self.ref_inp(ref.to(self.ref_inp.weight.dtype))
        rot_r = rope(r.shape[1], hd, r.device)
        for b in self.ref_blocks:
            r = b(r, ref_mask, rot_r)
        return self.ref_norm(r)

    def forward(self, z, mask, ref, ref_mask, t=None, mem=None):
        """z (B, T, c) the take (flow: the point x_t on the path), mask (B, T) bool; ref (B, Tr, c) the reference,
        ref_mask (B, Tr) bool (every row needs at least one True); t (B,) in [0, 1] for flow models.
        -> (B, T, c): flow=False the enhanced take, flow=True the velocity at (x_t, t)."""
        hd = self.cfg["d"] // self.heads
        r = self.memory(ref, ref_mask) if mem is None else mem
        x = self.inp(z.to(self.inp.weight.dtype))
        if self.flow:
            half = 128
            f = torch.exp(-math.log(10000) * torch.arange(half, device=z.device, dtype=torch.float32) / half)
            a = t.float()[:, None] * 1000 * f[None]
            x = x + self.t_mlp(torch.cat([a.sin(), a.cos()], -1).to(x.dtype))[:, None]
        rot = rope(x.shape[1], hd, x.device)
        for i, b in enumerate(self.blocks):
            x = b(x, mask, rot)
            if (i + 1) % self.cross_every == 0:
                x = self.cross[(i + 1) // self.cross_every - 1](x, r, ref_mask)
        o = self.out(self.norm(x)).to(z.dtype)
        return (o if self.flow else z + o) * mask[..., None].to(z.dtype)

    @torch.no_grad()
    def enhance(self, z, mask, ref, ref_mask, steps=4):
        """the take -> the enhanced take (flow=False: one call; flow=True: `steps` Euler steps from the take + noise)."""
        if not self.flow:
            return self(z, mask, ref, ref_mask)
        mem = self.memory(ref, ref_mask)
        x = (z + self.noise * torch.randn_like(z)) * mask[..., None].to(z.dtype)
        for i in range(steps):
            t = torch.full((z.shape[0],), i / steps, device=z.device)
            x = x + self(x, mask, ref, ref_mask, t=t, mem=mem).to(x.dtype) / steps
        return x

    def save(self, path, **meta):
        torch.save({"kind": "voice-enhancer", "cfg": self.cfg, "model": self.state_dict(), **meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        m = cls(**ck["cfg"])
        m.load_state_dict(ck["model"])
        return m.eval().to(device)
