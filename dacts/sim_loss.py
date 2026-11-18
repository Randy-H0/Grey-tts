"""Speaker-similarity loss: "does the model's own prediction sound like the person?"

Flow matching only asks the decoder to match latents frame by frame; nothing directly rewards sounding like
the speaker, which is what zero-shot cloning is judged on. Here, on a few rows per step at low noise (t high,
where the one-jump clean estimate x1_hat = x_t + (1 - t) v is already a recognisable voice), a ~3 s crop of
x1_hat is decoded to audio (frozen KVAE decoder), embedded by a frozen speaker-verification network (WavLM-SV,
the same fingerprint the voice encoder reads), and compared with the same crop of the real recording:
loss = 1 - cosine. Gradients flow through the frozen decoder / verifier into the TTS model only.

Evaluate with a DIFFERENT verifier (UniSpeech-SAT, eval_bottleneck.py) so a gain is not just the model
learning to please WavLM-SV.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

from .codec import KVAE_SR, load_kvae


class SimLoss(nn.Module):
    def __init__(self, latent, device="cuda", crop: int = 150, rows: int = 2):
        super().__init__()
        from transformers import WavLMForXVector

        self.latent = latent
        self.kvae = load_kvae(device)
        self.sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval().requires_grad_(False)
        self.crop, self.rows = crop, rows

    def embed(self, z):
        """normalized latents (B, T, 64) -> unit x-vectors (B, 512), differentiable."""
        a = self.kvae.decode(self.latent.denormalize(z.float()).transpose(1, 2))[:, 0]
        a = AF.resample(a, KVAE_SR, 16000)
        a = (a - a.mean(-1, keepdim=True)) / (a.var(-1, keepdim=True) + 1e-7).sqrt()  # the feature extractor's normalisation
        return F.normalize(self.sv(input_values=a).embeddings.float(), dim=-1)

    def forward(self, x1_hat, x1, gen):
        """x1_hat / x1: (N, T, 64) normalized latents (prediction / real); gen: (N, T) frames the model generated."""
        L = self.crop
        picks = []
        for i in torch.randperm(x1.shape[0]).tolist():
            g = gen[i]
            ok = (g.unfold(0, L, 1).all(-1)).nonzero().flatten() if g.shape[0] >= L else g.new_zeros(0)
            if len(ok):
                s = int(ok[torch.randint(len(ok), (1,))])
                picks.append((i, s))
            if len(picks) >= self.rows:
                break
        if not picks:
            return None
        zh = torch.stack([x1_hat[i, s:s + L] for i, s in picks])
        zr = torch.stack([x1[i, s:s + L] for i, s in picks])
        with torch.autocast(device_type=zh.device.type, enabled=False):
            with torch.no_grad():
                er = self.embed(zr)
            eh = self.embed(zh)
        return (1 - (eh * er).sum(-1)).mean()
