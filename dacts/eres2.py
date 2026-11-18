"""Second verifier for the voice critic (11-14, you via the Solver: "Yes, download it"): ERes2NetV2 trained on about
200,000 Chinese speakers (ModelScope iic/speech_eres2netv2_sv_zh-cn_16k-common), weights in
notes_solver/models/eres2netv2_zh_common/pretrained_eres2netv2.ckpt. Different architecture and speakers from CAM++
(VoxCeleb) and from the gate's judge, so a critic using both cannot satisfy one model's quirks.
(B, T) waves at 16 kHz in [-1, 1] -> (B, 192) unit fingerprints; frozen, fp32 inside, gradient to the wave.
From the Solver's notes_solver/eres2_critic.py + campp_fast.py (batched on-device Kaldi fbank, CPU-tested against
Kaldi.fbank: max diff 3.6e-3, fingerprint cosine 1.000000)."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

WEIGHTS = Path(__file__).resolve().parents[1] / "notes_solver" / "models" / "eres2netv2_zh_common" / "pretrained_eres2netv2.ckpt"


class BatchedFbank(nn.Module):
    """(B, T) waves at 16 kHz in [-1, 1] -> (B, frames, 80) log mel energies, as Kaldi.fbank(dither=0) row by row."""

    def __init__(self, n_mels: int = 80, sr: float = 16000.0, frame_ms: float = 25.0, shift_ms: float = 10.0,
                 low_freq: float = 20.0, preemph: float = 0.97):
        super().__init__()
        self.win, self.hop, self.preemph = int(sr * frame_ms * 0.001), int(sr * shift_ms * 0.001), preemph
        self.n_fft = 1 << (self.win - 1).bit_length()  # round_to_power_of_two: 400 -> 512
        self.register_buffer("window", torch.hann_window(self.win, periodic=False, dtype=torch.float64).pow(0.85).float(), persistent=False)
        mel, _ = Kaldi.get_mel_banks(n_mels, self.n_fft, sr, low_freq, 0.0, 100.0, -500.0, 1.0)  # (80, 256)
        self.register_buffer("mel", F.pad(mel, (0, 1)).t().contiguous().float(), persistent=False)  # (257, 80)
        self.eps = float(torch.finfo(torch.float32).eps)

    def forward(self, wav):
        x = wav.float().unfold(-1, self.win, self.hop)  # (B, frames, 400), snip_edges
        x = x - x.mean(-1, keepdim=True)  # remove_dc_offset
        x = x - self.preemph * torch.cat([x[..., :1], x[..., :-1]], -1)  # x[i] - 0.97 x[i-1], with x[-1] := x[0]
        x = F.pad(x * self.window, (0, self.n_fft - self.win))
        p = torch.fft.rfft(x).abs().pow(2)  # power spectrum (B, frames, 257)
        return (p @ self.mel).clamp_min(self.eps).log()


class Eres2Verifier(nn.Module):
    def __init__(self, weights=WEIGHTS):
        super().__init__()
        from funasr.models.eres2net.eres2netv2 import ERes2NetV2
        self.net = ERes2NetV2(feat_dim=80, embedding_size=192, baseWidth=26, scale=2, expansion=2)  # configuration.json
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.fbank = BatchedFbank()
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, wav16):
        with torch.autocast(device_type=wav16.device.type, enabled=False):
            f = self.fbank(wav16)
            e = self.net(f - f.mean(1, keepdim=True))
        return F.normalize(e.float(), dim=-1)
