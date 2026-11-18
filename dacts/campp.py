"""CAM++ (VoxCeleb) as the voice critic's verifier (11-14, you: "a stronger critic, one that can't be fooled"; Solver's drop-in).
wave (B, T) at 16 kHz, float in [-1, 1]  ->  unit x-vectors (B, 512), gradient flows back to the wave.
Weights: models/campplus_voxceleb.bin (ModelScope iic/speech_campplus_sv_en_voxceleb_16k).
Features as modelscope's SpeakerVerificationCAMPPlus: 80-bin Kaldi fbank, mean removed over time, per clip.
Test: notes_solver/campp_critic.py (same class). Picked by notes_solver/verifier_pick.py: 100 % best match, r 0.42 with the judge."""
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as Kaldi

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "models" / "campplus_voxceleb.bin"


class CamppVerifier(nn.Module):
    def __init__(self, weights=WEIGHTS):
        super().__init__()
        from funasr.models.campplus.model import CAMPPlus
        self.net = CAMPPlus(feat_dim=80, embedding_size=512)
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.net.eval().requires_grad_(False)  # frozen; eval() keeps its batch norms on their stored statistics

    def train(self, mode=True):  # never leave eval mode (a .train() on the parent model must not reach the batch norms)
        return super().train(False)

    def forward(self, wav16: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav16.device.type, enabled=False):  # fbank + log in fp32
            feats = []
            for w in wav16.float():  # Kaldi.fbank is per clip; the critic's batch is a handful of equal-length rows
                f = Kaldi.fbank(w[None], num_mel_bins=80, sample_frequency=16000.0, dither=0.0)
                feats.append(f - f.mean(0, keepdim=True))
            e = self.net(torch.stack(feats))
        return torch.nn.functional.normalize(e.float(), dim=-1)


