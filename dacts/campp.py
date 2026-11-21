"""CAM++ (VoxCeleb) as the voice critic's verifier (11-14, you: "a stronger critic, one that can't be fooled"; Solver's drop-in).
wave (B, T) at 16 kHz, float in [-1, 1]  ->  unit x-vectors (B, 512), gradient flows back to the wave.
Weights: models/campplus_voxceleb.bin (ModelScope iic/speech_campplus_sv_en_voxceleb_16k).
Features as modelscope's SpeakerVerificationCAMPPlus: 80-bin Kaldi fbank, mean removed over time, per clip.
Test: notes_solver/campp_critic.py (same class). Picked by notes_solver/verifier_pick.py: 100 % best match, r 0.42 with the judge."""
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as Kaldi

# (11-22, Solver: training speed) Kaldi.fbank's three constants made once and kept on the device: they were three
# blocking CPU->GPU copies per call, 24 forced waits per critic step (critic step 477 -> 320 ms). Bit-identical.
import math as _math

_KC = {}            # the constants, made once per device / dtype
_KDEV = [None]      # the device of the waves being featurised (set in CamppVerifier.forward)
_k_eps0, _k_mel0 = Kaldi._get_epsilon, Kaldi.get_mel_banks


def _k_eps(device, dtype):
    k = ("eps", str(device), dtype)
    if k not in _KC:
        _KC[k] = _k_eps0(device, dtype)
    return _KC[k]


def _k_mel(*a, **kw):
    k = ("mel", a, tuple(sorted(kw.items())))
    if k not in _KC:
        _KC[k] = _k_mel0(*a, **kw)          # (bins, centre frequencies) on the CPU, exactly as Kaldi builds them
    bins, cf = _KC[k]
    d = _KDEV[0]
    if d is None or d.type != "cuda":       # CPU callers (and loader workers) get what they always got
        return bins, cf
    kd = k + (str(d),)
    if kd not in _KC:
        _KC[kd] = bins.to(d)                # one copy, ever; fbank's own `.to(device, dtype)` is then a no-op
    return _KC[kd], cf


def _k_log_energy(strided_input, epsilon, energy_floor):
    device, dtype = strided_input.device, strided_input.dtype
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()
    if energy_floor == 0.0:
        return log_energy
    k = ("floor", str(device), dtype, energy_floor)
    if k not in _KC:
        _KC[k] = torch.tensor(_math.log(energy_floor), device=device, dtype=dtype)
    return torch.max(log_energy, _KC[k])


Kaldi._get_epsilon, Kaldi.get_mel_banks, Kaldi._get_log_energy = _k_eps, _k_mel, _k_log_energy

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
            _KDEV[0] = wav16.device
            feats = []
            for w in wav16.float():  # Kaldi.fbank is per clip; the critic's batch is a handful of equal-length rows
                f = Kaldi.fbank(w[None], num_mel_bins=80, sample_frequency=16000.0, dither=0.0)
                feats.append(f - f.mean(0, keepdim=True))
            e = self.net(torch.stack(feats))
        return torch.nn.functional.normalize(e.float(), dim=-1)


