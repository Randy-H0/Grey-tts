"""RMVPE pitch tracker (Wei et al. 2023, arXiv 2306.15412), the model code from RVC-Project/Retrieval-based-Voice-
Conversion-WebUI infer/rmvpe.py (MIT, see LICENSE next to this file), trimmed for Grey's singing mode (2026-10-31):
the network classes are unchanged (so the published rmvpe.pt state dict loads as is); removed: the conv-STFT /
DirectML / ONNX paths, CUDA graphs and RVC's config imports. The weights load with weights_only=True (a plain state
dict: nothing is unpickled). Decoding vectorised (same maths as to_local_average_cents)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        return self.conv(x) + (self.shortcut(x) if hasattr(self, "shortcut") else x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)]
                                  + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return (x, self.pool(x)) if self.kernel_size is not None else x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels, out_channels, in_size = out_channels, out_channels * 2, in_size // 2
        self.out_size, self.out_channel = in_size, out_channels

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            skips.append(t)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)]
                                    + [ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
                                       for _ in range(n_inters - 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride, (1, 1), output_padding=out_padding,
                               bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)]
                                   + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])

    def forward(self, x, skip):
        x = torch.cat((self.conv1(x), skip), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(ResDecoderBlock(in_channels, in_channels // 2, stride, n_blocks, momentum))
            in_channels //= 2

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers,
                                         n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(BiGRU(3 * 128, 256, n_gru), nn.Linear(512, 360), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        return self.fc(self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2))


class MelSpectrogram(nn.Module):
    def __init__(self, n_mel_channels=128, sampling_rate=16000, win_length=1024, hop_length=160, mel_fmin=30,
                 mel_fmax=8000, clamp=1e-5):
        super().__init__()
        from librosa.filters import mel
        basis = mel(sr=sampling_rate, n_fft=win_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(basis).float())
        self.register_buffer("window", torch.hann_window(win_length))
        self.n_fft, self.hop_length, self.clamp = win_length, hop_length, clamp

    def forward(self, audio):
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
                         window=self.window, center=True, return_complex=True)
        return torch.log(torch.clamp(self.mel_basis @ fft.abs(), min=self.clamp))


class RMVPE(nn.Module):
    """16 kHz mono audio -> F0 in Hz per 10 ms frame (0 = unvoiced)."""
    SR, HOP = 16000, 160

    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.model = E2E(4, 1, (2, 2))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.mel = MelSpectrogram()
        self.to(device).eval().requires_grad_(False)
        self.device = device
        self.cents = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))

    @torch.no_grad()
    def salience(self, audio16k) -> np.ndarray:
        x = torch.as_tensor(np.asarray(audio16k, dtype=np.float32), device=self.device)[None]
        mel = self.mel(x)
        n = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n))
        return self.model(mel)[0, :n].float().cpu().numpy()

    def decode(self, sal: np.ndarray, thred=0.03) -> np.ndarray:
        center = np.argmax(sal, axis=1) + 4
        s = np.pad(sal, ((0, 0), (4, 4)))
        idx = center[:, None] + np.arange(-4, 5)[None]
        w = np.take_along_axis(s, idx, 1)
        cents = (w * self.cents[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
        f0 = 10 * 2 ** (cents / 1200)
        f0[s.max(1) <= thred] = 0
        return f0

    def __call__(self, audio16k, thred=0.03) -> np.ndarray:
        return self.decode(self.salience(audio16k), thred)
