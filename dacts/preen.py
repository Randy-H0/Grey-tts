"""Preen-SE's damage, version 2 (owner 2026-10-27: "give preen some love, make it perform well"; the goal is an Adobe
Podcast Enhance-like "studio sound"). make_corrupt.py --kind preen damages studio-clean clips with this and keeps the
timing frame for frame, so the damaged and clean latents line up.

Compared with dacts/corrupt.py (prompt-robustness damage, all synthetic, steady levels):
  - REAL noise: DEMAND (18 environments at 48 kHz) + MUSAN noise / music / speech (16 kHz, so only below 8 kHz),
    next to the synthetic noise / hum / music and other speakers from our own data (babble)
  - levels from barely there to louder than the voice (SNR -5 .. 40 dB), and levels that CHANGE: fades, bursts,
    a background that starts partway
  - simulated rooms (tools/preen_rooms.py: real room shapes, 0.12-1.6 s echo, mic 0.2-4 m away)
  - bad microphones: thin (low cut), dull (high cut), boxy / harsh / nasal (EQ peaks and dips), cheap-mic distortion
  - low-bitrate codecs (MP3 / Opus through ffmpeg, re-aligned), clipping, phone bandwidth
  - 10% of clips untouched: Preen must learn to leave clean audio alone

  bank = PreenBank("cuda"); y, kinds = preen_damage(x, bank, others)      x: (N,) 48 kHz mono on the GPU
"""
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

from .corrupt import _mix, _rms, clip, coloured_noise, hum, music, phone

SR = 48000
ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "data" / "preen"


class PreenBank:
    """Rooms (data/preen/rooms.npz) and real noise excerpts (data/preen/noise_bank.npz, tools/preen_noise_bank.py),
    held on the CPU in fp16; an excerpt goes to the GPU when used."""

    def __init__(self, device="cuda"):
        self.device = device
        r = np.load(BANK / "rooms.npz")
        self.rirs, self.roff = r["rirs"], r["offsets"]
        self.rt60, self.dist = r["rt60"], r["dist"]
        self.noise, self.place = {}, None
        f = BANK / "noise_bank.npz"
        if f.exists():
            z = np.load(f)
            for kind in ("demand", "musan_noise", "musan_music", "musan_speech"):
                if f"{kind}_audio" in z:
                    self.noise[kind] = (z[f"{kind}_audio"], z[f"{kind}_offsets"])
            if "demand_place" in z:
                self.place = z["demand_place"]

    def room(self, rt=(0.0, 9.0), dist=(0.0, 99.0)):
        """A simulated room whose echo time and mic distance fall in these ranges (any room if none does)."""
        ok = np.flatnonzero((self.rt60 >= rt[0]) & (self.rt60 <= rt[1]) & (self.dist >= dist[0]) & (self.dist <= dist[1]))
        i = int(random.choice(ok)) if len(ok) else random.randrange(len(self.roff) - 1)
        return torch.from_numpy(self.rirs[self.roff[i]: self.roff[i + 1]].astype(np.float32)).to(self.device)

    def get(self, kind, n, places=None):
        """A real-noise excerpt; for DEMAND, from these places only (letters: D home, O office, P public indoors,
        S street, T transport, N nature)."""
        audio, off = self.noise[kind]
        i = random.randrange(len(off) - 1)
        if kind == "demand" and places and self.place is not None:
            ok = np.flatnonzero(np.isin(self.place, list(places)))
            i = int(random.choice(ok)) if len(ok) else i
        seg = audio[off[i]: off[i + 1]]
        if len(seg) < n:
            seg = np.tile(seg, n // len(seg) + 1)
        s = random.randrange(0, len(seg) - n + 1)
        return torch.from_numpy(seg[s: s + n].astype(np.float32)).to(self.device)


def envelope(n, device):
    """How a background's level moves over the clip: steady, fading in / out, in bursts, or starting partway."""
    t = torch.linspace(0, 1, n, device=device)
    r = random.random()
    if r < 0.45:
        return torch.ones(n, device=device)
    if r < 0.65:  # fade in or out
        a, b = sorted((random.random(), random.random()))
        e = torch.clamp((t - a) / max(b - a, 1e-3), 0, 1)
        return e if random.random() < 0.5 else 1 - e * random.uniform(0.6, 1.0)
    if r < 0.85:  # bursts (a door, a passing car, someone talking for a moment)
        e = torch.full((n,), random.uniform(0.0, 0.3), device=device)
        for _ in range(random.randint(1, 4)):
            c, w = random.random(), random.uniform(0.03, 0.3)
            e = torch.maximum(e, torch.exp(-((t - c) / w) ** 2))
        return e
    e = (t > random.uniform(0.15, 0.8)).float()  # starts partway, with a short ramp
    return AF.lowpass_biquad(e[None], SR, 5.0)[0].clamp(0, 1)


def _biquad_h(f, kind, f0, gain_db=0.0, q=0.707):
    """Complex frequency response of an RBJ-cookbook biquad at frequencies f (Hz)."""
    w0 = 2 * math.pi * f0 / SR
    al, c = math.sin(w0) / (2 * q), math.cos(w0)
    if kind == "hp":
        b, a = [(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + al, -2 * c, 1 - al]
    elif kind == "lp":
        b, a = [(1 - c) / 2, 1 - c, (1 - c) / 2], [1 + al, -2 * c, 1 - al]
    else:  # peaking EQ
        A = 10 ** (gain_db / 40)
        b, a = [1 + al * A, -2 * c, 1 - al * A], [1 + al / A, -2 * c, 1 - al / A]
    z1 = torch.exp(-1j * 2 * math.pi * f / SR)
    return (b[0] + b[1] * z1 + b[2] * z1 ** 2) / (a[0] + a[1] * z1 + a[2] * z1 ** 2)


def _nfft(n):
    return 1 << (int(n) - 1).bit_length()


def fft_filter(x, filters):
    """Apply a chain of biquads [(kind, f0, gain_db, q), ...] at once in the frequency domain (one FFT; the
    time-domain biquads ran sample by sample: ~120 ms each on a 7 s clip)."""
    n = len(x)
    N = _nfft(n + SR // 4)  # (room for the filters' ringing)
    f = torch.fft.rfftfreq(N, 1 / SR).to(x.device).to(torch.float64)
    H = torch.ones_like(f, dtype=torch.complex128)
    for kind, f0, g, q in filters:
        H = H * _biquad_h(f, kind, f0, g, q)
    return torch.fft.irfft(torch.fft.rfft(x.double(), n=N) * H, n=N)[:n].to(x.dtype)


def convolve(x, h):
    """Linear convolution through one power-of-two FFT (torchaudio's fftconvolve: ~130 ms per clip here)."""
    n = len(x) + len(h) - 1
    N = _nfft(n)
    return torch.fft.irfft(torch.fft.rfft(x, n=N) * torch.fft.rfft(h, n=N), n=N)[: len(x)]


def mic_colour(x):
    """A worse microphone / position: low cut (thin), high cut (dull), 1-3 EQ peaks or dips (boxy, nasal, harsh),
    sometimes a cheap preamp's soft distortion."""
    filters = []
    if random.random() < 0.5:
        filters.append(("hp", random.uniform(70, 400), 0.0, 0.707))
    if random.random() < 0.35:
        filters.append(("lp", random.uniform(3500, 14000), 0.0, 0.707))
    for _ in range(random.randint(1, 3)):
        filters.append(("pk", math.exp(random.uniform(math.log(150), math.log(8000))),
                        random.choice([-1, 1]) * random.uniform(3, 12), random.uniform(0.7, 3.0)))
    x = fft_filter(x, filters)
    if random.random() < 0.15:
        drive = random.uniform(1.5, 6.0)
        x = torch.tanh(drive * x / x.abs().max().clamp(min=1e-6)) * x.abs().max() / math.tanh(drive)
    return x


# ---------------------------------------------------------------- devices, transmission (v3: realistic chains)
CODECS = {  # name: (ffmpeg encoder args, codec sample rate) -- the real ones phones, landlines and apps use
    "amrnb": (lambda: ["-c:a", "libopencore_amrnb", "-b:a", random.choice(["4.75k", "5.9k", "7.4k", "10.2k", "12.2k"])], 8000),
    "gsm": (lambda: ["-c:a", "libgsm"], 8000),
    "g711u": (lambda: ["-c:a", "pcm_mulaw", "-f", "wav"], 8000),
    "g711a": (lambda: ["-c:a", "pcm_alaw", "-f", "wav"], 8000),
    "g726": (lambda: ["-c:a", "g726", "-b:a", random.choice(["16k", "24k", "32k"]), "-f", "wav"], 8000),
    "amrwb": (lambda: ["-c:a", "libvo_amrwbenc", "-b:a", random.choice(["6.6k", "8.85k", "12.65k", "23.85k"])], 16000),
    "g722": (lambda: ["-c:a", "g722", "-f", "wav"], 16000),
    "speex": (lambda: ["-c:a", "libspeex", "-f", "ogg"], 16000),
    "opus_nb": (lambda: ["-c:a", "libopus", "-b:a", random.choice(["6k", "8k", "10k"]), "-f", "ogg"], 16000),
    "opus_voip": (lambda: ["-c:a", "libopus", "-b:a", random.choice(["16k", "20k", "24k", "32k"]), "-f", "ogg"], 48000),
    "aac": (lambda: ["-c:a", "aac", "-b:a", random.choice(["48k", "64k", "96k", "128k"]), "-f", "adts"], 44100),
    "mp3": (lambda: ["-c:a", "libmp3lame", "-b:a", random.choice(["32k", "48k", "64k", "96k"]), "-f", "mp3"], 44100),
}
CONTAINER = {"amrnb": "amr", "amrwb": "amr", "gsm": "gsm"}


def codec(x, name=None):
    """Through a real codec with ffmpeg at its own sample rate, back to 48 kHz, re-aligned sample for sample to the
    input (codec delay removed). Returns (audio, tag) or (x, None) if ffmpeg fails."""
    name = name or random.choice(["mp3", "opus_voip"])
    args, rate = CODECS[name]
    args = args()
    if "-f" not in args:
        args += ["-f", CONTAINER.get(name, "ogg")]
    a = x.detach().float().cpu().numpy()
    peak = float(np.abs(a).max()) or 1.0
    pcm = (np.clip(a / peak, -1, 1) * 32767).astype(np.int16).tobytes()
    try:
        b = subprocess.run(["ffmpeg", "-v", "quiet", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:",
                            "-ar", str(rate), "-ac", "1", *args, "pipe:"], input=pcm, capture_output=True, check=True).stdout
        d = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:", "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:"],
                           input=b, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return x, None
    y = np.frombuffer(d, np.int16).astype(np.float32) / 32767 * peak
    if len(y) < SR // 10:
        return x, None
    m = min(len(a), len(y), SR * 3)  # re-align on the first seconds (codecs add their own delay)
    best, lag = -1.0, 0
    for k in range(0, min(8000, len(y) - m + 8), 8):
        c = float(np.dot(a[:m - 8], y[k: k + m - 8]))
        if c > best:
            best, lag = c, k
    y = y[lag: lag + len(a)]
    if len(y) < len(a):
        y = np.concatenate([y, np.zeros(len(a) - len(y), np.float32)])
    return torch.from_numpy(y).to(x.device), name


def band(x, lo, hi, order=2):
    """A device / network bandwidth: steep high-pass at lo and low-pass at hi (cascaded biquads)."""
    return fft_filter(x, [("hp", lo, 0.0, 0.707)] * order + [("lp", hi, 0.0, 0.707)] * order)


def agc(x, strength=None):
    """A phone / app's automatic gain control: level pushed toward a target with a slow follower (it pumps on loud
    syllables and lifts the noise in pauses), timing untouched."""
    strength = strength if strength is not None else random.uniform(0.3, 0.9)
    w = int(0.05 * SR)
    env = torch.nn.functional.avg_pool1d((x ** 2)[None, None], w, 1, w // 2, count_include_pad=False)[0, 0][: len(x)].sqrt()
    env = torch.nn.functional.avg_pool1d(env[None, None], 4 * w + 1, 1, 2 * w, count_include_pad=False)[0, 0][: len(x)]
    target = env.max() * random.uniform(0.25, 0.5)
    g = (target / env.clamp(min=float(target) * 0.03)) ** strength
    return x * g.clamp(max=random.uniform(4, 20))


def noise_suppression(x):
    """A call app's noise suppressor: spectral gating with over-subtraction -- the watery 'musical noise' and
    swallowed word tails real calls have."""
    n_fft, hop = 1024, 256
    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft, hop, window=win, return_complex=True)
    mag = X.abs()
    floor = torch.quantile(mag, 0.15, dim=1, keepdim=True)
    m = ((mag - random.uniform(1.5, 4.0) * floor) / mag.clamp(min=1e-8)).clamp(min=random.uniform(0.02, 0.2), max=1.0)
    return torch.istft(X * m, n_fft, hop, window=win, length=len(x))


def packet_loss(x):
    """Dropped network packets: 20-60 ms holes (silence, or the previous piece repeated as phones conceal them)."""
    x = x.clone()
    for _ in range(random.randint(1, 8)):
        L = int(random.uniform(0.02, 0.06) * SR)
        if len(x) < 3 * L:
            break
        s = random.randrange(L, len(x) - L)
        if random.random() < 0.5:
            x[s: s + L] = 0
        else:
            x[s: s + L] = x[s - L: s].clone()
    return x


def desk_comb(x):
    """A laptop / table mic: the voice plus its reflection off the desk a fraction of a millisecond later (comb filter)."""
    d = int(random.uniform(0.0005, 0.004) * SR)
    return x + random.uniform(0.25, 0.6) * torch.nn.functional.pad(x, (d, 0))[: len(x)]


def ambience(x, bank, places, snr, others=(), music_p=0.06, babble_p=0.06):
    """Background sound fitting the place: real recordings (DEMAND) or synthetic noise; music or other people
    talking only sometimes (owner: "music is allowed, background chatter too, just don't overdo it")."""
    n, dev = len(x), x.device
    r = random.random()
    if r < music_p:
        nz = bank.get("musan_music", n) if "musan_music" in bank.noise and random.random() < 0.6 else music(n, dev)
        tag, snr = "music", max(snr, random.uniform(8, 30))
    elif r < music_p + babble_p:
        if others and random.random() < 0.6:
            o = [y for y in others if len(y)][: random.randint(2, 4)]
            nz = sum(torch.cat([y] * (n // len(y) + 1))[:n] for y in o) if o else coloured_noise(n, dev)
        elif "musan_speech" in bank.noise:
            nz = sum(bank.get("musan_speech", n) for _ in range(random.randint(2, 4)))
        else:
            nz = coloured_noise(n, dev)
        tag, snr = "chatter", max(snr, random.uniform(8, 30))
    elif "demand" in bank.noise and random.random() < 0.7:
        nz, tag = bank.get("demand", n, places), "ambience"
    elif "musan_noise" in bank.noise and random.random() < 0.5:
        nz, tag = bank.get("musan_noise", n), "noise"
    else:
        nz, tag = coloured_noise(n, dev), "hiss"
    nz = nz * envelope(n, dev)
    return (_mix(x, nz, snr) if _rms(nz) > 1e-5 else x), tag


def _mic(x, harsh=False):
    """mic_colour, optionally leaning harsh (cheap headsets: a 2-5 kHz peak)."""
    x = mic_colour(x)
    if harsh:
        x = fft_filter(x, [("pk", random.uniform(2000, 5000), random.uniform(4, 10), random.uniform(1.0, 2.5))])
    return x


def _saturate(x):
    drive = random.uniform(1.5, 4.0)
    return torch.tanh(drive * x / x.abs().max().clamp(min=1e-6)) * x.abs().max() / math.tanh(drive)


SCENARIOS = {"clean": 8, "almost_clean": 14, "phone_call": 14, "speakerphone": 8, "laptop": 14, "voice_memo": 9,
             "headset": 8, "bluetooth": 6, "distant": 7, "noisy_place": 7, "old_recording": 5}
# Preen v3 (owner 2026-10-28: v2 left the quiet background of a phone memo in place, v1 removed it; "make the voice sound
# like it's coming from a true studio microphone"): + quiet_room, the most common real case -- a phone or laptop mic in an
# ordinary quiet room, steady low background, light compression. PREEN_V3=1 adds it (v2's pairs were made without it).
# Then (owner: "make preen v3 a little bigger and more varied data too") five more real situations: a home studio (a good
# mic in an untreated room), a vlog / on-camera mic outdoors with wind, a lavalier on the chest, an old low-quality web
# file, and an echoey room with no other damage.
if os.environ.get("PREEN_V3") == "1":
    SCENARIOS.update({"quiet_room": 20, "clean": 6, "home_studio": 10, "vlog": 6, "lavalier": 6, "old_web_file": 5,
                      "echoey_room": 7})


def wind(n, device):
    """Wind on a mic: brown noise below ~300 Hz in gusts."""
    nz = AF.lowpass_biquad(coloured_noise(n, device, -2.0)[None], SR, random.uniform(120, 350))[0]
    t = torch.linspace(0, 1, n, device=device)
    g = torch.full((n,), random.uniform(0.05, 0.3), device=device)
    for _ in range(random.randint(1, 5)):
        c, w = random.random(), random.uniform(0.05, 0.25)
        g = torch.maximum(g, torch.exp(-((t - c) / w) ** 2))
    return nz * g


def preen_damage(x, bank, others=()):
    """x: (N,) clean studio speech, 48 kHz, on the GPU -> (damaged copy, what was done). One realistic SCENARIO per
    clip (owner 2026-10-27: "a lot of random ways to make it sound bad, also include a lot of phone filters with echo
    etc, make sure to try to make it as realistic as you can ... music is allowed, background chatter too, just don't
    overdo it"), each a chain in the order it happens: room -> mic -> background -> device processing -> codec /
    network. Timing is kept frame for frame."""
    n, dev = len(x), x.device
    scen = random.choices(list(SCENARIOS), list(SCENARIOS.values()))[0]
    done = [scen]
    if scen == "clean":
        return x, done
    if scen == "quiet_room":  # a phone / laptop mic in a quiet room: small room, the mic's colour, steady low background
        x = convolve(x, bank.room((0.15, 0.6), (0.3, 1.5)))
        x = _mic(x)
        x, tag = ambience(x, bank, random.choice(["D", "O", "D"]), random.uniform(18, 42), others, 0.02, 0.02)
        done.append(tag)
        if random.random() < 0.5:
            x = agc(x, random.uniform(0.2, 0.5))
        if random.random() < 0.6:
            x, tag = codec(x, random.choice(["aac", "aac", "opus_voip", "mp3"]))
            done.append(tag or "codec_failed")
        return x, done
    if scen == "home_studio":  # a decent mic close up in an untreated room: the room's reflections, a computer fan
        x = convolve(x, bank.room((0.3, 0.9), (0.2, 0.8)))
        if random.random() < 0.4:
            x = fft_filter(x, [("pk", random.uniform(150, 400), random.uniform(2, 6), random.uniform(0.7, 1.5))])  # boomy
        x, tag = ambience(x, bank, "DO", random.uniform(32, 55), others, 0.01, 0.01)
        done.append(tag)
        if random.random() < 0.4:
            x, tag = codec(x, random.choice(["mp3", "aac"]))
            done.append(tag or "codec_failed")
        return x, done
    if scen == "vlog":  # an on-camera mic 1-3 m away, outdoors: little room, street / park sound, wind gusts
        x = convolve(x, bank.room((0.1, 0.35), (1.0, 3.0)))
        x = _mic(x)
        x, tag = ambience(x, bank, random.choice(["S", "T", "P", "N"]), random.uniform(5, 25), others, 0.05, 0.08)
        done.append(tag)
        if random.random() < 0.6:
            x = _mix(x, wind(n, dev), random.uniform(0, 18))
            done.append("wind")
        x = agc(x)
        x, tag = codec(x, "aac")
        done.append(tag or "codec_failed")
        return x, done
    if scen == "lavalier":  # a clip-on mic on the chest: dull top, chest boom, a little room, light background
        x = fft_filter(x, [("lp", random.uniform(4000, 9000), 0.0, 0.707),
                           ("pk", random.uniform(500, 900), random.uniform(3, 7), random.uniform(0.8, 1.6)),
                           ("pk", random.uniform(2500, 4500), -random.uniform(3, 8), random.uniform(0.8, 2.0))])
        if random.random() < 0.5:
            x = convolve(x, bank.room((0.2, 0.7), (0.3, 1.2)))
        x, tag = ambience(x, bank, random.choice(["D", "O", "P"]), random.uniform(20, 45), others)
        done.append(tag)
        if random.random() < 0.6:
            x, tag = codec(x, random.choice(["aac", "opus_voip"]))
            done.append(tag or "codec_failed")
        return x, done
    if scen == "old_web_file":  # an old video / podcast file: low sample rate, low-bitrate mp3, some hiss
        sr2 = random.choice([8000, 11025, 16000, 22050])
        x = AF.resample(AF.resample(x[None], SR, sr2), sr2, SR)[0][:n]
        if len(x) < n:
            x = torch.cat([x, x.new_zeros(n - len(x))])
        done.append(f"sr{sr2}")
        if random.random() < 0.5:
            x = _mic(x)
        if random.random() < 0.6:
            x = _mix(x, coloured_noise(n, dev), random.uniform(25, 45))
            done.append("hiss")
        x, tag = codec(x, "mp3")
        done.append(tag or "codec_failed")
        return x, done
    if scen == "echoey_room":  # a big or bare room, nothing else: a kitchen, a hall, a church, a stairwell
        x = convolve(x, bank.room((0.7, 2.5), (1.0, 4.0)))
        return x, done
    if scen == "almost_clean":  # what most "clean" datasets really have: a little room, a little hiss, a mic's tint
        if random.random() < 0.6:
            x = convolve(x, bank.room((0.12, 0.35), (0.2, 0.8)))
        if random.random() < 0.6:
            x = fft_filter(x, [("pk", math.exp(random.uniform(math.log(150), math.log(8000))), random.uniform(-5, 5),
                                random.uniform(0.7, 2.0))])
        x, _ = ambience(x, bank, "DO", random.uniform(30, 50), others, 0.02, 0.02)
        return x, done
    if scen in ("phone_call", "speakerphone"):
        if scen == "speakerphone":
            x = convolve(x, bank.room((0.3, 1.1), (0.5, 1.8)))
        elif random.random() < 0.4:
            x = convolve(x, bank.room((0.15, 0.6), (0.1, 0.4)))
        x = _mic(x)
        x, tag = ambience(x, bank, random.choice(["T", "S", "P", "D", "O", "N"]), random.uniform(0, 30), others)
        done.append(tag)
        if random.random() < 0.5:
            x = noise_suppression(x)
            done.append("noise_suppression")
        x = agc(x)
        narrow = random.random() < 0.55
        x = (band(x, random.uniform(250, 350), random.uniform(3200, 3500)) if narrow
             else band(x, random.uniform(50, 120), random.uniform(6800, 7600)))
        name = random.choice(["amrnb", "amrnb", "gsm", "g711u", "g711a", "g726"] if narrow
                             else ["amrwb", "amrwb", "g722", "opus_nb", "speex"])
        x, tag = codec(x, name)
        done.append(tag or "codec_failed")
        if random.random() < 0.35:
            x = packet_loss(x)
            done.append("packet_loss")
        return x, done
    if scen == "laptop":  # a video call from a laptop / webcam mic
        x = convolve(x, bank.room((0.25, 1.0), (0.4, 1.5)))
        x = desk_comb(x)
        x = _mic(x)
        x, tag = ambience(x, bank, random.choice(["O", "D", "O"]), random.uniform(10, 40), others)
        done.append(tag)
        if random.random() < 0.6:
            x = noise_suppression(x)
            done.append("noise_suppression")
        x = agc(x)
        if random.random() < 0.75:
            x, tag = codec(x, "opus_voip")
            done.append(tag or "codec_failed")
            if random.random() < 0.25:
                x = packet_loss(x)
                done.append("packet_loss")
        return x, done
    if scen == "voice_memo":  # a phone recording a voice memo / video
        if random.random() < 0.8:
            x = convolve(x, bank.room((0.2, 0.9), (0.2, 1.0)))
        x = _mic(x)
        x, tag = ambience(x, bank, random.choice(["D", "S", "P", "N", "T"]), random.uniform(10, 40), others)
        done.append(tag)
        x = agc(x, random.uniform(0.2, 0.6))
        x, tag = codec(x, random.choice(["aac", "aac", "mp3"]))
        done.append(tag or "codec_failed")
        return x, done
    if scen == "headset":  # a cheap headset / USB mic close to the mouth
        x = _mic(x, harsh=True)
        x, tag = ambience(x, bank, "DO", random.uniform(20, 45), others, 0.04, 0.03)
        done.append(tag)
        if random.random() < 0.3:
            x = clip(x)
            done.append("clip")
        if random.random() < 0.5:
            x, tag = codec(x, random.choice(["opus_voip", "mp3"]))
            done.append(tag or "codec_failed")
        return x, done
    if scen == "bluetooth":  # a bluetooth headset: narrow (CVSD, 8 kHz) or wideband (mSBC, 16 kHz) voice link
        x = _mic(x, harsh=random.random() < 0.5)
        x, tag = ambience(x, bank, random.choice(["T", "S", "P", "D"]), random.uniform(5, 35), others)
        done.append(tag)
        if random.random() < 0.6:
            x = noise_suppression(x)
            done.append("noise_suppression")
        x, tag = codec(x, random.choice(["g711u", "g726"]) if random.random() < 0.5 else random.choice(["speex", "opus_nb"]))
        done.append(tag or "codec_failed")
        return x, done
    if scen == "distant":  # across a room / conference room
        x = convolve(x, bank.room((0.6, 1.6), (1.8, 4.0)))
        if random.random() < 0.5:
            x = _mic(x)
        x, tag = ambience(x, bank, random.choice(["O", "P", "D"]), random.uniform(5, 30), others)
        done.append(tag)
        if random.random() < 0.4:
            x, tag = codec(x, random.choice(["opus_voip", "aac"]))
            done.append(tag or "codec_failed")
        return x, done
    if scen == "noisy_place":  # out and about: loud real backgrounds
        if random.random() < 0.5:
            x = convolve(x, bank.room((0.2, 0.8), (0.2, 1.5)))
        x, tag = ambience(x, bank, random.choice(["S", "T", "P", "N"]), random.uniform(-5, 15), others, 0.1, 0.15)
        done.append(tag)
        if random.random() < 0.4:
            x, tag = codec(x, random.choice(["aac", "opus_voip", "amrwb"]))
            done.append(tag or "codec_failed")
        return x, done
    # old_recording: narrow band, hiss, sometimes hum, slightly saturated, an old room
    if random.random() < 0.6:
        x = convolve(x, bank.room((0.3, 1.2), (0.5, 2.5)))
    x = band(x, random.uniform(80, 200), random.uniform(3500, 7000), 1)
    if random.random() < 0.4:
        x = _mix(x, hum(n, dev), random.uniform(15, 35))
    x = _mix(x, coloured_noise(n, dev), random.uniform(15, 35))
    if random.random() < 0.5:
        x = _saturate(x)
    return x, done
