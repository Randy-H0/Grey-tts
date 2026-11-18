"""Read any audio file: wav / flac / mp3 / ogg through soundfile, everything else (m4a / aac from phones,
webm, mp4, opus-in-mp4, ...) through PyAV's bundled ffmpeg."""
import io
from pathlib import Path

import numpy as np

EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".aiff", ".aif", ".m4a", ".aac", ".mp4", ".webm", ".wma",
        ".mka", ".mov", ".caf", ".amr", ".3gp"}


def read_any(src):
    """Path or raw bytes -> (float32 array (frames, channels), sample rate)."""
    import soundfile as sf

    try:
        return sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001  (format soundfile doesn't know: decode with ffmpeg)
        import av

        with av.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src)) as c:
            st = next(s for s in c.streams if s.type == "audio")
            sr = st.rate or st.codec_context.sample_rate
            rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
            out = []
            for frame in c.decode(st):
                for f in rs.resample(frame):
                    out.append(f.to_ndarray().reshape(-1))
            for f in rs.resample(None):
                out.append(f.to_ndarray().reshape(-1))
        if not out:
            raise ValueError("no audio found in that file")
        return np.concatenate(out)[:, None].astype(np.float32), int(sr)


def to_flac(src: Path, dst: Path) -> Path:
    """Convert anything readable to 16-bit FLAC (for the training pipeline, which reads seekable files)."""
    import soundfile as sf

    w, sr = read_any(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(w).max()) if len(w) else 0.0
    sf.write(str(dst), w / peak if peak > 1.0 else w, sr, subtype="PCM_16")
    return dst
