import importlib.util
import os
import sys
from pathlib import Path

# triton-windows: the installed MSVC (14.36) can't compile Triton's generated launcher C code,
# so point Triton at its bundled TinyCC for this process only. Set DACTS_CC to override.
if sys.platform == "win32":
    _spec = importlib.util.find_spec("triton")
    if _spec and _spec.origin:
        _tcc = Path(_spec.origin).parent / "runtime" / "tcc" / "tcc.exe"
        if "DACTS_CC" in os.environ:
            os.environ["CC"] = os.environ["DACTS_CC"]
        elif _tcc.exists():
            os.environ["CC"] = str(_tcc)

    # Background runs (pythonw, no console): every compiler call Triton/Inductor makes would pop up
    # its own cmd window. Start child console programs without a window instead.
    import subprocess

    if not getattr(subprocess.Popen, "_dacts_no_window", False):
        _popen_init = subprocess.Popen.__init__

        def _no_window_init(self, *args, **kwargs):
            if not kwargs.get("creationflags"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            _popen_init(self, *args, **kwargs)

        subprocess.Popen.__init__ = _no_window_init
        subprocess.Popen._dacts_no_window = True
