"""(11-18: a Grey Studio user, "Library cublas64_12.dll is not found or cannot be loaded" while transcribing) faster-whisper
(CTranslate2) looks for NVIDIA's cuBLAS / cuDNN DLLs on the system search path, so it worked only on PCs with the CUDA
toolkit installed. PyTorch already ships the same DLLs (torch/lib: cublas64_12, cublasLt64_12, cudnn64_9, cudart64_12):
put that folder (and pip's nvidia/*/bin, if there) on the DLL search path before faster-whisper loads. No-op off Windows."""
import importlib.util
import os
from pathlib import Path

_done = False


def expose_torch_dlls() -> list:
    global _done
    if _done or os.name != "nt":
        return []
    _done = True
    dirs = []
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        dirs.append(Path(spec.origin).parent / "lib")
    nv = importlib.util.find_spec("nvidia")
    if nv and nv.submodule_search_locations:
        for root in nv.submodule_search_locations:
            dirs += sorted(Path(root).glob("*/bin"))
    found = [d for d in dirs if d.is_dir()]
    for d in found:
        try:
            os.add_dll_directory(str(d))
        except OSError:
            pass
    os.environ["PATH"] = os.pathsep.join([*map(str, found), os.environ.get("PATH", "")])
    # CTranslate2 asks for the DLLs by bare name from its own folder / a CUDA install only (a user's diagnosis, 11-18):
    # load them by full path first -- Windows then hands that already-loaded module to its later LoadLibrary("cublas64_12.dll")
    import ctypes
    for name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
        for d in found:
            if (d / name).is_file():
                try:
                    ctypes.WinDLL(str(d / name))
                except OSError:
                    pass
                break
    return found
