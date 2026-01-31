import ctypes
import os
from ctypes.util import find_library


def _preload_libcuda() -> None:
    ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    for base in ld_paths:
        if not base or "stubs" in base:
            continue
        candidate = os.path.join(base, "libcuda.so.1")
        if os.path.isfile(candidate):
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                pass
    libname = find_library("cuda")
    if libname:
        try:
            ctypes.CDLL(libname, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_libcuda()

__all__ = ["allreduce"]
