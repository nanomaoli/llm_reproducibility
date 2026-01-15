import ctypes
import logging
import os
from ctypes.util import find_library
from typing import List, Tuple

import torch

from mini_allreduce.utils import is_cuda

logger = logging.getLogger(__name__)

IS_CUSTOM_AR_AVAILABLE = is_cuda()


def _preload_libcuda() -> None:
    # Avoid loading the CUDA driver stub from CUDA toolkit stubs/ if present.
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

try:
    _preload_libcuda()
    import sgl_kernel.allreduce as _custom_ar
except ImportError as exc:
    if IS_CUSTOM_AR_AVAILABLE:
        logger.warning("Failed to import sgl_kernel.allreduce: %r", exc)
    IS_CUSTOM_AR_AVAILABLE = False

if IS_CUSTOM_AR_AVAILABLE:

    def init_custom_ar(
        ipc_tensors: List[int],
        rank_data: torch.Tensor,
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return _custom_ar.init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)

    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
    ) -> None:
        _custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)

    def tree_all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
    ) -> None:
        _custom_ar.tree_all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)

    def dispose(fa: int) -> None:
        _custom_ar.dispose(fa)

    def meta_size() -> int:
        return _custom_ar.meta_size()

    def register_buffer(fa: int, ipc_tensors: List[int]) -> None:
        _custom_ar.register_buffer(fa, ipc_tensors)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        return _custom_ar.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        _custom_ar.register_graph_buffers(fa, handles, offsets)
