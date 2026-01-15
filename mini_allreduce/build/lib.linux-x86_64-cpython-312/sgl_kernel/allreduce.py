from typing import List, Tuple

import torch

try:
    from . import common_ops as _common_ops  # noqa: F401
except Exception as exc:
    raise ImportError(
        "sgl_kernel.common_ops failed to import. Build it with "
        "`python -m pip install -v ./mini_allreduce`."
    ) from exc


def init_custom_ar(
    ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
) -> int:
    return torch.ops.sgl_kernel.init_custom_ar.default(
        ipc_tensors, rank_data, rank, full_nvlink
    )


def dispose(fa: int) -> None:
    torch.ops.sgl_kernel.dispose.default(fa)


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops.sgl_kernel.all_reduce.default(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes
    )


def tree_all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops.sgl_kernel.tree_all_reduce.default(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes
    )


def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
    return torch.ops.sgl_kernel.get_graph_buffer_ipc_meta.default(fa)


def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
    torch.ops.sgl_kernel.register_buffer.default(fa, fake_ipc_ptrs)


def register_graph_buffers(
    fa: int, handles: List[List[int]], offsets: List[List[int]]
) -> None:
    torch.ops.sgl_kernel.register_graph_buffers.default(fa, handles, offsets)


def meta_size() -> int:
    return torch.ops.sgl_kernel.meta_size.default()
