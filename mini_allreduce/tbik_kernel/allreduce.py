from __future__ import annotations

import torch


try:
    import tbik_kernel.common_ops  # noqa: F401
except Exception as exc:
    raise ImportError(
        "tbik_kernel.common_ops failed to import. Build it with:\n"
        "  pip install -v --no-build-isolation /opt/tiger/llm_reproducibility/mini_allreduce\n"
    ) from exc


def init_custom_ar(*args, **kwargs):
    return torch.ops.tbik_kernel.init_custom_ar.default(*args, **kwargs)


def dispose(*args, **kwargs):
    return torch.ops.tbik_kernel.dispose.default(*args, **kwargs)


def all_reduce(*args, **kwargs):
    return torch.ops.tbik_kernel.all_reduce.default(*args, **kwargs)


def tree_all_reduce(*args, **kwargs):
    return torch.ops.tbik_kernel.tree_all_reduce.default(*args, **kwargs)


def get_graph_buffer_ipc_meta(*args, **kwargs):
    return torch.ops.tbik_kernel.get_graph_buffer_ipc_meta.default(*args, **kwargs)


def register_buffer(*args, **kwargs):
    return torch.ops.tbik_kernel.register_buffer.default(*args, **kwargs)


def register_graph_buffers(*args, **kwargs):
    return torch.ops.tbik_kernel.register_graph_buffers.default(*args, **kwargs)


def meta_size(*args, **kwargs):
    return torch.ops.tbik_kernel.meta_size.default(*args, **kwargs)