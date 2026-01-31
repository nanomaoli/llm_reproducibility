import ctypes
import logging
import os
import socket
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from mini_allreduce import custom_all_reduce_ops as ops
from mini_allreduce.cuda_wrapper import CudaRTLibrary
from mini_allreduce.custom_all_reduce_utils import (
    gpu_p2p_access_check,
    is_full_nvlink,
    is_weak_contiguous,
)
from mini_allreduce.utils import is_cuda, log_info_on_rank0

logger = logging.getLogger(__name__)


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def in_the_same_node_as(pg: ProcessGroup, source_rank: int = 0) -> List[bool]:
    # assert (
    #     dist.get_backend(pg) != dist.Backend.NCCL
    # ), "Use a non-NCCL group for in_the_same_node_as."
    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    hostname = socket.gethostname()
    hostnames: List[Optional[str]] = [None for _ in range(world_size)]
    dist.all_gather_object(hostnames, hostname, group=pg)
    return [h == hostnames[source_rank] for h in hostnames]


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CAR_SIZE = 8192 * 1024

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _MAX_CAR_SIZE,
    ) -> None:
        self._IS_CAPTURING = False
        self.disabled = True

        if not is_cuda() or not ops.IS_CUSTOM_AR_AVAILABLE:
            return

        self.group = group
        # assert (
        #     dist.get_backend(group) != dist.Backend.NCCL
        # ), "CustomAllreduce requires a non-NCCL process group."

        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning("Custom all-reduce disabled: multi-node group detected.")
            return

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1 and 1 not in self._SUPPORTED_WORLD_SIZES:
            return

        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom all-reduce disabled: unsupported world size %d.", world_size
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(torch.cuda.device_count()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cuda")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cuda") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        full_nvlink = is_full_nvlink(physical_device_ids, world_size)
        if world_size > 2 and not full_nvlink:
            logger.warning(
                "Custom all-reduce disabled: no full NVLink for world_size > 2."
            )
            return
            # pass

        if not _can_p2p(rank, world_size):
            logger.warning("Custom all-reduce disabled: P2P test failed.")
            return

        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink

        self.meta_ptrs = self.create_shared_buffer(
            ops.meta_size() + max_size, group=group
        )
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        self.rank_data = torch.empty(max_size, dtype=torch.uint8, device=self.device)
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.full_nvlink
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)

        self.disabled = False

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self) -> None:
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        log_info_on_rank0(logger, f"Registering {len(offset)} cuda graph addresses")
        all_data = [None] * dist.get_world_size(group=self.group)
        dist.all_gather_object(all_data, (handle, offset), group=self.group)
        handles = [d[0] for d in all_data]
        offsets = [d[1] for d in all_data]
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.world_size == 2 or self.full_nvlink:
            return inp_size < self.max_size
        return False

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor = None,
        registered: bool = False,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=False)
            return torch.zeros_like(input)
        return self.all_reduce(input, registered=False)

    def close(self) -> None:
        if not self.disabled and getattr(self, "_ptr", 0):
            ops.dispose(self._ptr)
            if dist.is_available() and dist.is_initialized():
                self.free_shared_buffer(self.meta_ptrs, group=self.group)
                self.free_shared_buffer(self.buffer_ptrs, group=self.group)
            self._ptr = 0

    def __del__(self):
        self.close()


class CustomTreeAllreduce(CustomAllreduce):
    _SUPPORTED_WORLD_SIZES = [1, 2, 4, 8]

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor = None,
        registered: bool = False,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.tree_all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.tree_all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_tree_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=False)
            return torch.zeros_like(input)
        return self.all_reduce(input, registered=False)
