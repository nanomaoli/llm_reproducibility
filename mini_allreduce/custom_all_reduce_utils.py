import logging
import os
from typing import List

import torch

logger = logging.getLogger(__name__)

try:
    import pynvml

    _HAS_NVML = True
except Exception as exc:
    _HAS_NVML = False
    logger.warning("pynvml not available: %r", exc)


def gpu_p2p_access_check(src: int, tgt: int) -> bool:
    skip_check = os.getenv("SGLANG_SKIP_P2P_CHECK", "0") == "1"
    if skip_check:
        logger.info("Skipping P2P check; using driver report only.")
    return torch.cuda.can_device_access_peer(src, tgt)


def is_full_nvlink(physical_device_ids: List[int], world_size: int) -> bool:
    if not _HAS_NVML:
        logger.warning("NVML unavailable; assuming no full NVLink.")
        return False
    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception("NVLink detection failed; assuming False.")
                        return False
        return True
    finally:
        pynvml.nvmlShutdown()


def is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )
