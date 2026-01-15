import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_cuda() -> bool:
    return torch.cuda.is_available() and torch.version.cuda is not None


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).lower()
    return value in ("true", "1")


def log_info_on_rank0(logger: logging.Logger, msg: str) -> None:
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info(msg)
    else:
        logger.info(msg)
