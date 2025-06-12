"""Utility functions"""

import os
import subprocess
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pyarrow
import ray
from filelock import FileLock
from huggingface_hub import snapshot_download
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit  # type: ignore
from ray.data import Dataset

from .logging import get_logger

logger = get_logger(__name__)


# The default local root directory to store models downloaded from S3.
# This path should always available on Anyscale platform. If not, then
# we will fallback to FALLBACK_LOCAL_MODEL_ROOT.
DEFAULT_LOCAL_MODEL_ROOT = "/mnt/local_storage/cache"
FALLBACK_LOCAL_MODEL_ROOT = "/tmp/cache"


def update_dict_recursive(
    orig: Dict[str, Any], update_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Update a dictionary (in-place) recursively.

    Args:
        orig: The original dictionary.
        update_dict: The dictionary to update.

    Returns:
        The updated dictionary.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict):
            orig[key] = update_dict_recursive(orig.get(key, {}), value)
        else:
            orig[key] = value
    return orig


def wait_for_gpu_memory_to_clear(threshold_bytes: int, timeout_s: float = 120) -> None:
    """Wait for GPU memory to be below a threshold.
    Use nvml instead of pytorch to reduce measurement error from torch cuda context.

    Args:
        threshold_bytes: The threshold in bytes.
        timeout_s: The timeout in seconds.

    Raises:
        ValueError: If the memory is not free after the timeout.
    """
    devices = [int(x) for x in ray.get_gpu_ids()]
    nvmlInit()
    start_time = time.monotonic()
    while True:
        output = {}
        output_raw = {}
        for device in devices:
            dev_handle = nvmlDeviceGetHandleByIndex(device)
            mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
            gb_used = mem_info.used / 2**30
            output_raw[device] = gb_used
            output[device] = f"{gb_used:.02f}"

        logger.info(
            "GPU memory used (GB): " + "; ".join(f"{k}={v}" for k, v in output.items())
        )

        dur_s = time.monotonic() - start_time
        if all(v <= (threshold_bytes / 2**30) for v in output_raw.values()):
            logger.info(
                "Done waiting for free GPU memory on devices %s (%.2f GB) %.02f s",
                devices,
                threshold_bytes / 2**30,
                dur_s,
            )
            break

        if dur_s >= timeout_s:
            raise ValueError(
                f"Memory of devices {devices=} not free after "
                f"{dur_s=:.02f} ({threshold_bytes/2**30=})"
            )

        time.sleep(5)


def run_s3_command(command: List[str], error_msg: Optional[str] = None) -> Any:
    """Run a S3 command and raise an exception if it fails.

    Args:
        command: The command to run.
        error_msg: The error message to raise if the command fails.

    Returns:
        The result of the command.
    """
    try:
        return subprocess.run(command, check=True, capture_output=True)
    except Exception as err:
        # Not using logger.exception since we raise anyway.
        if isinstance(err, (subprocess.TimeoutExpired, subprocess.CalledProcessError)):
            stdout_txt = f"\nSTDOUT: {err.stdout.decode()}" if err.stdout else ""
            stderr_txt = f"\nSTDERR: {err.stderr.decode()}" if err.stderr else ""
        else:
            stdout_txt = ""
            stderr_txt = ""

        if error_msg is not None:
            logger.error(
                "(%s) %s. Command %s.%s%s",
                str(err),
                error_msg,
                command,
                stdout_txt,
                stderr_txt,
            )
        raise


def download_hf_model_from_s3(s3_path: str, local_path_root: str) -> str:
    """Download model files from s3 to the local path. The model path prefix
    will be added to the local path.

    Args:
        s3_path: The s3 path to download from.
        local_path_root: The local path root to download to.

    Returns:
        The local path where the files are downloaded.
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid s3 path: {s3_path}")

    prefix = "/".join(s3_path.split("/")[3:])
    local_path = Path(local_path_root) / prefix

    # Use aws s3 sync to make sure we don't download the same files again.
    command = ["aws", "s3", "sync", s3_path, local_path]

    logger.info(
        "Downloading %s to %s using %s",
        s3_path,
        local_path,
        command,
    )
    with FileLock(local_path / ".lock", timeout=-1):
        run_s3_command(command, f"Failed to sync model from {s3_path} to {local_path}")
    return str(local_path)


def maybe_download_model_from_s3(
    model_path: str, local_path_root: Optional[str] = None
) -> str:
    """Download model from s3 to the local path, and return the local model path.

    Args:
        model_path: The maybe s3 path to download from.
        lora_path_root: The local path root to download to. If not provided,
            will use the default path (/mnt/local_storage/cache or /tmp/cache).

    Returns:
        The local path where the model is downloaded.
    """
    s3_path = os.path.expandvars(model_path)
    if not s3_path.startswith("s3://"):
        return model_path

    local_root = Path(local_path_root or DEFAULT_LOCAL_MODEL_ROOT)
    try:
        local_root.mkdir(parents=True, exist_ok=True)
        # Check if the directory is writable.
        with open(local_root / ".test", "w") as fp:
            fp.write("test")
    except PermissionError:
        logger.warning(
            "Failed to create local root directory at %s (Permission denied). "
            "Reset local root to %s",
            local_root,
            FALLBACK_LOCAL_MODEL_ROOT,
        )
        local_root = Path(FALLBACK_LOCAL_MODEL_ROOT)
        local_root.mkdir(parents=True, exist_ok=True)

    return download_hf_model_from_s3(s3_path, local_root)


def download_model_from_hf(
    model_name: str, local_path_root: Optional[str] = None
) -> str:
    """Download model files from Hugging Face to the local path.
    If the local path has permission issues, return the original model name, but warn the user.

    Args:
        model_name: The model name to download.
        local_path_root: The local path root to download to. If not provided,
            will use the default path (/mnt/local_storage/cache or /tmp/cache

    Returns:
        The local path where the files are downloaded.
    """
    # If the model_name is already a local path, skip downloading
    if model_name.startswith("/"):
        return model_name

    local_model_path = Path(local_path_root or DEFAULT_LOCAL_MODEL_ROOT) / model_name
    try:
        local_model_path.mkdir(parents=True, exist_ok=True)

        # Check directory is writable by trying to list files (avoiding .test file creation)
        if not os.access(local_model_path, os.W_OK):
            raise PermissionError
    except PermissionError:
        logger.warning(
            "Failed to create or write to the model directory at %s (Permission denied). "
            "Please grant permission, or each worker may download the model, hitting rate limits.",
            local_model_path,
        )
        return model_name  # Return the original model name

    snapshot_download(repo_id=model_name, local_dir=str(local_model_path))

    return str(local_model_path)


def async_caller_empty_batch_handler(func) -> Callable:
    """A decorator to handle the case where all rows are checkpointed.
    When all rows are checkpointed, we will still get a batch
    in pyarrow.Table format with empty rows. This is a bug and
    is being tracked here:
    https://github.com/anyscale/rayturbo/issues/1292

    Args:
        func: The function to wrap.

    Returns:
        The wrapped function.
    """

    @wraps(func)
    async def wrapper(self, batch):
        if not isinstance(batch, pyarrow.lib.Table) or batch.num_rows > 0:
            async for x in func(self, batch):
                yield x
        else:
            yield {}

    return wrapper


def has_materialized(ds: Dataset) -> bool:
    """Check if the dataset has been materialized.
    TODO: This API should be moved to Ray Data.

    Args:
        ds: The dataset to check.

    Returns:
        True if the dataset is materialized, False otherwise.
    """
    return bool(ds.stats())
