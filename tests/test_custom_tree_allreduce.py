import socket
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(
    rank: int,
    world_size: int,
    port: int,
    dtype: torch.dtype,
    numel: int,
    queue: mp.SimpleQueue,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    dist.barrier()
    ar_group = dist.new_group(backend="gloo")
    dist.barrier()

    repo_root = Path(__file__).resolve().parents[1]
    tbik_dir = repo_root / "src" / "tbik"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(tbik_dir))
    from tree_based_all_reduce import tree_all_reduce_sum_native

    from mini_allreduce import CustomTreeAllreduce

    custom_ar = CustomTreeAllreduce(group=ar_group, device=device)
    if custom_ar.disabled:
        queue.put({"rank": rank, "disabled": True})
        dist.destroy_process_group()
        return

    inp = torch.randn(numel, device=device, dtype=dtype)
    out = custom_ar.custom_tree_all_reduce(inp)
    if out is None:
        queue.put({"rank": rank, "disabled": True})
        dist.destroy_process_group()
        return

    ref = tree_all_reduce_sum_native(inp, device_group=dist.group.WORLD)
    eps = torch.finfo(ref.dtype).eps
    rel_diff = (ref - out).abs() / ref.abs().clamp_min(eps)
    rel_diff_f = rel_diff.float()
    stats = {
        "max": rel_diff_f.max().item(),
        "mean": rel_diff_f.mean().item(),
        "p50": rel_diff_f.quantile(0.50).item(),
        "p90": rel_diff_f.quantile(0.90).item(),
        "p99": rel_diff_f.quantile(0.99).item(),
        "nan": torch.isnan(rel_diff_f).sum().item(),
        "inf": torch.isinf(rel_diff_f).sum().item(),
    }
    queue.put({"rank": rank, "disabled": False, "stats": stats})

    custom_ar.close()
    dist.destroy_process_group()


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is required for custom tree allreduce.")
        return

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from mini_allreduce import custom_all_reduce_ops as ops

    if not ops.IS_CUSTOM_AR_AVAILABLE:
        print("Custom allreduce extension is not available.")
        return

    world_size = 8
    if torch.cuda.device_count() < world_size:
        print(f"Need at least {world_size} GPUs for this test.")
        return

    mp.set_start_method("spawn", force=True)
    port = _get_open_port()
    queue: mp.SimpleQueue[Dict[str, Any]] = mp.SimpleQueue()
    procs = []
    dtype = torch.float16
    numel = 256 * 1024

    for rank in range(world_size):
        proc = mp.Process(
            target=_worker,
            args=(rank, world_size, port, dtype, numel, queue),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Worker {proc.pid} failed with code {proc.exitcode}")

    disabled = False
    rank_stats: List[Dict[str, float]] = []
    for _ in range(world_size):
        msg = queue.get()
        disabled |= msg.get("disabled", False)
        if "stats" in msg:
            rank_stats.append(msg["stats"])

    if disabled:
        print("CustomTreeAllreduce disabled on this system.")
        return

    if not rank_stats:
        print("No stats collected from workers.")
        return

    keys = ["max", "mean", "p50", "p90", "p99", "nan", "inf"]
    print("relative error distribution across ranks:")
    for key in keys:
        values = [s[key] for s in rank_stats]
        values_sorted = sorted(values)
        mid = len(values_sorted) // 2
        if len(values_sorted) % 2 == 0:
            median = 0.5 * (values_sorted[mid - 1] + values_sorted[mid])
        else:
            median = values_sorted[mid]
        mean = sum(values) / len(values)
        print(
            f"  {key}: min={min(values):.6g}, median={median:.6g}, mean={mean:.6g}, max={max(values):.6g}"
        )


if __name__ == "__main__":
    main()
