import argparse
import socket
import sys
import time
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _benchmark_all_reduce(inp: torch.Tensor, iters: int) -> float:
    latencies = []
    for _ in range(iters):
        buf = inp.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    return median(latencies)


def _benchmark_custom_ar(inp: torch.Tensor, custom_ar, iters: int) -> float:
    latencies = []
    for _ in range(iters):
        buf = inp.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = custom_ar.custom_all_reduce(buf)
        torch.cuda.synchronize()
        if out is None:
            return float("nan")
        latencies.append(time.perf_counter() - start)
    return median(latencies)


def _benchmark_tree_ar(inp: torch.Tensor, tree_ar, iters: int) -> float:
    latencies = []
    for _ in range(iters):
        buf = inp.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        tree_ar(buf)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    return median(latencies)


def _worker(rank: int, world_size: int, port: int, args):
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
    sys.path.insert(0, str(tbik_dir))
    # from tree_based_all_reduce import tree_all_reduce_sum

    from mini_allreduce import CustomAllreduce

    custom_ar = CustomAllreduce(group=ar_group, device=device)

    # tree_enabled = world_size & (world_size - 1) == 0
    tree_enabled = False
    if rank == 0:
        if custom_ar.disabled:
            print("Custom all-reduce disabled; falling back to NCCL only.")
        else:
            print("Custom all-reduce enabled.")
        if not tree_enabled:
            print("Tree all-reduce requires power-of-two world size; skipping.")

    dtype = getattr(torch, args.dtype)
    for numel in args.numel_list:
        inp = torch.randn(numel, device=device, dtype=dtype)
        dist.barrier()

        lat_nccl = _benchmark_all_reduce(inp, args.iters)
        lat_custom = float("nan")
        if not custom_ar.disabled:
            lat_custom = _benchmark_custom_ar(inp, custom_ar, args.iters)

        lat_tree = float("nan")
        if tree_enabled:
            lat_tree = _benchmark_tree_ar(inp, tree_all_reduce_sum, args.iters)

        dist.barrier()
        if rank == 0:
            print(
                f"numel={numel:<10d} nccl={lat_nccl*1e3:>8.3f} ms "
                f"custom={lat_custom*1e3:>8.3f} ms tree={lat_tree*1e3:>8.3f} ms"
            )

    if not custom_ar.disabled:
        custom_ar.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Compare NCCL all_reduce vs custom all-reduce")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--numel",
        type=int,
        nargs="+",
        default=[256 * 1024, 512 * 1024, 1024 * 1024],
        help="List of element counts to benchmark",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    args = parser.parse_args()

    args.numel_list = args.numel

    available = torch.cuda.device_count()
    if available < args.world_size:
        raise RuntimeError(f"Need {args.world_size} GPUs, but only {available} visible.")

    mp.set_start_method("spawn", force=True)
    port = _get_open_port()
    procs = []
    for rank in range(args.world_size):
        p = mp.Process(target=_worker, args=(rank, args.world_size, port, args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker {p.pid} failed with code {p.exitcode}")


if __name__ == "__main__":
    main()
