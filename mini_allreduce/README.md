# Minimal custom all-reduce benchmark (NVIDIA only)

This folder is a minimal, standalone subset to exercise the custom all-reduce
kernel and compare against `torch.distributed.all_reduce` (NCCL).

## Requirements

- NVIDIA GPUs
- PyTorch with CUDA
- Build toolchain for CUDA extensions (nvcc, a C++17 compiler)

## Install

From the repo root:

```
uv pip install --no-build-isolation -v ./mini_allreduce
```


## Use CustomTreeAllreduce

The custom tree all-reduce is a drop-in replacement for the original
`tree_all_reduce_sum` in `src/tbik`. Use `CustomTreeAllreduce` instead.

Minimal example:

```python
import torch
import torch.distributed as dist

from mini_allreduce import CustomTreeAllreduce

def run(rank: int, world_size: int, port: int):
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

    custom_tree_ar = CustomTreeAllreduce(group=ar_group, device=device)
    if custom_tree_ar.disabled:
        print("CustomTreeAllreduce disabled.")
        dist.destroy_process_group()
        return

    inp = torch.randn(256 * 1024, device=device, dtype=torch.float16)
    out = custom_tree_ar.custom_tree_all_reduce(inp)
    custom_tree_ar.close()
    dist.destroy_process_group()
```

Notes:
- `CustomTreeAllreduce` only supports world sizes in (1, 2, 4, 8).
- Custom all-reduce only runs when the input size is < 8MB and the size is a
  multiple of 16 bytes.
- For world sizes > 2, full NVLink connectivity is required.

## Run the test and benchmark

From the repo root:

```
python tests/test_custom_tree_allreduce.py # ensure bitwise identical results
python tests/bench_allreduce.py
```
