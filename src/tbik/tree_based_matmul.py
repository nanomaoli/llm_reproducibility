from typing import Callable, Dict, Any

import torch
import triton
import triton.language as tl
import math


def _matmul_launch_metadata(
        grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n



def _get_tl_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16



# ---- kernel ----
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tp_persistent(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,
        LEVEL_K: tl.constexpr,  # = log2(TILE_K)
        TILE_K: tl.constexpr,  # = K//BLOCK_K
        FIRST_LEVEL_BLOCK: tl.constexpr,
        NEXT_POWER_OF_LEVEL: tl.constexpr,
        NEXT_POWER_OF_REMAIN_LEVEL: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        A_LARGE: tl.constexpr,
        B_LARGE: tl.constexpr,
        C_LARGE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    manual_acc = 3
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    S = tl.zeros((NEXT_POWER_OF_REMAIN_LEVEL, BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    S_mask = tl.arange(0, NEXT_POWER_OF_REMAIN_LEVEL)[:, None, None]

    offs_k = tl.arange(0, BLOCK_K)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    # set flatten to False further reduce 15% latency from 1724->1365 (I set it back but seems no change)
    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N

        offs_am = start_m + tl.arange(0, BLOCK_M)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)

        offs_bn = start_n + tl.arange(0, BLOCK_N)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)

        # this two line will results in 50% less latency!!! from 5709->2561 Idk why yet.
        # if I comment offs_am/bn = tl.where and these two lines, the performance remain unaffected
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

        # 改用指针递增寻址 不过没啥效果
        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk

        count = tl.zeros((NEXT_POWER_OF_LEVEL,), dtype=tl.int32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        for s_tile_idx in range(0, TILE_K):
            k0 = s_tile_idx * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & ((k0 + offs_k)[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=((k0 + offs_k)[:, None] < K) & (offs_bn[None, :] < N),
                other=0.0,
            )
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

            acc = tl.dot(a, b).to(ACC_DTYPE)

            break_flag = 0
            # using for instead of while reduce latency from 1365->1328
            for level in range(LEVEL_K):
                if break_flag == 0:
                    idx_mask = tl.arange(0, NEXT_POWER_OF_LEVEL) == level

                    count_value_added = tl.sum(count * idx_mask) + 1

                    table_value = FIRST_LEVEL_BLOCK if level == 0 else 2

                    carry_over = (table_value == count_value_added).to(tl.int1)

                    # 之前已经有值了 需要累加acc的值
                    if count_value_added > 1:
                        if level == 0:
                            acc = acc1 + acc
                        elif level == 1:
                            acc = acc2 + acc
                        elif level == 2:
                            acc = acc3 + acc
                        else:
                            tmp_acc_mask = (S_mask == (level - manual_acc))
                            acc = tl.sum(S * tmp_acc_mask, axis=0, dtype=ACC_DTYPE) + acc

                    count = tl.where(idx_mask, count_value_added * (1 - carry_over), count)
                    # 没进位
                    if not carry_over:
                        break_flag = 1
                        if level == 0:
                            acc1 = acc
                        elif level == 1:
                            acc2 = acc
                        elif level == 2:
                            acc3 = acc
                        else:
                            tmp_acc_mask = (S_mask == (level - manual_acc))
                            S = tl.where(tmp_acc_mask, acc[None, :, :], S)

        c_ptr = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        offs_cm = tl.where(offs_cm < M, offs_cm, 0)
        offs_cn = tl.where(offs_cn < N, offs_cn, 0)
        offs_cm = tl.max_contiguous(tl.multiple_of(offs_cm, BLOCK_M), BLOCK_M)
        offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_N), BLOCK_N)
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptr, acc.to(OUT_DTYPE), mask=mask_c)


def matmul_tp_persistent(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None):
    assert A.shape[-1] == B.shape[-2], "Dim doesn't match"

    out_dtype = A.dtype
    acc_dtype = A.dtype

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 8,
            "num_stages": 2,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 8,
            "num_stages": 2,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }

    BLOCK_M = configs[out_dtype]["BLOCK_SIZE_M"]
    BLOCK_N = configs[out_dtype]["BLOCK_SIZE_N"]
    BLOCK_K = configs[out_dtype]["BLOCK_SIZE_K"]
    GROUP_SIZE_M = configs[out_dtype]["GROUP_SIZE_M"]
    num_stages = configs[out_dtype]["num_stages"]
    num_warps = configs[out_dtype]["num_warps"]

    M, K = A.shape
    _, N = B.shape
    assert K % BLOCK_K == 0
    T = K // BLOCK_K
    FIRST_LEVEL_BLOCK = T

    LEVEL_K = 1
    while FIRST_LEVEL_BLOCK > 2 and FIRST_LEVEL_BLOCK % 2 == 0:
        FIRST_LEVEL_BLOCK //= 2
        LEVEL_K += 1

    C = torch.empty((M, N), device=A.device, dtype=out_dtype)

    manual_acc = 3

    NEXT_POWER_OF_LEVEL = 2 ** math.ceil(math.log2(LEVEL_K))
    # 这里设置最小为1, 即使用不到
    NEXT_POWER_OF_REMAIN_LEVEL = 2 ** math.ceil(math.log2(LEVEL_K - manual_acc)) if LEVEL_K > manual_acc else 1

    matmul_kernel_tp_persistent[grid](
        A, B, C,
        # Scratch,
        M, N, K,
        *A.stride(), *B.stride(), *C.stride(),
        # *Scratch.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=NUM_SMS,
        NEXT_POWER_OF_REMAIN_LEVEL=NEXT_POWER_OF_REMAIN_LEVEL,
        LEVEL_K=LEVEL_K, TILE_K=T, FIRST_LEVEL_BLOCK=FIRST_LEVEL_BLOCK, NEXT_POWER_OF_LEVEL=NEXT_POWER_OF_LEVEL,
        ACC_DTYPE=_get_tl_dtype(acc_dtype),
        OUT_DTYPE=_get_tl_dtype(out_dtype),
        A_LARGE=A.numel() > 2 ** 31,
        B_LARGE=B.numel() > 2 ** 31,
        C_LARGE=C.numel() > 2 ** 31,
        num_warps=num_warps, num_stages=num_stages
    )
    if bias is not None:
        C += bias
    return C