import os
import torch
from src.tbik.tree_based_matmul import matmul_tp_persistent

# dimension of Qwen3-1.7B down_proj
M, K, N = 1024, 6144, 2048

tp_invariant_mode = int(os.environ.get('TP_INVARIANT_MATMUL', '0'))

data_type = torch.bfloat16
A = torch.randn((M, K), device="cuda", dtype=data_type)
B = torch.randn((K, N), device="cuda", dtype=data_type)

matmul_kernel = matmul_tp_persistent if tp_invariant_mode else torch.matmul

def test_batch_inv():
    C1 = matmul_kernel(A, B)[:M // 2, :]
    C2 = matmul_kernel(A[:M // 2, :], B)
    diff = (C1 - C2).abs().max()
    return diff


def test_tp_inv():
    num_gpus = 4

    result = []
    for i in range(num_gpus):
        # tensor-parallel matmul along K dimension
        start_k = i * K // num_gpus
        end_k = (i + 1) * K // num_gpus
        result.append(
            matmul_kernel(A[:, start_k:end_k], B[start_k:end_k, :]))

    # simulate tree all_reduce
    for level in range(1, num_gpus.bit_length()):
        for left in range(0, 4, 1 << level):
            right = left + (1 << (level - 1))
            result[left] += result[right]

    C1 = result[0]
    C2 = matmul_kernel(A, B)
    diff = (C1 - C2).abs().max()
    return diff

if __name__ == '__main__':
    diff_bs = test_batch_inv()
    diff_tp = test_tp_inv()
    print(f'TP-Invariant Mode {tp_invariant_mode}')
    print('-'*50)
    print('Test Batch-Invariant:')
    print('\tMax Difference value:', diff_bs.abs().max().item())
    print('\tIs Batch-Invariant:', torch.all(diff_bs == 0.0).item())
    print('Test TP-Invariant:')
    print('\tMax Difference value:', diff_tp.abs().max().item())
    print('\tIs TP-Invariant:', torch.all(diff_tp == 0.0).item())
