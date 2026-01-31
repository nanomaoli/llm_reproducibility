import torch
import torch.distributed as dist

custom_ar = None

def tree_all_reduce_sum(x:torch.Tensor, device_group=None) -> torch.Tensor:
    global custom_ar

    if x.numel() > 2**16:
        return tree_all_reduce_sum_native(x, device_group)
    try:
        if custom_ar is None:
            from mini_allreduce import CustomTreeAllreduce
            custom_ar = CustomTreeAllreduce(group=device_group, device=x.device)
        if custom_ar.disabled:
            raise 'CustomTreeAllreduce is disabled.'
        y = custom_ar.custom_tree_all_reduce(x)
        return y
    except:
        return tree_all_reduce_sum_native(x, device_group)

def tree_all_reduce_sum_native(x: torch.Tensor, device_group=None) -> torch.Tensor:
    rank = dist.get_rank(device_group)
    world_size = dist.get_world_size(device_group)

    if world_size & (world_size - 1) != 0:
        raise ValueError("world_size must be the pow of 2 in order to use all_reduce_sum。")

    result = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(result, x, group=device_group)

    for level in range(1, world_size.bit_length()):
        for left in range(0, world_size, 1 << level):
            right = left + (1 << (level - 1))
            result[left] += result[right]

    return result[0]