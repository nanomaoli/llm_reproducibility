import torch
import torch.distributed as dist


def tree_all_reduce_sum(x: torch.Tensor, device_group=None) -> torch.Tensor:
    """
    使用树状归约和广播实现 all_reduce_sum。

    Args:
        x (torch.Tensor): 本地数据张量。
        device_group: 可选的分布式组。

    Returns:
        torch.Tensor: 归约并广播到所有进程后的张量。
    """
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