import random

import torch
import torch.distributed as dist


def extract_softmax_value(softmax_value, cu_seqlens):
    values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        value = softmax_value[start: end]
        values.append(value)
    return values


def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item():.3g}, "
                f"mean {a.abs().mean().item():.3g}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item():.3g}, "
                f"mean {a.abs().mean().item():.3g}",
                flush=True,
            )
        dist.barrier()
