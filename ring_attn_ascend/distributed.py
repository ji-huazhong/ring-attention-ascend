import torch
from torch import Tensor
from torch.distributed import ProcessGroup


# Copy from: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/distributed.py
# Raw operation, does not support autograd, but does support async
def all_gather_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    output = torch.empty(
        world_size * input_.shape[0], *input_.shape[1:], dtype=input_.dtype, device=input_.device
    )
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Copy from: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/distributed.py
# Raw operation, does not support autograd, but does support async
def reduce_scatter_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(
        input_.shape[0] // world_size, *input_.shape[1:], dtype=input_.dtype, device=input_.device
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Copy from: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/distributed.py
class AllGatherFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return grad_input, None


# Copy from: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/distributed.py
# Supports autograd, but does not support async
all_gather = AllGatherFunc.apply
