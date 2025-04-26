"""
torchrun --nproc_per_node=4 test_zigzag_ring_flash_attn_func.py
"""

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")
import torch.distributed as dist

from ring_attn_ascend import zigzag_ring_flash_attn_func
from utils import log, set_seed


def extract_local(value, rank, world_size, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.contiguous()


if __name__ == "__main__":
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"npu:{rank}")

    batch_size = 1
    seqlen = 3824
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True

    assert causal
    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_q = extract_local(q, rank, world_size).detach().clone()
    local_k = extract_local(k, rank, world_size).detach().clone()
    local_v = extract_local(v, rank, world_size).detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True
    local_dout = extract_local(dout, rank, world_size).detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)
    
    attn_mask = torch.triu(torch.ones([2048, 2048], device=q.device), diagonal=1).bool()
    out, softmax_max, softmax_sum, _, _, _, _ = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=q.shape[2],
        input_layout="BSND",
        atten_mask=attn_mask,
        scale=d ** (-0.5),
        pre_tockens=k.shape[1],
        next_tockens=0,
        sparse_mode=3,
        keep_prob=1.0-dropout_p,
    )

    local_out = extract_local(out, rank, world_size)
    # softmax_max shape is [batch_size, nheads, seqlen, 8]
    local_softmax_max = extract_local(softmax_max, rank, world_size, dim=2)
    local_softmax_sum = extract_local(softmax_sum, rank, world_size, dim=2)
    
    ring_out, ring_softmax_max, ring_softmax_sum = zigzag_ring_flash_attn_func(
        local_q,
        local_k,
        local_v,
        causal=causal,
    )
    log("out", out, rank0_only=True)
    log("softmax max", softmax_max, rank0_only=True)
    log("softmax sum", softmax_sum, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("softmax max diff", local_softmax_max - ring_softmax_max)
    log("softmax sum diff", local_softmax_sum - ring_softmax_sum)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)
    
    out.backward(dout)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    local_dq = extract_local(dq, rank, world_size)
    local_dk = extract_local(dk, rank, world_size)
    local_dv = extract_local(dv, rank, world_size)

    ring_out.backward(local_dout)
    ring_dq = local_q.grad
    ring_dk = local_k.grad
    ring_dv = local_v.grad

    log("dq diff", local_dq - ring_dq)
    log("dk diff", local_dk - ring_dk)
    log("dv diff", local_dv - ring_dv)

    dist.destroy_process_group()
