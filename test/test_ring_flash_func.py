"""
torchrun --nproc_per_node=8 test_ring_flash_func.py
"""

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")
import torch.distributed as dist

from ring_attn_ascend import ring_flash_attn_func
from utils import log, set_seed


def self_attention_cpu(q, k, v):
    import numpy as np
    # b, n, s, d
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    b, n, s, d = q.size()
    attn_mask = np.triu(np.ones((b, n, s, s)), k=1)
    attn_mask = torch.tensor(attn_mask).to(torch.float16)

    scale = d ** (-0.5)
    qk = torch.matmul(q, k.transpose(2, 3)).mul(scale)
    qk = qk + attn_mask * (-10000.0)
    attn_score = torch.nn.functional.softmax(qk, dim=-1)
    output = torch.matmul(attn_score, v)
    return output.transpose(1, 2).contiguous()


if __name__ == "__main__":
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"npu:{rank}")

    batch_size = 1
    seqlen = 3816
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True

    assert seqlen % world_size == 0
    assert d % 8 == 0

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
    local_v = v.chunk(world_size, dim=1)[rank].detach().clone()
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    attn_mask = torch.ones((q.shape[1], k.shape[1]), dtype=torch.bool, device=q.device)
    attn_mask = torch.triu(attn_mask, diagonal=1)
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
        keep_prob=1,
    )
    out_cpu = self_attention_cpu(q.cpu().float(), k.cpu().float(), v.cpu().float())
    torch.testing.assert_close(out.cpu().float(), out_cpu, rtol=1e-2, atol=1e-2)

    # out is (batch_size, seqlen, nheads, d)
    local_out = out.chunk(world_size, dim=1)[rank]
    # softmax_max shape is (batch_size, nheads, seqlen, 8)
    local_softmax_max = softmax_max.chunk(world_size, dim=2)[rank]
    local_softmax_sum = softmax_sum.chunk(world_size, dim=2)[rank]
    
    fn = ring_flash_attn_func
    ring_out, ring_softmax_max, ring_softmax_sum = fn(
        local_q,
        local_k,
        local_v,
        causal=causal,
    )

    log("out", out, rank0_only=True)
    log("sm", softmax_max, rank0_only=True)
    log("ss", softmax_sum, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("sm diff", local_softmax_max - ring_softmax_max)
    log("ss diff", local_softmax_sum - ring_softmax_sum)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    local_dq = dq.chunk(world_size, dim=1)[rank]
    local_dk = dk.chunk(world_size, dim=1)[rank]
    local_dv = dv.chunk(world_size, dim=1)[rank]

    ring_out.backward(local_dout)
    ring_dq = local_q.grad
    ring_dk = local_k.grad
    ring_dv = local_v.grad

    log("dq diff", local_dq - ring_dq)
    log("dk diff", local_dk - ring_dk)
    log("dv diff", local_dv - ring_dv)
