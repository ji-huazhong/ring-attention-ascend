"""
torchrun --nproc_per_node=8 test_llama3_flash_attn_varlen_func.py
"""
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")
import torch.distributed as dist

from ring_attn_ascend import (
    llama3_flash_attn_prepare_cu_seqlens,
    llama3_flash_attn_varlen_func,
)
from utils import log, set_seed


if __name__ == "__main__":
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"npu:{rank}")

    nheads = 5
    d = 8
    dropout_p = 0
    causal = True

    cu_seqlens = [0, 120, 1248, 4232]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    total_length = cu_seqlens[-1]
    local_length = total_length // world_size

    assert cu_seqlens_tensor[-1] % world_size == 0
    assert d % 8 == 0

    q = torch.randn(total_length, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_length, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_length, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    dout = torch.randn(total_length, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_q = q[rank * local_length : (rank + 1) * local_length].detach().clone()
    local_k = k[rank * local_length : (rank + 1) * local_length].detach().clone()
    local_v = v[rank * local_length : (rank + 1) * local_length].detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True
    local_dout = dout[rank * local_length : (rank + 1) * local_length].detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    attn_mask = torch.triu(torch.ones([2048, 2048], device=q.device), diagonal=1).bool()
    out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=q.shape[1],
        input_layout="TND",
        atten_mask=attn_mask,
        scale=d ** (-0.5),
        actual_seq_qlen=tuple(cu_seqlens_tensor[1:].cpu().numpy().tolist()),
        actual_seq_kvlen=tuple(cu_seqlens_tensor[1:].cpu().numpy().tolist()),
        sparse_mode=3,
        keep_prob=1.0-dropout_p,
    )

    local_out = out[rank * local_length : (rank + 1) * local_length]

    (
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(
        cu_seqlens_tensor,
        causal=causal,
        rank=rank,
        world_size=world_size,
    )

    llama3_out, _, _ = llama3_flash_attn_varlen_func(
        local_q,
        local_k,
        local_v,
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        heads_k_stride=1,
        local_k_slice=local_k_slice,
        dropout_p=dropout_p,
        causal=causal,
    )

    log("out", out, rank0_only=True)
    log("out diff", local_out - llama3_out)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    local_dq = dq[rank * local_length : (rank + 1) * local_length]
    local_dk = dk[rank * local_length : (rank + 1) * local_length]
    local_dv = dv[rank * local_length : (rank + 1) * local_length]

    llama3_out.backward(local_dout)
    llama3_dq = local_q.grad
    llama3_dk = local_k.grad
    llama3_dv = local_v.grad

    log("local_dq", local_dq[:, 0])
    log("local_dk", local_dk[:, 1])
    log("local_dv", local_dv[:, 2])
    log("dq diff", local_dq[:, 0] - llama3_dq[:, 0])
    log("dk diff", local_dk[:, 1] - llama3_dk[:, 1])
    log("dv diff", local_dv[:, 2] - llama3_dv[:, 2])

    dist.destroy_process_group()
