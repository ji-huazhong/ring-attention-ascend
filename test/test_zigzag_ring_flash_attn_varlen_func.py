"""
torchrun --nproc_per_node=4 test_zigzag_ring_flash_attn_varlen_func.py
"""
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")
import torch.distributed as dist

from ring_attn_ascend import zigzag_ring_flash_attn_varlen_func, flatten_softmax, get_sub_seq_lens
from utils import log, set_seed, extract_softmax_value


def extract_local(value, cu_seqlens, rank, world_size):
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        local_value = value[start:end].chunk(2 * world_size, dim=0)
        local_values.extend(
            [
                local_value[rank].detach().clone(),
                local_value[2 * world_size - 1 - rank].detach().clone(),
            ]
        )
    return torch.cat(local_values, dim=0).contiguous()


if __name__ == "__main__":
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"npu:{rank}")

    nheads = 5
    d = 128
    dropout_p = 0
    causal = True

    cu_seqlens = [0, 120, 1248, 4232]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    sub_seq_lens = get_sub_seq_lens(cu_seqlens)
    total_length = cu_seqlens[-1]

    assert torch.all(cu_seqlens_tensor % world_size == 0)
    assert d % 8 == 0

    q = torch.randn(total_length, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_length, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_length, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    dout = torch.randn(total_length, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_cu_seqlens_tensor = cu_seqlens_tensor // world_size
    local_sub_seq_lens = get_sub_seq_lens(local_cu_seqlens_tensor)

    local_q = extract_local(q, cu_seqlens, rank, world_size)
    local_k = extract_local(k, cu_seqlens, rank, world_size)
    local_v = extract_local(v, cu_seqlens, rank, world_size)
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True
    local_dout = extract_local(dout, cu_seqlens, rank, world_size)

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
        head_num=q.shape[1],
        input_layout="TND",
        atten_mask=attn_mask,
        scale=d ** (-0.5),
        actual_seq_qlen=tuple(cu_seqlens_tensor[1:].cpu().numpy().tolist()),
        actual_seq_kvlen=tuple(cu_seqlens_tensor[1:].cpu().numpy().tolist()),
        sparse_mode=3,
        keep_prob=1.0-dropout_p,
    )

    local_out = extract_local(out, cu_seqlens, rank, world_size)

    softmax_max = flatten_softmax(softmax_max, sub_seq_lens)
    local_softmax_max_list = extract_softmax_value(softmax_max, cu_seqlens)
    softmax_sum = flatten_softmax(softmax_sum, sub_seq_lens)
    local_softmax_sum_list = extract_softmax_value(softmax_sum, cu_seqlens)

    ring_out, ring_softmax_max, ring_softmax_sum = zigzag_ring_flash_attn_varlen_func(
        local_q,
        local_k,
        local_v,
        local_cu_seqlens_tensor,
        dropout_p=dropout_p,
        causal=causal,
    )

    ring_softmax_max = flatten_softmax(ring_softmax_max, local_sub_seq_lens)
    ring_softmax_max_list = extract_softmax_value(ring_softmax_max, local_cu_seqlens_tensor)
    ring_softmax_sum = flatten_softmax(ring_softmax_sum, local_sub_seq_lens)
    ring_softmax_sum_list = extract_softmax_value(ring_softmax_sum, local_cu_seqlens_tensor)

    log("out diff", local_out - ring_out)
    for i, (lsm, ring_lsm) in enumerate(zip(local_softmax_max_list, ring_softmax_max_list)):
        local_lsm = lsm.chunk(2 * world_size, dim=0)
        local_lsm = torch.cat(
            [local_lsm[rank], local_lsm[2 * world_size - 1 - rank]], dim=0
        )
        log(f"softmax max diff {i}", local_lsm - ring_lsm)
    for i, (lss, ring_lss) in enumerate(zip(local_softmax_sum_list, ring_softmax_sum_list)):
        local_lss = lss.chunk(2 * world_size, dim=0)
        local_lss = torch.cat(
            [local_lss[rank], local_lss[2 * world_size - 1 - rank]], dim=0
        )
        log(f"softmax sum diff {i}", local_lss - ring_lss)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    local_dq = extract_local(dq, cu_seqlens, rank, world_size)
    local_dk = extract_local(dk, cu_seqlens, rank, world_size)
    local_dv = extract_local(dv, cu_seqlens, rank, world_size)

    ring_out.backward(local_dout)
    ring_dq = local_q.grad
    ring_dk = local_k.grad
    ring_dv = local_v.grad

    log("dq diff", local_dq - ring_dq)
    log("dk diff", local_dk - ring_dk)
    log("dv diff", local_dv - ring_dv)

    dist.destroy_process_group()
