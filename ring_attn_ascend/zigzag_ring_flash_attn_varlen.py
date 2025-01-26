from typing import Optional, Tuple

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")

from .utils import RingComm, flatten_softmax, unflatten_softmax, get_sub_seq_lens


def _update_forward(
    prev_out: Optional[torch.Tensor],         # (total_length, nheads, hidden_dim)
    prev_softmax_max: Optional[torch.Tensor], # (total_length, nheads, 8)
    prev_softmax_sum: Optional[torch.Tensor], # (total_length, nheads, 8)
    cur_out: torch.Tensor, 
    cur_softmax_max: torch.Tensor, 
    cur_softmax_sum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # update softmax max
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    
    # update softmax sum
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # update out scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # (total_length, nheads, 8) -> (total_length, nheads, 1)
    prev_out_scale = prev_out_scale[..., 0].unsqueeze(2)
    cur_out_scale = cur_out_scale[..., 0].unsqueeze(2)

    # updata output
    out = prev_out * prev_out_scale + cur_out * cur_out_scale
    return out, softmax_max, softmax_sum


def update_forward(
    out: Optional[torch.Tensor], 
    softmax_max: Optional[torch.Tensor], 
    softmax_sum: Optional[torch.Tensor], 
    block_out: torch.Tensor, 
    block_softmax_max: torch.Tensor, 
    block_softmax_sum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = block_out.to(torch.float32)
        softmax_max = block_softmax_max
        softmax_sum = block_softmax_sum
    else:
        out, softmax_max, softmax_sum = _update_forward(
            out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum
        )

    return out, softmax_max, softmax_sum


def get_half_index(cu_seqlens, *, front: bool):
    if len(cu_seqlens) == 2:
        if front:
            return slice(None, cu_seqlens[-1] // 2)
        else:
            return slice(cu_seqlens[-1] // 2, None)

    index = torch.zeros((cu_seqlens[-1].item(),), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


def get_half_softmax(softmax_value, cu_seqlens, *, front: bool):
    new_softmax_value = torch.empty(
        (softmax_value.shape[0] // 2, softmax_value.shape[1], softmax_value.shape[2]),
        dtype=softmax_value.dtype,
        device=softmax_value.device,
    )

    sub_seq_lens = get_sub_seq_lens(cu_seqlens)
    softmax_value = flatten_softmax(softmax_value, sub_seq_lens)

    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        new_start, new_end = start // 2, end // 2
        if front:
            end -= (end - start) // 2
        else:
            start += (end - start) // 2
        new_softmax_value[new_start: new_end] = softmax_value[start: end]

    half_cu_seqlens = cu_seqlens // 2
    new_softmax_value = unflatten_softmax(new_softmax_value, half_cu_seqlens)

    return new_softmax_value


def zigzag_ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    half_index0,
    half_index1,
    softmax_scale,
    attn_mask,
    dropout_p=0,
    causal=True,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[0] // 2
    q1 = q[half_index1]

    out = None
    softmax_max = None
    softmax_sum = None
    next_k, next_v = None, None
    half_cu_seqlens = cu_seqlens // 2

    def forward(q, k, v, causal):
        seqlen_q = q.shape[0]
        seqlen_kv = k.shape[0]
        cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
        cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens

        sub_seq_lens = get_sub_seq_lens(cu_seqlens_q)

        outputs = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=q.shape[1],
            input_layout="TND",
            atten_mask=attn_mask if causal else None,
            scale=softmax_scale,
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_kv[1:].cpu().numpy().tolist()),
            sparse_mode=3,
            keep_prob=1.0-dropout_p,
        )

        block_out, block_softmax_max, block_softmax_sum, _, _, _, _ = outputs

        block_softmax_max = flatten_softmax(block_softmax_max, sub_seq_lens)
        block_softmax_sum = flatten_softmax(block_softmax_sum, sub_seq_lens)

        return block_out, block_softmax_max, block_softmax_sum

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if step == 0:
            block_out, block_softmax_max, block_softmax_sum = forward(q, k, v, causal=True)

            out, softmax_max, softmax_sum = update_forward(
                out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum
            )

        elif step <= comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            block_out, block_softmax_max, block_softmax_sum = forward(q, k0, v0, causal=False)

            out, softmax_max, softmax_sum = update_forward(
                out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum
            )

        else:
            block_out, block_softmax_max, block_softmax_sum = forward(q1, k, v, causal=False)

            out[half_index1], softmax_max[half_index1], softmax_sum[half_index1] = update_forward(
                out[half_index1], softmax_max[half_index1], softmax_sum[half_index1], block_out, block_softmax_max,
                block_softmax_sum
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    softmax_max = unflatten_softmax(softmax_max, cu_seqlens)
    softmax_sum = unflatten_softmax(softmax_sum, cu_seqlens)
    return out, softmax_max, softmax_sum


def zigzag_ring_flash_attn_varlen_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_max,
    softmax_sum,
    cu_seqlens,
    half_index0,
    half_index1,
    softmax_scale,
    attn_mask,
    dropout_p=0,
    causal=True,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = dout[half_index1]
    q1 = q[half_index1]
    out1 = out[half_index1]
    softmax_max1 = get_half_softmax(softmax_max, cu_seqlens, front=False)
    softmax_sum1 = get_half_softmax(softmax_sum, cu_seqlens, front=False)
    block_seq_len = q.shape[0] // 2

    half_cu_seqlens = cu_seqlens // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def backward(dout, q, k, v, out, softmax_max, softmax_sum, causal):
        seqlen_q = q.shape[0]
        seqlen_kv = k.shape[0]
        cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
        cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
        
        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            q,
            k,
            v,
            dout,
            head_num=q.shape[1],
            input_layout="TND",
            atten_mask=attn_mask if causal else None,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=out,
            scale_value=softmax_scale,
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_kv[1:].cpu().numpy().tolist()),
            sparse_mode=3,
            keep_prob=1.0-dropout_p,
        )

        dq_buffer[:seqlen_q] = attn_grad_outs[0]    # dq
        dk_buffer[:seqlen_kv] = attn_grad_outs[1]   # dk
        dv_buffer[:seqlen_kv] = attn_grad_outs[2]   # dv

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step == 0:
            backward(dout, q, k, v, out, softmax_max, softmax_sum, causal=True)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = k[half_index0]
                v0 = v[half_index0]
                backward(dout, q, k0, v0, out, softmax_max, softmax_sum, causal=False)
                dq += dq_buffer
            else:
                backward(dout1, q1, k, v, out1, softmax_max1, softmax_sum1, causal=False)
                dq[half_index1] += dq_buffer[:block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[half_index0] += dk_buffer[:block_seq_len]
                dv[half_index0] += dv_buffer[:block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        if dk_comm_buffer is None and dv_comm_buffer is None:
            dkv_comm_buffer = None
        else:
            dkv_comm_buffer = torch.stack((dk_comm_buffer, dv_comm_buffer), dim=0)

        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dkv_comm_buffer
        )

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class ZigZagRingFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        dropout_p,
        softmax_scale=None,
        attn_mask=None,
        causal=True,
        group=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if causal and attn_mask is None:
            # Ref: https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000156.html
            attn_mask = torch.triu(torch.ones([2048, 2048], device=q.device), diagonal=1).bool()

        k = k.contiguous()
        v = v.contiguous()
        half_index0 = get_half_index(cu_seqlens, front=True)
        half_index1 = get_half_index(cu_seqlens, front=False)
        out, softmax_max, softmax_sum = zigzag_ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            half_index0,
            half_index1,
            softmax_scale=softmax_scale,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            causal=causal,
        )
        # this should be out_padded
        is_half_index_tensor = isinstance(half_index0, torch.Tensor)
        ctx.is_half_index_tensor = is_half_index_tensor
        if is_half_index_tensor:
            ctx.save_for_backward(
                q, k, v, out, softmax_max, softmax_sum, cu_seqlens, half_index0, half_index1
            )
        else:
            ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum, cu_seqlens)
            ctx.half_index0 = half_index0
            ctx.half_index1 = half_index1
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.attn_mask = attn_mask
        ctx.causal = causal
        ctx.group = group
        return out, softmax_max, softmax_sum

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.is_half_index_tensor:
            (q, k, v, out, softmax_max, softmax_sum, cu_seqlens, half_index0, half_index1) = (
                ctx.saved_tensors
            )
        else:
            q, k, v, out, softmax_max, softmax_sum, cu_seqlens = ctx.saved_tensors
            half_index0 = ctx.half_index0
            half_index1 = ctx.half_index1
        dq, dk, dv = zigzag_ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_max,
            softmax_sum,
            cu_seqlens,
            half_index0,
            half_index1,
            softmax_scale=ctx.softmax_scale,
            attn_mask=ctx.attn_mask,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens,
    dropout_p=0.0,
    softmax_scale=None,
    attn_mask=None,
    causal=True,
    group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens,
        dropout_p,
        softmax_scale,
        attn_mask,
        causal,
        group,
    )
