from typing import Optional, Tuple

from einops import rearrange
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")

from .utils import RingComm


def _update_forward(
    prev_out: Optional[torch.Tensor],         # (batch_size, seqlen, nheads, d)
    prev_softmax_max: Optional[torch.Tensor], # (batch_size, nheads, seqlen, 8)
    prev_softmax_sum: Optional[torch.Tensor], # (batch_size, nheads, seqlen, 8)
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

    # [b, n, s, 8] -> [b, s, n, d]
    d = cur_out.shape[-1]
    prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d) # [b, n, s, 1] -> [b, n, s, d]
    prev_out_scale = rearrange(prev_out_scale, "b n s d -> b s n d").contiguous()
    cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    cur_out_scale = rearrange(cur_out_scale, "b n s d -> b s n d").contiguous()

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
        out, softmax_max, softmax_sum = _update_forward(out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum)
    return out, softmax_max, softmax_sum


def zigzag_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    attn_mask,
    dropout_p=0,
    causal=True,
):
    assert causal is True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    softmax_max = None
    softmax_sum = None
    next_k, next_v = None, None

    def forward(q, k, v, causal):
        outs = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=q.shape[2],
            input_layout="BSND",
            atten_mask=attn_mask if causal else None,
            scale=softmax_scale,
            pre_tockens=k.shape[1],
            next_tockens=0,
            sparse_mode=3,
            keep_prob=1.0-dropout_p,
        )
        block_out, block_softmax_max, block_softmax_sum, _, _, _, _ = outs
        return block_out, block_softmax_max, block_softmax_sum
    
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        
        if step == 0:
            block_out, block_softmax_max, block_softmax_sum = forward(q, k, v, causal=True)
            out, softmax_max, softmax_sum = update_forward(out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_softmax_max, block_softmax_sum = forward(q, k0, v0, causal=False)
            out, softmax_max, softmax_sum = update_forward(out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum)
        else:
            block_out, block_softmax_max, block_softmax_sum = forward(q1, k, v, causal=False)
            # [b, s, n, d] -> [b, 2, s//2, n, d]
            out = out.view(out.shape[0], 2, out.shape[1]//2, out.shape[2], out.shape[-1])
            # [b, n, s, 8] -> [b, n, 2, s//2, 8]
            softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], 2, softmax_max.shape[2]//2, softmax_max.shape[-1])
            softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], 2, softmax_sum.shape[2]//2, softmax_sum.shape[-1])
            updated_out, updated_softmax_max, updated_softmax_sum = update_forward(
                out[:, 1], 
                softmax_max[:, :, 1], 
                softmax_sum[:, :, 1],
                block_out,
                block_softmax_max,
                block_softmax_sum,
            )
            out[:, 1].copy_(updated_out)
            softmax_max[:, :, 1].copy_(updated_softmax_max)
            softmax_sum[:, :, 1].copy_(updated_softmax_sum)
            # [b, 2, s//2, n, d] -> [b, s, n, d]
            out = out.view(out.shape[0], -1, out.shape[-2], out.shape[-1])
            # [b, n, 2, s//2, 8] -> [b, n, s, 8]
            softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1, softmax_max.shape[-1])
            softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1, softmax_sum.shape[-1])

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    
    out = out.to(q.dtype)
    return out, softmax_max, softmax_sum


def zigzag_ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_max,
    softmax_sum,
    softmax_scale,
    attn_mask,
    dropout_p=0,
    causal=True,
):
    assert causal == True, "zigzag is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_max1 = softmax_max.chunk(2, dim=2)[1].contiguous()
    softmax_sum1 = softmax_sum.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def backward(dout, q, k, v, out, softmax_max, softmax_sum, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            q,
            k,
            v,
            dout,
            head_num=q.shape[2],
            input_layout="BSND",
            atten_mask=attn_mask if causal else None,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=out,
            scale_value=softmax_scale,
            pre_tockens=k.shape[1],
            next_tockens=0,
            sparse_mode=3,
            keep_prob=1.0-dropout_p,
        )

        dq_buffer[:, :seqlen_q] = attn_grad_outs[0]    # dq
        dk_buffer[:, :seqlen_kv] = attn_grad_outs[1]   # dk
        dv_buffer[:, :seqlen_kv] = attn_grad_outs[2]   # dv
    
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
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                backward(dout, q, k0, v0, out, softmax_max, softmax_sum, causal=False)
                dq += dq_buffer
            else:
                backward(dout1, q1, k, v, out1, softmax_max1, softmax_sum1, causal=False)
                # always use the first half in dq_buffer.
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
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


class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
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
        out, softmax_max, softmax_sum = zigzag_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            causal=causal,
        )
        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.attn_mask = attn_mask
        ctx.causal = causal
        ctx.group = group
        return out, softmax_max, softmax_sum

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_max, softmax_sum = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_max,
            softmax_sum,
            softmax_scale=ctx.softmax_scale,
            attn_mask=ctx.attn_mask,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    attn_mask=None,
    causal=True,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        attn_mask,
        causal,
        group,
    )
