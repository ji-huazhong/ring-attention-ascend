from typing import Optional, Tuple

from einops import rearrange
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")
import torch.distributed as dist

from .utils import RingComm


def _update_forward(
    prev_out: Optional[torch.Tensor],         # (batch_size, seqlen, nheads, d)
    prev_softmax_max: Optional[torch.Tensor], # (batch_size, seqlen, nheads, 1)
    prev_softmax_sum: Optional[torch.Tensor], # (batch_size, seqlen, nheads, 1)
    cur_out: torch.Tensor, 
    cur_softmax_max: torch.Tensor,            # (batch_size, nheads, seqlen, 8)
    cur_softmax_sum: torch.Tensor,            # (batch_size, nheads, seqlen, 8)
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    cur_softmax_max= cur_softmax_max[..., 0].transpose(-2, -1).unsqueeze(dim=-1) # b n s 8 -> b s n 1
    cur_softmax_sum= cur_softmax_sum[..., 0].transpose(-2, -1).unsqueeze(dim=-1) # b n s 8 -> b s n 1
    
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

    # #[b, n, s, 8] -> [b, s, n, d]
    # [b, s, n, 1] -> [b, s, n, d]
    d = cur_out.shape[-1]
    # prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d) # [b, n, s, 1] -> [b, n, s, d]
    # prev_out_scale = rearrange(prev_out_scale, "b n s d -> b s n d").contiguous()
    # cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    # cur_out_scale = rearrange(cur_out_scale, "b n s d -> b s n d").contiguous()
    prev_out_scale = prev_out_scale.repeat(1, 1, 1, d).contiguous()
    cur_out_scale = cur_out_scale.repeat(1, 1, 1, d).contiguous()

    # updata output
    out = prev_out * prev_out_scale + cur_out * cur_out_scale
    return out, softmax_max, softmax_sum


def update_forward(
    out: Optional[torch.Tensor], 
    mqk: Optional[torch.Tensor], 
    se: Optional[torch.Tensor], 
    block_out: torch.Tensor, 
    block_mqk: torch.Tensor, 
    block_se: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_forward should not pass slice_ args")
        out = block_out.to(torch.float32)
        mqk = block_mqk[..., 0].transpose(-2, -1).unsqueeze(dim=-1) # b n s 8 -> b s n 1
        se = block_se[..., 0].transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        # 请注意mqk se的shape跟out是不一致的！
        slice_out, slice_mqk, slice_se = out[slice_], mqk[slice_], se[slice_]
        slice_out, slice_mqk, slice_se = _update_forward(
            slice_out, slice_mqk, slice_se, block_out, block_mqk, block_se
        )
    else:
        out, mqk, se = _update_forward(out, mqk, se, block_out, block_mqk, block_se)
    return out, mqk, se


def zigzag_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    attn_mask=None,
    causal: bool = True,
):
    assert causal is True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    mqk = None
    se = None
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
            keep_prob=1,
        )
        block_out, block_mqk, block_se, _, _, _, _ = outs
        return block_out, block_mqk, block_se
    
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        
        if step == 0:
            block_out, block_mqk, block_se = forward(q, k, v, causal=True)
            out, mqk, se = update_forward(out, mqk, se, block_out, block_mqk, block_se)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_mqk, block_se = forward(q, k0, v0, causal=False)
            out, mqk, se = update_forward(out, mqk, se, block_out, block_mqk, block_se)
        else:
            block_out, block_mqk, block_se = forward(q1, k, v, causal=False)
            if comm.rank == 0:
                print(f">>>>{block_out.shape} {block_mqk.shape} {out.shape} {mqk.shape}")
            out, mqk, se = update_forward(
                out, 
                mqk, 
                se,
                block_out,
                block_mqk,
                block_se,
                slice_=(slice(None), slice(block_seq_len, None)), # [:, block_seq_len: ]
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    
    out = out.to(q.dtype)
    mqk = mqk.squeeze(dim=-1).transpose(1, 2).unsqueeze(dim=-1).repeat(1, 1, 1, 8)
    se = se.squeeze(dim=-1).transpose(1, 2).unsqueeze(dim=-1).repeat(1, 1, 1, 8)
    return out, mqk, se


def zigzag_ring_flash_attn_backward(
    process_group,
    dout,
    q, # [bsz, seqlen, nheads, headdim]
    k,
    v,
    out,
    softmax_scale,
    attn_mask,
    softmax_max,
    softmax_sum,
    causal=True,
):
    assert causal is True, "zigzag ring is meaningless for causal=False"
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
        )
        return attn_grad_outs
    
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
        
        if step == 0:
            grad_outs = backward(dout, q, k, v, out, softmax_max, softmax_sum, causal=True)
            dq = grad_outs[0].to(torch.float32)
            dk = grad_outs[1].to(torch.float32)
            dv = grad_outs[2].to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                grad_outs = backward(dout, q, k0, v0, out, softmax_max, softmax_sum, causal=False)
                dq += grad_outs[0]
            else:
                grad_outs = backward(dout1, q1, k, v, out1, softmax_max1, softmax_sum1, causal=False)
                dq[:, block_seq_len:] += grad_outs[0][:, :block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += grad_outs[1][:, :block_seq_len]
                dv[:, :block_seq_len] += grad_outs[2][:, :block_seq_len]
            else:
                dk += grad_outs[1]
                dv += grad_outs[2]

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        ) # TODO
    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale=None,
        attn_mask=None,
        causal=True,
        group=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if causal and attn_mask is None:
            attn_mask = torch.ones((q.shape[1], k.shape[1]), dtype=torch.bool, device=q.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)  

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_max, softmax_sum = zigzag_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            attn_mask=attn_mask,
            causal=causal,
        )
        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum)
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
            softmax_scale=ctx.softmax_scale,
            attn_mask=ctx.attn_mask,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
        )
        return dq, dk, dv, None, None, None, None


def zigzag_ring_flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    attn_mask=None,
    causal=True,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        attn_mask,
        causal,
        group,
    )
