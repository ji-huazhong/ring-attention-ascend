from typing import Optional, Tuple

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
    mqk: Optional[torch.Tensor], 
    se: Optional[torch.Tensor], 
    block_out: torch.Tensor, 
    block_mqk: torch.Tensor, 
    block_se: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = block_out.to(torch.float32)
        mqk = block_mqk
        se = block_se
    else:
        out, mqk, se = _update_forward(out, mqk, se, block_out, block_mqk, block_se)
    return out, mqk, se


def ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    attn_mask=None,

):
    comm = RingComm(process_group)
    
    out = None
    softmax_max = None
    softmax_sum = None
    next_k, next_v = None, None

    # old_softmax = False
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        if not causal or step <= comm.rank:
            outputs = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=q.shape[1],
                input_layout="TND",
                atten_mask=attn_mask if step == 0 else None,
                scale=softmax_scale,
                keep_prob=1-dropout_p,
                actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                sparse_mode=2 if step == 0 else 0,
            )
            block_out, block_softmax_max, block_softmax_sum, _, _, _, _ = outputs
            out, softmax_max, softmax_sum = update_forward(out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum)

    if step + 1 != comm.world_size:
        comm.wait()
        k, v = next_k, next_v
    
    out = out.to(q.dtype)
    return out, softmax_max, softmax_sum



def ring_flash_attn_varlen_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_max,
    softmax_sum,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    attn_mask=None,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    
    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
        
        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                q,
                k,
                v,
                dout,
                head_num=q.shape[1],
                input_layout="TND",
                atten_mask=attn_mask if bwd_causal else None,
                softmax_max=softmax_max,
                softmax_sum=softmax_sum,
                attention_in=out,
                scale_value=softmax_scale,
                keep_prob=1-dropout_p,
                actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                sparse_mode=2 if bwd_causal else 0,
            )

            if dq is None:
                dq == attn_grad_outs[0]
                d_kv_comm.wait()
                dk = attn_grad_outs[1] + next_dk
                dv = attn_grad_outs[2] + next_dv
            elif step != 0:
                d_kv_comm.wait()
                dk, dv = next_dk, next_dv
            
            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k, v = next_k, next_v
            
            next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlashAttnVarlenFunc(torch.autograd.Fucntion):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        attn_mask=None,
        group=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if causal and attn_mask is None:
            # Ref: https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000156.html
            attn_mask = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(q.device)

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_max, softmax_sum = ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            attn_mask=attn_mask
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.attn_mask = attn_mask
        ctx.group = group
        return out, softmax_max, softmax_sum

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_max, softmax_sum, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_max,
            softmax_sum,
            cu_seqlens,
            ctx.max_seqlen,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            attn_mask=ctx.attn_mask,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def ring_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    group=None,
):
    return RingFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        group,
    )
