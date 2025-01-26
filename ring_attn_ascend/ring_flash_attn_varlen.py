from typing import Optional, Tuple

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")

from .utils import RingComm, flatten_softmax, get_sub_seq_lens


def _update_forward(
    prev_out: Optional[torch.Tensor],         # (total_length, nheads, hidden_dim)
    prev_softmax_max: Optional[torch.Tensor], # (total_length, nheads, 8)
    prev_softmax_sum: Optional[torch.Tensor], # (total_length, nheads, 8)
    cur_out: torch.Tensor, 
    cur_softmax_max: torch.Tensor, 
    cur_softmax_sum: torch.Tensor,
    sub_seq_lens,
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
    prev_out_scale = flatten_softmax(prev_out_scale, sub_seq_lens)
    cur_out_scale = flatten_softmax(cur_out_scale, sub_seq_lens)
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
    sub_seq_lens,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = block_out.to(torch.float32)
        softmax_max = block_softmax_max
        softmax_sum = block_softmax_sum
    else:
        out, softmax_max, softmax_sum = _update_forward(
            out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum, sub_seq_lens
        )

    return out, softmax_max, softmax_sum


def ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor, # (total_length, nheads, hidden_dim)
    k: torch.Tensor, # (total_length, nheads, hidden_dim)
    v: torch.Tensor, # (total_length, nheads, hidden_dim)
    cu_seqlens,
    softmax_scale,
    attn_mask,
    dropout_p=0,
    causal=True,
):
    assert causal == True, "causal==False is not supported."

    comm = RingComm(process_group)
    
    out = None
    softmax_max = None
    softmax_sum = None
    next_k, next_v = None, None

    sub_seq_lens = get_sub_seq_lens(cu_seqlens)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if step <= comm.rank:
            outputs = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=q.shape[1],
                input_layout="TND",
                atten_mask=attn_mask if step == 0 else None,
                scale=softmax_scale,
                actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                sparse_mode=3,
                keep_prob=1.0-dropout_p,
            )
            block_out, block_softmax_max, block_softmax_sum, _, _, _, _ = outputs

            out, softmax_max, softmax_sum = update_forward(
                out, softmax_max, softmax_sum, block_out, block_softmax_max, block_softmax_sum, sub_seq_lens
            )

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
    softmax_scale,
    attn_mask,
    dropout_p=0,
    causal=True,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None

    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
        
        if step <= kv_comm.rank:
            attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                q,
                k,
                v,
                dout,
                head_num=q.shape[1],
                input_layout="TND",
                atten_mask=attn_mask if step == 0 else None,
                softmax_max=softmax_max,
                softmax_sum=softmax_sum,
                attention_in=out,
                scale_value=softmax_scale,
                actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                sparse_mode=3,
                keep_prob=1.0-dropout_p,
            )

            if dq is None:
                dq = attn_grad_outs[0].to(torch.float32)
                dk = attn_grad_outs[1].to(torch.float32)
                dv = attn_grad_outs[2].to(torch.float32)
            else:
                dq += attn_grad_outs[0]
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


class RingFlashAttnVarlenFunc(torch.autograd.Function):
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
        out, softmax_max, softmax_sum = ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            softmax_scale=softmax_scale,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            causal=causal,
        )

        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum, cu_seqlens)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.attn_mask = attn_mask
        ctx.causal = causal
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
            softmax_scale=ctx.softmax_scale,
            attn_mask=ctx.attn_mask,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None


def ring_flash_attn_varlen_func(
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
    return RingFlashAttnVarlenFunc.apply(
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
