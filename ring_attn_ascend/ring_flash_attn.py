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


def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor, # (batch_size, seqlen, nheads, headdim)
    k: torch.Tensor, # (batch_size, seqlen, nheads_k, headdim)
    v: torch.Tensor, # (batch_size, seqlen, nheads_k, headdim)
    softmax_scale,
    attn_mask,
    dropout_p=0,
    causal=True,
):
    assert causal == True, "causal==false is not supported."

    comm = RingComm(process_group)

    out = None
    mqk = None # The max value of each row of the matrix QK^T * scaling
    se = None # The sum exp of each row of the matrix QK^T * scaling 
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v) # 把k,v发给下一个rank，并从上一个rank接收下一组k, v

        if step <= comm.rank:
            outputs = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=q.shape[2],
                input_layout="BSND",
                atten_mask=attn_mask if step == 0 else None,
                scale=softmax_scale,
                pre_tockens=k.shape[1],
                next_tockens=0,
                sparse_mode=3,
                keep_prob=1.0-dropout_p,
            )
            block_out, block_mqk, block_se, _, _, _, _ = outputs
            out, mqk, se = update_forward(out, mqk, se, block_out, block_mqk, block_se)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    
    out = out.to(q.dtype)
    return out, mqk, se


# npu_fusion_attention_grad(Tensor query, Tensor key, Tensor value, Tensor dy, int head_num, str input_layout, *, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? softmax_max=None, Tensor? softmax_sum=None, Tensor? softmax_in=None, Tensor? attention_in=None, float scale_value=1., float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647, int inner_precise=0, int seed=0, int offset=0, int numels=0, int[]? prefix=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0, bool gen_mask_parallel=True, bool sync=False) -> (Tensor, Tensor, Tensor, Tensor)
def ring_flash_attn_backward(
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
                head_num=q.shape[2],
                input_layout="BSND",
                atten_mask=attn_mask if step == 0 else None,
                softmax_max=softmax_max,
                softmax_sum=softmax_sum,
                attention_in=out,
                scale_value=softmax_scale,
                pre_tockens=k.shape[1],
                next_tockens=0,
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


class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, # (batch_size, seqlen, nheads, headdim)
        k, # (batch_size, seqlen, nheads_k, headdim)
        v, # (batch_size, seqlen, nheads_k, headdim)
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
        out, softmax_max, softmax_sum = ring_flash_attn_forward(
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
        dq, dk, dv = ring_flash_attn_backward(
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
        return dq, dk, dv, None, None, None, None, None


def ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    attn_mask=None,
    causal=True,
    group=None,
):
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        attn_mask,
        causal,
        group,
    )
