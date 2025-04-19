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
            # [b, s, n, d] -> [b, 2, s//2, n, d]
            out = out.view(out.shape[0], 2, out.shape[1]//2, out.shape[2], out.shape[-1])
            # [b, n, s, 8] -> [b, n, 2, s//2, 8]
            mqk = mqk.view(mqk.shape[0], mqk.shape[1], 2, mqk.shape[2]//2, mqk.shape[-1])
            se = se.view(se.shape[0], se.shape[1], 2, se.shape[2]//2, se.shape[-1])
            updated_out, updated_mqk, updated_se = update_forward(
                out[:, 1], 
                mqk[:, :, 1], 
                se[:, :, 1],
                block_out,
                block_mqk,
                block_se,
            )
            out[:, 1].copy_(updated_out)
            mqk[:, :, 1].copy_(updated_mqk)
            se[:, :, 1].copy_(updated_se)
            # [b, 2, s//2, n, d] -> [b, s, n, d]
            out = out.view(out.shape[0], -1, out.shape[-2], out.shape[-1])
            # [b, n, 2, s//2, 8] -> [b, n, s, 8]
            mqk = mqk.view(mqk.shape[0], mqk.shape[1], -1, mqk.shape[-1])
            se = se.view(se.shape[0], se.shape[1], -1, se.shape[-1])

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    
    out = out.to(q.dtype)
    return out, mqk, se


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
        ...

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
