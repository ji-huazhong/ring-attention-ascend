from typing import Optional, Tuple

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")
import torch.distributed as dist

from .utils import RingComm


# https://github.com/zhuzilin/ring-flash-attention/blob/be3b01f5706f45245f9b6d78d6df231954b2ee64/ring_flash_attn/ring_flash_attn.py
# https://github.com/zouzias/ascend-op-plugin/blob/95b06297ee18d84746a4f11cbe99ea89ea9448a8/op_plugin/ops/v2r1/opapi/FlashAttentionKernelNpuOpApi.cpp#L248
# https://github.com/cosdt/op-plugin/blob/c8e7f492244dc4b56c040dac9db22420b8592ff6/test/test_custom_ops/test_npu_flash_attention_grad.py#L70
# https://gitee.com/ascend/MindSpeed/pulls/138/files


def _update_forward(
    prev_out: Optional[torch.Tensor], 
    prev_softmax_max: Optional[torch.Tensor], 
    prev_softmax_sum: Optional[torch.Tensor], 
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
    dropout_p=0,
    attn_mask=None,
    causal=True,
):
    assert causal == True, "causal==false is not supported."

    if causal and attn_mask is None:
        attn_mask = torch.ones((q.shape[1], k.shape[1]), dtype=torch.bool, device=q.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)

    comm = RingComm(process_group)

    out = None
    mqk = None # The max value of each row of the matrix QK^T * scaling
    se = None # The sum exp of each row of the matrix QK^T * scaling 
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size: # 不是最后一个rank
            next_k, next_v = comm.send_recv_kv(k, v) # 把k,v发给下一个rank，并从上一个rank接收下一组k, v

        if step <= comm.rank:
            outputs = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=q.shape[2],
                input_layer="BSND",
                atten_mask=attn_mask if step == 0 else None,
                scale=softmax_scale,
                pre_tockens=k.shape[1],
                next_tockens=0,
                keep_prob=1,
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
    softmax_scale,
):
    ...


class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        dropout_p,
        softmax_scale,
        causal,
        group,
    ):
        ...

    @staticmethod
    def backward(ctx, dout, *args):
        ...



def ring_flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=True,
    group=None,
):
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        group,
    )
