import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")


def ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    attn_mask=None,
    causal=True,
):
    ...


def ring_flash_attn_varlen_func():
    ...
