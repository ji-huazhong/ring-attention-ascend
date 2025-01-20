import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")
import torch.distributed as dist

from ring_attn_ascend import ring_flash_attn_func
from utils import log, set_seed


if __name__ == "__main__":
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"npu:{rank}")

    batch_size = 1
    seqlen = 32
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True

    assert seqlen % world_size == 0
    assert d % 8 == 0

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
    local_v = v.chunk(world_size, dim=1)[rank].detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True

    dist.barrier()

    outs = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=True,
    )
