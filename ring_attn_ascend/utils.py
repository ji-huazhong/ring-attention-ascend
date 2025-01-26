import fcntl
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def flatten_softmax(x, sub_seq_lens):
    orig_shape = x.shape
    section_len = [s * orig_shape[1] for s in sub_seq_lens]
    splits = x.view(-1, orig_shape[-1]).split(section_len, dim=0)
    merged = [item.view(orig_shape[1], -1, orig_shape[-1]).transpose(0, 1) for item in splits]
    merged = torch.cat(merged, dim=0)
    return merged


def unflatten_softmax(x, cu_seqlens):
    orig_shape = x.shape
    x_seq_list = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        x_seq_list.append(x[start:end].transpose(1, 0).reshape(-1, orig_shape[-2], orig_shape[-1]))
    return torch.cat(x_seq_list, dim=0)


def get_sub_seq_lens(cu_seqlens):
    sub_seq_lens = []
    for i in range(len(cu_seqlens) - 1):
        sub_seq_lens.append(cu_seqlens[i+1] - cu_seqlens[i])
    return sub_seq_lens


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 在npu上batch_isend_irecv不支持同一src device和dst device的多个isend操作
        kv = torch.stack((k, v), dim=0)
        next_kv = self.send_recv(kv, kv_buffer)
        self.commit()
        return next_kv[0], next_kv[1]


class AllGatherComm:
    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        handle = dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=self.group, async_op=True
        )
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []


def printflock(*msg):
    """solves multi-process interleaved print problem"""
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msg)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
