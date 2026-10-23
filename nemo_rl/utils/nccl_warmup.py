# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create the trainer's NCCL communicators at setup instead of at first use.

``init_process_group("nccl")`` is lazy: each process group gets its NCCL
communicator, and each pipeline peer its point-to-point connection, the first
time it is used, and NCCL then allocates its buffers with cudaMalloc. In the
first training step that can be after the PyTorch allocator has filled the GPU,
and the collective fails with ``Cuda failure 'out of memory'``. With
``NRL_NCCL_WARMUP=1`` one tiny all-reduce per process group a training step uses and one
send/receive between pipeline neighbours run right after setup, while memory is
free; the allocator later works around the buffers they hold.
"""

import os
import time

import torch
import torch.distributed as dist

# The groups a training step uses: gradient reduce-scatter and parameter all-gather
# (data parallel with CP, expert data parallel) and finalize_model_grads (tensor+data
# parallel with CP, pipeline). Each communicator takes about 0.5 GiB of GPU memory, so
# groups training never touches (e.g. the without-CP twins) are left out.
_GROUPS = [
    ("get_data_parallel_group", {"with_context_parallel": True}),
    ("get_tensor_and_data_parallel_group", {"with_context_parallel": True}),
    ("get_expert_data_parallel_group", {}),
    ("get_pipeline_model_parallel_group", {}),
]


def warm_up(rank: int) -> None:
    if os.environ.get("NRL_NCCL_WARMUP", "0") != "1":
        return
    from megatron.core import parallel_state as ps

    start = time.time()
    free_before = torch.cuda.mem_get_info()[0]
    done = []
    for name, kwargs in _GROUPS:
        getter = getattr(ps, name, None)
        if getter is None:
            continue
        try:
            group = getter(**kwargs)
        except Exception:
            continue
        if group is None or isinstance(group, list):
            continue
        if dist.get_rank(group) < 0 or dist.get_world_size(group) < 2:
            continue
        dist.all_reduce(torch.ones(1, device="cuda"), group=group)
        done.append(name + ("+cp" if kwargs else ""))

    pp = ps.get_pipeline_model_parallel_group()
    if not isinstance(pp, list) and dist.get_world_size(pp) > 1:
        send = torch.ones(1, device="cuda")
        recv = torch.empty(1, device="cuda")
        ops = [
            dist.P2POp(dist.isend, send, ps.get_pipeline_model_parallel_next_rank(), group=pp),
            dist.P2POp(dist.irecv, recv, ps.get_pipeline_model_parallel_prev_rank(), group=pp),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        done.append("pipeline_p2p")
    torch.cuda.synchronize()
    used = (free_before - torch.cuda.mem_get_info()[0]) / float(1 << 30)
    print(
        f"[nccl-warmup] rank={rank} {len(done)} communicators in {time.time() - start:.1f}s, "
        f"{used:.2f}G of GPU memory: {' '.join(done)}",
        flush=True,
    )
