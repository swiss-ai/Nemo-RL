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

"""Find and stop host memory that the NVIDIA driver moves into GPU memory.

On ATS systems (GH200) a GPU reads and writes pageable host memory directly, and
the UVM driver migrates pages a GPU accesses often into that GPU's HBM NUMA node
(access-counter migration, on by default there). The driver only honours a
memory policy set on the VMA itself (mbind), not the process-wide policy that
``srun --mem-bind`` sets, so those pages leave the CPU nodes despite a CPU-only
binding and take HBM that the CUDA allocator then cannot use.

``NRL_HBM_SPILL_CHECK=<n>`` logs, after each of the first n optimizer steps,
where the CPU-offloaded optimizer's host tensors live and which other anonymous
mappings of the process sit on GPU nodes. ``NRL_HBM_BIND_CPU=1`` binds every
large anonymous mapping to the CPU nodes after setup and after every optimizer
step, moving back pages already on HBM; the driver then keeps them there.
"""

import bisect
import ctypes
import os
import platform
from collections import defaultdict
from typing import Any

import torch

_SYSCALLS = {
    "aarch64": {"mbind": 235, "move_pages": 239},
    "x86_64": {"mbind": 237, "move_pages": 279},
}
_MPOL_BIND = 2
_MPOL_MF_MOVE = 1 << 1
_SAMPLE_BYTES = 8 << 20
_BIND_MIN_BYTES = 16 << 20
_GB = float(1 << 30)


def _cpu_and_gpu_nodes() -> tuple[list[int], list[int]]:
    cpu, gpu = [], []
    root = "/sys/devices/system/node"
    for name in os.listdir(root):
        if name.startswith("node") and name[4:].isdigit():
            with open(f"{root}/{name}/cpulist") as f:
                (cpu if f.read().strip() else gpu).append(int(name[4:]))
    return sorted(cpu), sorted(gpu)


class _HbmSpill:
    def __init__(self, rank: int, check_steps: int, bind: bool) -> None:
        self.rank = rank
        self.check_steps = check_steps
        self.bind = bind
        self.step = 0
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.syscall.restype = ctypes.c_long
        self.nr = _SYSCALLS[platform.machine()]
        self.cpu_nodes, self.gpu_nodes = _cpu_and_gpu_nodes()
        mask = sum(1 << n for n in self.cpu_nodes)
        self.mask = (ctypes.c_ulong * 1)(mask)

    def _host_tensors(self, optimizer: Any) -> dict[str, list[torch.Tensor]]:
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

        groups: dict[str, list[torch.Tensor]] = defaultdict(list)
        seen: set[int] = set()

        def add(kind: str, t: Any) -> None:
            if torch.is_tensor(t) and not t.is_cuda and t.numel() > 1:
                if t.data_ptr() not in seen:
                    seen.add(t.data_ptr())
                    groups[kind].append(t)

        for opt in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
            hdo = getattr(opt, "optimizer", None)
            if not isinstance(hdo, HybridDeviceOptimizer):
                continue
            for t in hdo.gpu_params_map_cpu_copy.values():
                add("param_copies", t)
            for t in hdo.param_to_fp32_param.values():
                add("fp32_params", t)
            for t in hdo.cpu_copy_map_grad.values():
                add("grads", t)
            for sub in hdo.cpu_optimizers:
                for state in sub.state.values():
                    for k, v in state.items():
                        add(f"adam_{k}", v)
        return groups

    def _node_gb(self, tensors: list[torch.Tensor]) -> dict[str, float]:
        addrs, weights = [], []
        for t in tensors:
            nbytes = t.numel() * t.element_size()
            n = max(1, nbytes // _SAMPLE_BYTES)
            for i in range(n):
                addrs.append(t.data_ptr() + i * (nbytes // n))
                weights.append(nbytes / n)
        out: dict[str, float] = defaultdict(float)
        if not addrs:
            return out
        pages = (ctypes.c_void_p * len(addrs))(*addrs)
        status = (ctypes.c_int * len(addrs))()
        self.libc.syscall(
            ctypes.c_long(self.nr["move_pages"]), ctypes.c_long(0),
            ctypes.c_ulong(len(addrs)), pages, None, status, ctypes.c_int(0),
        )
        for s, w in zip(status, weights):
            key = "hbm" if s in self.gpu_nodes else "cpu" if s >= 0 else "unmapped"
            out[key] += w / _GB
        return out

    def _anon_vmas(self) -> list[tuple[int, int, str, float]]:
        """(start, end, name, GB on GPU nodes) of every writable private anonymous mapping.

        Read-only and PROT_NONE mappings (CUDA's virtual address reservations among
        them) are left out.
        """
        ends, names = {}, {}
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                start, end = (int(x, 16) for x in parts[0].split("-"))
                name = parts[5] if len(parts) > 5 else ""
                if name in ("", "[heap]") and parts[1][1] == "w" and parts[1][3] == "p":
                    ends[start], names[start] = end, name or "anon"
        vmas = []
        with open("/proc/self/numa_maps") as f:
            for line in f:
                parts = line.split()
                start = int(parts[0], 16)
                if start not in ends:
                    continue
                kb = 4
                hbm_pages = 0
                for p in parts[2:]:
                    if p.startswith("kernelpagesize_kB="):
                        kb = int(p.split("=")[1])
                    elif p[0] == "N" and "=" in p:
                        node, pages = p[1:].split("=")
                        if int(node) in self.gpu_nodes:
                            hbm_pages += int(pages)
                vmas.append((start, ends[start], names[start], hbm_pages * kb / 1048576))
        return vmas

    def _bind_all(self, vmas: list[tuple[int, int, str, float]]) -> tuple[int, float, int]:
        bound, gb, failed = 0, 0.0, 0
        for start, end, _, _ in vmas:
            if end - start < _BIND_MIN_BYTES:
                continue
            rc = self.libc.syscall(
                ctypes.c_long(self.nr["mbind"]), ctypes.c_ulong(start),
                ctypes.c_ulong(end - start), ctypes.c_ulong(_MPOL_BIND), self.mask,
                ctypes.c_ulong(65), ctypes.c_ulong(_MPOL_MF_MOVE),
            )
            if rc == 0:
                bound += 1
                gb += (end - start) / _GB
            else:
                failed += 1
        return bound, gb, failed

    def _report(self, optimizer: Any, tag: str) -> list[tuple[int, int, str, float]]:
        groups = self._host_tensors(optimizer)
        known = []
        parts = []
        for kind, tensors in sorted(groups.items()):
            gb = self._node_gb(tensors)
            pinned = sum(t.is_pinned() for t in tensors)
            parts.append(
                f"{kind}={gb['hbm']:.1f}/{sum(gb.values()):.1f}G"
                f"(pinned {pinned}/{len(tensors)})"
            )
            for t in tensors:
                known.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
        known.sort()
        starts = [s for s, _ in known]

        def overlaps_known(start: int, end: int) -> bool:
            i = bisect.bisect_left(starts, end) - 1
            return i >= 0 and known[i][1] > start

        vmas = self._anon_vmas()
        other = [v for v in vmas if v[3] > 0 and not overlaps_known(v[0], v[1])]
        other.sort(key=lambda v: -v[3])
        top = " ".join(
            f"{hex(s)}+{(e - s) / _GB:.1f}G[{n}]:{g:.1f}G" for s, e, n, g in other[:4]
        )
        print(
            f"[hbm-spill] rank={self.rank} {tag} optimizer host tensors on HBM/total: "
            f"{' '.join(parts) or 'none'} | anon on HBM {sum(v[3] for v in vmas):.1f}G, "
            f"outside optimizer tensors {sum(v[3] for v in other):.1f}G {top}",
            flush=True,
        )
        return vmas

    def after_setup(self, optimizer: Any) -> None:
        vmas = self._report(optimizer, "setup") if self.check_steps else self._anon_vmas()
        if self.bind:
            self._log_bind(vmas, "setup")

    def after_step(self, optimizer: Any) -> None:
        self.step += 1
        if not (self.step <= self.check_steps or self.bind):
            return
        torch.cuda.synchronize()
        if self.step <= self.check_steps:
            vmas = self._report(optimizer, f"opt_step={self.step}")
        else:
            vmas = self._anon_vmas()
        if self.bind:
            self._log_bind(vmas, f"opt_step={self.step}")
            if self.step <= self.check_steps:
                self._report(optimizer, f"opt_step={self.step} after bind")

    def _log_bind(self, vmas: list[tuple[int, int, str, float]], tag: str) -> None:
        bound, gb, failed = self._bind_all(vmas)
        if tag == "setup" or self.step <= self.check_steps:
            print(
                f"[hbm-spill] rank={self.rank} {tag} bound {bound} anon mappings "
                f"({gb:.1f}G) to CPU nodes {self.cpu_nodes}, {failed} failed",
                flush=True,
            )


def install(optimizer: Any, rank: int) -> None:
    """Hook the checks/binding onto every CPU-offloading sub-optimizer."""
    check_steps = int(os.environ.get("NRL_HBM_SPILL_CHECK", "0"))
    bind = os.environ.get("NRL_HBM_BIND_CPU", "0") == "1"
    if optimizer is None or not (check_steps or bind):
        return
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    spill = _HbmSpill(rank, check_steps, bind)
    hdos = [
        opt.optimizer
        for opt in getattr(optimizer, "chained_optimizers", None) or [optimizer]
        if isinstance(getattr(opt, "optimizer", None), HybridDeviceOptimizer)
    ]
    if not hdos:
        return

    def guarded(fn: Any) -> None:
        try:
            fn(optimizer)
        except Exception as e:
            print(f"[hbm-spill] rank={rank} check failed: {e!r}", flush=True)

    hdos[-1].register_step_post_hook(lambda *_: guarded(spill.after_step))
    guarded(spill.after_setup)
