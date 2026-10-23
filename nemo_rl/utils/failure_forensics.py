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

"""Leave enough behind on a GPU memory failure to see where and why it happened.

Enabled by ``NRL_FORENSICS_DIR`` (one directory per run, shared by all ranks):

- ``mem-rank<r>.tsv``: every ``NRL_MEM_SAMPLE_S`` seconds (default 10) the CUDA
  allocator counters, the device's free memory and what the process holds outside
  PyTorch (NCCL, CUDA context, other libraries), tagged with the last phase the
  worker logged.
- ``failure-rank<r>-<time>.txt``: written when a decorated worker method raises
  (and on any PyTorch out-of-memory error): the exception, the same counters and
  the allocator summary.
- ``oom-rank<r>-<time>.pickle``: with ``NRL_MEM_HISTORY=1``, a memory snapshot
  of every live allocation with the Python stack that made it, written for
  out-of-memory errors only (PyTorch's, and NCCL's ``Cuda failure 'out of
  memory'``). Open it at https://pytorch.org/memory_viz.

``NRL_FORENSICS_TEST_OOM=1`` makes the first ``train`` call request 1 TiB of GPU
memory, to check the whole path in a debug run.
"""

import functools
import os
import socket
import threading
import time
import traceback
from typing import Any, Callable, Optional

import torch

_GB = float(1 << 30)
_MAX_REPORTS = 5

_state: dict[str, Any] = {
    "dir": None,
    "rank": -1,
    "device": None,
    "phase": "init",
    "history": False,
    "reports": 0,
    "snapshots": 0,
}


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _counters() -> str:
    dev = _state["device"]
    free, total = torch.cuda.mem_get_info(dev)
    reserved = torch.cuda.memory_reserved(dev)
    return (
        f"free={free / _GB:.2f}G total={total / _GB:.2f}G "
        f"allocated={torch.cuda.memory_allocated(dev) / _GB:.2f}G "
        f"reserved={reserved / _GB:.2f}G "
        f"max_allocated={torch.cuda.max_memory_allocated(dev) / _GB:.2f}G "
        f"max_reserved={torch.cuda.max_memory_reserved(dev) / _GB:.2f}G "
        f"outside_torch={(total - free - reserved) / _GB:.2f}G"
    )


def set_phase(tag: str) -> None:
    """Record the phase the worker is in; it tags the memory samples."""
    _state["phase"] = tag


def _sampler(period_s: float) -> None:
    torch.cuda.set_device(_state["device"])
    path = os.path.join(_state["dir"], f"mem-rank{_state['rank']}.tsv")
    with open(path, "a", buffering=1) as f:
        f.write(f"# host={socket.gethostname()} pid={os.getpid()} device={_state['device']}\n")
        while True:
            try:
                f.write(f"{time.strftime('%F %T')}\t{_state['phase']}\t{_counters()}\n")
            except Exception as e:
                f.write(f"{time.strftime('%F %T')}\tsample failed: {e!r}\n")
            time.sleep(period_s)


def _snapshot() -> Optional[str]:
    if not _state["history"] or _state["snapshots"] >= 1:
        return None
    _state["snapshots"] += 1
    path = os.path.join(_state["dir"], f"oom-rank{_state['rank']}-{_stamp()}.pickle")
    torch.cuda.memory._dump_snapshot(path)
    return path


def report(reason: str, exc: Optional[BaseException] = None) -> None:
    """Write the memory state now, plus a snapshot if this is an out-of-memory error."""
    if _state["dir"] is None or _state["reports"] >= _MAX_REPORTS:
        return
    _state["reports"] += 1
    text = f"{reason}\n{exc!r}" if exc is not None else reason
    oom = "out of memory" in text.lower()
    path = os.path.join(_state["dir"], f"failure-rank{_state['rank']}-{_stamp()}.txt")
    try:
        with open(path, "w") as f:
            f.write(
                f"rank={_state['rank']} host={socket.gethostname()} pid={os.getpid()} "
                f"device={_state['device']} phase={_state['phase']}\n{reason}\n"
            )
            if exc is not None:
                f.write("".join(traceback.format_exception(exc)))
            f.write(f"\n{_counters()}\n")
            snap = _snapshot() if oom else None
            if snap:
                f.write(f"memory snapshot: {snap}\n")
            f.write("\n" + torch.cuda.memory_summary(_state["device"], abbreviated=True))
        print(f"[forensics] rank={_state['rank']} wrote {path}", flush=True)
    except Exception as e:
        print(f"[forensics] rank={_state['rank']} report failed: {e!r}", flush=True)


def _on_torch_oom(device: int, alloc: int, device_total: int, device_free: int) -> None:
    report(
        f"PyTorch out of memory: requested {alloc / _GB:.2f}G, "
        f"device total {device_total / _GB:.2f}G, device free {device_free / _GB:.2f}G"
    )


def dump_on_failure(fn: Callable) -> Callable:
    """Decorate a worker method so a failure inside it leaves a report behind."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            if fn.__name__ == "train" and os.environ.get("NRL_FORENSICS_TEST_OOM") == "1":
                torch.empty(1 << 40, dtype=torch.uint8, device="cuda")
            return fn(*args, **kwargs)
        except Exception as e:
            report(f"exception in {fn.__name__}", e)
            raise

    return wrapper


def install(rank: int) -> None:
    """Start the forensics for this worker; call once its CUDA device is set."""
    out_dir = os.environ.get("NRL_FORENSICS_DIR")
    if not out_dir or not torch.cuda.is_available():
        return
    os.makedirs(out_dir, exist_ok=True)
    _state.update(dir=out_dir, rank=rank, device=torch.cuda.current_device())
    if os.environ.get("NRL_MEM_HISTORY", "0") == "1":
        torch.cuda.memory._record_memory_history(
            context="alloc",
            stacks="python",
            max_entries=int(os.environ.get("NRL_MEM_HISTORY_ENTRIES", "100000")),
        )
        _state["history"] = True
    torch._C._cuda_attach_out_of_memory_observer(_on_torch_oom)
    period = float(os.environ.get("NRL_MEM_SAMPLE_S", "10"))
    threading.Thread(
        target=_sampler, args=(period,), name="forensics-sampler", daemon=True
    ).start()
    print(
        f"[forensics] rank={rank} writing to {out_dir} "
        f"(memory history {'on' if _state['history'] else 'off'})",
        flush=True,
    )
