# Copyright (c) 2026, ETH Zurich / EPFL / Swiss AI Initiative.
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

import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.bridge")
offloading = pytest.importorskip(
    "megatron.core.transformer.moe.experts_offloading_fp8_util"
)

from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore

MANAGER_MODULE = "megatron.core.transformer.moe.experts_offloading_fp8_util"


def _worker(**overrides):
    config = {
        "moe_use_offloading_experts": True,
        "moe_use_inplace_fp8_param": True,
        "moe_offloading_experts_debug_mode": False,
        "moe_offloading_mode": "coarse-grained",
        "moe_use_extra_fp8_param_storage": False,
    }
    config.update(overrides)
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = torch.nn.Linear(2, 2, bias=False)
    worker.model.config = SimpleNamespace(**config)
    worker.cfg = {"megatron_cfg": {"empty_unused_memory_level": 0}}
    worker.should_disable_forward_pre_hook = False
    worker.sampling_params = None
    return worker


@pytest.fixture
def manager(monkeypatch):
    cls = offloading.FP8ExpertsParameterManager
    cache = object.__new__(cls)
    cache._uid_to_is_first_microbatch = {1: False}
    monkeypatch.setattr(cls, "_instance", cache)
    return cache


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_use_offloading_experts": False},
        {"moe_use_inplace_fp8_param": False},
        {"moe_offloading_experts_debug_mode": True},
    ],
)
def test_disabled_cache_does_not_import_optional_manager(monkeypatch, overrides):
    monkeypatch.setitem(sys.modules, MANAGER_MODULE, None)
    _worker(**overrides)._mark_offloaded_fp8_weights_dirty()


def test_requested_cache_without_core_api_fails_explicitly(monkeypatch):
    monkeypatch.setitem(sys.modules, MANAGER_MODULE, None)
    with pytest.raises(RuntimeError, match="FP8ExpertsParameterManager"):
        _worker()._mark_offloaded_fp8_weights_dirty()


def test_fine_mode_with_extra_storage_marks_cache_dirty(manager):
    worker = _worker(
        moe_offloading_mode="fine-grained",
        moe_use_extra_fp8_param_storage=True,
    )
    worker._mark_offloaded_fp8_weights_dirty()
    assert manager._uid_to_is_first_microbatch[1]


def test_fine_mode_requires_intact_bf16_source_weights():
    with pytest.raises(ValueError, match="moe_use_extra_fp8_param_storage=True"):
        _worker(moe_offloading_mode="fine-grained")._mark_offloaded_fp8_weights_dirty()


def test_rank_without_local_manager_needs_no_creation(monkeypatch):
    monkeypatch.setattr(offloading.FP8ExpertsParameterManager, "_instance", None)
    _worker()._mark_offloaded_fp8_weights_dirty()
    assert offloading.FP8ExpertsParameterManager._instance is None


def test_reference_copy_and_policy_restore_each_mark_cache_dirty(manager):
    worker = _worker()
    original = worker.model.weight.detach().clone()
    worker.reference_state_dict = {"weight": torch.ones_like(original)}
    with worker.use_reference_model():
        assert torch.equal(worker.model.weight, torch.ones_like(original))
        assert manager._uid_to_is_first_microbatch[1]
        # Model the flag consumption on the reference forward's first cache access.
        manager._uid_to_is_first_microbatch[1] = False
    assert torch.equal(worker.model.weight, original)
    assert manager._uid_to_is_first_microbatch[1]


def test_state_application_without_tensor_writes_leaves_cache_clean(manager):
    _worker()._apply_state_dict_to_model({})
    assert not manager._uid_to_is_first_microbatch[1]
