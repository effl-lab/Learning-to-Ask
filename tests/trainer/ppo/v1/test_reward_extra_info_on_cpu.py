# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""V1 trainer: reward_extra_info plumbing (advantage estimators + train metrics).

Covers the two legacy-trainer behaviours that depend on ``batch.non_tensor_batch`` carrying
the reward components: the ``grpo_segmented`` estimator (seg_pre/seg_post) and the
``reward_extra/*/mean`` train metrics. In V1 those components live in each trajectory's
TransferQueue ``extra_fields["reward_extra_info"]`` (or, for a colocated reward model, in
top-level fields), so the trainer has to fetch and expose them explicitly.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.utils import compute_reward_extra_metrics, extract_reward_extra_info
from verl.utils import tensordict_utils as tu


def test_extract_reward_extra_info_fills_missing_with_none():
    extra_fields = [
        {"reward_extra_info": {"seg_pre": 1.0, "query": "q"}},
        {"reward_extra_info": {"seg_pre": 0.5, "called": 1}},
        {"min_global_steps": 3},  # trajectory without reward info
        "not a dict",
    ]
    out = extract_reward_extra_info(extra_fields)
    assert set(out) == {"seg_pre", "query", "called"}
    assert out["seg_pre"].tolist() == [1.0, 0.5, None, None]
    assert out["query"].tolist() == ["q", None, None, None]
    assert out["called"].tolist() == [None, 1, None, None]
    assert all(isinstance(v, np.ndarray) and v.dtype == object and v.shape == (4,) for v in out.values())
    assert extract_reward_extra_info([]) == {}


def test_compute_reward_extra_metrics_numeric_only():
    reward_extra = {
        "called": np.array([1, 0, None, 1], dtype=object),  # None rows excluded from the mean
        "flag": np.array([True, False], dtype=object),
        "score": np.array([0.5, 1.5]),
        "query": np.array(["a", "b"], dtype=object),  # strings skipped
        "nested": np.array([[1, 2], [3]], dtype=object),  # ragged lists skipped
        "empty": np.array([None, None], dtype=object),
    }
    metrics = compute_reward_extra_metrics(reward_extra)
    assert set(metrics) == {"reward_extra/called/mean", "reward_extra/flag/mean", "reward_extra/score/mean"}
    assert metrics["reward_extra/called/mean"] == pytest.approx(2 / 3)
    assert metrics["reward_extra/flag/mean"] == pytest.approx(0.5)
    assert metrics["reward_extra/score/mean"] == pytest.approx(1.0)
    assert compute_reward_extra_metrics({}) == {}


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _nested(rows, dtype):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=dtype) for r in rows], layout=torch.jagged)


class _FakeTransferQueue:
    """One partition; every field holds one entry per key, in key order."""

    def __init__(self, keys: list[str], tensor_fields: dict, list_fields: dict):
        self.keys = keys
        self.tensor_fields = tensor_fields
        self.list_fields = list_fields
        self.puts: list[dict] = []
        self.get_calls: list[list[str] | None] = []

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        assert list(keys) == self.keys
        self.get_calls.append(select_fields)
        wanted = self.keys if select_fields is None else select_fields
        # TransferQueue only returns the fields that were written (values, ref_log_prob, ... may be absent).
        tensor_dict = {k: v for k, v in self.tensor_fields.items() if k in wanted}
        tensor_dict.update({k: list(v) for k, v in self.list_fields.items() if k in wanted})
        return tu.get_tensordict(tensor_dict)

    def kv_batch_put(self, keys, partition_id, fields):
        assert list(keys) == self.keys
        self.puts.append(dict(fields.items()))
        return KVBatchMeta(keys=list(keys), tags=[{} for _ in keys], partition_id=partition_id)


@pytest.fixture
def segmented_batch(monkeypatch):
    # Two prompt groups of three trajectories. In group A the first two rows contain a masked tool
    # span of >= 8 tokens (the grpo_segmented pre/post boundary); the third row never called.
    keys = ["A_0_0", "A_1_0", "A_2_0", "B_0_0", "B_1_0", "B_2_0"]
    masks = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1],
    ]
    seg_pre = [2.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    seg_post = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rm_scores = [[0.0] * (len(m) - 1) + [pre + post] for m, pre, post in zip(masks, seg_pre, seg_post, strict=True)]
    extra_fields = [
        {"reward_extra_info": {"seg_pre": pre, "seg_post": post, "score": pre + post, "query": "q"}}
        for pre, post in zip(seg_pre, seg_post, strict=True)
    ]
    fake = _FakeTransferQueue(
        keys,
        tensor_fields={"response_mask": _nested(masks, torch.int64), "rm_scores": _nested(rm_scores, torch.float32)},
        list_fields={"uid": [k.split("_")[0] for k in keys], "extra_fields": extra_fields},
    )
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", fake.kv_batch_get)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", fake.kv_batch_put)

    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo_segmented",
                "use_kl_in_reward": False,
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
                "rollout_correction": None,
            },
            "actor_rollout_ref": {"rollout": {"n": 3}},
        }
    )
    batch = KVBatchMeta(keys=keys, tags=[{} for _ in keys], partition_id="train")
    return trainer, batch, fake, masks


def test_compute_advantage_grpo_segmented_reads_seg_components_from_extra_fields(segmented_batch):
    trainer, batch, fake, masks = segmented_batch

    trainer._compute_advantage(batch, metrics={})

    assert len(fake.puts) == 1
    advantages = fake.puts[0]["advantages"]
    assert advantages.is_nested
    rows = [row for row in advantages.unbind()]
    assert [len(r) for r in rows] == [len(m) for m in masks]

    # Group A: seg_pre [2, 1, 0] -> group-normalized [1, 0, -1]; seg_post over the two rows that
    # have a post segment [1, 0] -> [0.707, -0.707]. Group B: seg_pre [1, 1, 0] -> [0.577, 0.577, -1.155].
    # (GRPO normalization uses the unbiased std.) Masked tool-span tokens get 0.
    p, b = 1 / np.sqrt(2), 1 / np.sqrt(3)
    expected = [
        [1.0] * 3 + [0.0] * 8 + [p] * 2,
        [0.0] * 2 + [0.0] * 8 + [-p],
        [-1.0] * 4,
        [b] * 3,
        [b] * 5,
        [-2 * b] * 2,
    ]
    for row, exp in zip(rows, expected, strict=True):
        torch.testing.assert_close(row, torch.tensor(exp, dtype=torch.float32), atol=1e-3, rtol=0)
    torch.testing.assert_close(fake.puts[0]["returns"].values(), advantages.values())


def test_fetch_reward_extra_info_caches_and_merges_colocated_fields(segmented_batch):
    trainer, batch, fake, _ = segmented_batch
    fake.list_fields["acc"] = [1, 0, 1, 0, 1, 0]
    trainer._colocate_reward_extra_keys = ["acc", "score"]  # "score" already comes from extra_fields

    reward_extra = trainer._fetch_reward_extra_info(batch)
    assert set(reward_extra) == {"seg_pre", "seg_post", "score", "query", "acc"}
    assert reward_extra["acc"].tolist() == [1, 0, 1, 0, 1, 0]
    assert fake.get_calls == [["extra_fields"], ["acc"]]

    # Second call for the same batch hits the cache.
    assert trainer._fetch_reward_extra_info(batch) is reward_extra
    assert len(fake.get_calls) == 2

    metrics = compute_reward_extra_metrics(reward_extra)
    assert metrics["reward_extra/seg_pre/mean"] == pytest.approx(5 / 6)
    assert metrics["reward_extra/acc/mean"] == pytest.approx(0.5)
    assert "reward_extra/query/mean" not in metrics


def test_fetch_reward_extra_info_without_extra_fields(segmented_batch, monkeypatch):
    trainer, batch, _, _ = segmented_batch

    def failing_get(keys, partition_id, select_fields=None):
        raise ValueError("field extra_fields not found")

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", failing_get)
    assert trainer._fetch_reward_extra_info(batch) == {}
