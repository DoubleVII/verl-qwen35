import asyncio
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.group import (
    FusedFlashGPEMarkdownRewardModelProcessor,
    GroupRewardManager,
)
from verl.utils.reward_score.group import myers_insert_delete_distance


class FakeTokenizer:
    def __init__(self, decoded=None, encoded=None):
        self.decoded = decoded
        self.encoded = encoded or {}

    def decode(self, ids, skip_special_tokens=True):
        if self.decoded is not None:
            return self.decoded[int(ids[0])]
        return {0: "A", 1: "B", 2: "A", 3: "C"}[int(ids[0])]

    def encode(self, text, add_special_tokens=False):
        if text in self.encoded:
            return self.encoded[text]
        return list(range(len(text)))


class FakeGroupManager(GroupRewardManager):
    calls = 0

    async def _request(self, prompt):
        self.calls += 1
        return "analysis\nA: 2, B: 8"


class FakeFusedManager(FusedFlashGPEMarkdownRewardModelProcessor):
    def __init__(self, *args, **kwargs):
        self.prompts = []
        super().__init__(*args, **kwargs)

    async def _request(self, prompt):
        self.prompts.append(prompt)
        return "analysis\nA: 2, B: 8"


def _batch():
    td = TensorDict(
        {
            "prompts": torch.zeros((4, 1), dtype=torch.long),
            "responses": torch.arange(4).view(4, 1),
            "attention_mask": torch.ones((4, 2), dtype=torch.long),
        },
        batch_size=[4],
    )
    extra = np.array([{"src_text": "source", "lang_pair": "en-zh"}] * 4, dtype=object)
    return DataProto(td, {"extra_info": extra, "uid": np.array(["g", "g", "h", "h"], dtype=object)})


def _fused_markdown(candidates: Sequence[str], final_translation="final translation"):
    rendered = "\n\n".join(f"# Candidate {index}\n{candidate}" for index, candidate in enumerate(candidates, 1))
    return (
        f"<thinking>generate</thinking><response>{rendered}</response>"
        f"<thinking>edit</thinking><response># Final Translation\n{final_translation}</response>"
    )


def _fused_batch(responses, infos=None, uids=None):
    size = len(responses)
    td = TensorDict(
        {
            "prompts": torch.zeros((size, 1), dtype=torch.long),
            "responses": torch.arange(size).view(size, 1),
            "attention_mask": torch.ones((size, 2), dtype=torch.long),
        },
        batch_size=[size],
    )
    if infos is None:
        infos = [
            {
                "src_text": "source",
                "lang_pair": "en-zh",
                "prompt_type": "markdown",
                "max_candidates": 4,
            }
        ] * size
    return DataProto(
        td,
        {
            "extra_info": np.array(infos, dtype=object),
            "uid": np.array(uids or ["g"] * size, dtype=object),
        },
    )


def _fused_config(**overrides):
    reward_kwargs = {
        "group_prompt_type": "score",
        "score_scale_factor": 0.1,
        "default_reward": -1.0,
        "diversity_algorithm": "none",
    }
    reward_kwargs.update(overrides)
    return OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": reward_kwargs,
                "custom_processor": {},
                "reward_model": {"model_path": "model"},
            },
            "trainer": {},
        }
    )


def test_group_manager_scores_each_uid_once():
    cfg = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {"group_prompt_type": "score", "score_scale_factor": 0.1},
                "custom_processor": {},
                "reward_model": {"model_path": "model"},
            },
            "trainer": {},
        }
    )
    manager = FakeGroupManager(cfg, FakeTokenizer(), reward_router_address="router")
    result = asyncio.run(manager.run_batch(_batch()))
    assert manager.calls == 2
    assert [item["reward_score"] for item in result] == [0.2, 0.8, 0.2, 0.8]


def test_fused_markdown_manager_scores_only_final_translations():
    responses = [
        _fused_markdown(["one", "two", "three", "four"], "final one"),
        _fused_markdown(["five", "six", "seven", "eight"], "final two"),
    ]
    manager = FakeFusedManager(_fused_config(), FakeTokenizer(responses), reward_router_address="router")

    result = asyncio.run(manager.run_batch(_fused_batch(responses)))

    assert [item["reward_score"] for item in result] == pytest.approx([0.2, 0.8])
    assert len(manager.prompts) == 1
    assert "final one" in manager.prompts[0]
    assert "final two" in manager.prompts[0]
    assert "# Candidate" not in manager.prompts[0]


def test_fused_markdown_manager_rejects_malformed_and_wrong_candidate_counts():
    responses = [
        "<thinking>incomplete</thinking>",
        _fused_markdown(["one", "two", "three"], "wrong count"),
    ]
    manager = FakeFusedManager(_fused_config(), FakeTokenizer(responses), reward_router_address="router")

    result = asyncio.run(manager.run_batch(_fused_batch(responses)))

    assert [item["reward_score"] for item in result] == [-1.0, -1.0]
    assert manager.prompts == []


def test_fused_markdown_target_candidate_count_and_myers_penalty():
    responses = [
        _fused_markdown(["a b", "a c", "z"], "final one"),
        _fused_markdown(["x", "y", "z"], "final two"),
    ]
    infos = [
        {
            "src_text": "source",
            "lang_pair": "en-zh",
            "prompt_type": "adaptive",
            "max_candidates": 8,
            "target_candidate_count": 3,
        }
    ] * 2
    tokenizer = FakeTokenizer(
        responses,
        encoded={"a b": [1, 2], "a c": [1, 3], "z": [4], "x": [5], "y": [6]},
    )
    manager = FakeFusedManager(
        _fused_config(diversity_algorithm="token_myers", diversity_penalty_weight=0.6),
        tokenizer,
        reward_router_address="router",
    )

    result = asyncio.run(manager.run_batch(_fused_batch(responses, infos)))

    # Pairwise normalized distances are 0.5, 1.0, and 1.0, so penalty is 0.1.
    assert [item["reward_score"] for item in result] == pytest.approx([0.1, 0.8])
    assert myers_insert_delete_distance([1, 2], [1, 3]) == 2


def test_fused_markdown_applies_diversity_penalty_when_group_cannot_be_scored():
    responses = [_fused_markdown(["a b", "a c"], "only final")]
    infos = [
        {
            "src_text": "source",
            "lang_pair": "en-zh",
            "prompt_type": "markdown",
            "max_candidates": 2,
        }
    ]
    tokenizer = FakeTokenizer(responses, encoded={"a b": [1, 2], "a c": [1, 3]})
    manager = FakeFusedManager(
        _fused_config(diversity_algorithm="token_myers", diversity_penalty_weight=1.0),
        tokenizer,
        reward_router_address="router",
    )

    result = asyncio.run(manager.run_batch(_fused_batch(responses, infos)))

    assert result[0]["reward_score"] == pytest.approx(-1.5)
    assert manager.prompts == []


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"diversity_algorithm": "cosine"}, "diversity_algorithm"),
        ({"diversity_penalty_weight": -0.1}, "non-negative"),
        ({"diversity_penalty_clip": 1.1}, "diversity_penalty_clip"),
        ({"diversity_penalty_max": -0.1}, "diversity_penalty_max"),
    ],
)
def test_fused_markdown_manager_rejects_invalid_diversity_config(overrides, error):
    with pytest.raises(ValueError, match=error):
        FakeFusedManager(_fused_config(**overrides), FakeTokenizer([]), reward_router_address="router")
