import asyncio
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.group import GroupRewardManager


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return {0: "A", 1: "B", 2: "A", 3: "C"}[int(ids[0])]

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class FakeGroupManager(GroupRewardManager):
    calls = 0

    async def _request(self, prompt):
        self.calls += 1
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
