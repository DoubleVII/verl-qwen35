from __future__ import annotations

import aiohttp
import numpy as np

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score.group import build_prompt, extract_response, language_pair, overlong_penalty, parse_scores


@register("group")
class GroupRewardManager(RewardManagerBase):
    """Batch GQM manager: one generative RM request per uid group."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.router = reward_router_address
        self.rm_tokenizer = reward_model_tokenizer or tokenizer
        cfg = config.reward.get("reward_kwargs", {})
        cfg = dict(cfg)
        cfg.update(dict(config.reward.get("custom_processor", {})))
        self.prompt_type = cfg.get("group_prompt_type", "ranking_score")
        self.extractor = cfg.get("extractor_type", "line")
        self.max_prompt_length = int(cfg.get("max_prompt_length", 2048))
        self.scale = float(cfg.get("score_scale_factor", 0.1))
        self.default = float(cfg.get("default_reward", 0.0))
        self.add_example = bool(cfg.get("group_add_example", False))
        self.overlong = cfg.get("overlong_buffer", None)
        self.model = config.reward.reward_model.model_path

    async def _request(self, prompt: str) -> str:
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 8192}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.post(f"http://{self.router}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                return (await response.json())["choices"][0]["message"]["content"]

    async def run_batch(self, data: DataProto) -> list[dict]:
        n = len(data)
        responses = []
        for i in range(n):
            ids = data.batch["responses"][i]
            length = int(data.batch["attention_mask"][i][-ids.shape[-1]:].sum())
            responses.append(extract_response(self.tokenizer.decode(ids[:length], skip_special_tokens=True), self.extractor))
        infos = list(data.non_tensor_batch.get("extra_info", [{}] * n))
        uids = list(data.non_tensor_batch.get("uid", np.arange(n)))
        groups = {}
        for i, uid in enumerate(uids):
            groups.setdefault(str(uid), []).append(i)
        scores = [self.default] * n
        metadata = [{"group_reward_output": "", "group_reward_prompt": None} for _ in range(n)]
        for indices in groups.values():
            valid = []
            seen = {}
            for i in indices:
                text = responses[i]
                if text is not None and text not in seen:
                    seen[text] = len(valid); valid.append(text)
            if len(valid) <= 1:
                continue
            prompt = build_prompt(infos[indices[0]], valid, self.prompt_type, self.add_example)
            if len(self.rm_tokenizer.encode(prompt, add_special_tokens=False)) > self.max_prompt_length:
                continue
            try:
                output = await self._request(prompt)
            except Exception:
                output = ""
            parsed = parse_scores(output, self.prompt_type, len(valid))
            if parsed is None:
                continue
            for i in indices:
                if responses[i] in seen:
                    score = parsed[seen[responses[i]]] * self.scale
                    ids = data.batch["responses"][i]
                    length = int(data.batch["attention_mask"][i][-ids.shape[-1]:].sum())
                    scores[i] = score - overlong_penalty(length, self.overlong)
                    metadata[i] = {"group_reward_output": output, "group_reward_prompt": prompt}
        return [{"reward_score": score, "reward_extra_info": metadata[i]} for i, score in enumerate(scores)]

    async def run_single(self, data: DataProto) -> dict:
        return (await self.run_batch(data))[-1]
