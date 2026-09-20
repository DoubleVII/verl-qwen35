from __future__ import annotations

import aiohttp

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score.group import (
    build_prompt,
    extract_response,
    is_language_match,
    is_valid_fused_candidate_count,
    language_pair,
    normalize_fused_candidate,
    overlong_penalty,
    parse_fused_flash_gpe_markdown_response,
    parse_fused_flash_gpe_response,
    parse_fused_flash_gpe_simple_markdown_response,
    parse_scores,
    token_myers_diversity,
)


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
        self.enable_language_detection = bool(cfg.get("enable_language_detection", False))
        self.overlong = cfg.get("overlong_buffer", None)
        self.model = config.reward.reward_model.model_path
        rollout_cfg = config.reward.reward_model.get("rollout", {})
        self.max_tokens = int(rollout_cfg.get("response_length") or 2048)
        self.sampling_params = {
            "temperature": rollout_cfg.get("temperature", 1.0),
            "top_p": rollout_cfg.get("top_p", 1.0),
            "top_k": rollout_cfg.get("top_k", -1),
        }

    def _prepare_response(self, response: str, info: dict) -> tuple[str | None, float]:
        return extract_response(response, self.extractor), 0.0

    def _encode_prompt(self, prompt: str) -> list[int]:
        """Use the same local chat template for length filtering and RM inference."""
        text = self.rm_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return self.rm_tokenizer.encode(text, add_special_tokens=False)

    async def _request(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            # The old processor sent templated token IDs directly to vLLM. Avoid
            # server-side chat templating/reasoning extraction changing that contract.
            "prompt": self._encode_prompt(prompt),
            "max_tokens": self.max_tokens,
            **self.sampling_params,
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.post(f"http://{self.router}/v1/completions", json=payload) as response:
                response.raise_for_status()
                response_json = await response.json()
                choice = response_json["choices"][0]
                result = choice["text"]
                return result

    async def run_batch(self, data: DataProto) -> list[dict]:
        n = len(data)
        responses = []
        response_penalties = []
        lengths = []
        infos = list(data.non_tensor_batch["extra_info"])
        for i in range(n):
            ids = data.batch["responses"][i]
            length = int(data.batch["attention_mask"][i][-ids.shape[-1] :].sum())
            lengths.append(length)
            text = self.tokenizer.decode(ids[:length], skip_special_tokens=True)
            if self.tokenizer.eos_token:
                text = text.replace(self.tokenizer.eos_token, "")
            response, penalty = self._prepare_response(text, infos[i])
            responses.append(response)
            response_penalties.append(penalty)
        uids = list(data.non_tensor_batch["uid"])
        groups = {}
        for i, uid in enumerate(uids):
            groups.setdefault(str(uid), []).append(i)
        scores = [self.default - penalty for penalty in response_penalties]
        metadata = [{"group_reward_output": "", "group_reward_prompt": None} for _ in range(n)]
        for indices in groups.values():
            valid = []
            seen = {}
            target_lang = language_pair(infos[indices[0]])[1]
            for i in indices:
                text = responses[i]
                if text is None:
                    continue
                language_ok = not self.enable_language_detection or is_language_match(text, target_lang)
                if not language_ok:
                    continue
                if text not in seen:
                    seen[text] = []
                    valid.append(text)
                seen[text].append(i)
            if len(valid) <= 1:
                continue
            prompt = build_prompt(infos[indices[0]], valid, self.prompt_type, self.add_example)
            if len(self._encode_prompt(prompt)) > self.max_prompt_length:
                continue
            # Transport/server failures should fail the step as in the old engine,
            # not silently turn an entire batch into negative training rewards.
            output = await self._request(prompt)
            parsed = parse_scores(output, self.prompt_type, len(valid))
            if parsed is None:
                # Compatibility with compute_group_translation_scores: only RM
                # parse failures are scaled again; rejected inputs keep default.
                parsed = [self.default] * len(valid)
            for text, raw_score in zip(valid, parsed, strict=True):
                targets = seen[text]
                # Legacy GQM deduplicates translations before scoring and shares
                # the first occurrence's length penalty across duplicate outputs.
                score = raw_score * self.scale - overlong_penalty(lengths[targets[0]], self.overlong)
                for i in targets:
                    scores[i] = score - response_penalties[i]
                    metadata[i] = {"group_reward_output": output, "group_reward_prompt": prompt}
        return [{"reward_score": score, "reward_extra_info": metadata[i]} for i, score in enumerate(scores)]

    async def run_single(self, data: DataProto) -> dict:
        return (await self.run_batch(data))[-1]


@register("fused_flash_gpe")
class FusedFlashGPERewardModelProcessor(GroupRewardManager):
    """Score final translations from fused JSON candidate-generation/post-edit responses."""

    _DIVERSITY_ALGORITHMS = {"none", "exact_match", "token_myers"}

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)
        cfg = dict(config.reward.get("reward_kwargs", {}))
        cfg.update(dict(config.reward.get("custom_processor", {})))
        self.diversity_algorithm = cfg.get("diversity_algorithm", "none")
        self.diversity_penalty_weight = float(cfg.get("diversity_penalty_weight", 1.0))
        self.diversity_penalty_clip = float(cfg.get("diversity_penalty_clip", 0.0))
        penalty_max = cfg.get("diversity_penalty_max")
        self.diversity_penalty_max = None if penalty_max is None else float(penalty_max)
        if self.diversity_algorithm not in self._DIVERSITY_ALGORITHMS:
            raise ValueError(
                f"diversity_algorithm must be one of {sorted(self._DIVERSITY_ALGORITHMS)}, "
                f"got {self.diversity_algorithm!r}"
            )
        if self.diversity_penalty_weight < 0:
            raise ValueError("diversity_penalty_weight must be non-negative")
        if not 0.0 <= self.diversity_penalty_clip <= 1.0:
            raise ValueError("diversity_penalty_clip must be between 0 and 1")
        if self.diversity_penalty_max is not None and self.diversity_penalty_max < 0:
            raise ValueError("diversity_penalty_max must be non-negative or None")

    def _cap_diversity_penalty(self, penalty: float) -> float:
        return penalty if self.diversity_penalty_max is None else min(penalty, self.diversity_penalty_max)

    def _diversity_penalty(self, candidates: list[str]) -> float:
        if self.diversity_algorithm == "none":
            return 0.0
        has_duplicate = len({normalize_fused_candidate(candidate) for candidate in candidates}) != len(candidates)
        if self.diversity_algorithm == "exact_match":
            penalty = self.diversity_penalty_weight if has_duplicate else 0.0
        else:
            diversity = token_myers_diversity(candidates, self.tokenizer)
            penalty = max(1.0 - diversity - self.diversity_penalty_clip, 0.0) * self.diversity_penalty_weight
        return self._cap_diversity_penalty(penalty)

    def _prepare_response(self, response: str, info: dict) -> tuple[str | None, float]:
        parsed = self._parse_response(response)
        if parsed is None:
            return None, 0.0
        candidates, final_translation = parsed
        if not is_valid_fused_candidate_count(info, len(candidates)):
            return None, 0.0
        return final_translation, self._diversity_penalty(candidates)

    def _parse_response(self, response: str) -> tuple[list[str], str] | None:
        return parse_fused_flash_gpe_response(response)


@register("fused_flash_gpe_markdown")
class FusedFlashGPEMarkdownRewardModelProcessor(FusedFlashGPERewardModelProcessor):
    """Use the Markdown output protocol with the shared fused Flash GPE scorer."""

    def _parse_response(self, response: str) -> tuple[list[str], str] | None:
        return parse_fused_flash_gpe_markdown_response(response)


@register("fused_flash_gpe_simple_markdown")
class FusedFlashGPESimpleMarkdownRewardModelProcessor(FusedFlashGPERewardModelProcessor):
    """Use the visible-analysis simple Markdown protocol with the shared scorer."""

    def _parse_response(self, response: str) -> tuple[list[str], str] | None:
        return parse_fused_flash_gpe_simple_markdown_response(response)
