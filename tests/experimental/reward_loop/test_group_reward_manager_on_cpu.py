import asyncio
import json
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.group import (
    FusedFlashGPEMarkdownRewardModelProcessor,
    FusedFlashGPERewardModelProcessor,
    FusedFlashGPESimpleMarkdownRewardModelProcessor,
    GroupRewardManager,
)
from verl.utils.reward_score.group import (
    build_prompt,
    myers_insert_delete_distance,
    parse_fused_flash_gpe_simple_markdown_response,
    parse_scores,
)


class FakeTokenizer:
    eos_token = "<eos>"

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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return f"<user>{messages[0]['content']}</user><assistant>"


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


class FakeJSONFusedManager(FusedFlashGPERewardModelProcessor):
    def __init__(self, *args, **kwargs):
        self.prompts = []
        super().__init__(*args, **kwargs)

    async def _request(self, prompt):
        self.prompts.append(prompt)
        return "analysis\nA: 2, B: 8"


class FakeSimpleFusedManager(FusedFlashGPESimpleMarkdownRewardModelProcessor):
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


def _fused_json(candidates: Sequence[str], final_translation="final translation"):
    rendered = json.dumps({"translations": candidates}, ensure_ascii=False)
    return (
        f"<thinking>generate</thinking><response>{rendered}</response>"
        f"<thinking>edit</thinking><response>```text\n{final_translation}\n```</response>"
    )


def _fused_simple_markdown(candidates: Sequence[str], final_translation="final translation"):
    rendered = "\n\n".join(f"# Candidate {index}\n{candidate}" for index, candidate in enumerate(candidates, 1))
    return (
        "# Step-by-step Analysis\n\nGenerate diverse candidates.\n\n"
        f"{rendered}\n\n"
        "---\n\n"
        "Now, review the candidates and produce the best final translation.\n\n"
        "# Step-by-step Analysis\n\nCompare candidate fidelity.\n\n"
        f"# Final Translation\n\n```\n{final_translation}\n```"
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


def test_fused_simple_markdown_manager_scores_only_final_translations():
    responses = [
        _fused_simple_markdown(["one", "two", "three", "four"], "final one"),
        _fused_simple_markdown(["five", "six", "seven", "eight"], "final two"),
    ]
    manager = FakeSimpleFusedManager(_fused_config(), FakeTokenizer(responses), reward_router_address="router")

    result = asyncio.run(manager.run_batch(_fused_batch(responses)))

    assert [item["reward_score"] for item in result] == pytest.approx([0.2, 0.8])
    assert len(manager.prompts) == 1
    assert "final one" in manager.prompts[0]
    assert "final two" in manager.prompts[0]
    assert "# Candidate" not in manager.prompts[0]


@pytest.mark.parametrize(
    "response",
    [
        _fused_simple_markdown(["one", "two"]).replace(
            "Now, review the candidates and produce the best final translation.", "Different transition."
        ),
        _fused_simple_markdown(["one", "two"]).replace("\n\n---\n\n", "\n---\n\n"),
        _fused_simple_markdown(["one", "two"]).replace(
            "---\n\nNow, review the candidates and produce the best final translation.",
            "---\n\nNow, review the candidates and produce the best final translation.\n\n"
            "---\n\nNow, review the candidates and produce the best final translation.",
        ),
        _fused_simple_markdown(["one", "two"]).replace("Generate diverse candidates.", ""),
        _fused_simple_markdown(["one", "two"]).replace("Compare candidate fidelity.", ""),
        _fused_simple_markdown(["one", "two"]).replace("```\nfinal translation\n```", "final translation"),
        _fused_simple_markdown(["one", "two"]).replace(
            "final translation\n```", "first draft\n```\n\nRevised final translation.\n```"
        ),
        _fused_simple_markdown(["one", "two"]) + "\ntrailing text",
    ],
)
def test_fused_simple_markdown_parser_rejects_malformed_protocol(response):
    assert parse_fused_flash_gpe_simple_markdown_response(response) is None


def test_fused_simple_markdown_parser_allows_other_horizontal_rules():
    response = _fused_simple_markdown(["one", "two"]).replace(
        "Generate diverse candidates.", "Generate diverse candidates.\n\n---\n\nContinue analysis."
    )
    assert parse_fused_flash_gpe_simple_markdown_response(response) == (["one", "two"], "final translation")


@pytest.mark.parametrize(
    ("prompt_type", "max_candidates", "candidate_count"),
    [("fixed_4", 4, 4), ("fixed_16", 16, 16), ("adaptive", 8, 2)],
)
def test_fused_json_manager_scores_only_final_translations(prompt_type, max_candidates, candidate_count):
    responses = [
        _fused_json([f"candidate {i}" for i in range(candidate_count)], "final one"),
        _fused_json([f"alternative {i}" for i in range(candidate_count)], "final two"),
    ]
    infos = [
        {
            "src_text": "source",
            "lang_pair": "en-zh",
            "prompt_type": prompt_type,
            "max_candidates": max_candidates,
        }
    ] * 2
    manager = FakeJSONFusedManager(_fused_config(), FakeTokenizer(responses), reward_router_address="router")

    result = asyncio.run(manager.run_batch(_fused_batch(responses, infos)))

    assert [item["reward_score"] for item in result] == pytest.approx([0.2, 0.8])
    assert len(manager.prompts) == 1
    assert "final one" in manager.prompts[0]
    assert "final two" in manager.prompts[0]
    assert "candidate 0" not in manager.prompts[0]


def test_fused_json_manager_honors_target_candidate_count():
    responses = [
        _fused_json(["one", "two", "three"], "final one"),
        _fused_json(["four", "five", "six"], "final two"),
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
    manager = FakeJSONFusedManager(_fused_config(), FakeTokenizer(responses), reward_router_address="router")

    result = asyncio.run(manager.run_batch(_fused_batch(responses, infos)))

    assert [item["reward_score"] for item in result] == pytest.approx([0.2, 0.8])


@pytest.mark.parametrize(
    ("prompt_type", "max_candidates", "candidate_count"),
    [
        ("fixed_4", 4, 3),
        ("fixed_4", 8, 4),
        ("fixed_16", 16, 15),
        ("adaptive", 8, 1),
        ("adaptive", 2, 3),
        ("unknown", 4, 4),
    ],
)
def test_fused_json_manager_rejects_invalid_candidate_counts(prompt_type, max_candidates, candidate_count):
    responses = [
        _fused_json([f"candidate {i}" for i in range(candidate_count)], "final one"),
        _fused_json([f"alternative {i}" for i in range(candidate_count)], "final two"),
    ]
    infos = [
        {
            "src_text": "source",
            "lang_pair": "en-zh",
            "prompt_type": prompt_type,
            "max_candidates": max_candidates,
        }
    ] * 2
    manager = FakeJSONFusedManager(_fused_config(), FakeTokenizer(responses), reward_router_address="router")

    result = asyncio.run(manager.run_batch(_fused_batch(responses, infos)))

    assert [item["reward_score"] for item in result] == [-1.0, -1.0]
    assert manager.prompts == []


@pytest.mark.parametrize(
    "response",
    [
        "<thinking>incomplete</thinking>",
        _fused_json(["one", "two"]).replace('"translations"', '"candidates"'),
        _fused_json(["one", ""]),
        _fused_json(["one", "two"]).replace("```text\nfinal translation\n```", "final translation"),
    ],
)
def test_fused_json_manager_rejects_malformed_responses(response):
    info = {
        "src_text": "source",
        "lang_pair": "en-zh",
        "prompt_type": "adaptive",
        "max_candidates": 4,
    }
    manager = FakeJSONFusedManager(_fused_config(), FakeTokenizer([response]), reward_router_address="router")

    result = asyncio.run(manager.run_batch(_fused_batch([response], [info])))

    assert result[0]["reward_score"] == -1.0
    assert manager.prompts == []


def test_fused_json_manager_applies_exact_match_diversity_penalty():
    responses = [
        _fused_json(["Same  Translation", " same translation "], "final one"),
        _fused_json(["different one", "different two"], "final two"),
    ]
    infos = [
        {
            "src_text": "source",
            "lang_pair": "en-zh",
            "prompt_type": "adaptive",
            "max_candidates": 2,
        }
    ] * 2
    manager = FakeJSONFusedManager(
        _fused_config(diversity_algorithm="exact_match", diversity_penalty_weight=1.0),
        FakeTokenizer(responses),
        reward_router_address="router",
    )

    result = asyncio.run(manager.run_batch(_fused_batch(responses, infos)))

    assert [item["reward_score"] for item in result] == pytest.approx([-0.8, 0.8])


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


def test_group_prompt_preserves_legacy_text_and_context_order():
    info = {"lang_pair": "en-zh", "src_text": "source", "ref_text": " ref ", "ref_lang": " en ", "notes": " note "}
    expected = (
        "Given a source text in English and multiple translation candidates in Chinese. "
        "Perform a step by step analysis and comparison of the translation quality for the candidates. "
        "Finally, rank and score the candidates with integer scores on a scale from 0 to 10.\n\n"
        "Source text:\n```\nsource\n```\n\nTranslation A:\n```\none\n```\nTranslation B:\n```\ntwo\n```\n"
        "\n\nYou may refer to the following reference, if helpful, when evaluating the translations.\n\n"
        "English reference:\n```\nref\n```\n"
        "\n\nYou may refer to the following notes, if helpful, when evaluating the translations.\n\n"
        "Notes:\n```\nnote\n```\n"
    )
    assert build_prompt(info, ["one", "two"], "ranking_score", False) == expected


@pytest.mark.parametrize(
    ("kind", "text", "count", "expected"),
    [
        ("ranking", "analysis\nB > A = C", 3, [0, 1, 0]),
        ("ranking", "A = B", 2, [0, 0]),
        ("ranking", "A > A", 2, None),
        ("ranking", "A < B", 2, None),
        ("ranking_score", "analysis\nB: 8, A: 2", 2, [2, 8]),
        ("ranking_score", "A: 2, B: 8, C: 9", 2, [2, 8]),
        ("ranking_score", "A: 1, A: 2, B: 8", 2, [2, 8]),
        ("ranking_score", "A: 2, C: 8", 2, None),
        ("ranking_score", "A: 2.5, B: 8", 2, None),
        ("score", "B: 8, A: 2", 2, [8, 2]),
        ("score", "A: 2, B: 8, C: 9", 2, None),
    ],
)
def test_group_score_parser_matches_legacy_rules(kind, text, count, expected):
    assert parse_scores(text, kind, count) == expected


@pytest.mark.parametrize("rm_output", ["A: 2, B: 8", "invalid RM output"])
def test_group_duplicates_share_first_length_penalty_and_scale_parse_failure(rm_output):
    class Manager(FakeGroupManager):
        async def _request(self, prompt):
            return rm_output

    cfg = _fused_config(
        extractor_type="line",
        group_prompt_type="ranking_score",
        score_scale_factor=0.01,
        default_reward=-0.04,
        overlong_buffer={"enable": True, "max_resp_len": 10, "len": 5, "penalty_factor": 0.04},
    )
    manager = Manager(cfg, FakeTokenizer(["same", "other", "same", " "]), reward_router_address="router")
    data = _batch()
    data.non_tensor_batch["uid"][:] = "g"
    data.batch["responses"] = torch.arange(4).view(4, 1).expand(4, 10)
    # Same final translation at lengths 6 and 10 shares the first one's 0.008 penalty.
    data.batch["attention_mask"] = torch.cat(
        [torch.ones(4, 1, dtype=torch.long), (torch.arange(10)[None, :] < torch.tensor([6, 2, 10, 1])[:, None]).long()],
        dim=1,
    )
    scores = [item["reward_score"] for item in asyncio.run(manager.run_batch(data))]
    expected = [0.012, 0.08, 0.012, -0.04] if rm_output.startswith("A:") else [-0.0084, -0.0004, -0.0084, -0.04]
    assert scores == pytest.approx(expected)


@pytest.mark.parametrize("fits", [False, True])
def test_group_prompt_limit_includes_chat_template(fits):
    tokenizer = FakeTokenizer()
    prompt = build_prompt({"src_text": "source", "lang_pair": "en-zh"}, ["A", "B"], "score", False)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
    limit = len(rendered) if fits else len(rendered) - 1
    assert len(prompt) < limit
    cfg = _fused_config(max_prompt_length=limit)
    manager = FakeGroupManager(cfg, tokenizer, reward_router_address="router")
    result = asyncio.run(manager.run_batch(_batch().select_idxs([0, 1])))
    assert manager.calls == int(fits)
    assert [item["reward_score"] for item in result] == pytest.approx([0.2, 0.8] if fits else [-1, -1])


def test_group_request_sends_templated_tokens_and_explicit_sampling(monkeypatch):
    captured = {}

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        async def json(self):
            return {"choices": [{"text": "analysis\nA: 2, B: 8"}]}

    class Session(Response):
        def __init__(self, **kwargs):
            pass

        def post(self, url, json):
            captured.update(url=url, payload=json)
            return Response()

    monkeypatch.setattr("verl.experimental.reward_loop.reward_manager.group.aiohttp.ClientSession", Session)
    cfg = _fused_config()
    cfg.reward.reward_model.rollout = {"temperature": 0.6, "top_p": 0.9, "top_k": -1, "response_length": 6400}
    tokenizer = FakeTokenizer()
    manager = GroupRewardManager(cfg, tokenizer, reward_router_address="router")
    assert asyncio.run(manager._request("prompt")) == "analysis\nA: 2, B: 8"
    assert captured == {
        "url": "http://router/v1/completions",
        "payload": {
            "model": "model",
            "prompt": list(range(len("<user>prompt</user><assistant>"))),
            "max_tokens": 6400,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": -1,
        },
    }


def test_group_request_failure_is_not_silently_scored():
    class Manager(FakeGroupManager):
        async def _request(self, prompt):
            raise RuntimeError("RM unavailable")

    manager = Manager(_fused_config(), FakeTokenizer(), reward_router_address="router")
    with pytest.raises(RuntimeError, match="RM unavailable"):
        asyncio.run(manager.run_batch(_batch()))
