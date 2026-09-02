# Copyright 2026 Individual Contributor: Yangs
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Rule-based ranking and score reward functions for MT training."""

from __future__ import annotations

from itertools import combinations
import json
from typing import Any, Optional, Union


def _split_last_line(text: str) -> Optional[tuple[str, str]]:
    text = text.strip()
    index = text.rfind("\n")
    if index < 0:
        return None
    return text[:index].strip(), text[index:].strip()


def parse_order(order_str: str) -> list[set[str]]:
    return [set(x.strip() for x in group.split("=")) for group in order_str.split(">")]


def pair_relation(tiers: list[set[str]], x: str, y: str) -> int:
    positions = {item: i for i, tier in enumerate(tiers) for item in tier}
    ix, iy = positions[x], positions[y]
    return 1 if ix < iy else -1 if ix > iy else 0


def compare_orderings(test_str: str, ref_str: str) -> float:
    try:
        test_tiers, ref_tiers = parse_order(test_str), parse_order(ref_str)
        items = sorted(set().union(*ref_tiers))
        if len(items) < 2:
            return 1.0 if test_tiers == ref_tiers else 0.0
        return sum(
            pair_relation(test_tiers, x, y) == pair_relation(ref_tiers, x, y)
            for x, y in combinations(items, 2)
        ) / (len(items) * (len(items) - 1) / 2)
    except (KeyError, ValueError, TypeError):
        return 0.0


def validate_ranking(test_str: str, ref_str: str) -> bool:
    try:
        if "<" in test_str:
            return False
        ref_tiers = parse_order(ref_str)
        count = sum(map(len, ref_tiers))
        if len(test_str) != (count - 1) * 3 + count:
            return False
        if sum(map(len, parse_order(test_str))) != count:
            return False
        return all(test_str.count(item) == 1 for tier in ref_tiers for item in tier)
    except (ValueError, TypeError):
        return False


def _score_to_rank(scores: dict[str, int]) -> str:
    groups: list[list[str]] = []
    for key, score in sorted(scores.items(), key=lambda x: (-x[1], x[0])):
        if not groups or scores[groups[-1][0]] != score:
            groups.append([key])
        else:
            groups[-1].append(key)
    return " > ".join(" = ".join(group) for group in groups)


def _parse_score_text(text: str) -> Optional[dict[str, int]]:
    try:
        return {k.strip(): int(v.strip()) for k, v in (item.split(":") for item in text.split(","))}
    except (ValueError, TypeError):
        return None


def _compare_scores(test: dict[str, int], ref: dict[str, int]) -> float:
    if len(ref) < 2:
        return 1.0 if test == ref else 0.0
    total = 0.0
    for x, y in combinations(ref, 2):
        error = abs((ref[x] - ref[y]) - (test[x] - test[y]))
        total += {0: 1.0, 1: 0.6, 2: 0.2}.get(error, 0.0)
    return total / (len(ref) * (len(ref) - 1) / 2)


def ranking_score_reward_fn(
    data_source: str, solution_str: str, ground_truth: Union[str, dict], extra_info: Any = None,
    score_scale_factor: float = 1.0,
) -> dict[str, Any]:
    """Score a response containing ``ranking`` and ``scores`` on its final two lines."""
    result = {"score": 0.0, "valid_answer": 0, "ranking_reward": 0.0, "score_reward": 0.0}
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth) if ground_truth.lstrip().startswith("{") else ground_truth
        except json.JSONDecodeError:
            return result
    lines = solution_str.strip().splitlines()
    if len(lines) < 3:
        return result
    ranking, score_text = lines[-2:]
    scores = _parse_score_text(score_text)
    if scores is None or not isinstance(ground_truth, dict) or set(scores) != set(ground_truth):
        return result
    predicted = _score_to_rank(scores)
    if ranking != predicted or not validate_ranking(ranking, predicted):
        return result
    result["ranking_reward"] = compare_orderings(predicted, _score_to_rank(ground_truth))
    result["score_reward"] = _compare_scores(scores, ground_truth)
    result["score"] = (result["ranking_reward"] + result["score_reward"]) * score_scale_factor
    result["valid_answer"] = 1
    return result


compute_score = ranking_score_reward_fn
