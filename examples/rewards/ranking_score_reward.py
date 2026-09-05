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
    last_line_index = text.rfind("\n")
    if last_line_index == -1:
        return None
    last_line = text[last_line_index:].strip()
    residual_text = text[:last_line_index].strip()
    return residual_text, last_line


def parse_order(order_str: str) -> list[set[str]]:
    tiers = []
    for group in order_str.split(">"):
        tiers.append(set(x.strip() for x in group.split("=")))
    return tiers


def pair_relation(tiers: list[set[str]], x: str, y: str) -> int:
    positions = {item: i for i, tier in enumerate(tiers) for item in tier}
    ix, iy = positions[x], positions[y]
    if ix < iy:
        return 1
    if ix > iy:
        return -1
    return 0


def compare_orderings(test_str: str, ref_str: str) -> float:
    try:
        test_tiers = parse_order(test_str)
        ref_tiers = parse_order(ref_str)
        items = sorted(set().union(*test_tiers))

        total = 0
        score = 0
        for x, y in combinations(items, 2):
            r_ref = pair_relation(ref_tiers, x, y)
            r_test = pair_relation(test_tiers, x, y)
            total += 1
            if r_ref == r_test:
                score += 1
            elif 0 in (r_ref, r_test):
                score += 0  # Ties are treated as incorrect.
        return score / total
    except Exception:
        return 0


def validate_ranking(test_str: str, ref_str: str) -> bool:
    try:
        if "<" in test_str:
            return False
        ref_tiers = parse_order(ref_str)
        ref_count = sum(len(tier) for tier in ref_tiers)
        if len(test_str) != (ref_count - 1) * 3 + ref_count:
            return False

        test_tiers = parse_order(test_str)
        if sum(len(tier) for tier in test_tiers) != ref_count:
            return False
        for tier in ref_tiers:
            for candidate_identifier in tier:
                if test_str.count(candidate_identifier) != 1:
                    return False
        return True
    except Exception:
        return False


def ranking_reward_fn_no_cot(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    score_scale_factor: float = 1.0,
) -> dict[str, Union[int, float]]:
    pred_ranking_str = solution_str.strip()
    if not validate_ranking(pred_ranking_str, ground_truth):
        return {"score": 0, "valid_answer": 0}
    if solution_str.count(pred_ranking_str) != 1:
        return {"score": 0, "valid_answer": 0}

    base_score = compare_orderings(pred_ranking_str, ground_truth)
    return {
        "score": float(base_score) * float(score_scale_factor),
        "valid_answer": 1,
    }


def _score_to_rank(scores: dict[str, int]) -> str:
    sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    result_parts = []
    current_score = None
    current_group = []
    for candidate_identifier, score in sorted_items:
        if score != current_score:
            if current_group:
                result_parts.append(" = ".join(current_group))
            current_score = score
            current_group = [candidate_identifier]
        else:
            current_group.append(candidate_identifier)
    if current_group:
        result_parts.append(" = ".join(current_group))
    return " > ".join(result_parts)


def compare_ranking_scores(
    test_score_dict: dict[str, int],
    ref_score_dict: dict[str, int],
    error_to_reward: Optional[dict[int, float]] = None,
) -> float:
    if not error_to_reward:
        error_to_reward = {0: 1, 1: 0.6, 2: 0.2}
    try:
        candidate_items = list(ref_score_dict.keys())
        total = 0
        score = 0.0
        for x, y in combinations(candidate_items, 2):
            margin_ref = ref_score_dict[x] - ref_score_dict[y]
            margin_test = test_score_dict[x] - test_score_dict[y]
            total += 1
            margin_error = abs(margin_ref - margin_test)
            if margin_error in error_to_reward:
                score += error_to_reward[margin_error]
        return score / total
    except Exception:
        return 0


def _extract_ranking_score(output_text: str) -> Optional[dict[str, str]]:
    try:
        residual_text, score_text = _split_last_line(output_text)
        residual_text, score_header = _split_last_line(residual_text)
        residual_text, ranking_text = _split_last_line(residual_text)
        residual_text, ranking_header = _split_last_line(residual_text)
        return {
            "cot_text": residual_text,
            "score_text": score_text,
            "ranking_text": ranking_text,
            "score_header": score_header,
            "ranking_header": ranking_header,
        }
    except (TypeError, ValueError):
        return None


def _parse_score_text(score_text: str) -> Optional[dict[str, int]]:
    """Parse a score line such as B: 6, A: 5, C: 2."""
    try:
        score_dict = {}
        for item in score_text.strip().split(","):
            candidate_identifier, score = item.strip().split(":")
            score_dict[candidate_identifier.strip()] = int(score.strip())
        return score_dict
    except (AttributeError, TypeError, ValueError):
        return None


def ranking_score_reward_fn_no_cot(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, dict[str, int]],
    extra_info: Any = None,
    score_scale_factor: float = 1.0,
) -> dict[str, Union[int, float]]:
    reward_out: dict[str, Union[int, float]] = {
        "score": 0,
        "valid_answer": 0,
        "ranking_reward": 0,
        "score_reward": 0,
    }
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return reward_out

    solution_str = solution_str.strip()
    if solution_str.count("\n") != 1:
        return reward_out
    ranking_text, score_text = solution_str.split("\n")

    pred_score_dict = _parse_score_text(score_text)
    if pred_score_dict is None:
        return reward_out

    pred_score_to_rank = _score_to_rank(pred_score_dict)
    consistency_check = ranking_reward_fn_no_cot(
        data_source=data_source,
        solution_str=ranking_text,
        ground_truth=pred_score_to_rank,
        extra_info=extra_info,
        score_scale_factor=1.0,
    )
    if consistency_check["score"] != 1:
        return reward_out

    if not isinstance(ground_truth, dict) or len(pred_score_dict) != len(ground_truth):
        return reward_out
    for candidate_identifier in pred_score_dict:
        if candidate_identifier not in ground_truth:
            return reward_out

    ref_score_to_rank = _score_to_rank(ground_truth)
    ranking_reward = compare_orderings(pred_score_to_rank, ref_score_to_rank)
    score_reward = compare_ranking_scores(pred_score_dict, ground_truth)
    reward = ranking_reward + score_reward
    reward_out["score"] = float(reward) * float(score_scale_factor)
    reward_out["valid_answer"] = 1
    reward_out["ranking_reward"] = ranking_reward
    reward_out["score_reward"] = score_reward
    return reward_out


def ranking_reward_fn_no_cot_ranking_score_response(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    score_scale_factor: float = 1.0,
) -> dict[str, Union[int, float]]:
    reward_out: dict[str, Union[int, float]] = {
        "score": 0,
        "valid_answer": 0,
        "ranking_reward": 0,
        "score_reward": 0,
    }
    if ground_truth.strip().startswith("{"):
        return reward_out

    solution_str = solution_str.strip()
    if solution_str.count("\n") != 1:
        return reward_out
    ranking_text, score_text = solution_str.split("\n")

    pred_score_dict = _parse_score_text(score_text)
    if pred_score_dict is None:
        return reward_out

    pred_score_to_rank = _score_to_rank(pred_score_dict)
    consistency_check = ranking_reward_fn_no_cot(
        data_source,
        ranking_text,
        pred_score_to_rank,
        extra_info,
        score_scale_factor=1.0,
    )
    if consistency_check["score"] != 1:
        return reward_out
    if not validate_ranking(pred_score_to_rank, ground_truth):
        return reward_out

    ranking_reward = compare_orderings(pred_score_to_rank, ground_truth)
    reward_out["score"] = float(ranking_reward) * float(score_scale_factor)
    reward_out["valid_answer"] = 1
    reward_out["ranking_reward"] = ranking_reward
    return reward_out


def ranking_score_reward_fn(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, dict[str, int]],
    extra_info: Any = None,
    score_scale_factor: float = 1.0,
) -> dict[str, Union[int, float]]:
    reward_out: dict[str, Union[int, float]] = {
        "score": 0,
        "valid_answer": 0,
        "ranking_reward": 0,
        "score_reward": 0,
    }
    extract_out = _extract_ranking_score(solution_str)
    if extract_out is None:
        return reward_out

    cot_text = extract_out["cot_text"]
    score_text = extract_out["score_text"]
    ranking_text = extract_out["ranking_text"]
    if len(cot_text) == 0:
        return reward_out

    no_cot_solution_str = f"{ranking_text}\n{score_text}"
    if isinstance(ground_truth, dict) or ground_truth.strip().startswith("{"):
        return ranking_score_reward_fn_no_cot(
            data_source,
            no_cot_solution_str,
            ground_truth,
            extra_info,
            score_scale_factor=score_scale_factor,
        )
    return ranking_reward_fn_no_cot_ranking_score_response(
        data_source,
        no_cot_solution_str,
        ground_truth,
        extra_info,
        score_scale_factor=score_scale_factor,
    )


compute_score = ranking_score_reward_fn
