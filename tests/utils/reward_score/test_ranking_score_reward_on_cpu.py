import pytest

from examples.rewards.ranking_score_reward import (
    _extract_ranking_score,
    ranking_score_reward_fn,
    ranking_score_reward_fn_no_cot,
)


GROUND_TRUTH = '{"A": 10, "B": 10, "C": 8, "D": 5}'


def _response(
    ranking: str = "A = B > C > D",
    scores: str = "A: 10, B: 10, C: 8, D: 5",
) -> str:
    return f"""Detailed candidate analysis.

### Final Ranking:

{ranking}

### Scores:

{scores}
"""


def test_extract_ranking_score_matches_model_markdown_format():
    extracted = _extract_ranking_score(_response())

    assert extracted == {
        "cot_text": "Detailed candidate analysis.",
        "ranking_header": "### Final Ranking:",
        "ranking_text": "A = B > C > D",
        "score_header": "### Scores:",
        "score_text": "A: 10, B: 10, C: 8, D: 5",
    }


@pytest.mark.parametrize("ground_truth", [GROUND_TRUTH, {"A": 10, "B": 10, "C": 8, "D": 5}])
def test_ranking_score_reward_accepts_string_and_dict_ground_truth(ground_truth):
    result = ranking_score_reward_fn(
        "TowerBlocks-MT-Ranking.ranking_score",
        _response(),
        ground_truth,
        score_scale_factor=0.5,
    )

    assert result == {
        "score": pytest.approx(1.0),
        "valid_answer": 1,
        "ranking_reward": pytest.approx(1.0),
        "score_reward": pytest.approx(1.0),
    }


def test_ranking_score_reward_preserves_partial_reward_semantics():
    result = ranking_score_reward_fn(
        "TowerBlocks-MT-Ranking.ranking_score",
        _response(ranking="A > C = D > B", scores="A: 5, C: 3, D: 3, B: 1"),
        '{"A": 10, "B": 1, "C": 8, "D": 7}',
    )

    assert result["valid_answer"] == 1
    assert 0 < result["ranking_reward"] < 1
    assert 0 <= result["score_reward"] < 1
    assert result["score"] == pytest.approx(result["ranking_reward"] + result["score_reward"])


@pytest.mark.parametrize(
    "response",
    [
        "A = B > C > D\nA: 10, B: 10, C: 8, D: 5",
        _response(ranking="A > B > C > D"),
        _response(scores="A: 10, B: 10, C: 8"),
        _response(scores="A: ten, B: 10, C: 8, D: 5"),
    ],
)
def test_ranking_score_reward_rejects_invalid_responses(response):
    assert ranking_score_reward_fn("source", response, GROUND_TRUTH) == {
        "score": 0,
        "valid_answer": 0,
        "ranking_reward": 0,
        "score_reward": 0,
    }


def test_no_cot_entry_point_matches_original_two_line_contract():
    result = ranking_score_reward_fn_no_cot(
        "source",
        "A = B > C > D\nA: 10, B: 10, C: 8, D: 5",
        GROUND_TRUTH,
    )

    assert result["score"] == pytest.approx(2.0)
    assert result["valid_answer"] == 1


def test_ranking_string_ground_truth_matches_original_ranking_only_branch():
    result = ranking_score_reward_fn(
        "source",
        _response(ranking="A > B > C > D", scores="A: 10, B: 9, C: 8, D: 5"),
        "A > C > B > D",
        score_scale_factor=0.5,
    )

    assert result == {
        "score": pytest.approx(5 / 12),
        "valid_answer": 1,
        "ranking_reward": pytest.approx(5 / 6),
        "score_reward": 0,
    }
