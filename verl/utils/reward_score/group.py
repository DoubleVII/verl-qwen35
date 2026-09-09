"""Core helpers for group generative translation rewards."""

from __future__ import annotations

import re
from typing import Any

LANG_MAP = {
    "ar": "Arabic",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
IDENTIFIERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_FUSED_THINKING_OPEN = "<thinking>"
_FUSED_THINKING_CLOSE = "</thinking>"
_FUSED_RESPONSE_OPEN = "<response>"
_FUSED_RESPONSE_CLOSE = "</response>"
_FUSED_FLASH_GPE_PATTERN = re.compile(
    rf"\s*{re.escape(_FUSED_THINKING_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(_FUSED_THINKING_CLOSE)}\s*"
    rf"{re.escape(_FUSED_RESPONSE_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(_FUSED_RESPONSE_CLOSE)}\s*"
    rf"{re.escape(_FUSED_THINKING_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(_FUSED_THINKING_CLOSE)}\s*"
    rf"{re.escape(_FUSED_RESPONSE_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(_FUSED_RESPONSE_CLOSE)}\s*",
    re.DOTALL,
)


def normalize_fused_candidate(text: str) -> str:
    return " ".join(text.split()).casefold()


def parse_fused_flash_gpe_markdown_response(text: str | None) -> tuple[list[str], str] | None:
    """Parse candidate generation and post-edit sections from the fused Markdown protocol."""
    if not isinstance(text, str):
        return None
    tags = (_FUSED_THINKING_OPEN, _FUSED_THINKING_CLOSE, _FUSED_RESPONSE_OPEN, _FUSED_RESPONSE_CLOSE)
    if any(text.count(tag) != 2 for tag in tags):
        return None
    match = _FUSED_FLASH_GPE_PATTERN.fullmatch(text)
    if match is None:
        return None
    candidate_thinking, candidate_response, post_edit_thinking, post_edit_response = (
        value.strip() for value in match.groups()
    )
    if not all((candidate_thinking, candidate_response, post_edit_thinking, post_edit_response)):
        return None

    matches = list(re.finditer(r"(?m)^# Candidate ([1-9][0-9]*)[ \t]*$", candidate_response))
    if not matches:
        return None
    candidates = []
    for index, candidate_match in enumerate(matches):
        if int(candidate_match.group(1)) != index + 1:
            return None
        end = matches[index + 1].start() if index + 1 < len(matches) else len(candidate_response)
        candidate = candidate_response[candidate_match.end() : end].strip()
        if not candidate or "```" in candidate:
            return None
        candidates.append(candidate)
    if len({normalize_fused_candidate(candidate) for candidate in candidates}) != len(candidates):
        return None

    marker = "# Final Translation"
    marker_index = post_edit_response.find(marker)
    if marker_index != -1:
        final_translation = post_edit_response[marker_index + len(marker) :].strip()
        if final_translation.startswith("# ") or "```" in final_translation:
            return None
    else:
        final_translation = extract_response(post_edit_response, "codeblock")
    if not final_translation:
        return None
    return candidates, final_translation


def is_valid_fused_candidate_count(extra_info: Any, candidate_count: int) -> bool:
    if not isinstance(extra_info, dict):
        return False
    target_candidate_count = extra_info.get("target_candidate_count")
    try:
        max_candidates = int(extra_info.get("max_candidates"))
    except (TypeError, ValueError):
        max_candidates = None

    if target_candidate_count is not None:
        try:
            target_candidate_count = int(target_candidate_count)
        except (TypeError, ValueError):
            return False
        if target_candidate_count < 2:
            return False
        if max_candidates is not None and target_candidate_count > max_candidates:
            return False
        return candidate_count == target_candidate_count

    return (
        extra_info.get("prompt_type") == "markdown"
        and max_candidates is not None
        and max_candidates >= 2
        and candidate_count == max_candidates
    )


def myers_insert_delete_distance(left: list[int], right: list[int]) -> int:
    """Return the shortest insert/delete edit distance using Myers' algorithm."""
    if not left:
        return len(right)
    if not right:
        return len(left)

    frontier = {1: 0}
    for distance in range(len(left) + len(right) + 1):
        next_frontier = {}
        for diagonal in range(-distance, distance + 1, 2):
            if diagonal == -distance or (
                diagonal != distance and frontier.get(diagonal - 1, -1) < frontier.get(diagonal + 1, -1)
            ):
                x = frontier.get(diagonal + 1, 0)
            else:
                x = frontier.get(diagonal - 1, 0) + 1
            y = x - diagonal
            while x < len(left) and y < len(right) and left[x] == right[y]:
                x += 1
                y += 1
            next_frontier[diagonal] = x
            if x >= len(left) and y >= len(right):
                return distance
        frontier = next_frontier
    return len(left) + len(right)


def token_myers_diversity(candidates: list[str], tokenizer: Any) -> float:
    tokenized = [list(tokenizer.encode(candidate, add_special_tokens=False)) for candidate in candidates]
    distances = []
    for left_index, left in enumerate(tokenized):
        for right in tokenized[left_index + 1 :]:
            denominator = len(left) + len(right)
            distance = myers_insert_delete_distance(left, right)
            distances.append(distance / denominator if denominator else 0.0)
    return sum(distances) / len(distances) if distances else 0.0


def is_language_match(text: str, target_lang: str) -> bool:
    """Return whether text matches the configured target language when lingua is available."""
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError as exc:
        raise ImportError("enable_language_detection=True requires the lingua-language-detector package") from exc
    languages = {
        "en": Language.ENGLISH,
        "zh": Language.CHINESE,
        "de": Language.GERMAN,
        "ru": Language.RUSSIAN,
        "ko": Language.KOREAN,
        "fr": Language.FRENCH,
        "es": Language.SPANISH,
        "pt": Language.PORTUGUESE,
        "it": Language.ITALIAN,
        "nl": Language.DUTCH,
    }
    expected = languages.get(str(target_lang).lower())
    if expected is None or not text.strip():
        return True
    detector = LanguageDetectorBuilder.from_languages(*languages.values()).build()
    return detector.detect_language_of(text) == expected


def extract_response(text: str, kind: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    if kind == "none":
        return text
    if kind == "line":
        return text.splitlines()[-1].strip() or None
    if kind == "oneline":
        return text if "\n" not in text else None
    if kind == "codeblock":
        if text.count("```") != 2 or not text.endswith("```"):
            return None
        block = text[:-3]
        block = block[block.rfind("```") + 3 :]
        return block.split("\n", 1)[-1].strip() or None
    raise ValueError(f"Unknown extractor_type: {kind}")


def language_pair(info: dict[str, Any]) -> tuple[str, str]:
    if "src_lang" in info and "trg_lang" in info:
        return str(info["src_lang"]), str(info["trg_lang"])
    if "lang_pair" in info:
        parts = str(info["lang_pair"]).split("-", 1)
        if len(parts) == 2:
            return tuple(parts)
    raise ValueError(f"extra_info must contain a language pair: {info}")


def build_prompt(info: dict[str, Any], candidates: list[str], prompt_type: str, add_example: bool) -> str:
    src, tgt = language_pair(info)
    src, tgt = LANG_MAP.get(src, src), LANG_MAP.get(tgt, tgt)
    task = {
        "score": "score the candidates with integer scores on a scale from 0 to 10",
        "ranking": "rank the candidates in order of quality from best to worst",
        "ranking_score": "rank and score the candidates with integer scores on a scale from 0 to 10",
    }.get(prompt_type)
    if task is None:
        raise ValueError(f"Unsupported group_prompt_type: {prompt_type}")
    example = " For example, use `B > A = C` and `B: 9, A: 7, C: 7`." if add_example else ""
    body = "\n\n".join(f"Translation {IDENTIFIERS[i]}:\n```\n{text}\n```" for i, text in enumerate(candidates))
    extra = ""
    if info.get("notes"):
        extra += f"\n\nNotes:\n```\n{str(info['notes']).strip()}\n```"
    if info.get("ref_text") and info.get("ref_lang"):
        extra += f"\n\n{LANG_MAP.get(str(info['ref_lang']), info['ref_lang'])} reference:\n```\n{info['ref_text']}\n```"
    return (
        f"Given a source text in {src} and multiple translation candidates in {tgt}. "
        f"Perform a step by step analysis and comparison of translation quality, then finally {task}.{example}\n\n"
        f"Source text:\n```\n{info['src_text']}\n```\n\n{body}{extra}"
    )


def parse_scores(text: str, prompt_type: str, count: int) -> list[int] | None:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    score_line = lines[-1]
    if prompt_type == "ranking":
        ranking = score_line.split(">")
        if sum(len(t.split("=")) for t in ranking) != count:
            return None
        mapped = {name.strip(): count - i for i, tier in enumerate(ranking) for name in tier.split("=")}
        return (
            [mapped.get(IDENTIFIERS[i], -1) for i in range(count)] if set(mapped) == set(IDENTIFIERS[:count]) else None
        )

    if prompt_type == "score":
        try:
            scores = [int(item.strip().split(":")[-1].strip()) for item in score_line.split(",")]
        except (AttributeError, TypeError, ValueError):
            return None
        return scores if len(scores) == count else None

    if prompt_type != "ranking_score":
        return None

    # Keep this parser in lockstep with examples/rewards/ranking_score_reward.py:
    # GQM uses the final non-empty line as the score line and does not require
    # the preceding analysis/ranking text to have a particular shape.
    try:
        scores = {}
        for item in score_line.strip().split(","):
            candidate_identifier, score = item.strip().split(":")
            scores[candidate_identifier.strip()] = int(score.strip())
    except (AttributeError, TypeError, ValueError):
        return None
    if len(scores) != count or set(scores) != set(IDENTIFIERS[:count]):
        return None
    return [scores[IDENTIFIERS[i]] for i in range(count)]


def overlong_penalty(length: int, cfg: Any) -> float:
    if not cfg or not cfg.get("enable", False):
        return 0.0
    max_len, buffer = cfg.get("max_resp_len"), cfg.get("len", 0)
    factor = cfg.get("penalty_factor", 0.0)
    if max_len is None or buffer <= 0 or factor <= 0 or length <= max_len - buffer:
        return 0.0
    return min((length - max_len + buffer) / buffer, 1.0) * factor
