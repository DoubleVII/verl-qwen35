"""Core helpers for group generative translation rewards."""

from __future__ import annotations

import json
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
_FUSED_SIMPLE_SEPARATOR = "---"
_FUSED_SIMPLE_CONNECTOR = "Now, review the candidates and produce the best final translation."
_FUSED_SIMPLE_ANALYSIS_HEADING = "# Step-by-step Analysis"


def normalize_fused_candidate(text: str) -> str:
    return " ".join(text.split()).casefold()


def _parse_fused_flash_gpe_sections(text: str | None) -> tuple[str, str] | None:
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
    return candidate_response, post_edit_response


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()
    start = text.find("{")
    if start == -1:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def parse_fused_flash_gpe_response(text: str | None) -> tuple[list[str], str] | None:
    """Parse JSON candidates and the final translation from the fused protocol."""
    sections = _parse_fused_flash_gpe_sections(text)
    if sections is None:
        return None
    candidate_response, post_edit_response = sections

    payload = _extract_json_object(candidate_response)
    raw_candidates = payload.get("translations") if payload is not None else None
    if not isinstance(raw_candidates, list):
        return None
    if any(not isinstance(candidate, str) or not candidate.strip() for candidate in raw_candidates):
        return None
    candidates = [candidate.strip() for candidate in raw_candidates]
    final_translation = extract_response(post_edit_response, "codeblock")
    if final_translation is None:
        return None
    return candidates, final_translation


def parse_fused_flash_gpe_markdown_response(text: str | None) -> tuple[list[str], str] | None:
    """Parse Markdown candidates and the final translation from the fused protocol."""
    sections = _parse_fused_flash_gpe_sections(text)
    if sections is None:
        return None
    candidate_response, post_edit_response = sections

    candidates = _extract_fused_markdown_candidates(candidate_response)
    if candidates is None:
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


def _extract_fused_markdown_candidates(response: str) -> list[str] | None:
    matches = list(re.finditer(r"(?m)^# Candidate ([1-9][0-9]*)[ \t]*$", response))
    if not matches:
        return None
    candidates = []
    for index, candidate_match in enumerate(matches):
        if int(candidate_match.group(1)) != index + 1:
            return None
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        candidate = response[candidate_match.end() : end].strip()
        if not candidate or "```" in candidate:
            return None
        candidates.append(candidate)
    if len({normalize_fused_candidate(candidate) for candidate in candidates}) != len(candidates):
        return None
    return candidates


def _split_fused_simple_analysis_section(text: str, body_pattern: str) -> tuple[str, str] | None:
    text = text.strip()
    analysis_match = re.match(rf"^{re.escape(_FUSED_SIMPLE_ANALYSIS_HEADING)}[ \t]*\n", text)
    if analysis_match is None:
        return None
    body_match = re.search(body_pattern, text, re.MULTILINE)
    if body_match is None:
        return None
    analysis = text[analysis_match.end() : body_match.start()].strip()
    response = text[body_match.start() :].strip()
    if not analysis or not response:
        return None
    return analysis, response


def parse_fused_flash_gpe_simple_markdown_response(text: str | None) -> tuple[list[str], str] | None:
    """Parse the visible-analysis simple Markdown Fused FlashGPE protocol."""
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n")
    connector_matches = list(
        re.finditer(
            rf"(?m)^[ \t]*{re.escape(_FUSED_SIMPLE_SEPARATOR)}[ \t]*\n{{2,}}"
            rf"[ \t]*{re.escape(_FUSED_SIMPLE_CONNECTOR)}[ \t]*$",
            text,
        )
    )
    if len(connector_matches) != 1:
        return None

    connector_match = connector_matches[0]
    candidate_stage = text[: connector_match.start()]
    post_edit_stage = text[connector_match.end() :]
    if not candidate_stage.endswith("\n\n") or not post_edit_stage.startswith("\n\n"):
        return None
    candidate_sections = _split_fused_simple_analysis_section(candidate_stage, r"^# Candidate 1[ \t]*$")
    post_edit_sections = _split_fused_simple_analysis_section(post_edit_stage, r"^# Final Translation[ \t]*$")
    if candidate_sections is None or post_edit_sections is None:
        return None

    _, candidate_response = candidate_sections
    _, post_edit_response = post_edit_sections
    candidates = _extract_fused_markdown_candidates(candidate_response)
    if candidates is None or post_edit_response.count("# Final Translation") != 1:
        return None
    final_section = post_edit_response[len("# Final Translation") :].strip()
    if not final_section.startswith("```"):
        return None
    final_translation = extract_response(final_section, "codeblock")
    if not final_translation or final_translation.startswith("# ") or "```" in final_translation:
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

    if max_candidates is None:
        return False

    prompt_type = extra_info.get("prompt_type")
    if prompt_type == "markdown":
        return max_candidates >= 2 and candidate_count == max_candidates
    if prompt_type == "fixed_4":
        return max_candidates == 4 and candidate_count == 4
    if prompt_type == "fixed_16":
        return max_candidates == 16 and candidate_count == 16
    if prompt_type == "adaptive":
        return max_candidates >= 2 and 2 <= candidate_count <= max_candidates
    return False


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
    if kind == "none":
        return text
    if not text:
        return None
    if kind == "line":
        return text.split("\n")[-1].strip() or None
    if kind == "oneline":
        return text if "\n" not in text else None
    if kind == "codeblock":
        if text.count("```") != 2 or not text.endswith("```"):
            return None
        block = text[:-3]
        block = block[block.rfind("```") + 3 :]
        return block.split("\n", 1)[-1].strip() or None
    if kind == "markdown":
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.strip() == "# Final Translation":
                return "\n".join(lines[index + 1 :]).strip() or None
        return None
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
    """Render the original GQM prompt, including whitespace and optional context."""
    if len(candidates) == 1 or len(candidates) > len(IDENTIFIERS):
        raise ValueError(f"GQM requires multiple candidates, up to {len(IDENTIFIERS)}")
    src, tgt = language_pair(info)
    src, tgt = LANG_MAP.get(src, src), LANG_MAP.get(tgt, tgt)
    task = {
        "score": "Finally, score the candidates with integer scores on a scale from 0 to 10.",
        "ranking": "Finally, rank the candidates in order of quality from best to worst.",
        "ranking_score": "Finally, rank and score the candidates with integer scores on a scale from 0 to 10.",
    }.get(prompt_type)
    if task is None:
        raise ValueError(f"Unsupported group_prompt_type: {prompt_type}")
    if add_example:
        example = {
            "score": "Output the scores on the last line, for example: `A: 4, B: 9, C: 7, D: 9`.",
            "ranking": "Output the rankings in descending order on the last line, for example: `B > A = D > C`.",
            "ranking_score": (
                "At the end section, first output the rankings in descending order, for example: `B > A = D > C`. "
                "Then, on the last line, output the scores, for example: `B: 9, A: 7, D: 7, C: 2`."
            ),
        }[prompt_type]
        task += f" {example}"
    body = "".join(f"Translation {IDENTIFIERS[i]}:\n```\n{text}\n```\n" for i, text in enumerate(candidates))
    extra = ""
    ref_text, ref_lang = (info.get("ref_text") or "").strip(), (info.get("ref_lang") or "").strip()
    if ref_text and ref_lang:
        extra += (
            "\n\nYou may refer to the following reference, if helpful, when evaluating the translations.\n\n"
            f"{LANG_MAP.get(ref_lang, ref_lang)} reference:\n```\n{ref_text}\n```\n"
        )
    notes = (info.get("notes") or "").strip()
    if notes:
        extra += (
            "\n\nYou may refer to the following notes, if helpful, when evaluating the translations.\n\n"
            f"Notes:\n```\n{notes}\n```\n"
        )
    return (
        f"Given a source text in {src} and multiple translation candidates in {tgt}. "
        f"Perform a step by step analysis and comparison of the translation quality for the candidates. {task}\n\n"
        f"Source text:\n```\n{info['src_text']}\n```\n\n{body}{extra}"
    )


def parse_scores(text: str, prompt_type: str, count: int) -> list[int] | None:
    """Match the original GQM last-line parser and ranking score scale."""
    text = (text or "").strip()
    if not text:
        return None
    score_line = text.split("\n")[-1].strip()
    if prompt_type == "ranking":
        if "<" in score_line:
            return None
        ranking = score_line.split(">")
        if sum(len({name.strip() for name in tier.split("=")}) for tier in ranking) != count:
            return None
        if any(score_line.count(name) != 1 for name in IDENTIFIERS[:count]):
            return None
        # Legacy ranking rewards are tier-based: the worst tier gets zero.
        mapped = {name.strip(): len(ranking) - 1 - i for i, tier in enumerate(ranking) for name in tier.split("=")}
        return (
            [mapped[name] for name in IDENTIFIERS[:count]]
            if all(name in mapped for name in IDENTIFIERS[:count])
            else None
        )

    if prompt_type == "score":
        try:
            scores = [int(item.strip().split(":")[-1].strip()) for item in score_line.split(",")]
        except (AttributeError, TypeError, ValueError):
            return None
        return scores if len(scores) == count else None

    if prompt_type != "ranking_score":
        return None

    # Legacy GQM requires the expected identifiers but tolerates extra identifiers;
    # duplicate identifiers take their last value. Do not tighten this during migration.
    try:
        scores = {}
        for item in score_line.strip().split(","):
            candidate_identifier, score = item.strip().split(":")
            scores[candidate_identifier.strip()] = int(score.strip())
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(name in scores for name in IDENTIFIERS[:count]):
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
