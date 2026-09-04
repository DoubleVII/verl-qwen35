"""Core helpers for group generative translation rewards."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

LANG_MAP = {"ar":"Arabic","de":"German","el":"Greek","en":"English","es":"Spanish","fr":"French","it":"Italian","ja":"Japanese","ko":"Korean","nl":"Dutch","pt":"Portuguese","ro":"Romanian","ru":"Russian","th":"Thai","uk":"Ukrainian","vi":"Vietnamese","zh":"Chinese"}
IDENTIFIERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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
    task = {"score":"score the candidates with integer scores on a scale from 0 to 10",
            "ranking":"rank the candidates in order of quality from best to worst",
            "ranking_score":"rank and score the candidates with integer scores on a scale from 0 to 10"}.get(prompt_type)
    if task is None:
        raise ValueError(f"Unsupported group_prompt_type: {prompt_type}")
    example = " For example, use `B > A = C` and `B: 9, A: 7, C: 7`." if add_example else ""
    body = "\n\n".join(f"Translation {IDENTIFIERS[i]}:\n```\n{text}\n```" for i, text in enumerate(candidates))
    extra = ""
    if info.get("notes"):
        extra += f"\n\nNotes:\n```\n{str(info['notes']).strip()}\n```"
    if info.get("ref_text") and info.get("ref_lang"):
        extra += f"\n\n{LANG_MAP.get(str(info['ref_lang']), info['ref_lang'])} reference:\n```\n{info['ref_text']}\n```"
    return (f"Given a source text in {src} and multiple translation candidates in {tgt}. "
            f"Perform a step by step analysis and comparison of translation quality, then finally {task}.{example}\n\n"
            f"Source text:\n```\n{info['src_text']}\n```\n\n{body}{extra}")


def parse_scores(text: str, prompt_type: str, count: int) -> list[int] | None:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    score_line = lines[-1]
    try:
        scores = {k.strip(): int(v.strip()) for k, v in (part.split(":", 1) for part in score_line.split(","))}
    except (ValueError, TypeError):
        scores = {}
    if prompt_type == "ranking":
        ranking = score_line.split(">")
        if sum(len(t.split("=")) for t in ranking) != count:
            return None
        mapped = {name.strip(): count - i for i, tier in enumerate(ranking) for name in tier.split("=")}
        return [mapped.get(IDENTIFIERS[i], -1) for i in range(count)] if set(mapped) == set(IDENTIFIERS[:count]) else None
    if set(scores) != set(IDENTIFIERS[:count]) or len(scores) != count:
        return None
    if prompt_type == "ranking_score" and len(lines) >= 2:
        ranking = lines[-2].split(">")
        tiers = [{x.strip() for x in tier.split("=")} for tier in ranking]
        score_tiers = [{k for k, v in scores.items() if v == value} for value in sorted(set(scores.values()), reverse=True)]
        if tiers != score_tiers:
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
