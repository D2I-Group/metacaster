
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from .config import (
    COMPACT_RECENT_KEEP,
    COMPACT_SUMMARY_MAX_TOKENS,
    PRESERVE_RESULT_TOOLS,
    TOOL_RESULT_AGE_LIMIT,
    TOOL_RESULT_MAX_CHARS,
)


def _serialise_for_count(item: Any) -> str:
    if hasattr(item, "model_dump"):
        return json.dumps(item.model_dump(), ensure_ascii=False, default=str)
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False, default=str)
    return str(item)


def estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        total += len(_serialise_for_count(m))
    return total // 4


def should_compact(messages: list, *, threshold_tokens: int) -> bool:
    return estimate_tokens(messages) >= threshold_tokens


def _get_attr(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _item_type(item: Any) -> str | None:
    return _get_attr(item, "type")


def _build_call_id_to_name(messages: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in messages:
        if _item_type(m) == "function_call":
            cid = _get_attr(m, "call_id")
            name = _get_attr(m, "name")
            if cid and name:
                out[cid] = name
    return out


def microcompact(
    messages: list,
    *,
    current_turn: int,
    log_dir: Path,
) -> int:
    log_dir = Path(log_dir)
    call_id_to_name = _build_call_id_to_name(messages)
    folded = 0

    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("type") != "function_call_output":
            continue
        if m.get("_compacted"):
            continue
        produced_turn = m.get("_turn", 0)
        age = current_turn - produced_turn
        if age <= TOOL_RESULT_AGE_LIMIT:
            continue

        call_id = m.get("call_id")
        tool_name = call_id_to_name.get(call_id, "unknown")
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue

        original = str(m.get("output", ""))
        if not original:
            continue

        dump_dir = log_dir / "tool_results"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / f"{produced_turn:04d}_{tool_name}.txt"
        try:
            dump_path.write_text(original, encoding="utf-8")
        except OSError as exc:
            logger.warning("microcompact: dump failed for {}: {}", dump_path, exc)
            continue

        m["output"] = (
            f"[turn {produced_turn}: used {tool_name} — "
            f"full at {dump_path}]"
        )
        m["_compacted"] = True
        folded += 1

    if folded:
        logger.info(
            "[microcompact] turn {:02d}: folded {} stale tool result(s)",
            current_turn,
            folded,
        )
    return folded


def truncate_oversized(messages: list, *, log_dir: Path) -> int:
    log_dir = Path(log_dir)
    truncated = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("type") != "function_call_output":
            continue
        if m.get("_truncated"):
            continue
        out = str(m.get("output", ""))
        if len(out) <= TOOL_RESULT_MAX_CHARS:
            continue
        produced_turn = m.get("_turn", 0)
        dump_dir = log_dir / "tool_results"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / f"{produced_turn:04d}_oversized.txt"
        try:
            dump_path.write_text(out, encoding="utf-8")
        except OSError as exc:
            logger.warning("truncate_oversized: dump failed: {}", exc)
            continue
        head = out[: TOOL_RESULT_MAX_CHARS // 2]
        tail = out[-TOOL_RESULT_MAX_CHARS // 2 :]
        m["output"] = (
            head
            + f"\n\n... [TRUNCATED {len(out) - TOOL_RESULT_MAX_CHARS} chars; "
            f"full at {dump_path}] ...\n\n"
            + tail
        )
        m["_truncated"] = True
        truncated += 1
    return truncated


SUMMARY_PROMPT = """You are summarising the conversation of an HPAgent
mid-execution. Your summary will replace the early turns of the transcript so
the agent can continue working with reduced context. Preserve everything
future turns will need; condense verbose tool output.

Produce a markdown document with these EXACT sections (omit a section only if
genuinely empty for this transcript):

## 1. Goal
Restate, in two sentences, what the agent is trying to accomplish in this
session.

## 2. Accomplishments
Bullet points of completed work in chronological order. Include concrete
artefacts produced (files written, datasets generated, jobs finished).

## 3. Work in progress
What was being worked on at the moment this summary was generated. Be precise
about state — e.g. "12 training jobs submitted, 7 finished, 5 still running".

## 4. Files touched
Bullet list of file paths the agent has read, edited, or created, with a one-
line note about what changed or why it was read.

## 5. Key facts learned
The numerical values, table rows, error messages, and other concrete data
points that future turns will reference. Quote numbers verbatim — do not
round or summarise. Preserve any tool-result file pointers exactly:
``[turn N: used <tool> — full at <path>]`` MUST appear unchanged.

## 6. Decisions made
What the agent decided and the reasoning behind each decision.

## 7. Errors / blockers seen
What went wrong (if anything) and how it was handled.

## 8. Next steps
What the agent was about to do when this summary was triggered.

## 9. Open questions
Anything unresolved that the agent should still investigate.

Hard rules:
- Use only information present in the transcript. Do NOT speculate.
- File paths and absolute numbers are sacred — quote them verbatim.
- Tool result pointers ``full at <path>`` MUST be preserved verbatim so the
  agent can read the original later.
- Output ONLY the 9-section markdown document. No preamble, no postamble.
"""


def _strip_heavy(item: Any) -> Any:
    if isinstance(item, str):
        if item.startswith("data:image"):
            return "[image]"
        return item
    if isinstance(item, dict):
        return {
            k: _strip_heavy(v)
            for k, v in item.items()
            if not k.startswith("_")
        }
    if isinstance(item, list):
        return [_strip_heavy(v) for v in item]
    if hasattr(item, "model_dump"):
        return _strip_heavy(item.model_dump())
    return item


def _safe_cut_boundary(messages: list, recent_keep: int) -> int:
    n = len(messages)
    if n <= recent_keep + 1:
        return 0

    cut = n - recent_keep


    needed: set[str] = set()
    for m in messages[cut:]:
        if isinstance(m, dict) and m.get("type") == "function_call_output":
            cid = m.get("call_id")
            if cid:
                needed.add(cid)
    if not needed:
        return cut


    for i in range(cut - 1, 0, -1):
        m = messages[i]
        if _item_type(m) == "function_call":
            cid = _get_attr(m, "call_id")
            if cid in needed:
                cut = i

    return cut if cut >= 2 else 0


def full_compact(
    messages: list,
    *,
    client,
    model: str,
    log_dir: Path,
    current_turn: int,
    api_caller: Callable | None = None,
) -> list:
    log_dir = Path(log_dir)
    compact_dir = log_dir / "compact"
    compact_dir.mkdir(parents=True, exist_ok=True)

    pre_path = compact_dir / f"turn_{current_turn:04d}_pre.jsonl"
    with pre_path.open("w", encoding="utf-8") as f:
        for m in messages:
            entry = m.model_dump() if hasattr(m, "model_dump") else m
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    cut = _safe_cut_boundary(messages, COMPACT_RECENT_KEEP)
    if cut == 0:


        raise RuntimeError(
            f"[full_compact] turn {current_turn:02d}: nothing safe to "
            "summarise (recent window pulls all the way to start)."
        )

    initial = messages[0]
    early = messages[1:cut]
    recent = messages[cut:]

    early_for_summary = [_strip_heavy(m) for m in early]
    serialised = "\n".join(
        json.dumps(m, ensure_ascii=False, default=str) for m in early_for_summary
    )

    summary_input: list = [
        {
            "role": "user",
            "content": (
                "Transcript to summarise (one JSON object per line):\n\n"
                f"{serialised}"
            ),
        }
    ]

    create_kwargs = {
        "model": model,
        "instructions": SUMMARY_PROMPT,
        "input": summary_input,
        "max_output_tokens": COMPACT_SUMMARY_MAX_TOKENS,
    }

    started = time.time()
    if api_caller is not None:
        response = api_caller(lambda: client.responses.create(**create_kwargs))
    else:
        response = client.responses.create(**create_kwargs)
    elapsed = time.time() - started

    summary_text = ""
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    summary_text += part.text

    if not summary_text.strip():
        logger.error("[full_compact] empty summary returned; aborting compact")
        raise RuntimeError("empty summary from compact LLM call")

    summary_msg = {
        "role": "user",
        "content": (
            f"[CONVERSATION SUMMARY — replaces turns 1..{current_turn} except "
            f"the most recent {len(recent)} items]\n\n"
            f"{summary_text}\n\n"
            f"[END SUMMARY]"
        ),
    }

    post_path = compact_dir / f"turn_{current_turn:04d}_summary.md"
    post_path.write_text(summary_text, encoding="utf-8")

    new_messages = [initial, summary_msg, *recent]

    logger.success(
        "[full_compact] turn {:02d}: summarised {} items -> 1 summary msg "
        "({} chars) in {:.1f}s; {} recent items preserved",
        current_turn,
        len(early),
        len(summary_text),
        elapsed,
        len(recent),
    )
    return new_messages
