
import json
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from .compact import (
    full_compact,
    microcompact,
    should_compact,
    truncate_oversized,
)
from .config import (
    COMPACT_MAX_FAILURES,
    COMPACT_THRESHOLD_TOKENS,
    HP_REASONING_EFFORT,
    MAX_CONTINUATIONS,
    MAX_EMPTY,
)
from .image import ImageResult, build_content_with_images

_LOG_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}"

_MAX_LOG = 2000
_BANNER = "=" * 60
_MAX_TOOL_OUTPUT_CHARS = 16000


def call_with_retry(
    client_call: Callable,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
):
    retryable = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )
    for attempt in range(max_retries):
        try:
            return client_call()
        except retryable as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            sys.stderr.write(
                f"[retry] {type(e).__name__}: sleeping {delay}s before "
                f"retry {attempt + 1}/{max_retries}\n"
            )
            time.sleep(delay)
        except APIError as e:
            status = getattr(e, "status_code", None)
            if (
                status is not None
                and 500 <= int(status) < 600
                and attempt < max_retries - 1
            ):
                delay = base_delay * (2**attempt)
                sys.stderr.write(
                    f"[retry] APIError {status}: sleeping {delay}s before "
                    f"retry {attempt + 1}/{max_retries}\n"
                )
                time.sleep(delay)
                continue
            raise


def _cut(text: str) -> str:
    if len(text) <= _MAX_LOG:
        return text
    return text[:_MAX_LOG] + f" ... [{len(text)} chars]"


def _strip_internal_keys(messages: list) -> list:
    out: list = []
    for m in messages:
        if isinstance(m, dict):
            out.append({k: v for k, v in m.items() if not k.startswith("_")})
        else:
            out.append(m)
    return out


def _strip_base64(obj):
    if isinstance(obj, str):
        return "[base64 image omitted]" if obj.startswith("data:image") else obj
    if isinstance(obj, dict):
        return {k: _strip_base64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_base64(v) for v in obj]
    return obj


def make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"{ts}_{uid}"


def _dump_context(messages: list, log_dir: Path) -> None:
    path = log_dir / "conversation.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            entry = msg.model_dump() if hasattr(msg, "model_dump") else msg
            f.write(json.dumps(_strip_base64(entry), ensure_ascii=False) + "\n")
    logger.info("Context saved: {}", path)


def _extract_text(response) -> str:
    parts = []
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    parts.append(part.text)
    return "".join(parts)


def agent_loop(
    user_input: str,
    system: str,
    *,
    client: OpenAI,
    model: str,
    max_turns: int,
    tools: list,
    tool_handlers: dict,
    log_dir: Path,
    done_check: Callable[[], bool],
    images: list[str] | None = None,
) -> str:
    run_id = make_run_id()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _sink_id = logger.add(
        log_dir / f"run_{run_id}.log",
        encoding="utf-8",
        format=_LOG_FMT,
    )


    if images:
        content = build_content_with_images(user_input, images)
        messages: list = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": user_input}]

    continuations = 0
    empty_responses = 0
    compact_failures = 0

    logger.info(
        "{}\nOPTIMIZER START | run_id={} | model={} | max_turns={}\nUser: {}\n{}",
        _BANNER,
        run_id,
        model,
        max_turns,
        _cut(user_input),
        _BANNER,
    )

    for turn in range(1, max_turns + 1):

        if should_compact(messages, threshold_tokens=COMPACT_THRESHOLD_TOKENS):
            try:
                messages = full_compact(
                    messages,
                    client=client,
                    model=model,
                    log_dir=log_dir,
                    current_turn=turn,
                    api_caller=call_with_retry,
                )
                compact_failures = 0
            except Exception as exc:
                compact_failures += 1
                logger.error(
                    "[Turn {:02d}] full_compact failed ({}/{}): {}",
                    turn,
                    compact_failures,
                    COMPACT_MAX_FAILURES,
                    exc,
                )
                if compact_failures >= COMPACT_MAX_FAILURES:
                    _dump_context(messages, log_dir)
                    logger.remove(_sink_id)
                    raise RuntimeError(
                        f"Auto-compact failed {COMPACT_MAX_FAILURES} times "
                        f"in a row at turn {turn}; aborting round."
                    ) from exc

        create_kwargs: dict = {
            "model": model,
            "instructions": system,
            "input": _strip_internal_keys(messages),
            "tools": tools,
        }
        if HP_REASONING_EFFORT:
            create_kwargs["reasoning"] = {"effort": HP_REASONING_EFFORT}

        response = call_with_retry(
            lambda kw=create_kwargs: client.responses.create(**kw)
        )
        logger.debug("[Turn {:02d}] Usage: {}", turn, response.usage)

        llm_text = _extract_text(response)
        if llm_text:
            logger.info("[Turn {:02d}] LLM: {}", turn, _cut(llm_text))

        messages.extend(response.output)

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        hosted_tool_calls = [
            item for item in response.output if item.type in ("web_search_call",)
        ]

        if hosted_tool_calls and not function_calls:
            logger.info("[Turn {:02d}] Hosted tool (web_search), continuing", turn)
            continue


        if not function_calls:
            text = llm_text

            if done_check():
                logger.success("[Turn {:02d}] DONE — round marked complete", turn)
                _dump_context(messages, log_dir)
                logger.remove(_sink_id)
                return text or "(no text response)"

            if not text and empty_responses < MAX_EMPTY:
                empty_responses += 1
                logger.warning(
                    "[Turn {:02d}] Nudge: empty response ({}/{})",
                    turn,
                    empty_responses,
                    MAX_EMPTY,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Continue working on the task — call a tool to proceed.",
                    }
                )
                continue

            if continuations < MAX_CONTINUATIONS:
                continuations += 1
                logger.warning(
                    "[Turn {:02d}] Nudge: round not complete ({}/{})",
                    turn,
                    continuations,
                    MAX_CONTINUATIONS,
                )
                nudge = (
                    (
                        "This round is not yet marked complete. "
                        "Create the done marker or continue working."
                    )
                    if text
                    else ("Your response was empty. Call a tool to continue working.")
                )
                messages.append({"role": "user", "content": nudge})
                continue

            logger.warning("[Turn {:02d}] Nudges exhausted, accepting response", turn)
            _dump_context(messages, log_dir)
            logger.remove(_sink_id)
            return text or "(no text response)"


        for call in function_calls:
            try:
                args = json.loads(call.arguments)
                logger.info(
                    "[Turn {:02d}] Call {}: {}",
                    turn,
                    call.name,
                    _cut(json.dumps(args, ensure_ascii=False)),
                )
                handler = tool_handlers.get(call.name)
                output = handler(**args) if handler else f"Unknown tool: {call.name}"
            except Exception as exc:
                output = f"Error executing {call.name}: {exc}"
                logger.error("[Turn {:02d}] Error {}: {}", turn, call.name, exc)

            if isinstance(output, ImageResult):
                api_output = output.to_tool_output()
                logger.info(
                    "[Turn {:02d}] Result {}: [Image] {} ({:.1f} KB)",
                    turn,
                    call.name,
                    output.path,
                    output.size_kb,
                )
            else:
                api_output = str(output)
                if len(api_output) > _MAX_TOOL_OUTPUT_CHARS:
                    head = api_output[: _MAX_TOOL_OUTPUT_CHARS // 2]
                    tail = api_output[-_MAX_TOOL_OUTPUT_CHARS // 2 :]
                    api_output = (
                        head
                        + f"\n\n... [TRUNCATED {len(api_output) - _MAX_TOOL_OUTPUT_CHARS} chars "
                        f"(total {len(api_output)}); re-grep or read_file with offset/limit for more] ...\n\n"
                        + tail
                    )
                logger.info(
                    "[Turn {:02d}] Result {}: {}", turn, call.name, _cut(api_output)
                )

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": api_output,
                    "_turn": turn,
                }
            )


        truncate_oversized(messages, log_dir=log_dir)
        microcompact(messages, current_turn=turn, log_dir=log_dir)

    logger.error("EXCEEDED max_turns ({})", max_turns)
    _dump_context(messages, log_dir)
    logger.remove(_sink_id)
    raise RuntimeError(f"Optimizer did not finish within {max_turns} turns.")
