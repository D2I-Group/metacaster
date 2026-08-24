
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import OpenAI

from ._runtime import strip_base64
from .config import MAX_CONTINUATIONS, MAX_EMPTY, REASONING_EFFORT
from .image import ImageResult, build_content_with_images

_LOG_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}"

_MAX_LOG = 2000
_BANNER = "=" * 60


def _cut(text: str) -> str:
    if len(text) <= _MAX_LOG:
        return text
    return text[:_MAX_LOG] + f" ... [{len(text)} chars]"


def make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"{ts}_{uid}"


def _dump_context(messages: list, log_dir: Path) -> None:
    path = log_dir / "conversation.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            entry = msg.model_dump() if hasattr(msg, "model_dump") else msg
            f.write(json.dumps(strip_base64(entry), ensure_ascii=False) + "\n")
    logger.info("Context saved: {}", path)


    try:
        from scripts.summarize_log import process_file

        process_file(path, stdout=False)
    except Exception as exc:
        logger.warning("Failed to generate summary: {}", exc)


def _extract_text(response) -> str:
    parts = []
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    parts.append(part.text)
    return "".join(parts)


def _output_exists(output_dir: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, "dataset.npy"))


def agent_loop(
    user_input: str,
    system: str,
    images: list[str] | None = None,
    *,
    client: OpenAI,
    model: str,
    max_turns: int,
    tools: list,
    tool_handlers: dict,
    output_dir: str = "./work_dir",
) -> str:
    run_id = make_run_id()
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)


    log_dir = Path(output_dir) / "log" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    _sink_id = logger.add(
        log_dir / "run.log",
        encoding="utf-8",
        format=_LOG_FMT,
    )


    system = system.replace("{output_dir}", ".")


    if images:
        content = build_content_with_images(user_input, images)
        messages: list = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": user_input}]

    continuations = 0
    empty_responses = 0

    logger.info(
        "{}\nAGENT START | run_id={} | model={} | max_turns={}\nUser: {}\n{}",
        _BANNER,
        run_id,
        model,
        max_turns,
        _cut(user_input),
        _BANNER,
    )


    for turn in range(1, max_turns + 1):
        create_kwargs: dict = dict(
            model=model,
            instructions=system,
            input=messages,
            tools=tools,
        )
        if REASONING_EFFORT:
            create_kwargs["reasoning"] = {"effort": REASONING_EFFORT}

        response = client.responses.create(**create_kwargs)
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


            if _output_exists(output_dir):
                logger.success("[Turn {:02d}] DONE — output exists", turn)
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
                    "[Turn {:02d}] Nudge: output missing ({}/{})",
                    turn,
                    continuations,
                    MAX_CONTINUATIONS,
                )
                nudge = (
                    (
                        f"No output file found in {output_dir}/. "
                        f"The task is not complete — call a tool to continue."
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
                logger.info(
                    "[Turn {:02d}] Result {}: {}", turn, call.name, _cut(api_output)
                )

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": api_output,
                }
            )

    logger.error("EXCEEDED max_turns ({})", max_turns)
    _dump_context(messages, log_dir)
    logger.remove(_sink_id)
    raise RuntimeError(f"Agent did not finish within {max_turns} turns.")
