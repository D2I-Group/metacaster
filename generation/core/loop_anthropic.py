
import os

import anthropic
from loguru import logger

from ._runtime import (
    BANNER,
    cut,
    dump_context,
    filter_tools,
    output_exists,
    setup_run,
)
from .config import MAX_CONTINUATIONS, MAX_EMPTY
from .image import ImageResult, load_image

MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192"))


def _build_client() -> anthropic.Anthropic:
    kwargs = {"api_key": os.environ["ANTHROPIC_API_KEY"]}
    if os.getenv("ANTHROPIC_BASE_URL"):
        kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    return anthropic.Anthropic(**kwargs)


def _convert_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in filter_tools(tools):

        if "input_schema" in t:
            out.append(t)
            continue
        out.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _data_uri_parts(data_uri: str) -> tuple[str, str]:
    head, b64 = data_uri.split(",", 1)
    mime = head.split(";", 1)[0].removeprefix("data:")
    return mime, b64


def _image_block(img: ImageResult) -> dict:
    mime, b64 = _data_uri_parts(img.data_uri)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": b64},
    }


def _build_user_content(text: str, images: list[str] | None) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": text}]
    if not images:
        return content
    for path in images:
        r = load_image(path)
        if isinstance(r, ImageResult):
            content.append(_image_block(r))
        else:
            content.append({"type": "text", "text": f"[image error: {r}]"})
    return content


def _tool_result_blocks(call_id: str, output) -> list[dict]:
    if isinstance(output, ImageResult):
        return [{
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": [
                {"type": "text", "text": str(output)},
                _image_block(output),
            ],
        }]
    return [{
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": str(output),
    }]


def _extract_text(response) -> str:
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def agent_loop(
    user_input: str,
    system: str,
    images: list[str] | None = None,
    *,
    model: str,
    max_turns: int,
    tools: list,
    tool_handlers: dict,
    output_dir: str = "./work_dir",
) -> str:
    output_dir, log_dir, sink_id, system = setup_run(output_dir, system)
    client = _build_client()
    anth_tools = _convert_tools(tools)

    messages: list[dict] = [
        {"role": "user", "content": _build_user_content(user_input, images)}
    ]

    continuations = 0
    empty_responses = 0

    logger.info(
        "{}\nAGENT START | provider=anthropic | model={} | max_turns={}\nUser: {}\n{}",
        BANNER, model, max_turns, cut(user_input), BANNER,
    )

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=anth_tools,
            messages=messages,
        )
        logger.debug("[Turn {:02d}] Usage: {}", turn, response.usage)

        text = _extract_text(response)
        if text:
            logger.info("[Turn {:02d}] LLM: {}", turn, cut(text))


        messages.append({
            "role": "assistant",
            "content": [b.model_dump() for b in response.content],
        })

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            if output_exists(output_dir):
                logger.success("[Turn {:02d}] DONE — output exists", turn)
                dump_context(messages, log_dir)
                logger.remove(sink_id)
                return text or "(no text response)"

            if not text and empty_responses < MAX_EMPTY:
                empty_responses += 1
                logger.warning(
                    "[Turn {:02d}] Nudge: empty response ({}/{})",
                    turn, empty_responses, MAX_EMPTY,
                )
                messages.append({
                    "role": "user",
                    "content": "Continue working on the task — call a tool to proceed.",
                })
                continue

            if continuations < MAX_CONTINUATIONS:
                continuations += 1
                logger.warning(
                    "[Turn {:02d}] Nudge: output missing ({}/{})",
                    turn, continuations, MAX_CONTINUATIONS,
                )
                nudge = (
                    f"No output file found in {output_dir}/. "
                    f"The task is not complete — call a tool to continue."
                    if text else
                    "Your response was empty. Call a tool to continue working."
                )
                messages.append({"role": "user", "content": nudge})
                continue

            logger.warning("[Turn {:02d}] Nudges exhausted, accepting response", turn)
            dump_context(messages, log_dir)
            logger.remove(sink_id)
            return text or "(no text response)"


        tool_results: list[dict] = []
        for call in tool_uses:
            try:
                logger.info(
                    "[Turn {:02d}] Call {}: {}",
                    turn, call.name, cut(str(call.input)),
                )
                handler = tool_handlers.get(call.name)
                output = handler(**call.input) if handler else f"Unknown tool: {call.name}"
            except Exception as exc:
                output = f"Error executing {call.name}: {exc}"
                logger.error("[Turn {:02d}] Error {}: {}", turn, call.name, exc)

            if isinstance(output, ImageResult):
                logger.info(
                    "[Turn {:02d}] Result {}: [Image] {} ({:.1f} KB)",
                    turn, call.name, output.path, output.size_kb,
                )
            else:
                logger.info(
                    "[Turn {:02d}] Result {}: {}", turn, call.name, cut(str(output)),
                )
            tool_results.extend(_tool_result_blocks(call.id, output))

        messages.append({"role": "user", "content": tool_results})

    logger.error("EXCEEDED max_turns ({})", max_turns)
    dump_context(messages, log_dir)
    logger.remove(sink_id)
    raise RuntimeError(f"Agent did not finish within {max_turns} turns.")


def run(task, *, system, model, max_turns, tools, tool_handlers, output_dir, images=None):
    return agent_loop(
        task, system=system, images=images, model=model, max_turns=max_turns,
        tools=tools, tool_handlers=tool_handlers, output_dir=output_dir,
    )
