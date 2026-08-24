
import json
import os

from loguru import logger
from openai import OpenAI

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


def _build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )


def _convert_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in filter_tools(tools):
        if t.get("type") == "function" and "function" in t:
            out.append(t)
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return out


def _image_block(img: ImageResult) -> dict:
    return {"type": "image_url", "image_url": {"url": img.data_uri}}


def _user_content(text: str, images: list[str] | None):
    if not images:
        return text
    content: list = [{"type": "text", "text": text}]
    for path in images:
        r = load_image(path)
        if isinstance(r, ImageResult):
            content.append(_image_block(r))
        else:
            content.append({"type": "text", "text": f"[image error: {r}]"})
    return content


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
    cc_tools = _convert_tools(tools)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": _user_content(user_input, images)},
    ]

    continuations = 0
    empty_responses = 0

    logger.info(
        "{}\nAGENT START | provider=vllm | model={} | max_turns={}\nUser: {}\n{}",
        BANNER, model, max_turns, cut(user_input), BANNER,
    )

    for turn in range(1, max_turns + 1):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=cc_tools,
        )
        logger.debug("[Turn {:02d}] Usage: {}", turn, response.usage)

        msg = response.choices[0].message
        text = msg.content or ""
        if text:
            logger.info("[Turn {:02d}] LLM: {}", turn, cut(text))


        assistant_entry: dict = {"role": "assistant", "content": text}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        tool_calls = msg.tool_calls or []

        if not tool_calls:
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

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                logger.info(
                    "[Turn {:02d}] Call {}: {}",
                    turn, tc.function.name,
                    cut(json.dumps(args, ensure_ascii=False)),
                )
                handler = tool_handlers.get(tc.function.name)
                output = handler(**args) if handler else f"Unknown tool: {tc.function.name}"
            except Exception as exc:
                output = f"Error executing {tc.function.name}: {exc}"
                logger.error(
                    "[Turn {:02d}] Error {}: {}", turn, tc.function.name, exc,
                )

            if isinstance(output, ImageResult):
                logger.info(
                    "[Turn {:02d}] Result {}: [Image] {} ({:.1f} KB)",
                    turn, tc.function.name, output.path, output.size_kb,
                )


                tool_text = str(output)
            else:
                tool_text = str(output)
                logger.info(
                    "[Turn {:02d}] Result {}: {}",
                    turn, tc.function.name, cut(tool_text),
                )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_text,
            })

    logger.error("EXCEEDED max_turns ({})", max_turns)
    dump_context(messages, log_dir)
    logger.remove(sink_id)
    raise RuntimeError(f"Agent did not finish within {max_turns} turns.")


def run(task, *, system, model, max_turns, tools, tool_handlers, output_dir, images=None):
    return agent_loop(
        task, system=system, images=images, model=model, max_turns=max_turns,
        tools=tools, tool_handlers=tool_handlers, output_dir=output_dir,
    )
