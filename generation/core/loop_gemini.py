
import contextlib
import os

from google import genai
from google.genai import types
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


def _build_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _sanitize_schema(schema: dict) -> dict:
    drop = {"$schema", "additionalProperties", "default"}
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
        elif isinstance(v, dict):
            out[k] = _sanitize_schema(v)
        elif isinstance(v, list):
            out[k] = [_sanitize_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


def _convert_tools(tools: list[dict]) -> list[types.Tool]:
    decls = []
    for t in filter_tools(tools):
        decls.append(types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=_sanitize_schema(t.get("parameters", {"type": "object", "properties": {}})),
        ))
    if not decls:
        return []
    return [types.Tool(function_declarations=decls)]


def _data_uri_parts(data_uri: str) -> tuple[str, bytes]:
    import base64
    head, b64 = data_uri.split(",", 1)
    mime = head.split(";", 1)[0].removeprefix("data:")
    return mime, base64.b64decode(b64)


def _image_part(img: ImageResult) -> types.Part:
    mime, raw = _data_uri_parts(img.data_uri)
    return types.Part.from_bytes(data=raw, mime_type=mime)


def _user_parts(text: str, images: list[str] | None) -> list[types.Part]:
    parts: list[types.Part] = [types.Part.from_text(text=text)]
    if not images:
        return parts
    for path in images:
        r = load_image(path)
        if isinstance(r, ImageResult):
            parts.append(_image_part(r))
        else:
            parts.append(types.Part.from_text(text=f"[image error: {r}]"))
    return parts


def _content_to_dict(c) -> dict:
    try:
        return c.model_dump(exclude_none=True)
    except Exception:
        return {"role": getattr(c, "role", None), "parts": [str(p) for p in getattr(c, "parts", [])]}


def _extract_text(response) -> str:
    parts = []
    for cand in response.candidates or []:
        for p in (cand.content.parts or []) if cand.content else []:
            if getattr(p, "text", None):
                parts.append(p.text)
    return "".join(parts)


def _extract_function_calls(response):
    calls = []
    for cand in response.candidates or []:
        for p in (cand.content.parts or []) if cand.content else []:
            fc = getattr(p, "function_call", None)
            if fc and fc.name:
                calls.append(fc)
    return calls


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
    gem_tools = _convert_tools(tools)
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=gem_tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=_user_parts(user_input, images)),
    ]

    continuations = 0
    empty_responses = 0

    logger.info(
        "{}\nAGENT START | provider=gemini | model={} | max_turns={}\nUser: {}\n{}",
        BANNER, model, max_turns, cut(user_input), BANNER,
    )

    for turn in range(1, max_turns + 1):
        response = client.models.generate_content(
            model=model, contents=contents, config=config,
        )
        with contextlib.suppress(Exception):
            logger.debug("[Turn {:02d}] Usage: {}", turn, response.usage_metadata)

        text = _extract_text(response)
        if text:
            logger.info("[Turn {:02d}] LLM: {}", turn, cut(text))


        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        fcalls = _extract_function_calls(response)

        if not fcalls:
            if output_exists(output_dir):
                logger.success("[Turn {:02d}] DONE — output exists", turn)
                dump_context([_content_to_dict(c) for c in contents], log_dir)
                logger.remove(sink_id)
                return text or "(no text response)"

            if not text and empty_responses < MAX_EMPTY:
                empty_responses += 1
                logger.warning(
                    "[Turn {:02d}] Nudge: empty response ({}/{})",
                    turn, empty_responses, MAX_EMPTY,
                )
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Continue working on the task — call a tool to proceed.")],
                ))
                continue

            if continuations < MAX_CONTINUATIONS:
                continuations += 1
                logger.warning(
                    "[Turn {:02d}] Nudge: output missing ({}/{})",
                    turn, continuations, MAX_CONTINUATIONS,
                )
                msg = (
                    f"No output file found in {output_dir}/. "
                    f"The task is not complete — call a tool to continue."
                    if text else
                    "Your response was empty. Call a tool to continue working."
                )
                contents.append(types.Content(
                    role="user", parts=[types.Part.from_text(text=msg)],
                ))
                continue

            logger.warning("[Turn {:02d}] Nudges exhausted, accepting response", turn)
            dump_context([_content_to_dict(c) for c in contents], log_dir)
            logger.remove(sink_id)
            return text or "(no text response)"


        result_parts: list[types.Part] = []
        for fc in fcalls:
            try:
                args = dict(fc.args) if fc.args else {}
                logger.info(
                    "[Turn {:02d}] Call {}: {}", turn, fc.name, cut(str(args)),
                )
                handler = tool_handlers.get(fc.name)
                output = handler(**args) if handler else f"Unknown tool: {fc.name}"
            except Exception as exc:
                output = f"Error executing {fc.name}: {exc}"
                logger.error("[Turn {:02d}] Error {}: {}", turn, fc.name, exc)

            if isinstance(output, ImageResult):
                logger.info(
                    "[Turn {:02d}] Result {}: [Image] {} ({:.1f} KB)",
                    turn, fc.name, output.path, output.size_kb,
                )


                result_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": str(output)},
                ))
                result_parts.append(_image_part(output))
            else:
                logger.info(
                    "[Turn {:02d}] Result {}: {}", turn, fc.name, cut(str(output)),
                )
                result_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": str(output)},
                ))

        contents.append(types.Content(role="user", parts=result_parts))

    logger.error("EXCEEDED max_turns ({})", max_turns)
    dump_context([_content_to_dict(c) for c in contents], log_dir)
    logger.remove(sink_id)
    raise RuntimeError(f"Agent did not finish within {max_turns} turns.")


def run(task, *, system, model, max_turns, tools, tool_handlers, output_dir, images=None):
    return agent_loop(
        task, system=system, images=images, model=model, max_turns=max_turns,
        tools=tools, tool_handlers=tool_handlers, output_dir=output_dir,
    )
