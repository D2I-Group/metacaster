
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

_LOG_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}"
_MAX_LOG = 2000
_BANNER = "=" * 60


def cut(text: str) -> str:
    if len(text) <= _MAX_LOG:
        return text
    return text[:_MAX_LOG] + f" ... [{len(text)} chars]"


def strip_base64(obj):
    if isinstance(obj, str):
        if obj.startswith("data:image"):
            return "[base64 image omitted]"
        obj = re.sub(r"/(?:Users|home|root)/[^/\s]+/", "<local-root>/", obj)
        return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[token omitted]", obj)
    if isinstance(obj, dict):
        return {
            key: (
                "[secret omitted]"
                if any(
                    marker in str(key).upper()
                    for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
                )
                else strip_base64(value)
            )
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [strip_base64(v) for v in obj]
    return obj


def make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def output_exists(output_dir: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, "dataset.npy"))


def setup_run(output_dir: str, system: str) -> tuple[str, Path, int, str]:
    run_id = make_run_id()
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)
    log_dir = Path(output_dir) / "log" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(log_dir / "run.log", encoding="utf-8", format=_LOG_FMT)
    system = system.replace("{output_dir}", ".")
    return output_dir, log_dir, sink_id, system


def dump_context(messages: list, log_dir: Path) -> None:
    path = log_dir / "conversation.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            if isinstance(msg, dict):
                entry = msg
            elif hasattr(msg, "model_dump"):
                entry = msg.model_dump()
            else:
                entry = str(msg)
            f.write(json.dumps(strip_base64(entry), ensure_ascii=False) + "\n")
    logger.info("Context saved: {}", path)
    try:
        from scripts.summarize_log import process_file
        process_file(path, stdout=False)
    except Exception as exc:
        logger.warning("Failed to generate summary: {}", exc)


def filter_tools(tools: list[dict]) -> list[dict]:
    return [t for t in tools if t.get("type") != "web_search"]


BANNER = _BANNER
LOG_FMT = _LOG_FMT
