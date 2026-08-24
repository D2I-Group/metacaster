
from pathlib import Path

SCHEMA = {
    "type": "function",
    "name": "read_file",
    "description": "Read text file. offset=skip leading lines, limit=max lines.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
    },
}


def handler(
    path: str,
    offset: int = 0,
    limit: int | None = None,
) -> str:
    from optimizer.tools import TOOL_CONTEXT

    if not TOOL_CONTEXT:
        return "Error: HPAgent tool context is not initialized"
    run_root = TOOL_CONTEXT["run_root"].resolve()
    candidate = Path(path).expanduser()
    candidate = candidate.resolve() if candidate.is_absolute() else (run_root / candidate).resolve()
    try:
        candidate.relative_to(run_root)
    except ValueError:
        return f"Error: read path must remain inside the HPAgent run workspace: {run_root}"
    try:
        lines = candidate.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"

    start = max(0, int(offset or 0))
    if limit is not None:
        end = start + max(0, int(limit))
        sliced = lines[start:end]
    else:
        sliced = lines[start:]
    return "\n".join(sliced)
