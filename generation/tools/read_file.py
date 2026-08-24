
from pathlib import Path

from generation.tools._paths import resolve_within

SCHEMA = {
    "type": "function",
    "name": "read_file",
    "description": "Read the text contents of a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read."},
            "limit": {
                "type": "integer",
                "description": "Max lines to return (optional).",
            },
        },
        "required": ["path"],
    },
}


def handler(path: str, limit: int | None = None, *, root: str | Path = ".") -> str:
    try:
        lines = resolve_within(path, root).read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[:limit] if limit else lines)
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except PermissionError as exc:
        return f"Error: {exc}"
    except Exception as e:
        return f"Error reading {path}: {e}"
