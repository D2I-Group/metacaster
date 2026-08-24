
from pathlib import Path

from generation.tools._paths import resolve_within

SCHEMA = {
    "type": "function",
    "name": "write_file",
    "description": "Write (or overwrite) content to a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
    },
}


def handler(path: str, content: str, *, root: str | Path = ".") -> str:
    try:
        p = resolve_within(path, root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except Exception as exc:
        return f"Error writing {path}: {exc}"
