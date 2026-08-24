
from ._whitelist import check_allowed

SCHEMA = {
    "type": "function",
    "name": "write_file",
    "description": "Write/overwrite a file. Whitelist: harness/, run_root, and LT-Lib runtime directories.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
}


def handler(path: str, content: str) -> str:
    checked = check_allowed(path)
    if isinstance(checked, str):
        return checked
    abs_path, rel_str = checked
    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {rel_str}"
    except Exception as e:
        return f"Error writing {rel_str}: {e}"
