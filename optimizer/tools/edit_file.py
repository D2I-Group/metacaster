
from ._whitelist import check_allowed

SCHEMA = {
    "type": "function",
    "name": "edit_file",
    "description": "Replace exact string. Must be unique unless replace_all=true. Whitelist same as write_file.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    },
}


def handler(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    checked = check_allowed(path)
    if isinstance(checked, str):
        return checked
    abs_path, rel_str = checked

    if not abs_path.exists():
        return f"Error: file not found: {rel_str}"
    try:
        text = abs_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {rel_str}: {e}"

    count = text.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {rel_str}"

    if replace_all:
        new_text = text.replace(old_string, new_string)
        n = count
    else:
        if count > 1:
            return (
                f"Error: old_string occurs {count} times in {rel_str}; "
                f"pass replace_all=true or provide a more specific string."
            )
        new_text = text.replace(old_string, new_string, 1)
        n = 1

    try:
        abs_path.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return f"Error writing {rel_str}: {e}"

    return f"Edited {rel_str} ({n} replacement{'s' if n != 1 else ''})"
