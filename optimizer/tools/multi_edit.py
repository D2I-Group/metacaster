
from ._whitelist import check_allowed

SCHEMA = {
    "type": "function",
    "name": "multi_edit",
    "description": "Atomic batched replaces on one file. All-or-nothing. Whitelist same as edit_file.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean", "default": False},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["file_path", "edits"],
    },
}


def handler(file_path: str, edits: list[dict]) -> str:
    checked = check_allowed(file_path)
    if isinstance(checked, str):
        return checked
    abs_path, rel_str = checked

    if not abs_path.exists():
        return f"Error: file not found: {rel_str}"
    try:
        text = abs_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {rel_str}: {e}"

    working = text
    applied = []
    for i, edit in enumerate(edits):
        old = edit.get("old_string", "")
        new = edit.get("new_string", "")
        replace_all = bool(edit.get("replace_all", False))
        if old == "":
            return f"Error: edit[{i}] has empty old_string."
        count = working.count(old)
        if count == 0:
            return f"Error: edit[{i}] old_string not found in current text."
        if not replace_all and count > 1:
            return (
                f"Error: edit[{i}] old_string occurs {count} times; "
                f"pass replace_all=true or make it unique."
            )
        if replace_all:
            working = working.replace(old, new)
            applied.append(f"edit[{i}]: {count} replacements")
        else:
            working = working.replace(old, new, 1)
            applied.append(f"edit[{i}]: 1 replacement")

    try:
        abs_path.write_text(working, encoding="utf-8")
    except Exception as e:
        return f"Error writing {rel_str}: {e}"
    return f"multi_edit {rel_str}: {len(edits)} edits applied.\n" + "\n".join(applied)
