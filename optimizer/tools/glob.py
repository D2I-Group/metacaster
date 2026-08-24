
from pathlib import Path

SCHEMA = {
    "type": "function",
    "name": "glob",
    "description": "Glob files relative to root.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "root": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["pattern"],
    },
}


def handler(pattern: str, root: str = ".", max_results: int = 500) -> str:
    try:
        base = Path(root).expanduser().resolve()
    except Exception as e:
        return f"Error: cannot resolve root {root}: {e}"
    if not base.exists():
        return f"Error: root does not exist: {root}"

    try:
        if "/" in pattern or "**" in pattern:
            matches = (
                base.rglob(pattern) if pattern.startswith("**/") else base.glob(pattern)
            )
        else:
            matches = base.glob(pattern)
        results: list[str] = []
        for p in matches:
            try:
                results.append(str(p.relative_to(base)))
            except ValueError:
                results.append(str(p))
            if len(results) >= max_results:
                break
    except Exception as e:
        return f"Error in glob: {e}"

    if not results:
        return "(no matches)"
    return "\n".join(results)
