from __future__ import annotations

from pathlib import Path


def resolve_within(path: str, root: str | Path) -> Path:
    base = Path(root).resolve()
    candidate = Path(path)
    resolved = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise PermissionError(f"Path is outside the agent workspace: {path}") from exc
    return resolved
