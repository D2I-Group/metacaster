
from pathlib import Path


def assert_write_allowed(path: Path | str, ctx: dict) -> None:
    path = Path(path).resolve()
    allowed_prefixes = [
        ctx["run_root"].resolve(),
        (ctx["agent_root"] / "lt_lib" / "dataset").resolve(),
        (ctx["agent_root"] / "lt_lib" / "configs" / "exp").resolve(),
        (ctx["agent_root"] / "lt_lib" / "work_dirs").resolve(),
    ]
    forbidden_prefixes = [
        (ctx["agent_root"] / "generation").resolve(),
    ]
    for fp in forbidden_prefixes:
        try:
            path.relative_to(fp)
            raise PermissionError(
                f"Write to {path} is forbidden (under {fp}); the optimizer "
                f"must not modify the repo harness. Use "
                f"{ctx['harness_root']} instead."
            )
        except ValueError:
            continue
    for ap in allowed_prefixes:
        try:
            path.relative_to(ap)
            return
        except ValueError:
            continue
    raise PermissionError(
        f"Write to {path} is not allowed. Allowed roots: {allowed_prefixes}"
    )


def check_allowed(path: str) -> tuple[Path, str] | str:

    from optimizer.tools import TOOL_CONTEXT

    if not TOOL_CONTEXT:
        return (
            "Error: TOOL_CONTEXT is empty; optimizer.agent must populate "
            "run_root/harness_root/agent_root/repo_root before tool use."
        )

    try:
        abs_path = Path(path).expanduser().resolve()
    except Exception as e:
        return f"Error: cannot resolve path {path}: {e}"

    try:
        assert_write_allowed(abs_path, TOOL_CONTEXT)
    except PermissionError as e:
        return f"Error: {e}"


    run_root = TOOL_CONTEXT["run_root"].resolve()
    agent_root = TOOL_CONTEXT["agent_root"].resolve()
    try:
        display = abs_path.relative_to(run_root).as_posix()
    except ValueError:
        try:
            display = abs_path.relative_to(agent_root).as_posix()
        except ValueError:
            display = str(abs_path)
    return abs_path, display
