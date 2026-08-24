
import shutil
import subprocess

SCHEMA = {
    "type": "function",
    "name": "grep",
    "description": "ripgrep search; returns file:line matches.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["pattern"],
    },
}


def handler(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    max_results: int = 100,
) -> str:
    if shutil.which("rg") is None:
        return "Error: ripgrep (rg) is not installed. Install it (e.g. `apt install ripgrep`) to use this tool."

    cmd = ["rg", "-n", "--no-heading", "--color=never"]
    if glob:
        cmd.extend(["--glob", glob])
    cmd.extend(["--", pattern, path])

    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "Error: grep timed out after 60s"
    except Exception as e:
        return f"Error running rg: {e}"

    if result.returncode == 1 and not result.stdout:
        return "(no matches)"
    if result.returncode not in (0, 1):
        return f"Error: rg exited {result.returncode}: {result.stderr.strip()}"

    lines = result.stdout.splitlines()
    truncated = len(lines) > max_results
    lines = lines[:max_results]
    out = "\n".join(lines) if lines else "(no matches)"
    if truncated:
        out += f"\n... (truncated to {max_results} lines)"
    return out
