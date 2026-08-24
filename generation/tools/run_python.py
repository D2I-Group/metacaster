
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_MAX_OUTPUT_CHARS = 10_000

SCHEMA = {
    "type": "function",
    "name": "run_python",
    "description": (
        "Execute Python code using the project environment. "
        "All uv-installed packages are immediately available. "
        "Prefer this over bash for any Python work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete, self-contained Python code to execute.",
            },
        },
        "required": ["code"],
    },
}


def handler(code: str, *, work_dir: str | Path = ".") -> str:
    workspace = Path(work_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="_agent_run_",
        dir=workspace,
        delete=False,
    ) as f:
        f.write(code)
        tmp = Path(f.name)

    try:
        env = {
            key: value
            for key, value in os.environ.items()
            if not any(
                marker in key.upper()
                for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
            )
        }
        result = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=workspace,
            env=env,
        )
        out = result.stdout
        if result.returncode != 0 or result.stderr:
            out += f"\n--- stderr ---\n{result.stderr}"
        out = out.strip() or "(no output)"
        if len(out) > _MAX_OUTPUT_CHARS:
            out = out[:_MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(out)} chars total]"
        return out
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (120s)"
    except FileNotFoundError:
        return "Error: Python executable not found"
    finally:
        tmp.unlink(missing_ok=True)
