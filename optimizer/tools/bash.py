
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

_GIT_DENY_RE = re.compile(r"\bgit\s+(commit|add|push)\b")

SCHEMA = {
    "type": "function",
    "name": "bash",
    "description": "Run shell command. background=true returns job_id; poll with command='poll' + job_id.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout": {"type": "integer"},
            "background": {"type": "boolean"},
            "job_id": {"type": "string"},
        },
        "required": ["command"],
    },
}

_BLACKLIST_PATTERNS = (
    "rm -rf /",
    ":(){ :|:& };:",
    "git push",
    "git reset --hard",
    "mkfs",
    "dd if=",
    "> /dev/",
    "shutdown",
    "reboot",
)

_MAX_OUTPUT = 20_000


def _runtime_context() -> tuple[Path, Path]:
    from optimizer.tools import TOOL_CONTEXT

    if not TOOL_CONTEXT:
        raise RuntimeError("HPAgent tool context is not initialized")
    run_root = TOOL_CONTEXT["run_root"].resolve()
    return run_root, run_root / ".bash_jobs"


def _sanitized_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not any(
            marker in key.upper()
            for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
        )
    }


def _blacklist_hit(command: str) -> str | None:
    for pat in _BLACKLIST_PATTERNS:
        if pat in command:
            return pat
    return None


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT:
        return text
    return text[:_MAX_OUTPUT] + f"\n... [output truncated at {_MAX_OUTPUT} chars]"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _poll_job(job_id: str) -> str:
    try:
        _, jobs_dir = _runtime_context()
    except RuntimeError as exc:
        return f"Error: {exc}"
    meta_path = jobs_dir / f"{job_id}.json"
    log_path = jobs_dir / f"{job_id}.log"
    if not meta_path.exists():
        return f"Error: no such job_id {job_id}"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"Error reading job metadata: {e}"

    pid = int(meta.get("pid", 0))
    running = _pid_alive(pid) if pid else False
    exit_code = meta.get("exit_code")


    if not running and exit_code is None and pid:
        try:
            _pid, status = os.waitpid(pid, os.WNOHANG)
            if _pid != 0:
                exit_code = (
                    os.waitstatus_to_exitcode(status)
                    if hasattr(os, "waitstatus_to_exitcode")
                    else status
                )
                meta["exit_code"] = exit_code
                meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except ChildProcessError:
            pass
        except Exception:
            pass

    tail = ""
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-200:])
        except Exception as e:
            tail = f"(could not read log: {e})"

    return json.dumps(
        {
            "job_id": job_id,
            "running": running,
            "exit_code": exit_code,
            "pid": pid,
            "log_path": str(log_path),
            "tail": _truncate(tail),
        },
        ensure_ascii=False,
    )


def handler(
    command: str,
    cwd: str | None = None,
    timeout: int = 600,
    background: bool = False,
    job_id: str | None = None,
) -> str:

    if job_id and (not command or command.strip() in ("", "poll")):
        return _poll_job(job_id)

    if not command or not command.strip():
        return "Error: empty command"

    hit = _blacklist_hit(command)
    if hit:
        return f"Error: command blocked by blacklist pattern '{hit}'"

    if _GIT_DENY_RE.search(command):
        return (
            "Error: git commit/add/push is disabled for the optimizer "
            "(harness edits are not committed)."
        )

    try:
        run_root, jobs_dir = _runtime_context()
    except RuntimeError as exc:
        return f"Error: {exc}"
    workdir = Path(cwd).expanduser().resolve() if cwd else run_root
    if not workdir.exists():
        return f"Error: cwd does not exist: {workdir}"
    try:
        workdir.relative_to(run_root)
    except ValueError:
        return f"Error: cwd must remain inside the HPAgent run workspace: {run_root}"

    child_env = _sanitized_env()
    if background:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        jid = job_id or f"job_{uuid.uuid4().hex[:6]}"
        log_path = jobs_dir / f"{jid}.log"
        meta_path = jobs_dir / f"{jid}.json"
        try:
            with log_path.open("w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=str(workdir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=child_env,
                )
        except Exception as e:
            return f"Error starting background job: {e}"

        meta = {
            "job_id": jid,
            "pid": proc.pid,
            "command": command,
            "cwd": str(workdir),
            "start_time": time.time(),
            "exit_code": None,
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return json.dumps(
            {"job_id": jid, "pid": proc.pid, "log_path": str(log_path)},
            ensure_ascii=False,
        )


    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=child_env,
        )
    except subprocess.TimeoutExpired as e:
        partial = e.output or ""
        return _truncate(f"Error: command timed out after {timeout}s\n{partial}")
    except Exception as e:
        return f"Error running command: {e}"

    output = result.stdout or "(no output)"
    header = f"[exit={result.returncode}] "
    return _truncate(header + output)
