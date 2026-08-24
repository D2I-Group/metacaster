
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HP_MODEL: str = os.getenv("HP_MODEL", "gpt-5.4")
HP_REASONING_EFFORT: str = os.getenv("HP_REASONING_EFFORT", "high")
HP_MAX_TURNS: int = int(os.getenv("HP_MAX_TURNS", "500"))
HP_MAX_ROUNDS: int = int(os.getenv("HP_MAX_ROUNDS", "8"))
MAX_CONTINUATIONS = 3
MAX_EMPTY = 3


COMPACT_THRESHOLD_TOKENS = 120_000


TOOL_RESULT_AGE_LIMIT = 3


TOOL_RESULT_MAX_CHARS = 8000


PRESERVE_RESULT_TOOLS = frozenset({"read_file", "read_image", "glob", "grep"})


COMPACT_RECENT_KEEP = 10


COMPACT_MAX_FAILURES = 3


COMPACT_SUMMARY_MAX_TOKENS = 3000


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT

RUNS_ROOT = AGENT_ROOT / "work_dir" / "runs"


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def harness_dir(run_id: str) -> Path:
    return run_dir(run_id) / "harness"


def round_dir(run_id: str, round_n: int) -> Path:
    return run_dir(run_id) / "rounds" / f"round_{round_n}"


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
