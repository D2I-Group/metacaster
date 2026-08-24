
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MG_MODEL: str = os.getenv("MG_MODEL", "gpt-5.4")
MG_MAX_TURNS: int = int(os.getenv("MG_MAX_TURNS", "50"))


LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()


REASONING_EFFORT: str = os.getenv("REASONING_EFFORT", "")


MAX_CONTINUATIONS = 3
MAX_EMPTY = 2


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
