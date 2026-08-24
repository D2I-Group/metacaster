
from .base import BenchmarkConfig
from .gift_eval import GIFT_EVAL

BENCHMARKS: dict[str, BenchmarkConfig] = {
    "gift_eval": GIFT_EVAL,
}

__all__ = ["BENCHMARKS", "GIFT_EVAL", "BenchmarkConfig"]
