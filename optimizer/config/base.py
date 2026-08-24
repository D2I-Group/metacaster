
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:

    name: str
    data_root: Path
    train_datasets: list[str]
    test_datasets: list[str]
    excluded_train_domains: list[str]
    model_pool: list[str]
    sampler_script: str | None = None
    support_shot_range: tuple[int, int] = field(default_factory=tuple)


    def _format_support_shots(self) -> str:
        if not self.support_shot_range:
            return "(none — few-shot is pre-baked for both roles)"
        lo, hi = self.support_shot_range
        return f"K ~ Uniform([{lo}, {hi}])"

    def _format_model_pool(self) -> str:
        return ", ".join(f"`{m}`" for m in self.model_pool)

    def to_prompt_kwargs(self) -> dict[str, str]:
        return {
            "benchmark_name": self.name,
            "data_root": str(self.data_root),
            "n_train": str(len(self.train_datasets)),
            "n_test": str(len(self.test_datasets)),
            "train_datasets": ", ".join(self.train_datasets),
            "test_datasets": ", ".join(self.test_datasets),
            "excluded_domains": ", ".join(self.excluded_train_domains),
            "model_pool_desc": self._format_model_pool(),
            "sampler_script": self.sampler_script or "(none)",
            "support_shot_protocol": self._format_support_shots(),
        }
