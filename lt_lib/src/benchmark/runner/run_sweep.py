
from __future__ import annotations

from collections.abc import Iterable

from benchmark.runner.run_one import run_one


def run_sweep(configs: Iterable) -> list:
    results = []
    for loaded in configs:
        results.append(
            run_one(loaded.config, loaded.raw, loaded.sweep_keys, loaded.config_name)
        )
    return results
