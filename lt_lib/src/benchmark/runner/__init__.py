from __future__ import annotations


def run_one(*args, **kwargs):
    """Lazily import the single-run entry point to avoid package import cycles."""
    from benchmark.runner.run_one import run_one as _run_one

    return _run_one(*args, **kwargs)


def run_sweep(*args, **kwargs):
    """Lazily import the sweep entry point to avoid package import cycles."""
    from benchmark.runner.run_sweep import run_sweep as _run_sweep

    return _run_sweep(*args, **kwargs)


__all__ = ["run_one", "run_sweep"]
