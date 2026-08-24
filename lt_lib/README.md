# MetaCaster LT-Lib

This package contains the 23 Forecasters used by MetaCaster, their model configurations, the `pre_split` and `raw_idx` dataset interfaces, and a uniform training/evaluation runtime.

Install the environment from the repository root:

```bash
uv sync --project lt_lib
```

The target Linux environment uses PyTorch 2.5.1 with the CUDA 12.4 package index. Profiling dependencies are optional:

```bash
uv sync --project lt_lib --extra profile
```

Validate every model with a forward, backward, and optimizer step:

```bash
uv run --project lt_lib \
  python -m unittest discover -s lt_lib/tests -v
```

Use FTAgent for isolated parallel training, hyperparameter search, and validation-MSE Top-1 selection. See [`../README.md`](../README.md).
