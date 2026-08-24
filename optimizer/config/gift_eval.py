
from __future__ import annotations

from pathlib import Path

from optimizer.runtime.sample_few_shot import (
    N_FEWSHOT_TRAIN_RANGE,
)
from scripts.prepare_gift_eval import TEST_BATCH, TRAIN_BATCH

from .base import BenchmarkConfig

_LIGHTWEIGHT_MODEL_POOL = [

    "Linear",
    "DLinear",
    "NLinear",
    "RLinear",
    "MixLinear",

    "TSMixer",
    "LightTS",
    "PatchMLP",
    "xPatch",
    "CMoS",
    "PatchTSMixer",

    "FITS",
    "CycleNet",
    "PaiFilter",
    "TexFilter",

    "TimeMixer",
    "TimeBridge",
    "TimeEmb",
    "Amplifier",
    "SparseTSF",
]


GIFT_EVAL = BenchmarkConfig(
    name="GIFT-Eval",
    data_root=Path("data/GIFT-Eval"),
    train_datasets=[entry[0] for entry in TRAIN_BATCH],
    test_datasets=[entry[0] for entry in TEST_BATCH],
    excluded_train_domains=["hydro", "demographics"],
    model_pool=_LIGHTWEIGHT_MODEL_POOL,
    sampler_script="optimizer/runtime/sample_few_shot.py",
    support_shot_range=N_FEWSHOT_TRAIN_RANGE,
)
