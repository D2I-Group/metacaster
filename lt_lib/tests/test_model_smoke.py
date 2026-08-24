from __future__ import annotations

import importlib
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark.registry import MODEL_REGISTRY
from benchmark.registry.models import MODEL_NAME_MAP

MODELS = [
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
    "CrossLinear",
    "TimeBase",
    "FreqCycle",
]


class ModelSmokeTests(unittest.TestCase):
    def test_all_models_forward_and_backward(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = SimpleNamespace(
            task=SimpleNamespace(
                seq_len=336,
                pred_len=192,
                label_len=0,
                features="M",
            )
        )
        for model_name in MODELS:
            with self.subTest(model=model_name):
                module = importlib.import_module(MODEL_NAME_MAP[model_name])
                module.register()
                factory, schema = MODEL_REGISTRY.get(model_name)
                model_config = tomllib.loads(
                    (root / "configs" / "models" / f"{model_name}.toml").read_text(
                        encoding="utf-8"
                    )
                )
                params = model_config["model"]["params"]
                params.update(enc_in=1, dec_in=1, c_out=1)
                if schema is not None:
                    params = schema.model_validate(params).model_dump()
                model = factory(config, params)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                x = torch.randn(1, 336, 1, requires_grad=True)
                x_mark = torch.zeros(1, 336, 6)
                decoder = torch.zeros(1, 192, 1)
                y_mark = torch.zeros(1, 192, 6)
                try:
                    prediction = model(x, x_mark, decoder, y_mark)
                except TypeError:
                    prediction = model(x)
                self.assertEqual(tuple(prediction.shape), (1, 192, 1))
                loss = prediction.square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    unittest.main()
