
from __future__ import annotations

import importlib
from collections.abc import Callable

from pydantic import BaseModel


class ModelRegistry:

    def __init__(self) -> None:
        self._models: dict[str, tuple[Callable, type[BaseModel] | None]] = {}

    def register(
        self, name: str, factory: Callable, schema: type[BaseModel] | None = None
    ) -> None:
        self._models[name] = (factory, schema)

    def get(self, name: str) -> tuple[Callable, type[BaseModel] | None]:
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")
        return self._models[name]

    def names(self) -> list[str]:
        return sorted(self._models.keys())


MODEL_REGISTRY = ModelRegistry()


MODEL_NAME_MAP = {

    "Linear": "models.linear.registry",
    "DLinear": "models.dlinear.registry",
    "NLinear": "models.nlinear.registry",
    "RLinear": "models.rlinear.registry",
    "CrossLinear": "models.crosslinear.registry",
    "MixLinear": "models.mixlinear.registry",

    "TSMixer": "models.tsmixer.registry",
    "LightTS": "models.lightts.registry",
    "PatchMLP": "models.patchmlp.registry",
    "xPatch": "models.xpatch.registry",
    "CMoS": "models.cmos.registry",

    "FITS": "models.fits.registry",
    "CycleNet": "models.cyclenet.registry",
    "PaiFilter": "models.paifilter.registry",
    "TexFilter": "models.texfilter.registry",

    "TimeMixer": "models.timemixer.registry",
    "TimeBase": "models.timebase.registry",
    "TimeBridge": "models.timebridge.registry",
    "TimeEmb": "models.timeemb.registry",
    "Amplifier": "models.amplifier.registry",
    "SparseTSF": "models.sparsetsf.registry",
    "PatchTSMixer": "models.patchtsmixer.registry",
    "FreqCycle": "models.freqcycle.registry",
}

_REGISTERED_MODELS: set[str] = set()


def register_model_by_name(name: str) -> None:
    if name in _REGISTERED_MODELS:
        return
    module_name = MODEL_NAME_MAP.get(name)
    if module_name is None:
        available = ", ".join(sorted(MODEL_NAME_MAP.keys())) or "<none>"
        raise KeyError(
            f"Model '{name}' is not mapped. Update MODEL_NAME_MAP in "
            f"benchmark.registry.models. Available: {available}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                f"Model registry module not found: {module_name}. "
                "Check that the LT-Lib is installed correctly."
            ) from exc
        raise ImportError(
            f"Failed to import '{module_name}' due to missing dependency: {exc}"
        ) from exc

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise AttributeError(
            f"Model registry '{module_name}' must define a register() function"
        )
    register_fn()
    _REGISTERED_MODELS.add(name)
