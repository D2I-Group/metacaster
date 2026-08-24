
from __future__ import annotations

import importlib
from collections.abc import Callable

import torch.nn as nn


class LossRegistry:

    def __init__(self) -> None:
        self._losses: dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str, factory: Callable[..., nn.Module]) -> None:
        self._losses[name.lower()] = factory

    def get(self, name: str) -> Callable[..., nn.Module]:
        key = name.lower()
        if key not in self._losses:
            raise KeyError(f"Loss '{name}' is not registered")
        return self._losses[key]

    def names(self) -> list[str]:
        return sorted(self._losses.keys())


LOSS_REGISTRY = LossRegistry()

LOSS_NAME_MAP = {
    "mse": "benchmark.losses",
    "mae": "benchmark.losses",
    "l1": "benchmark.losses",
}

_REGISTERED_LOSSES: set[str] = set()


def register_loss_by_name(name: str) -> None:
    key = name.lower()
    if key in _REGISTERED_LOSSES:
        return
    module_name = LOSS_NAME_MAP.get(key)
    if module_name is None:
        available = ", ".join(sorted(LOSS_NAME_MAP.keys())) or "<none>"
        raise KeyError(
            f"Loss '{name}' is not mapped. Update LOSS_NAME_MAP in "
            f"benchmark.registry.losses. Available: {available}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                f"Loss registry module not found: {module_name}. "
                "Expected module path in LOSS_NAME_MAP"
            ) from exc
        raise ImportError(
            f"Failed to import '{module_name}' due to missing dependency: {exc}"
        ) from exc

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise AttributeError(
            f"Loss registry '{module_name}' must define a register() function"
        )
    register_fn()
    _REGISTERED_LOSSES.add(key)


def get_loss(name: str, **kwargs) -> nn.Module:
    register_loss_by_name(name)
    factory = LOSS_REGISTRY.get(name)
    return factory(**kwargs)
