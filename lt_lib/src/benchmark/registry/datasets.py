
from __future__ import annotations

import importlib

from pydantic import BaseModel


class DatasetRegistry:

    def __init__(self) -> None:
        self._datasets: dict[str, tuple[type, type[BaseModel] | None]] = {}

    def register(
        self, name: str, dataset_cls: type, schema: type[BaseModel] | None = None
    ) -> None:
        self._datasets[name] = (dataset_cls, schema)

    def get(self, name: str) -> tuple[type, type[BaseModel] | None]:
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' is not registered")
        return self._datasets[name]

    def names(self) -> list[str]:
        return sorted(self._datasets.keys())


DATASET_REGISTRY = DatasetRegistry()

DATASET_NAME_MAP = {
    "pre_split": "data.datasets.pre_split",
    "raw_idx": "data.datasets.raw_idx",
}

_REGISTERED_DATASETS: set[str] = set()


def register_dataset_by_name(name: str) -> None:
    if name in _REGISTERED_DATASETS:
        return
    module_name = DATASET_NAME_MAP.get(name)
    if module_name is None:
        available = ", ".join(sorted(DATASET_NAME_MAP.keys())) or "<none>"
        raise KeyError(
            f"Dataset '{name}' is not mapped. Update DATASET_NAME_MAP in "
            f"benchmark.registry.datasets. Available: {available}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                f"Dataset registry module not found: {module_name}. "
                "Expected module path in DATASET_NAME_MAP"
            ) from exc
        raise ImportError(
            f"Failed to import '{module_name}' due to missing dependency: {exc}"
        ) from exc

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise AttributeError(
            f"Dataset registry '{module_name}' must define a register() function"
        )
    register_fn()
    _REGISTERED_DATASETS.add(name)
