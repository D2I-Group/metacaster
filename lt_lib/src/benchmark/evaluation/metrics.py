
from __future__ import annotations

import numpy as np

from benchmark.registry import METRIC_REGISTRY


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((true - pred) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs((true - pred) / true)))


def mspe(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.square((true - pred) / true)))


def collect_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(pred, true),
        "mse": mse(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
        "mspe": mspe(pred, true),
    }


def register() -> None:
    METRIC_REGISTRY.register("mae", mae)
    METRIC_REGISTRY.register("mse", mse)
    METRIC_REGISTRY.register("rmse", rmse)
    METRIC_REGISTRY.register("mape", mape)
    METRIC_REGISTRY.register("mspe", mspe)
