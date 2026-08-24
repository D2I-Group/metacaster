
from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable


def write_csv_summary(
    path: str,
    row: dict,
    header: Iterable[str] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    if header is None:
        header = list(row.keys())

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _flatten_params(params: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in params.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, path))
        elif isinstance(value, (list, tuple)):
            flat[path] = json.dumps(value, ensure_ascii=True)
        else:
            flat[path] = value
    return flat


def _append_sweep_values(row: dict, raw: dict, sweep_keys: list[str]) -> None:
    if not sweep_keys:
        return
    flattened = _flatten_params(raw)
    for key in sweep_keys:
        if key in flattened:
            row[f"sweep.{key}"] = flattened[key]


def default_summary_row(
    base: dict,
    metrics: dict[str, float],
    raw: dict | None = None,
    sweep_keys: list[str] | None = None,
) -> dict:
    row = {
        "dataset": base.get("dataset"),
        "model": base.get("model"),
        "seq_len": base.get("seq_len"),
        "pred_len": base.get("pred_len"),
        "seed": base.get("seed"),
        "run_id": base.get("run_id"),
        "validation_mse": base.get("validation_mse"),
        "checkpoint_path": base.get("checkpoint_path"),
    }

    metric_order = ["mae", "mse", "rmse", "mape", "mspe"]
    for name in metric_order:
        if name in metrics:
            row[name] = metrics[name]
    for name, value in metrics.items():
        if name not in row:
            row[name] = value

    if raw and sweep_keys:
        _append_sweep_values(row, raw, sweep_keys)
    return row
