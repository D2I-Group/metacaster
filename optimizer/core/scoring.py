
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from statistics import mean, median

MetricTable = Mapping[str, Mapping[str, Mapping[str, float]]]


AUDIT_FULL_MSE_THRESHOLD = 1e-3


@dataclass(frozen=True)
class HingeScore:

    raw_relative: float
    hinge: float


def hinge_full_loss(
    candidate_mse: float,
    full_mse: float,
    *,
    eps: float = 1e-12,
) -> HingeScore:

    candidate = float(candidate_mse)
    full = float(full_mse)
    if not (isfinite(candidate) and isfinite(full)):
        raise ValueError("candidate_mse and full_mse must be finite")
    if candidate < 0:

        candidate = 0.0


    full_safe = max(full, eps)
    raw_relative = (candidate - full_safe) / full_safe
    hinge = max(raw_relative, 0.0)
    return HingeScore(raw_relative=raw_relative, hinge=hinge)


def _winsorized_mean(values: list[float], ceiling: float) -> float:
    if not values:
        return 0.0
    return mean(min(v, ceiling) for v in values)


def score_against_full(
    candidate: MetricTable,
    full_baseline: MetricTable,
    *,
    winsorize_ceiling: float = 5.0,
) -> dict:

    per_pair: list[dict] = []
    per_ds_hinge: dict[str, dict[str, float]] = {}
    missing: list[str] = []

    for dataset, model_metrics in candidate.items():
        for model, metrics in model_metrics.items():
            full_metrics = full_baseline.get(dataset, {}).get(model)
            if not full_metrics:
                missing.append(f"{dataset}/{model}")
                continue
            score = hinge_full_loss(metrics["mse"], full_metrics["mse"])
            full_mse_val = float(full_metrics["mse"])
            row: dict = {
                "dataset": dataset,
                "model": model,
                "mse": float(metrics["mse"]),
                "full_mse": full_mse_val,
                "raw_relative_mse": score.raw_relative,
                "hinge": score.hinge,
                "audit_only": full_mse_val < AUDIT_FULL_MSE_THRESHOLD,
            }
            if "mae" in metrics and "mae" in full_metrics:
                mae_score = hinge_full_loss(metrics["mae"], full_metrics["mae"])
                row.update(
                    {
                        "mae": float(metrics["mae"]),
                        "full_mae": float(full_metrics["mae"]),
                        "raw_relative_mae": mae_score.raw_relative,
                        "mae_hinge": mae_score.hinge,
                    }
                )
            per_pair.append(row)
            per_ds_hinge.setdefault(dataset, {})[model] = score.hinge

    if missing:
        raise KeyError("Missing full baseline entries: " + ", ".join(missing))
    if not per_pair:
        raise ValueError("No candidate metrics were provided")

    per_dataset: dict[str, dict] = {}
    for ds, by_model in sorted(per_ds_hinge.items()):
        h_values = list(by_model.values())
        per_dataset[ds] = {
            "models": sorted(by_model.keys()),
            "hinge_per_model": {m: by_model[m] for m in sorted(by_model)},
            "median_hinge": median(h_values),
            "max_hinge": max(h_values),
            "n_models": len(h_values),
        }

    all_h = [row["hinge"] for row in per_pair]
    primary_h = [row["hinge"] for row in per_pair if not row.get("audit_only")]
    n_audit_only = sum(1 for row in per_pair if row.get("audit_only"))

    audit_summary = {
        "n_pairs": len(per_pair),
        "n_audit_only": n_audit_only,
        "n_primary_pairs": len(primary_h),
        "n_datasets": len(per_dataset),


        "raw_mean_hinge": mean(all_h),
        "median_hinge_over_pairs": median(all_h),
        "winsorized_mean_hinge_at_ceiling": _winsorized_mean(all_h, winsorize_ceiling),


        "raw_mean_hinge_primary": mean(primary_h) if primary_h else None,
        "median_hinge_primary": median(primary_h) if primary_h else None,
        "winsorized_mean_hinge_primary": (
            _winsorized_mean(primary_h, winsorize_ceiling) if primary_h else None
        ),
        "winsorize_ceiling": float(winsorize_ceiling),
        "audit_full_mse_threshold": AUDIT_FULL_MSE_THRESHOLD,
    }

    return {
        "per_pair": per_pair,
        "per_dataset": per_dataset,
        "audit_summary": audit_summary,
    }
