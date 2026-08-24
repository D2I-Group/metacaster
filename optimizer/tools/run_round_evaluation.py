
from __future__ import annotations

import json
import re
from pathlib import Path

from optimizer.core.config import AGENT_ROOT
from optimizer.runtime.round_evaluator import run_round_evaluation

SCHEMA = {
    "type": "function",
    "name": "run_round_evaluation",
    "description": (
        "Run all generations + trainings + scoring for this round. Blocks "
        "until complete (~30 min). Returns model_pool, per_pair (with "
        "failed/skipped flags), per_dataset, per_dataset_dist, delta_vs_best, "
        "audit_summary, n_failed/n_skipped, phase_timings. Pass the same "
        "fields to finalize_round."
    ),
    "parameters": {
        "type": "object",
        "properties": {"round_n": {"type": "integer"}},
        "required": ["round_n"],
    },
}


def _load_best_history(run_root: Path) -> tuple[list[dict] | None, dict | None]:
    best_md = run_root / "best" / "best_round.md"
    if not best_md.exists():
        return None, None
    text = best_md.read_text(encoding="utf-8")

    m = re.search(r"round_n[^0-9]+(\d+)", text)
    if not m:
        return None, None
    best_n = int(m.group(1))
    summary_path = run_root / "rounds" / f"round_{best_n}" / "summary.json"
    if not summary_path.exists():
        return None, None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("per_pair"), summary.get("per_dataset_dist")


def handler(round_n: int) -> str:
    from optimizer.tools import TOOL_CONTEXT

    run_root: Path = TOOL_CONTEXT["run_root"]
    harness_root: Path = TOOL_CONTEXT["harness_root"]
    gpus: list[int] = list(TOOL_CONTEXT.get("allowed_gpus") or [0])

    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    full = json.loads((run_root / "full_baseline.json").read_text(encoding="utf-8"))
    full_clean = {k: v for k, v in full.items() if not k.startswith("_")}

    model_pool = config.get("model_pool")
    if not model_pool:
        return (
            "ERROR: config.json has no model_pool. Driver must populate it at "
            "init_run. Cannot proceed."
        )
    datasets: list[str] = config["datasets"]

    best_per_pair, best_per_dataset_dist = _load_best_history(run_root)

    result = run_round_evaluation(
        run_root=run_root,
        run_id=run_root.name,
        round_n=round_n,
        datasets=datasets,
        pool_models=model_pool,
        harness_root=harness_root,
        data_root=AGENT_ROOT / "data" / "GIFT-Eval",
        full_baseline=full_clean,
        best_per_pair=best_per_pair,
        best_per_dataset_dist=best_per_dataset_dist,
        gpus=gpus,
    )


    return json.dumps(result, indent=2)
