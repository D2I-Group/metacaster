
from __future__ import annotations

import json
import os
import shutil
from math import isfinite
from pathlib import Path

from optimizer.runtime.round_evaluator import _jsonable

from ._whitelist import check_allowed

SCHEMA = {
    "type": "function",
    "name": "finalize_round",
    "description": (
        "Close the round: writes report.md + summary.json, copies harness "
        "to best/ if is_new_best=True, touches the round-done marker. "
        "per_pair / per_dataset_dist / delta_vs_best / audit_summary are "
        "OPTIONAL — when omitted, the handler auto-loads them from "
        "rounds/round_<N>/round_eval_result.json (always written by "
        "run_round_evaluation)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "round_n": {"type": "integer"},
            "model_pool": {"type": "array", "items": {"type": "string"}},
            "narrative": {
                "type": "string",
                "description": "What changed + reading of evidence + hypothesis next.",
            },
            "is_new_best": {"type": "boolean"},
            "rationale": {"type": "string"},
            "per_pair": {
                "type": "array", "items": {"type": "object"},
                "description": (
                    "Optional. If omitted, handler reads from "
                    "round_eval_result.json::per_pair."
                ),
            },
            "per_dataset_dist": {
                "type": "object",
                "description": "Optional. Auto-loaded if omitted.",
            },
            "delta_vs_best": {
                "type": "object",
                "description": "Optional. Auto-loaded if omitted.",
            },
            "audit_summary": {
                "type": "object",
                "description": "Optional. Auto-loaded if omitted.",
            },
            "headline_metrics": {"type": "object"},
            "extras": {"type": "object"},
        },
        "required": [
            "round_n",
            "model_pool",
            "narrative",
            "is_new_best",
            "rationale",
        ],
    },
}


def _table_per_pair(rows: list[dict]) -> str:
    if not rows:
        return "_no per-pair rows_\n"
    cols_pref = [
        "dataset", "model", "mse", "full_mse", "hinge", "raw_relative_mse",
        "mae", "full_mae", "mae_hinge",
    ]
    seen = {k for r in rows for k in r}
    cols = [c for c in cols_pref if c in seen] + sorted(seen - set(cols_pref))
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.4g}")
            elif v is None:
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _table_per_ds_dist(per_ds: dict) -> str:
    if not per_ds:
        return "_no distribution metrics_\n"
    metric_keys: list[str] = []
    for d in per_ds.values():
        for k in d:
            if k not in metric_keys:
                metric_keys.append(k)
    cols = ["dataset", *metric_keys]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for ds in sorted(per_ds):
        row = per_ds[ds]
        cells = [ds]
        for k in metric_keys:
            v = row.get(k)
            cells.append(f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _table_delta(delta: dict) -> str:
    if not delta:
        return "_no cross-round delta (round 0 or no prior best)_\n"
    cols_pref = [
        "this_hinge", "best_hinge", "delta_hinge",
        "this_mmd", "best_mmd", "delta_mmd", "verdict",
    ]
    seen: list[str] = []
    for d in delta.values():
        for k in d:
            if k not in seen:
                seen.append(k)
    cols = ["dataset"] + [c for c in cols_pref if c in seen] + [c for c in seen if c not in cols_pref]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for ds in sorted(delta):
        row = delta[ds]
        cells = [ds]
        for c in cols[1:]:
            v = row.get(c)
            cells.append(f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _render_md(
    round_n: int,
    model_pool: list[str],
    per_pair: list[dict],
    per_dataset_dist: dict | None,
    delta_vs_best: dict | None,
    narrative: str,
    audit_summary: dict | None,
) -> str:
    parts = [
        f"# Round {round_n} Report",
        "",
        f"**LT-Lib optimization pool** (K={len(model_pool)}): {', '.join(model_pool) or '(none)'}",
        "",
        "## Per-(dataset, model) prediction metrics",
        "",
        _table_per_pair(per_pair),
        "## Per-dataset distribution metrics",
        "",
        _table_per_ds_dist(per_dataset_dist or {}),
        "## Cross-round delta vs current best",
        "",
        _table_delta(delta_vs_best or {}),
        "## Narrative",
        "",
        narrative.strip() or "_(empty)_",
        "",
    ]
    if audit_summary:
        parts.extend([
            "## Audit summary (primary score and diagnostics)",
            "",
            "```json",
            json.dumps(_jsonable(audit_summary), indent=2),
            "```",
            "",
        ])
    return "\n".join(parts)


def _primary_score(audit_summary: dict | None) -> float | None:
    """Return the paper objective: mean hinge over every evaluated pair."""
    if not audit_summary:
        return None
    value = audit_summary.get("raw_mean_hinge")
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def _is_objective_improvement(
    run_root: Path,
    round_n: int,
    audit_summary: dict | None,
    per_pair: list[dict] | None = None,
) -> bool:
    # Eq. (1) is defined over the complete dataset-LT-Lib optimization pool. Never accept
    # an incomplete candidate whose failed cells disappeared from the mean.
    if not per_pair or any(
        row.get("failed")
        or row.get("skipped")
        or row.get("hinge") is None
        for row in per_pair
    ):
        return False
    score = _primary_score(audit_summary)
    if score is None:
        return False
    best_md = run_root / "best" / "best_round.md"
    if not best_md.is_file():
        return True
    import re

    match = re.search(r"round_n:\s*(\d+)", best_md.read_text(encoding="utf-8"))
    if match is None:
        return True
    best_round = int(match.group(1))
    if best_round == round_n:
        return False
    best_summary_path = run_root / "rounds" / f"round_{best_round}" / "summary.json"
    if not best_summary_path.is_file():
        return True
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
    best_score = _primary_score(best_summary.get("audit_summary"))
    return best_score is None or score < best_score


def handler(
    round_n: int,
    model_pool: list[str],
    narrative: str,
    is_new_best: bool,
    rationale: str,
    per_pair: list[dict] | None = None,
    per_dataset_dist: dict | None = None,
    delta_vs_best: dict | None = None,
    audit_summary: dict | None = None,
    headline_metrics: dict | None = None,
    extras: dict | None = None,
) -> str:
    from optimizer.tools import TOOL_CONTEXT

    run_root: Path = TOOL_CONTEXT["run_root"]
    harness_root: Path = TOOL_CONTEXT["harness_root"]

    r_dir = run_root / "rounds" / f"round_{round_n}"
    md_path = r_dir / f"round_{round_n}_report.md"
    json_path = r_dir / "summary.json"
    marker_path = r_dir / "round_done.marker"


    cached: dict | None = None
    cached_path = r_dir / "round_eval_result.json"
    if cached_path.exists() and (
        per_pair is None or per_dataset_dist is None
        or delta_vs_best is None or audit_summary is None
    ):
        try:
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
        except Exception:
            cached = None
    if cached is not None:
        if per_pair is None:
            per_pair = cached.get("per_pair") or []
        if per_dataset_dist is None:
            per_dataset_dist = cached.get("per_dataset_dist") or {}
        if delta_vs_best is None:
            delta_vs_best = cached.get("delta_vs_best") or {}
        if audit_summary is None:
            audit_summary = cached.get("audit_summary") or {}
        extras = dict(extras or {})
        if cached.get("support_k") is not None:
            extras.setdefault("support_k", cached["support_k"])

    for p in (md_path, json_path, marker_path):
        checked = check_allowed(str(p))
        if isinstance(checked, str):
            return checked

    r_dir.mkdir(parents=True, exist_ok=True)

    # The LLM may recommend a verdict, but the driver owns acceptance.
    is_new_best = _is_objective_improvement(
        run_root, round_n, audit_summary, per_pair
    )

    md = _render_md(
        round_n=round_n,
        model_pool=list(model_pool or []),
        per_pair=per_pair or [],
        per_dataset_dist=per_dataset_dist,
        delta_vs_best=delta_vs_best,
        narrative=narrative,
        audit_summary=audit_summary,
    )
    md_path.write_text(md, encoding="utf-8")


    summary: dict = {
        "round_n": round_n,
        "status": "complete",
        "model_pool": list(model_pool or []),
        "per_pair": per_pair or [],
        "per_dataset_dist": per_dataset_dist or {},
        "delta_vs_best": delta_vs_best or {},
        "audit_summary": audit_summary or {},
        "is_new_best": bool(is_new_best),
        "rationale": rationale,
    }
    if headline_metrics:
        summary["headline_metrics"] = headline_metrics
    if extras:
        summary["extras"] = extras


    json_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")


    best_dir = run_root / "best"
    best_md = best_dir / "best_round.md"
    best_harness = best_dir / "harness"
    best_status = ""
    if is_new_best:
        check2 = check_allowed(str(best_md))
        if isinstance(check2, str):
            return check2
        best_dir.mkdir(parents=True, exist_ok=True)


        tmp = best_dir / f".harness.tmp.{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(
            harness_root,
            tmp,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        if best_harness.exists():
            shutil.rmtree(best_harness)
        os.replace(tmp, best_harness)
        md_lines = [
            f"# Best so far: Round {round_n}",
            "",
            f"round_n: {round_n}",
            "",
        ]
        if headline_metrics:
            md_lines.append("## Headline metrics")
            md_lines.append("")
            for k, v in headline_metrics.items():
                md_lines.append(f"- {k}: {v}")
            md_lines.append("")
        md_lines += [
            "## Rationale",
            "",
            rationale.strip() or "(none provided)",
            "",
            f"Harness snapshot copied to `best/harness/` from `{harness_root}`.",
            "",
        ]
        best_md.write_text("\n".join(md_lines), encoding="utf-8")
        best_status = "  Best updated; harness copied to best/harness/."
    else:
        best_status = "  Not new best; driver will roll harness/ back to best/harness/."


    marker_path.write_text(f"finalize_round({round_n})\n", encoding="utf-8")

    return (
        f"Round {round_n} closed. "
        f"Wrote {md_path.relative_to(run_root)} ({len(md)} chars) and "
        f"{json_path.relative_to(run_root)} (rows={len(per_pair or [])}, "
        f"lt_lib_pool={len(model_pool or [])}).{best_status} Marker touched."
    )
