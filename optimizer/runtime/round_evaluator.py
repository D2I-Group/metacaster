
from __future__ import annotations

import contextlib
import json
import subprocess
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from threading import Lock

from agents.ftagent import run_panel_training
from optimizer.core.scoring import score_against_full


def _jsonable(obj):
    try:
        import numpy as _np
    except ImportError:
        _np = None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj

AGENT_ROOT = Path(__file__).resolve().parents[2]
LT_LIB_ROOT = AGENT_ROOT / "lt_lib"


DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 5


DEFAULT_GEN_TIMEOUT = 900
DEFAULT_TRAIN_TIMEOUT = 600
DEFAULT_TRAIN_PER_GPU = 2
DEFAULT_GEN_CONCURRENCY = 10


@dataclass
class _ProgressLog:
    path: Path
    lock: Lock = field(default_factory=Lock)

    def event(self, **kwargs):
        kwargs["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = json.dumps(kwargs, default=str)
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def _run_one_generation(
    *,
    run_id: str,
    round_n: int,
    dataset_key: str,
    dataset_dir: Path,
    out_dir: Path,
    harness_root: Path,
    timeout_s: int,
    log_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"gen_{dataset_key}.log"
    started = time.time()


    sampler_cmd = [
        "uv", "run", "python",
        str(AGENT_ROOT / "optimizer" / "runtime" / "sample_few_shot.py"),
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(out_dir),
        "--run-id", run_id,
        "--round-n", str(round_n),
    ]
    try:
        sampler_proc = subprocess.run(
            sampler_cmd,
            cwd=AGENT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ds": dataset_key, "ok": False, "dataset_npy": None,
                "n_target": None, "error": f"sample_few_shot timeout: {exc}",
                "elapsed": time.time() - started}

    log_path.write_text(
        f"# {dataset_key} sample_few_shot\n"
        f"cmd: {' '.join(sampler_cmd)}\n"
        f"rc: {sampler_proc.returncode}\n"
        f"--- stdout ---\n{sampler_proc.stdout}\n"
        f"--- stderr ---\n{sampler_proc.stderr}\n",
        encoding="utf-8",
    )
    if sampler_proc.returncode != 0:
        return {"ds": dataset_key, "ok": False, "dataset_npy": None,
                "n_target": None,
                "error": f"sample_few_shot rc={sampler_proc.returncode}; see {log_path}",
                "elapsed": time.time() - started}


    n_target = None
    support_k = None
    for line in sampler_proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "N_target" in line:
            try:
                sampling_record = json.loads(line)
                n_target = int(sampling_record["N_target"])
                support_k = int(sampling_record["n_few_shot"])
                break
            except Exception:
                continue
    if n_target is None:

        sl = out_dir / "sampling_log.json"
        if sl.exists():
            with contextlib.suppress(Exception):
                sampling_record = json.loads(sl.read_text())
                n_target = int(sampling_record["N_target"])
                support_k = int(sampling_record["n_few_shot"])
    if n_target is None:
        return {"ds": dataset_key, "ok": False, "dataset_npy": None,
                "n_target": None,
                "error": "could not determine N_target from sampler",
                "elapsed": time.time() - started}


    task = (
        f"Generate {n_target} synthetic windows from the few-shot set at "
        f"{out_dir}/agent_view/. Output shape ({n_target}, total_len, "
        f"C_eff) float32 to {out_dir}/dataset.npy."
    )
    gen_cmd = [
        "uv", "run", "python", "-m", "generation.agent",
        "--input-dir", str(out_dir / "agent_view"),
        "--harness-root", str(harness_root),
        "--output", str(out_dir),
        "--n-target", str(n_target),
        task,
    ]
    dataset_npy = out_dir / "dataset.npy"
    dataset_npy.unlink(missing_ok=True)
    elapsed_remaining = max(60, timeout_s - int(time.time() - started))
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n# {dataset_key} generation.agent\n")
        log_f.write(f"cmd: {' '.join(gen_cmd[:4])} ... (task elided)\n")
        log_f.flush()
        try:
            proc = subprocess.run(
                gen_cmd,
                cwd=AGENT_ROOT,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=elapsed_remaining,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            return {"ds": dataset_key, "ok": False, "dataset_npy": None,
                    "n_target": n_target,
                    "error": f"generation.agent timeout after {elapsed_remaining}s: {exc}",
                    "elapsed": time.time() - started}
        except Exception as exc:
            return {"ds": dataset_key, "ok": False, "dataset_npy": None,
                    "n_target": n_target,
                    "error": f"generation.agent crash: {exc}",
                    "elapsed": time.time() - started}

    if rc != 0 or not dataset_npy.exists():
        return {"ds": dataset_key, "ok": False, "dataset_npy": None,
                "n_target": n_target,
                "error": f"generation.agent rc={rc}, dataset.npy {'exists' if dataset_npy.exists() else 'missing'}; see {log_path}",
                "elapsed": time.time() - started}

    return {"ds": dataset_key, "ok": True, "dataset_npy": str(dataset_npy),
            "n_target": n_target, "support_k": support_k, "error": None,
            "elapsed": time.time() - started}


def _phase_generation(
    *,
    run_id: str,
    round_n: int,
    datasets: list[str],
    data_root: Path,
    round_dir: Path,
    harness_root: Path,
    log_dir: Path,
    timeout_s: int,
    concurrency: int,
    progress: _ProgressLog,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    log_dir.mkdir(parents=True, exist_ok=True)
    progress.event(phase="gen", event="phase_start", n=len(datasets))

    futures = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for ds in datasets:
            ds_dir = data_root / "train" / ds
            out_dir = round_dir / ds
            progress.event(phase="gen", event="submit", ds=ds)
            fut = pool.submit(
                _run_one_generation,
                run_id=run_id, round_n=round_n,
                dataset_key=ds, dataset_dir=ds_dir, out_dir=out_dir,
                harness_root=harness_root,
                timeout_s=timeout_s, log_dir=log_dir,
            )
            futures[fut] = ds

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                ds = futures.pop(fut)
                try:
                    r = fut.result()
                except Exception as exc:
                    r = {"ds": ds, "ok": False, "dataset_npy": None,
                         "n_target": None, "error": f"future crash: {exc}",
                         "elapsed": 0.0}
                results[ds] = r
                progress.event(phase="gen", event="done",
                               ds=ds, ok=r["ok"], elapsed=r["elapsed"],
                               error=r["error"])

    progress.event(phase="gen", event="phase_done",
                   n_ok=sum(1 for r in results.values() if r["ok"]),
                   n_failed=sum(1 for r in results.values() if not r["ok"]))
    return results


def _phase_scoring(
    train_results: list[dict],
    full_baseline: dict,
) -> tuple[list[dict], dict, dict]:

    candidate: dict[str, dict[str, dict[str, float]]] = {}
    for r in train_results:
        if not r.get("ok"):
            continue
        candidate.setdefault(r["dataset"], {})[r["model"]] = {
            "mse": r["mse"], "mae": r["mae"]
        }

    if candidate:
        scored = score_against_full(candidate, full_baseline)
        per_pair = list(scored["per_pair"])
        per_dataset = scored["per_dataset"]
        audit = scored["audit_summary"]
    else:
        per_pair, per_dataset, audit = [], {}, {
            "n_pairs": 0, "n_datasets": 0,
            "raw_mean_hinge": None, "median_hinge_over_pairs": None,
            "winsorized_mean_hinge_at_ceiling": None, "winsorize_ceiling": 5.0,
        }


    for r in train_results:
        if r.get("ok"):
            continue
        ds, model = r["dataset"], r["model"]
        full_metric = full_baseline.get(ds, {}).get(model, {})
        per_pair.append({
            "dataset": ds, "model": model,
            "mse": None, "full_mse": full_metric.get("mse"),
            "hinge": None, "raw_relative_mse": None,
            "skipped": bool(r.get("skipped")),
            "failed": bool(r.get("failed")),
            "error": r.get("error"),
        })

    return per_pair, per_dataset, audit


def _phase_distribution(
    *,
    datasets: list[str],
    train_pool_root: Path,
    gen_results: dict[str, dict],
    round_dir: Path,
    progress: _ProgressLog,
) -> dict[str, dict]:
    progress.event(phase="dist", event="phase_start")
    out: dict[str, dict] = {}


    try:
        from eval.generation_quality import evaluate_run
    except Exception as exc:
        progress.event(phase="dist", event="import_error", error=str(exc))
        return {ds: {"error": f"eval import failed: {exc}"} for ds in datasets}

    for ds in datasets:
        gr = gen_results.get(ds)
        if not gr or not gr.get("ok"):
            out[ds] = {"error": "generation failed; no synthetic to evaluate"}
            progress.event(phase="dist", event="skip", ds=ds)
            continue

        real_path = train_pool_root / f"{ds}.npy"
        if not real_path.exists():
            out[ds] = {"error": f"train_pool missing at {real_path}"}
            progress.event(phase="dist", event="skip", ds=ds, reason="no train_pool")
            continue

        eval_dir = round_dir / ds / "dist_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        progress.event(phase="dist", event="start", ds=ds)
        try:
            metrics = evaluate_run(
                full_path=str(real_path),
                gen_path=gr["dataset_npy"],
                sample_path=None,
                output_dir=str(eval_dir),
                dataset_name=ds,
            )
            out[ds] = {k: v for k, v in metrics.items() if not k.startswith("_")}
            progress.event(phase="dist", event="done", ds=ds)
        except Exception as exc:
            out[ds] = {"error": f"{exc}"}
            progress.event(phase="dist", event="error", ds=ds, error=str(exc))
    progress.event(phase="dist", event="phase_done")
    return out


def _classify_delta(d: float | None, *, flat_band: float = 0.01,
                    catastrophe: float = 0.5) -> str:
    if d is None:
        return "n/a"
    if abs(d) < flat_band:
        return "flat"
    if d <= -flat_band and d > -catastrophe:
        return "improved"
    if d >= flat_band and d < catastrophe:
        return "regressed"
    if d < 0:
        return "huge_improvement"
    return "catastrophe"


def _phase_delta(
    *,
    this_per_pair: list[dict],
    this_per_dataset_dist: dict[str, dict],
    best_per_pair: list[dict] | None,
    best_per_dataset_dist: dict[str, dict] | None,
    datasets: list[str],
) -> dict[str, dict]:
    if best_per_pair is None:
        return {}

    def _ds_median_hinge(rows: list[dict], ds: str) -> float | None:
        vals = [r["hinge"] for r in rows
                if r.get("dataset") == ds and r.get("hinge") is not None]
        return median(vals) if vals else None

    out: dict[str, dict] = {}
    for ds in datasets:
        this_h = _ds_median_hinge(this_per_pair, ds)
        best_h = _ds_median_hinge(best_per_pair, ds)
        delta_h = (this_h - best_h) if (this_h is not None and best_h is not None) else None
        row = {
            "this_hinge": this_h,
            "best_hinge": best_h,
            "delta_hinge": delta_h,
        }

        if this_per_dataset_dist and best_per_dataset_dist:
            this_mmd = this_per_dataset_dist.get(ds, {}).get("mmd")
            best_mmd = best_per_dataset_dist.get(ds, {}).get("mmd")
            row["this_mmd"] = this_mmd
            row["best_mmd"] = best_mmd
            row["delta_mmd"] = (this_mmd - best_mmd) if (this_mmd is not None and best_mmd is not None) else None
        row["verdict"] = _classify_delta(delta_h)
        out[ds] = row
    return out


def run_round_evaluation(
    *,
    run_root: Path,
    run_id: str,
    round_n: int,
    datasets: list[str],
    pool_models: list[str],
    harness_root: Path,
    data_root: Path,
    full_baseline: dict,
    best_per_pair: list[dict] | None = None,
    best_per_dataset_dist: dict[str, dict] | None = None,
    gpus: list[int],
    gen_concurrency: int = DEFAULT_GEN_CONCURRENCY,
    train_per_gpu: int = DEFAULT_TRAIN_PER_GPU,
    gen_timeout_s: int = DEFAULT_GEN_TIMEOUT,
    train_timeout_s: int = DEFAULT_TRAIN_TIMEOUT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
) -> dict:
    run_root = Path(run_root)
    round_dir = run_root / "rounds" / f"round_{round_n}"
    round_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = round_dir / "eval_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress = _ProgressLog(progress_dir / "progress.jsonl")
    progress.event(
        event="run_start",
        round_n=round_n,
        datasets=datasets,
        model_pool=pool_models,
        gpus=gpus,
    )

    test_pool_root = run_root / "_shared" / "test_pool"
    train_pool_root = run_root / "_shared" / "train_pool"
    gen_log_dir = progress_dir / "gen_logs"
    train_log_dir = progress_dir / "train_logs"

    overall_started = time.time()
    timings: dict[str, float] = {}
    phase_errors: dict[str, str] = {}


    t0 = time.time()
    try:
        gen_results = _phase_generation(
            run_id=run_id, round_n=round_n,
            datasets=datasets, data_root=data_root,
            round_dir=round_dir, harness_root=harness_root,
            log_dir=gen_log_dir,
            timeout_s=gen_timeout_s, concurrency=gen_concurrency,
            progress=progress,
        )
    except Exception as exc:
        phase_errors["gen"] = f"{exc!r}"
        gen_results = {ds: {"ok": False, "error": f"phase crash: {exc!r}",
                            "dataset_npy": None} for ds in datasets}
        progress.event(phase="gen", event="phase_crash", error=str(exc))
    timings["gen"] = time.time() - t0


    t0 = time.time()
    try:
        train_results = run_panel_training(
            run_id=run_id, round_n=round_n,
            pool_models=pool_models, gen_results=gen_results,
            data_root=data_root, round_dir=round_dir,
            test_pool_root=test_pool_root,
            log_dir=train_log_dir,
            epochs=epochs, batch_size=batch_size, patience=patience,
            timeout_s=train_timeout_s,
            gpus=gpus, train_per_gpu=train_per_gpu,
            progress=progress,
        )
    except Exception as exc:
        phase_errors["train"] = f"{exc!r}"
        train_results = []
        progress.event(phase="train", event="phase_crash", error=str(exc))
    timings["train"] = time.time() - t0


    t0 = time.time()
    try:
        per_pair, per_dataset, audit_summary = _phase_scoring(
            train_results=train_results,
            full_baseline=full_baseline,
        )
    except Exception as exc:
        phase_errors["score"] = f"{exc!r}"
        per_pair, per_dataset, audit_summary = [], {}, {
            "n_pairs": 0, "n_datasets": 0,
            "raw_mean_hinge": None, "median_hinge_over_pairs": None,
            "winsorized_mean_hinge_at_ceiling": None, "winsorize_ceiling": 5.0,
            "phase_error": f"{exc!r}",
        }
        progress.event(phase="score", event="phase_crash", error=str(exc))
    timings["score"] = time.time() - t0


    t0 = time.time()
    try:
        per_dataset_dist = _phase_distribution(
            datasets=datasets,
            train_pool_root=train_pool_root,
            gen_results=gen_results,
            round_dir=round_dir,
            progress=progress,
        )
    except Exception as exc:
        phase_errors["dist"] = f"{exc!r}"
        per_dataset_dist = {ds: {"error": f"phase crash: {exc!r}"} for ds in datasets}
        progress.event(phase="dist", event="phase_crash", error=str(exc))
    timings["dist"] = time.time() - t0


    t0 = time.time()
    try:
        delta_vs_best = _phase_delta(
            this_per_pair=per_pair,
            this_per_dataset_dist=per_dataset_dist,
            best_per_pair=best_per_pair,
            best_per_dataset_dist=best_per_dataset_dist,
            datasets=datasets,
        )
    except Exception as exc:
        phase_errors["delta"] = f"{exc!r}"
        delta_vs_best = {}
        progress.event(phase="delta", event="phase_crash", error=str(exc))
    timings["delta"] = time.time() - t0


    t0 = time.time()
    try:
        from optimizer.runtime.auto_plots import generate_round_plots
        plot_paths = generate_round_plots(
            round_dir=round_dir,
            datasets=datasets,
            train_pool_root=train_pool_root,
            gen_results=gen_results,
            per_pair=per_pair,
            per_dataset_dist=per_dataset_dist,
            run_root=run_root,
            round_n=round_n,
        )
    except Exception as exc:
        phase_errors["plots"] = f"{exc!r}"
        plot_paths = {}
        progress.event(phase="plots", event="phase_crash", error=str(exc))
    timings["plots"] = time.time() - t0

    n_failed = sum(1 for r in per_pair if r.get("failed"))
    n_skipped = sum(1 for r in per_pair if r.get("skipped"))

    gen_failures = [f"{ds}: {r['error']}" for ds, r in gen_results.items() if not r["ok"]]
    train_failures = [
        f"{r['dataset']}/{r['model']}: {r.get('error', '?')}"
        for r in train_results if not r.get("ok")
    ]

    support_counts = {
        int(record["support_k"])
        for record in gen_results.values()
        if record.get("ok") and record.get("support_k") is not None
    }
    if len(support_counts) > 1:
        phase_errors["support_k"] = (
            "Paper protocol violation: datasets used different support-shot counts"
        )

    result = {
        "round_n": round_n,
        "support_k": next(iter(support_counts)) if len(support_counts) == 1 else None,
        "model_pool": pool_models,
        "datasets": datasets,
        "per_pair": per_pair,
        "per_dataset": per_dataset,
        "per_dataset_dist": per_dataset_dist,
        "delta_vs_best": delta_vs_best,
        "audit_summary": audit_summary,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "phase_timings": {k: round(v, 1) for k, v in timings.items()},
        "phase_errors": phase_errors,
        "plot_paths": plot_paths,
        "elapsed_seconds": round(time.time() - overall_started, 1),
        "logs": {
            "progress_jsonl": str(progress.path),
            "gen_failures": gen_failures,
            "train_failures": train_failures,
        },
    }
    progress.event(event="run_done",
                   elapsed=result["elapsed_seconds"],
                   n_failed=n_failed, n_skipped=n_skipped)


    result = _jsonable(result)
    (round_dir / "round_eval_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
