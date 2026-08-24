
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from agents.ftagent import ensure_full_baseline
from optimizer.config import BENCHMARKS
from optimizer.core.config import (
    AGENT_ROOT,
    HP_MAX_ROUNDS,
    HP_MAX_TURNS,
    HP_MODEL,
    HP_REASONING_EFFORT,
    build_client,
    harness_dir,
    round_dir,
    run_dir,
)
from optimizer.core.loop import agent_loop
from optimizer.core.prompt import METRIC_DIRECTIVE, SYSTEM
from optimizer.tools import TOOL_CONTEXT, TOOL_HANDLERS, TOOLS


def _summarise_round(summary_json_path: Path) -> dict:
    if not summary_json_path.exists():
        return {}
    s = json.loads(summary_json_path.read_text(encoding="utf-8"))
    per_pair = s.get("per_pair") or []
    return {
        "round_n": s.get("round_n"),
        "is_new_best": s.get("is_new_best"),
        "model_pool": s.get("model_pool", []),
        "n_pairs": len(per_pair),
        "n_failed": sum(1 for r in per_pair if r.get("failed")),
        "n_skipped": sum(1 for r in per_pair if r.get("skipped")),
        "per_pair": per_pair,
        "per_dataset_dist": s.get("per_dataset_dist", {}),
        "delta_vs_best": s.get("delta_vs_best", {}),
        "audit_summary": s.get("audit_summary", {}),
        "narrative_excerpt": (s.get("rationale") or "")[:1500],
    }


def _format_full_baseline(full: dict) -> str:
    rows: list[str] = ["| dataset | model | full_mse | full_mae |",
                       "|---|---|---|---|"]
    for ds in sorted(k for k in full if not k.startswith("_")):
        for model in sorted(full[ds]):
            mt = full[ds][model]
            rows.append(
                f"| {ds} | {model} | "
                f"{mt.get('mse', '?'):.6g} | {mt.get('mae', '?'):.6g} |"
            )
    return "\n".join(rows)


def _read_skill_files(harness_root: Path) -> list[tuple[str, str]]:
    skills_dir = harness_root / "skills"
    out: list[tuple[str, str]] = []
    if not skills_dir.exists():
        return out
    for p in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            body = p.read_text(encoding="utf-8")
        except Exception as exc:
            body = f"[failed to read: {exc}]"
        out.append((str(p.relative_to(harness_root)), body))
    return out


_LT_LIB_POOL_DESC = """\
| model | family | notes |
|---|---|---|
| Linear        | Linear     | linear projection over flattened input |
| DLinear       | Linear     | seasonal/trend decomposition + linear |
| NLinear       | Linear     | last-step normalised linear |
| RLinear       | Linear     | RevIN-normalised linear |
| MixLinear     | Linear     | mixed linear basis |
| TSMixer       | MLP        | MLP-Mixer for time series |
| LightTS       | MLP        | lightweight MLP backbone |
| PatchMLP      | MLP        | MLP over patches |
| xPatch        | MLP        | extended patch MLP |
| CMoS          | MLP        | channel-mixing; patched for C=1 |
| PatchTSMixer  | MLP        | IBM KDD'23 patch mixer |
| FITS          | Frequency  | frequency interpolation |
| CycleNet      | Frequency  | cycle-aware net |
| PaiFilter     | Frequency  | filter-based |
| TexFilter     | Frequency  | filter-based |
| TimeMixer     | Mixing     | multi-scale mixing |
| TimeBridge    | Mixing     | bridge architecture |
| TimeEmb       | Mixing     | time-embedding enhanced |
| Amplifier     | Mixing     | amplifier-based |
| SparseTSF     | Mixing     | sparse time-series forecaster |
"""


def build_state_digest(
    *,
    run_id: str,
    round_n: int,
    config: dict,
    benchmark,
) -> str:
    from optimizer.runtime.skill_registry import (
        load_skills,
        render_full_bodies,
        render_health_report,
        render_manifest_table,
    )

    rroot = run_dir(run_id)
    hroot = harness_dir(run_id)


    router_path = hroot / "core" / "router.md"
    router_text = (
        router_path.read_text(encoding="utf-8")
        if router_path.exists()
        else "(missing — driver setup error)"
    )
    skills = load_skills(hroot)
    skills_manifest = render_manifest_table(skills)
    skills_bodies = render_full_bodies(skills)


    skills_health = render_health_report(skills)


    history_block = ""
    for n in range(round_n):
        summary_path = rroot / "rounds" / f"round_{n}" / "summary.json"
        s = _summarise_round(summary_path)
        if not s:
            history_block += f"\n## Round {n}\n_(no summary.json found)_\n"
            continue
        history_block += (
            f"\n## Round {n}  (is_new_best={s.get('is_new_best')})\n"
            f"- model_pool: {s.get('model_pool')}\n"
            f"- n_pairs: {s.get('n_pairs')} "
            f"(failed={s.get('n_failed')}, skipped={s.get('n_skipped')})\n"
            f"- audit_summary: ```json\n{json.dumps(s.get('audit_summary'), indent=2)}\n```\n"
            f"- per_pair (full):\n```json\n{json.dumps(s.get('per_pair'), indent=2, default=str)}\n```\n"
            f"- per_dataset_dist:\n```json\n{json.dumps(s.get('per_dataset_dist'), indent=2)}\n```\n"
            f"- delta_vs_best:\n```json\n{json.dumps(s.get('delta_vs_best'), indent=2, default=str)}\n```\n"
            f"- narrative excerpt:\n> {s.get('narrative_excerpt')[:1500]}\n"
        )
    if not history_block:
        history_block = "\n_(no prior rounds — this is round 0)_\n"


    best_md_path = rroot / "best" / "best_round.md"
    best_block = (
        best_md_path.read_text(encoding="utf-8") if best_md_path.exists()
        else "_(no best round yet)_"
    )


    log_path = rroot / "optimization_log.md"
    log_block = (
        log_path.read_text(encoding="utf-8") if log_path.exists()
        else "_(empty — round 0 will write the first entry)_"
    )


    full_path = rroot / "full_baseline.json"
    if full_path.exists():
        full = json.loads(full_path.read_text(encoding="utf-8"))
        full_block = _format_full_baseline(full)
    else:
        full_block = "_(missing — driver setup error)_"


    ds_block_lines = ["| dataset | T | C | seq_len | pred_len | freq | domain |",
                      "|---|---|---|---|---|---|---|"]
    for ds in config["datasets"]:
        meta_path = AGENT_ROOT / benchmark.data_root / "train" / ds / "meta.json"
        if not meta_path.exists():
            ds_block_lines.append(f"| {ds} | _missing_ | | | | | |")
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        ds_block_lines.append(
            f"| {ds} | {m.get('T', '?')} | {m.get('C_eff', '?')} | "
            f"{m.get('seq_len', '?')} | {m.get('pred_len', '?')} | "
            f"{m.get('freq', '?')} | {m.get('domain', '?')} |"
        )
    ds_block = "\n".join(ds_block_lines)

    model_pool = config.get("model_pool") or []


    plot_block = ""
    if round_n > 0:
        prev_dir = rroot / "rounds" / f"round_{round_n - 1}"
        plot_paths: list[str] = []
        for p in sorted(prev_dir.rglob("*.png")):
            try:
                plot_paths.append(str(p.relative_to(rroot)))
            except ValueError:
                continue
        if plot_paths:
            plot_block = (
                f"_(driver-generated diagnostic PNGs from round "
                f"{round_n - 1}; READ AT LEAST 2 OF THESE BEFORE PROPOSING "
                f"A HARNESS CHANGE)_\n\n"
                + "\n".join(f"- `{p}`" for p in plot_paths)
            )
        else:
            plot_block = "_(no PNGs from previous round — diagnose from numbers + read_image any plots you produce yourself)_"
    else:
        plot_block = "_(round 0 — no prior plots to read)_"

    return f"""# Round {round_n} state digest (driver-injected)

## Pinned LT-Lib optimization pool (same every round of this run)
**Forecasters trained per round:** {', '.join(model_pool) if model_pool else '_(none — driver setup error)_'}

`run_round_evaluation` trains all 20 non-held-out LT-Lib Forecasters on every
source dataset and reports per-(dataset, Forecaster) hinge. CrossLinear,
TimeBase, and FreqCycle are reserved for the paper's held-out architecture
evaluation.

## LT-Lib optimization pool
{_LT_LIB_POOL_DESC}

## Datasets ({len(config["datasets"])})
{ds_block}

## Current harness — `harness/core/router.md`
```markdown
{router_text}
```

## Current harness — skill library

{skills_manifest}

### Skill library health (must clear before round closes)
{skills_health}

### Full skill bodies
{skills_bodies}

## Diagnostic PNGs (read via `read_image`)
{plot_block}

## Run history (rounds 0..{round_n - 1})
{history_block}

## Current best
{best_block}

## Optimization log (driver-maintained)
{log_block}

## Full baseline (hinge denominators — `(cand_mse - full_mse) / full_mse`)
{full_block}
"""


def _append_optimization_log(run_id: str, round_n: int) -> None:
    rroot = run_dir(run_id)
    summary_path = rroot / "rounds" / f"round_{round_n}" / "summary.json"
    if not summary_path.exists():
        return
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = s.get("audit_summary", {}) or {}
    entry = (
        f"\n## Round {round_n} ({_iso_now()})\n"
        f"- is_new_best: {s.get('is_new_best')}\n"
        f"- median_hinge: {audit.get('median_hinge_over_pairs')}\n"
        f"- raw_mean_hinge: {audit.get('raw_mean_hinge')}\n"
        f"- n_failed: {sum(1 for r in (s.get('per_pair') or []) if r.get('failed'))}, "
        f"n_skipped: {sum(1 for r in (s.get('per_pair') or []) if r.get('skipped'))}\n"
        f"- rationale (first 300 chars): {(s.get('rationale') or '')[:300]}\n"
    )
    log_path = rroot / "optimization_log.md"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)


def _materialize_train_pool(
    run_id: str, datasets: list[str], data_root: Path
) -> dict[str, Path]:
    import numpy as np

    pool_dir = run_dir(run_id) / "_shared" / "train_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, Path] = {}
    for ds in datasets:
        out = pool_dir / f"{ds}.npy"
        materialized[ds] = out
        if out.exists():
            continue
        ds_dir = data_root / "train" / ds
        meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
        total_len = int(meta["total_len"])
        raw = np.load(ds_dir / "raw.npy", mmap_mode="r")
        train_idx = np.load(ds_dir / "train_idx.npy")
        if raw.ndim == 3:
            windows = np.empty(
                (len(train_idx), total_len, raw.shape[-1]), dtype=np.float32
            )
            for k, (s, p) in enumerate(train_idx):
                s_i, p_i = int(s), int(p)
                windows[k] = np.asarray(
                    raw[s_i, p_i : p_i + total_len, :], dtype=np.float32
                )
        else:
            windows = np.stack(
                [
                    np.asarray(raw[int(i) : int(i) + total_len], dtype=np.float32)
                    for i in train_idx
                ],
                axis=0,
            )
        if meta.get("values_normalized"):
            mean = np.asarray(meta["normalization_mean"], dtype=np.float32)
            scale = np.asarray(meta["normalization_scale"], dtype=np.float32)
            windows = ((windows - mean) / scale).astype(np.float32)
        np.save(out, windows)
    return materialized


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(AGENT_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


_ROUTER_STUB = """\
You are the MGAgent.

Your single deliverable is `{output_dir}/dataset.npy`: a float32 numpy
array of synthetic time-series windows with shape
`(N_target, total_len, C_eff)`, matching the few-shot tensor you are given.

Inputs you may read:
    {input_dir}/few_shot.npy   # (N_few_shot, total_len, C_eff) float32
    {input_dir}/meta.json      # semantic + shape fields
    {input_dir}/context.txt    # de-identified domain context

Available skills:
{skill_descriptions}

# Hard contract (non-negotiable — HPAgent cannot remove these)

- Output shape exactly `(N_target, total_len, C_eff)` float32, all finite.
- Never save a 2-D array — always preserve the trailing channel dim even
  for univariate (C_eff = 1) datasets.
- Preserve the few-shot tensor's empirical numeric domain.
- Save once to `{output_dir}/dataset.npy`.

(Everything else — workflow, strategy selection, validation, regime
detection — is up to the skill library the HPAgent has authored.
If no skills are present, design a conservative bootstrap from the
few-shot tensor on your own.)
"""


def _copy_harness(dst_harness: Path) -> None:
    dst_harness.mkdir(parents=True, exist_ok=True)
    (dst_harness / "core").mkdir(parents=True, exist_ok=True)
    router_path = dst_harness / "core" / "router.md"
    if not router_path.exists():
        router_path.write_text(_ROUTER_STUB, encoding="utf-8")

    dst_skills = dst_harness / "skills"
    if dst_skills.exists():
        shutil.rmtree(dst_skills)
    dst_skills.mkdir(parents=True, exist_ok=True)


def _read_best_round(run_id: str) -> int | None:
    best_md = run_dir(run_id) / "best" / "best_round.md"
    if not best_md.exists():
        return None
    text = best_md.read_text(encoding="utf-8")

    m = re.search(r"round_n:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _rollback_harness_to_best(run_id: str) -> bool:
    best_harness = run_dir(run_id) / "best" / "harness"
    if not best_harness.is_dir():
        return False
    target = harness_dir(run_id)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        best_harness,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return True


def init_run(
    run_id: str,
    *,
    seed: int,
    model: str,
    max_rounds: int,
    datasets: list[str],
    benchmark: str,
    resume: bool,
    gpus: list[int] | None = None,
) -> dict:
    rdir = run_dir(run_id)
    cfg_path = rdir / "config.json"


    if cfg_path.exists() and not resume:
        raise SystemExit(
            f"Error: run {run_id} is already initialised ({cfg_path}).\n"
            f"Pass --resume to reuse it, or choose a different --run-id."
        )

    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "rounds").mkdir(parents=True, exist_ok=True)
    (rdir / "logs").mkdir(parents=True, exist_ok=True)

    hdir = harness_dir(run_id)
    if not hdir.exists():
        _copy_harness(hdir)

    if resume and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cfg

    bench = BENCHMARKS[benchmark]
    # The paper objective averages over the complete LT-Lib optimization
    # pool. The three held-out Forecasters are absent from bench.model_pool.
    model_pool = list(bench.model_pool)

    cfg = {
        "run_id": run_id,
        "seed": seed,
        "model": model,
        "max_rounds": max_rounds,
        "started_at": _iso_now(),
        "git_head_sha": _git_head_sha(),
        "benchmark": benchmark,
        "datasets": datasets,
        "gpus": list(gpus) if gpus else [],
        "max_turns": HP_MAX_TURNS,
        "reasoning_effort": HP_REASONING_EFFORT,
        "pool_size": len(model_pool),
        "model_pool": model_pool,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def _materialize_test_pool(
    run_id: str, datasets: list[str], data_root: Path
) -> dict[str, Path]:
    import numpy as np

    pool_dir = run_dir(run_id) / "_shared" / "test_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, Path] = {}
    for ds in datasets:
        out = pool_dir / f"{ds}.npy"
        materialized[ds] = out
        if out.exists():
            continue
        ds_dir = data_root / "train" / ds
        meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
        total_len = int(meta["total_len"])
        raw = np.load(ds_dir / "raw.npy", mmap_mode="r")
        test_idx = np.load(ds_dir / "test_idx.npy")
        if raw.ndim == 3:
            windows = np.empty(
                (len(test_idx), total_len, raw.shape[-1]), dtype=np.float32
            )
            for k, (s, p) in enumerate(test_idx):
                s_i, p_i = int(s), int(p)
                windows[k] = np.asarray(
                    raw[s_i, p_i : p_i + total_len, :], dtype=np.float32
                )
        else:
            windows = np.stack(
                [
                    np.asarray(raw[int(i) : int(i) + total_len], dtype=np.float32)
                    for i in test_idx
                ],
                axis=0,
            )
        if meta.get("values_normalized"):
            mean = np.asarray(meta["normalization_mean"], dtype=np.float32)
            scale = np.asarray(meta["normalization_scale"], dtype=np.float32)
            windows = ((windows - mean) / scale).astype(np.float32)
        np.save(out, windows)
    return materialized


def detect_next_round(run_id: str) -> int:
    rdir = run_dir(run_id) / "rounds"
    rdir.mkdir(parents=True, exist_ok=True)
    done: list[int] = []
    for p in rdir.glob("round_*/round_done.marker"):
        m = re.match(r"round_(\d+)", p.parent.name)
        if m:
            done.append(int(m.group(1)))
    return (max(done) + 1) if done else 0


def _snapshot_harness(run_id: str, r_dir: Path) -> None:
    snap = r_dir / "snapshot"
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(
        harness_dir(run_id),
        snap,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _rollback_if_needed(run_id: str, round_n: int) -> None:
    summary_path = run_dir(run_id) / "rounds" / f"round_{round_n}" / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Round {round_n} did not produce {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("is_new_best") and not _rollback_harness_to_best(run_id):
        raise RuntimeError(
            f"Round {round_n} was rejected but no best Harness is available for rollback"
        )


def run_round(round_n: int, run_id: str, config: dict) -> str:
    r_dir = round_dir(run_id, round_n)
    r_dir.mkdir(parents=True, exist_ok=True)
    (r_dir / "optimizer_log").mkdir(parents=True, exist_ok=True)
    _snapshot_harness(run_id, r_dir)

    benchmark = BENCHMARKS[config["benchmark"]]
    prompt_kwargs = benchmark.to_prompt_kwargs()
    prompt_kwargs["train_datasets"] = ", ".join(config["datasets"])

    gpus = config.get("gpus") or []

    TOOL_CONTEXT.clear()
    TOOL_CONTEXT.update(
        {
            "run_root": run_dir(run_id),
            "harness_root": harness_dir(run_id),
            "agent_root": AGENT_ROOT,
            "repo_root": AGENT_ROOT.parent,
            "allowed_gpus": list(gpus),
        }
    )

    system = SYSTEM.format(
        round_n=round_n,
        repo_root=str(AGENT_ROOT.parent),
        harness_root=str(harness_dir(run_id)),
        run_root=str(run_dir(run_id)),
        primary_metric_directive=METRIC_DIRECTIVE,
        **prompt_kwargs,
    )
    marker = r_dir / "round_done.marker"


    state_digest = build_state_digest(
        run_id=run_id, round_n=round_n, config=config, benchmark=benchmark
    )
    user_msg = (
        f"# Round {round_n} task\n\n"
        f"You are the supervising HPAgent for run_id={run_id}. The "
        f"state digest below is everything you need to start. Diagnose, "
        f"design ONE harness change, edit harness/, call run_round_evaluation, "
        f"reason about the result, then call finalize_round. Marker file "
        f"is touched automatically by finalize_round.\n\n"
        f"---\n\n"
        f"{state_digest}"
    )

    return agent_loop(
        user_input=user_msg,
        system=system,
        client=build_client(),
        model=config["model"],
        max_turns=config["max_turns"],
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
        log_dir=r_dir / "optimizer_log",
        done_check=lambda: marker.exists(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HPAgent for MetaCaster")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique run identifier (default: timestamp_s<seed>).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default=HP_MODEL,
        help="Optimizer model (default: $HP_MODEL).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=HP_MAX_ROUNDS,
        help="Max rounds (default: $HP_MAX_ROUNDS or 8).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Force a specific round. Default: auto-detect next unfinished round.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing run_dir instead of erroring.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gift_eval",
        choices=sorted(BENCHMARKS.keys()),
        help="Benchmark config (default: gift_eval). Determines dataset list, model pool, sampler, paths.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Override training dataset list for this run (default: all from --benchmark).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU indices this run may use, e.g. '0,1,2'. "
            "Sets CUDA_VISIBLE_DEVICES for this process and all children; "
            "also injected into the prompt so training fan-out cycles only "
            "through these devices. Default: all visible GPUs."
        ),
    )
    parser.add_argument(
        "--skip-full-baseline-precompute",
        action="store_true",
        help=(
            "Skip the startup pass that fills full_baseline.json for every "
            "dataset/model pair. By default the optimizer precomputes this "
            "fixed cache before round 0."
        ),
    )
    args = parser.parse_args()

    gpus: list[int] = []
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

    benchmark = BENCHMARKS[args.benchmark]
    seed = args.seed
    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_s{seed}"
    datasets = args.datasets if args.datasets is not None else benchmark.train_datasets

    config = init_run(
        run_id,
        seed=seed,
        model=args.model,
        max_rounds=args.max_rounds,
        datasets=datasets,
        benchmark=args.benchmark,
        resume=args.resume,
        gpus=gpus,
    )
    if args.resume and not gpus and config.get("gpus"):

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config["gpus"])

    print(f"HPAgent run_id={run_id}")
    print(f"  run_dir:     {run_dir(run_id)}")
    print(f"  harness_dir: {harness_dir(run_id)}")
    print(f"  max_rounds:  {args.max_rounds}")
    print(f"  pool_size:    {config['pool_size']}")

    print("Materialising shared test pool...")
    pool = _materialize_test_pool(
        run_id=run_id,
        datasets=datasets,
        data_root=AGENT_ROOT / benchmark.data_root,
    )
    print(f"  test pool: {len(pool)} datasets at {run_dir(run_id) / '_shared' / 'test_pool'}")

    print("Materialising shared train pool (for distribution metrics)...")
    train_pool = _materialize_train_pool(
        run_id=run_id,
        datasets=datasets,
        data_root=AGENT_ROOT / benchmark.data_root,
    )
    print(f"  train pool: {len(train_pool)} datasets at {run_dir(run_id) / '_shared' / 'train_pool'}")

    if not args.skip_full_baseline_precompute:
        print("Precomputing full_baseline.json for all dataset/model pairs...")
        baseline_result = ensure_full_baseline(
            run_root=run_dir(run_id),
            datasets=datasets,
            models=benchmark.model_pool,
            data_root=AGENT_ROOT / benchmark.data_root,
            gpus=gpus,
        )
        print(
            "Full baseline cache ready: "
            f"{baseline_result['cache_path']} "
            f"(trained={len(baseline_result['trained'])}, "
            f"skipped={len(baseline_result['skipped'])})"
        )

    if args.round is not None:
        run_round(args.round, run_id, config)
        _rollback_if_needed(run_id, args.round)
        return

    start = detect_next_round(run_id)
    print(f"Starting from round {start}, max {args.max_rounds}")
    for n in range(start, args.max_rounds):
        run_round(n, run_id, config)
        _rollback_if_needed(run_id, n)
        _append_optimization_log(run_id, n)


    _write_final_summary(run_id, args.max_rounds)


def _write_final_summary(run_id: str, max_rounds: int) -> None:
    rroot = run_dir(run_id)
    cfg = json.loads((rroot / "config.json").read_text(encoding="utf-8"))
    model_pool = cfg.get("model_pool", [])

    rows: list[dict] = []
    for n in range(max_rounds):
        sp = rroot / "rounds" / f"round_{n}" / "summary.json"
        if not sp.exists():
            continue
        try:
            s = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        audit = s.get("audit_summary", {}) or {}
        rows.append({
            "round": n,
            "is_new_best": s.get("is_new_best"),
            "n_pairs": len(s.get("per_pair") or []),
            "median_primary": audit.get("median_hinge_primary"),
            "median_all": audit.get("median_hinge_over_pairs"),
            "raw_mean_primary": audit.get("raw_mean_hinge_primary"),
            "rationale_excerpt": (s.get("rationale") or "")[:160],
        })

    best_round_n = _read_best_round(run_id)

    parts = [
        f"# Final summary — run `{run_id}`",
        "",
        f"- LT-Lib optimization pool: {', '.join(model_pool) if model_pool else '_(none)_'}",
        f"- max_rounds: {max_rounds}",
        f"- rounds_completed: {len(rows)}",
        f"- best_round: {best_round_n if best_round_n is not None else '_(none)_'}",
        "",
        "## Round-by-round trajectory",
        "",
        "| round | new_best | n_pairs | median (primary) | median (all) | mean (primary) | rationale |",
        "|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
        parts.append(
            f"| {r['round']} | "
            f"{'✅' if r['is_new_best'] else '—'} | "
            f"{r['n_pairs']} | "
            f"{fmt(r['median_primary'])} | {fmt(r['median_all'])} | "
            f"{fmt(r['raw_mean_primary'])} | "
            f"{r['rationale_excerpt']} |"
        )


    parts.extend(["", "## Best skill library (saved at `best/harness/`)", ""])
    best_h = rroot / "best" / "harness"
    if best_h.exists():
        try:
            from optimizer.runtime.skill_registry import (
                load_skills,
                render_manifest_table,
            )
            skills = load_skills(best_h)
            parts.append(render_manifest_table(skills))
        except Exception as exc:
            parts.append(f"_(failed to render skill manifest: {exc})_")
    else:
        parts.append("_(no `best/harness/` saved)_")

    parts.extend([
        "",
        "## Output",
        "",
        f"- Optimized Harness: `{rroot / 'best' / 'harness'}`",
        f"- Run summary: `{rroot / 'final_summary.md'}`",
        "",
    ])

    out = rroot / "final_summary.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Final summary written to {out}")


if __name__ == "__main__":
    main()
