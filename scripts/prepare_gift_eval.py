
from __future__ import annotations

import argparse
import concurrent.futures as _futures
import json
import math
import os
import re
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

LONG = {"seq_len": 336, "pred_len": 192, "total_len": 528, "min_len": 700}
MAX_RAW_T = 50000
NAN_RATIO_MAX = 0.05
N_TRAIN = 5000
N_TEST = 100
N_FEWSHOT_TEST = (10, 30, 50)
SAMPLING_SEED = 42
PIPELINE_VERSION = "gift-eval-5.0"


DOMAIN_MAP = {

    "australian_electricity_demand": ("energy",   "Australian state electricity demand"),
    "solar_power":                   ("energy",   "Renewable PV historical generation"),
    "PEMS_BAY":                      ("traffic",  "PEMS-BAY road-network traffic"),
    "traffic_hourly":                ("traffic",  "Aggregate hourly traffic flow"),
    "weather":                       ("weather",  "Daily station weather"),
    "cdc_fluview_ilinet":            ("health",   "CDC FluView ILINet epidemiology"),
    "m5":                            ("retail",   "M5 / Walmart product sales"),
    "alibaba_cluster_trace_2018":    ("cloud",    "Alibaba 2018 cluster compute trace"),
    "fred_md":                       ("finance",  "FRED-MD macroeconomic indicators"),
    "tourism_quarterly":             ("tourism",  "Quarterly tourism arrivals"),

    "electricity":                   ("energy",       "UCI Portugal residential electricity"),
    "ett1":                          ("energy",       "Electricity transformer oil temperature"),
    "LOOP_SEATTLE":                  ("traffic",      "Seattle inductive loop sensors"),
    "SZ_TAXI":                       ("traffic",      "Shenzhen taxi demand"),
    "hierarchical_sales":            ("retail",       "Hierarchical / grouped retail sales"),
    "temperature_rain_with_missing": ("weather",      "Station temperature/rain with NaN"),
    "bitbrains_fast_storage":        ("cloud",        "Bitbrains fast storage IO traces"),
    "solar":                         ("energy",       "Solar PV power generation"),
    "saugeenday":                    ("hydro",        "Daily Saugeen river flow"),
    "m4_hourly":                     ("mixed",        "M4 competition hourly mixed"),
    "m4_daily":                      ("mixed",        "M4 competition daily mixed"),
    "us_births":                     ("demographics", "Daily US births"),
}


TRAIN_BATCH = [


    ("australian_electricity_demand", None,  "energy",   "national-grid-australia"),
    ("solar_power",                   None,  "energy",   "renewable-PV-historical"),
    ("PEMS_BAY",                      None,  "traffic",  "road-network"),
    ("traffic_hourly",                None,  "traffic",  "aggregate-hourly"),
    ("weather",                       None,  "weather",  "station-daily"),
    ("cdc_fluview_ilinet",            None,  "health",   "epidemiology"),
    ("m5",                            None,  "retail",   "walmart-sales"),
    ("alibaba_cluster_trace_2018",    None,  "cloud",    "compute-trace"),
]

TEST_BATCH = [


    ("electricity",                    "H",   "energy",       "UCI-PT-residential",   "in"),
    ("ett1",                           "15T", "energy",       "transformer-oil-temp", "in"),
    ("LOOP_SEATTLE",                   "H",   "traffic",      "seattle-loop",         "in"),
    ("SZ_TAXI",                        "15T", "traffic",      "shenzhen-taxi",        "in"),
    ("hierarchical_sales",             "D",   "retail",       "grouped-sales",        "in"),
    ("bitbrains_fast_storage",         "5T",  "cloud",        "storage-IO",           "in"),
    ("solar",                          "H",   "energy",       "renewable-PV",         "in"),
    ("saugeenday",                     "D",   "hydro",        "river-flow",           "out"),
    ("m4_daily",                       None,  "mixed",        "m4-comp-daily",        "out"),
    ("us_births",                      "D",   "demographics", "birth-count",          "out"),
]


def normalize_freq(freq_str: str) -> str:
    freq_str = freq_str.strip()
    if re.match(r"^\d+S$", freq_str):
        return "S"
    if re.match(r"^\d+T$", freq_str) or re.match(r"^\d+min$", freq_str, re.I):
        return "T"
    m = re.match(r"^([A-Z]+)", freq_str)
    if m:
        return m.group(1)
    return freq_str


def _build_dataset_key(name: str, freq: str | None) -> str:
    key = name.replace("/", "_")
    if freq is not None:
        key = f"{key}_{freq}"
    return key


def _interpolate_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    for c in range(arr.shape[1]):
        col = arr[:, c]
        mask = np.isnan(col)
        if not mask.any():
            continue
        if mask.all():
            continue
        valid = np.where(~mask)[0]
        col[mask] = np.interp(np.where(mask)[0], valid, col[valid])
        arr[:, c] = col
    return arr


def load_series(
    source_path: str, name: str, freq: str | None = None
) -> tuple[list[np.ndarray], str]:
    from datasets import Dataset as HFDataset, load_from_disk

    ds_path = (
        os.path.join(source_path, name, freq)
        if freq is not None
        else os.path.join(source_path, name)
    )
    single_arrow = os.path.join(ds_path, "data-00000-of-00001.arrow")
    if os.path.exists(single_arrow):
        ds = HFDataset.from_file(single_arrow)
    else:
        ds = load_from_disk(ds_path)
    freq_raw = ds[0]["freq"]
    series_list = []

    for i in range(len(ds)):
        target = np.array(ds[i]["target"], dtype=np.float32)
        if target.ndim == 1:
            target = target[:, None]
        elif target.ndim == 2:

            target = target.T
        series_list.append(target)

    return series_list, freq_raw


def _pick_single_series(
    series_list: list[np.ndarray],
    dataset_name: str,
    *,
    min_required_T: int = LONG["min_len"],
    max_T: int = MAX_RAW_T,
    nan_max: float = NAN_RATIO_MAX,
) -> tuple[np.ndarray, dict]:
    if not series_list:
        raise ValueError(f"{dataset_name}: empty series list")

    normalised: list[np.ndarray] = []
    C_values: list[int] = []
    for s in series_list:
        s = s if s.ndim == 2 else s[:, None]
        s = s.astype(np.float32)
        normalised.append(s)
        C_values.append(s.shape[1])

    N_source_total = len(normalised)

    c_mode, _ = Counter(C_values).most_common(1)[0]
    if not all(c == c_mode for c in C_values):
        kept = [s for s in normalised if s.shape[1] == c_mode]
        print(
            f"  Channel-filter: kept {len(kept)}/{len(normalised)} series "
            f"with C={c_mode}"
        )
        normalised = kept

    if not normalised:
        raise ValueError(f"{dataset_name}: no series after channel filtering")

    candidates = []
    for s in normalised:
        size = s.size
        if size == 0:
            continue
        raw_nan = float(np.isnan(s).sum()) / size
        if raw_nan > 0.5:
            continue
        s_filled = _interpolate_nan(s) if raw_nan > 0 else s

        residual_nan = float(np.isnan(s_filled).sum()) / s_filled.size
        if residual_nan > nan_max:
            continue
        T_original = int(s_filled.shape[0])
        truncated = T_original > max_T
        if truncated:
            s_filled = s_filled[-max_T:]
        T_i = int(s_filled.shape[0])
        if T_i < min_required_T:
            continue
        candidates.append((T_i, raw_nan, T_original, truncated, s_filled))

    N_after_nan_filter = len(candidates)
    if N_after_nan_filter == 0:
        raise ValueError(
            f"{dataset_name}: no series survived NaN/length filtering "
            f"(N_source_total={N_source_total})"
        )


    candidates.sort(key=lambda t: (-t[0], t[1]))
    T_i, raw_nan, T_original, truncated, raw = candidates[0]

    info = {
        "C_eff": int(raw.shape[1]),
        "N_source_total": int(N_source_total),
        "N_after_nan_filter": int(N_after_nan_filter),
        "T": int(T_i),
        "T_original": int(T_original),
        "truncated": bool(truncated),
        "raw_nan_ratio": float(raw_nan),
    }
    print(
        f"  Picker: N_source={N_source_total}, after_filter={N_after_nan_filter}, "
        f"T={T_i} (orig={T_original}, truncated={truncated}), "
        f"raw_nan={raw_nan:.4f}"
    )
    return raw, info


def _train_validation_window_count(raw_length: int, total_len: int) -> int:
    train_boundary = math.floor(0.8 * raw_length)
    val_boundary = math.floor(0.9 * raw_length)
    train_windows = max(0, train_boundary - total_len + 1)
    validation_windows = max(0, val_boundary - train_boundary - total_len + 1)
    return train_windows + validation_windows


def _split_and_sample(
    raw: np.ndarray,
    *,
    total_len: int,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    T = int(raw.shape[0])
    train_boundary = math.floor(0.8 * T)
    val_boundary = math.floor(0.9 * T)
    train_starts = np.arange(0, train_boundary - total_len + 1, dtype=np.int64)
    val_starts = np.arange(
        train_boundary, val_boundary - total_len + 1, dtype=np.int64
    )
    test_starts = np.arange(val_boundary, T - total_len + 1, dtype=np.int64)
    if min(len(train_starts), len(val_starts), len(test_starts)) <= 0:
        raise ValueError(
            f"Empty chronological split pool (T={T}, total_len={total_len})"
        )

    rng = np.random.default_rng(seed)
    n_val = max(1, round(n_train / 9))
    n_train_only = n_train - n_val
    sampled_train = rng.choice(
        train_starts, size=n_train_only, replace=len(train_starts) < n_train_only
    ).astype(np.int32)
    sampled_val = rng.choice(
        val_starts, size=n_val, replace=len(val_starts) < n_val
    ).astype(np.int32)
    test_idx = rng.choice(
        test_starts, size=n_test, replace=len(test_starts) < n_test
    ).astype(np.int32)
    rng.shuffle(sampled_train)
    rng.shuffle(sampled_val)
    rng.shuffle(test_idx)
    train_idx = np.concatenate([sampled_train, sampled_val])
    return train_idx, test_idx


def _materialize_windows(
    raw: np.ndarray, idx: np.ndarray, total_len: int
) -> np.ndarray:
    N = len(idx)
    C = int(raw.shape[1])
    out = np.empty((N, total_len, C), dtype=np.float32)
    for k in range(N):
        p = int(idx[k])
        out[k] = raw[p : p + total_len]
    return out


def _deidentified_context(domain: str, frequency: str, channels: int) -> str:
    return (
        f"Time-series observations in the {domain} domain, sampled at frequency "
        f"{frequency}, with {channels} channel(s).\n"
    )


AGENT_META_FIELDS = (
    "freq",
    "domain",
    "total_len",
    "seq_len",
    "pred_len",
    "C_eff",
    "n_few_shot",
    "n_target",
    "values_normalized",
    "normalization_mean",
    "normalization_scale",
)
"""Whitelist for ``agent_view/meta.json``. Topology-revealing fields
(T, T_original, sub_domain, sampling_seed, ...) MUST NOT appear here -
MGAgent strict-reads only this file."""


def _process_common(
    name: str,
    freq: str | None,
    domain: str,
    sub_domain: str,
    source_path: str,
    output_root: Path,
    role: str,
    domain_class: str | None,
    seed: int,
    materialize_agent_view: bool,
) -> None:
    label = f"{name}/{freq}" if freq else name
    print(f"\n{'=' * 60}\n{role.upper()} {label}\n{'=' * 60}")

    raw_series, freq_raw = load_series(source_path, name, freq=freq)
    freq_norm = normalize_freq(freq_raw)
    print(f"  Loaded {len(raw_series)} raw series, freq={freq_raw}")

    raw, pick_info = _pick_single_series(
        raw_series,
        dataset_name=label,
        min_required_T=LONG["min_len"],
        max_T=MAX_RAW_T,
        nan_max=NAN_RATIO_MAX,
    )
    C = pick_info["C_eff"]
    train_boundary = math.floor(0.8 * len(raw))
    train_values = np.asarray(raw[:train_boundary], dtype=np.float64)
    normalization_mean = train_values.mean(axis=0)
    normalization_scale = train_values.std(axis=0)
    normalization_scale[normalization_scale == 0] = 1.0

    train_idx, test_idx = _split_and_sample(
        raw,
        total_len=LONG["total_len"],
        n_train=N_TRAIN,
        n_test=N_TEST,
        seed=seed,
    )
    print(f"  Sampled idx: train={len(train_idx)}, test={len(test_idx)}")

    ds_key = _build_dataset_key(name, freq)
    out_dir = output_root / ds_key
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "raw.npy", raw)
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "test_idx.npy", test_idx)
    test_windows = _materialize_windows(raw, test_idx, LONG["total_len"])
    test_windows = (
        (test_windows - normalization_mean) / normalization_scale
    ).astype(np.float32)
    np.save(out_dir / "test.npy", test_windows)

    n_few_shot_meta = max(N_FEWSHOT_TEST) if materialize_agent_view else 0

    meta = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_key": ds_key,
        "name": name,
        "freq": freq_norm,
        "domain": domain,
        "sub_domain": sub_domain,
        "role": role,
        "domain_class": domain_class,
        "T": int(pick_info["T"]),
        "train_boundary": int(train_boundary),
        "T_original": int(pick_info["T_original"]),
        "truncated": bool(pick_info["truncated"]),
        "C_eff": int(C),
        "N_source_total": int(pick_info["N_source_total"]),
        "raw_nan_ratio": float(pick_info["raw_nan_ratio"]),
        "seq_len": LONG["seq_len"],
        "pred_len": LONG["pred_len"],
        "total_len": LONG["total_len"],
        "min_len": LONG["min_len"],
        "n_train": len(train_idx),
        "n_val": max(1, round(len(train_idx) / 9)),
        "n_test": len(test_idx),
        "n_few_shot": int(n_few_shot_meta),
        "n_target": _train_validation_window_count(
            int(pick_info["T"]), LONG["total_len"]
        ),
        "values_normalized": True,
        "normalization_mean": normalization_mean.tolist(),
        "normalization_scale": normalization_scale.tolist(),
        "few_shot_budgets": list(N_FEWSHOT_TEST) if materialize_agent_view else [],
        "sampling_seed": int(seed),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    context_text = _deidentified_context(domain, freq_norm, C)
    (out_dir / "context.txt").write_text(context_text, encoding="utf-8")

    if materialize_agent_view:
        fs_rng = np.random.default_rng(seed + 1)
        n_val = max(1, round(len(train_idx) / 9))
        support_pool = train_idx[:-n_val]
        ordered_positions = fs_rng.permutation(support_pool)
        for budget in N_FEWSHOT_TEST:
            n_fs = min(budget, len(support_pool))
            windows = _materialize_windows(
                raw, ordered_positions[:n_fs], LONG["total_len"]
            )
            windows = (
                (windows - normalization_mean) / normalization_scale
            ).astype(np.float32)
            agent_dir = out_dir / f"agent_view_k{budget}"
            agent_dir.mkdir(parents=True, exist_ok=True)
            np.save(agent_dir / "few_shot.npy", windows)
            agent_meta = {k: meta[k] for k in AGENT_META_FIELDS if k in meta}
            agent_meta["n_few_shot"] = int(windows.shape[0])
            (agent_dir / "meta.json").write_text(json.dumps(agent_meta, indent=2))
            (agent_dir / "context.txt").write_text(context_text, encoding="utf-8")
            print(f"  Wrote agent_view_k{budget}/few_shot.npy {windows.shape}")

    print(
        f"  Saved -> {out_dir} "
        f"(raw={raw.shape}, agent_view={'yes' if materialize_agent_view else 'no'})"
    )


def process_train_dataset(
    name: str,
    freq: str | None,
    domain: str,
    sub_domain: str,
    source_path: str,
    output_root: Path,
    seed: int = SAMPLING_SEED,
) -> None:
    _process_common(
        name=name,
        freq=freq,
        domain=domain,
        sub_domain=sub_domain,
        source_path=source_path,
        output_root=output_root,
        role="train",
        domain_class=None,
        seed=seed,
        materialize_agent_view=False,
    )


def process_test_dataset(
    name: str,
    freq: str | None,
    domain: str,
    sub_domain: str,
    domain_class: str,
    source_path: str,
    output_root: Path,
    seed: int = SAMPLING_SEED,
) -> None:
    _process_common(
        name=name,
        freq=freq,
        domain=domain,
        sub_domain=sub_domain,
        source_path=source_path,
        output_root=output_root,
        role="test",
        domain_class=domain_class,
        seed=seed,
        materialize_agent_view=True,
    )


DEFAULT_TRAIN_SRC = os.getenv("GIFT_EVAL_PRETRAIN", "data/GiftEvalPretrain")
DEFAULT_TEST_SRC = os.getenv("GIFT_EVAL", "data/GIFT_Eval")
DEFAULT_OUTPUT_ROOT = os.getenv("GIFT_EVAL_BENCHMARK_ROOT", "data/GIFT-Eval")
DEFAULT_WORKERS = 4


def _resolve_source(
    arg_val: str | None, env_primary: str, env_fallback: str, default: str
) -> str:
    if arg_val:
        return arg_val
    return os.environ.get(env_primary) or os.environ.get(env_fallback) or default


def _validate_source(path: str, label: str) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(
            f"{label} source directory does not exist: {source}. "
            "Set the corresponding .env variable or pass an explicit source path."
        )
    return str(source)


def _filter_only(entries: list[tuple], only: list[str] | None) -> list[tuple]:
    if not only:
        return entries
    only_set = set(only)
    out = []
    for e in entries:
        base = e[0]
        if base in only_set or base.split("/")[0] in only_set:
            out.append(e)
    return out


def _already_done(
    output_root: Path, name: str, freq: str | None, *, expect_agent_view: bool
) -> bool:
    ds_dir = output_root / _build_dataset_key(name, freq)
    core_ok = (
        (ds_dir / "raw.npy").exists()
        and (ds_dir / "train_idx.npy").exists()
        and (ds_dir / "test_idx.npy").exists()
        and (ds_dir / "test.npy").exists()
        and (ds_dir / "meta.json").exists()
        and (ds_dir / "context.txt").exists()
    )
    if not core_ok:
        return False
    try:
        prepared_meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if prepared_meta.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if expect_agent_view:
        return all(
            (ds_dir / f"agent_view_k{budget}" / filename).exists()
            for budget in N_FEWSHOT_TEST
            for filename in ("few_shot.npy", "meta.json", "context.txt")
        )
    return True


def _train_worker(task: dict) -> tuple[str, str, str]:
    name = task["name"]
    freq = task["freq"]
    freq_label = freq or "-"
    output_root = Path(task["output_root"])
    try:
        if not task["force"] and _already_done(
            output_root, name, freq, expect_agent_view=False
        ):
            return (name, freq_label, "skip")
        process_train_dataset(
            name=name,
            freq=freq,
            domain=task["domain"],
            sub_domain=task["sub_domain"],
            source_path=task["source_path"],
            output_root=output_root,
            seed=task["seed"],
        )
        return (name, freq_label, "ok")
    except Exception as e:
        tb = traceback.format_exc()
        return (name, freq_label, f"error:{type(e).__name__}: {e}\n{tb}")


def _test_worker(task: dict) -> tuple[str, str, str]:
    name = task["name"]
    freq = task["freq"]
    freq_label = freq or "-"
    output_root = Path(task["output_root"])
    try:
        if not task["force"] and _already_done(
            output_root, name, freq, expect_agent_view=True
        ):
            return (name, freq_label, "skip")
        process_test_dataset(
            name=name,
            freq=freq,
            domain=task["domain"],
            sub_domain=task["sub_domain"],
            domain_class=task["domain_class"],
            source_path=task["source_path"],
            output_root=output_root,
            seed=task["seed"],
        )
        return (name, freq_label, "ok")
    except Exception as e:
        tb = traceback.format_exc()
        return (name, freq_label, f"error:{type(e).__name__}: {e}\n{tb}")


def _run_parallel(worker, tasks: list[dict], workers: int, label: str) -> None:
    if not tasks:
        print(f"\n{label}: nothing to do.")
        return

    failures: list[str] = []
    print(f"\n{label}: {len(tasks)} tasks, workers={workers}")
    if workers <= 1:
        for task in tasks:
            name, freq, status = worker(task)
            print(f"  [{label}] {name}/{freq}: {status.splitlines()[0]}")
            if status.startswith("error:"):
                print(status, flush=True)
                failures.append(f"{name}/{freq}")
    else:
        with _futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_label = {
                executor.submit(worker, task): f"{task['name']}/{task['freq'] or '-'}"
                for task in tasks
            }
            for future in _futures.as_completed(future_to_label):
                task_label = future_to_label[future]
                try:
                    name, freq, status = future.result()
                except Exception as exc:
                    print(
                        f"  [{label}] {task_label}: worker crashed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    failures.append(task_label)
                    continue
                first_line = status.splitlines()[0] if status else status
                print(f"  [{label}] {name}/{freq}: {first_line}")
                if status.startswith("error:"):
                    print(status, flush=True)
                    failures.append(f"{name}/{freq}")
    if failures:
        raise RuntimeError(
            f"{label} preparation failed for {len(failures)} dataset(s): "
            + ", ".join(sorted(failures))
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Prepare Harness-training subset (8 datasets)")
    group.add_argument("--test", action="store_true", help="Prepare test subset (10 datasets)")
    group.add_argument("--all", action="store_true", help="Prepare both train and test subsets")

    p.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these dataset names (freq-agnostic; matches base name)",
    )
    p.add_argument("--source-train", default=None, help="Path to train source")
    p.add_argument("--test-source", "--source-test", dest="source_test", default=None,
                   help="Path to test source")
    p.add_argument("--train-source", dest="source_train_alt", default=None,
                   help="Alias of --source-train")
    p.add_argument(
        "--output-root",
        default=None,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    p.add_argument("--seed", type=int, default=SAMPLING_SEED)
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers (default: %(default)s; one process per dataset)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-generate datasets whose output already exists",
    )
    args = p.parse_args()

    agent_root = Path(
        os.environ.get("AGENT_ROOT", Path(__file__).resolve().parent.parent)
    )
    output_root_arg = args.output_root if args.output_root else DEFAULT_OUTPUT_ROOT
    output_root_rel = Path(output_root_arg)
    output_root = (
        output_root_rel if output_root_rel.is_absolute() else agent_root / output_root_rel
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {output_root}")
    print(f"Pipeline version: {PIPELINE_VERSION}")

    do_train = args.train or args.all
    do_test = args.test or args.all

    if do_train:
        source_train = _validate_source(
            _resolve_source(
                args.source_train or args.source_train_alt,
                "GIFT_EVAL_PRETRAIN",
                "GIFT_EVAL_PRETRAIN",
                DEFAULT_TRAIN_SRC,
            ),
            "Training",
        )
        train_root = output_root / "train"
        entries = _filter_only(TRAIN_BATCH, args.only)
        if args.only and not entries:
            p.error(f"--only matched no train datasets: {args.only}")
        tasks = [
            {
                "name": name,
                "freq": freq,
                "domain": domain,
                "sub_domain": sub_domain,
                "source_path": source_train,
                "output_root": str(train_root),
                "seed": args.seed,
                "force": args.force,
            }
            for (name, freq, domain, sub_domain) in entries
        ]
        _run_parallel(_train_worker, tasks, workers=args.workers, label="TRAIN")
        print(f"\nTrain done. Output: {train_root}")

    if do_test:
        source_test = _validate_source(
            _resolve_source(
                args.source_test, "GIFT_EVAL", "GIFT_EVAL", DEFAULT_TEST_SRC
            ),
            "Evaluation",
        )
        test_root = output_root / "test"
        entries = _filter_only(TEST_BATCH, args.only)
        if args.only and not entries:
            p.error(f"--only matched no test datasets: {args.only}")
        tasks = [
            {
                "name": name,
                "freq": freq,
                "domain": domain,
                "sub_domain": sub_domain,
                "domain_class": domain_class,
                "source_path": source_test,
                "output_root": str(test_root),
                "seed": args.seed,
                "force": args.force,
            }
            for (name, freq, domain, sub_domain, domain_class) in entries
        ]
        _run_parallel(_test_worker, tasks, workers=args.workers, label="TEST")
        print(f"\nTest done. Output: {test_root}")


if __name__ == "__main__":
    main()
