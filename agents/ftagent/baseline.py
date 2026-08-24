
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

AGENT_ROOT = Path(__file__).resolve().parents[2]
LT_LIB_ROOT = AGENT_ROOT / "lt_lib"


DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 5
NORMALIZATION_PROTOCOL = "train-split-zscore-v1"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _quote_list(values: Iterable[str]) -> str:
    return "[\n" + "\n".join(f'    "{v}",' for v in values) + "\n]"


def _write_full_config(
    *,
    config_path: Path,
    dataset_key: str,
    dataset_dir: Path,
    model_names: list[str],
    work_key: str,
    epochs: int,
    batch_size: int,
    patience: int,
) -> str:

    meta = _load_json(dataset_dir / "meta.json")
    channels = int(meta["C_eff"])
    seq_len = int(meta["seq_len"])
    pred_len = int(meta["pred_len"])
    alias = f"full_{dataset_key}"
    model_paths = [f"../models/{name}.toml" for name in model_names]

    config = f'''extends = ["../base.toml"]

[experiment]
description = "full baseline cache for {dataset_key}"
work_dir = "./work_dirs/{work_key}"

[experiment.runtime]
device = "cuda"
use_multi_gpu = false
device_ids = [0]
amp = false
num_workers = 4

[task]
seq_len = {seq_len}
label_len = 0
pred_len = {pred_len}
features = "M"
inverse = false

[dataset]
name = "raw_idx"
alias = "{alias}"
root_path = "{dataset_dir.resolve()}"
data_path = ""

[dataset.params]
val_ratio = 0.1111111111
scale = true

[model.params]
enc_in = {channels}
dec_in = {channels}
c_out = {channels}

[training]
epochs = {epochs}
batch_size = {batch_size}
patience = {patience}

[evaluation]
enable_profile = false
metrics = ["mae", "mse"]

[sweep.extend]
models = {_quote_list(model_paths)}
'''
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config, encoding="utf-8")
    return alias


def _read_latest_metric(perf_path: Path) -> dict[str, float]:
    if not perf_path.exists():
        raise FileNotFoundError(f"Missing performance.csv: {perf_path}")
    with perf_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {perf_path}")
    row = rows[-1]
    return {"mse": float(row["mse"]), "mae": float(row["mae"])}


def _run_dataset_sweep(
    *,
    dataset_key: str,
    missing: list[str],
    config_path: Path,
    alias: str,
    work_key: str,
    gpu: int,
    log_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    dataset_dir: Path,
) -> tuple[dict[str, dict[str, float]], list[str]]:

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset_key}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    results: dict[str, dict[str, float]] = {}
    failed: list[str] = []

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(
            f"\n# {dataset_key} on GPU {gpu} (per-model isolated subprocesses)\n"
        )
        log_f.flush()
        for model in missing:

            single_cfg_path = (
                config_path.parent
                / f"{config_path.stem}__{model}.toml"
            )
            _write_full_config(
                config_path=single_cfg_path,
                dataset_key=dataset_key,
                dataset_dir=dataset_dir,
                model_names=[model],
                work_key=work_key,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
            )
            cmd = ["uv", "run", "metacaster-lt-lib", "--config", str(single_cfg_path)]
            log_f.write(
                f"\n>>> {dataset_key}/{model} cmd: {' '.join(cmd)}\n"
            )
            log_f.flush()
            perf_path = (
                LT_LIB_ROOT
                / "work_dirs"
                / work_key
                / alias
                / model
                / "performance.csv"
            )
            perf_path.unlink(missing_ok=True)
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=LT_LIB_ROOT,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                rc = proc.returncode
            except Exception as exc:
                log_f.write(f"!!! launcher exception {model}: {exc}\n")
                log_f.flush()
                rc = -1

            if rc != 0:
                log_f.write(f"!!! {dataset_key}/{model} FAILED rc={rc}\n")
                log_f.flush()
                failed.append(model)
                continue
            try:
                metric = _read_latest_metric(perf_path)
            except Exception as exc:
                log_f.write(
                    f"!!! {dataset_key}/{model} FAILED rc={rc} ({exc})\n"
                )
                log_f.flush()
                failed.append(model)
                continue
            results[model] = metric

    return results, failed


def ensure_full_baseline(
    *,
    run_root: Path,
    datasets: list[str],
    models: list[str],
    data_root: Path = AGENT_ROOT / "data" / "GIFT-Eval",
    gpus: list[int] | None = None,
    force: bool = False,
    dry_run: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
) -> dict:

    run_root = Path(run_root)
    cache_path = run_root / "full_baseline.json"
    cache = _load_json(cache_path)
    if cache.get("_meta", {}).get("normalization_protocol") not in (
        None,
        NORMALIZATION_PROTOCOL,
    ):
        cache = {}
    elif cache and cache.get("_meta", {}).get("normalization_protocol") is None:
        # Legacy entries used validation-contaminated or synthetic-fitted
        # scalers and are not comparable with the paper protocol.
        cache = {}
    cache.setdefault("_meta", {})
    cache["_meta"].update(
        {
            "kind": "full_baseline",
            "data_root": str(Path(data_root).resolve()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "normalization_protocol": NORMALIZATION_PROTOCOL,
        }
    )

    work_key = f"full_baseline/{run_root.name}"
    config_root = LT_LIB_ROOT / "configs" / "exp"


    log_dir = LT_LIB_ROOT / "work_dirs" / work_key / "_subprocess_logs"
    trained: list[str] = []
    skipped: list[str] = []


    todo: list[tuple[str, list[str], Path, str]] = []
    for dataset_key in datasets:
        cache.setdefault(dataset_key, {})
        missing = [
            model
            for model in models
            if force or model not in cache.get(dataset_key, {})
        ]
        skipped.extend(
            f"{dataset_key}/{model}" for model in models if model not in missing
        )
        if not missing:
            continue

        dataset_dir = Path(data_root) / "train" / dataset_key
        safe_run = run_root.name.replace("/", "_")
        config_path = config_root / f"full_baseline_{safe_run}_{dataset_key}.toml"
        alias = _write_full_config(
            config_path=config_path,
            dataset_key=dataset_key,
            dataset_dir=dataset_dir,
            model_names=missing,
            work_key=work_key,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        if dry_run:
            print(
                f"DRY RUN: would sweep {dataset_key} ({len(missing)} models) "
                f"via {config_path}"
            )
            continue
        todo.append((dataset_key, missing, config_path, alias))

    if dry_run or not todo:
        cache["_meta"]["n_pairs"] = sum(
            len(v) for k, v in cache.items() if not k.startswith("_")
        )
        _write_json(cache_path, cache)
        return {"cache_path": str(cache_path), "trained": trained, "skipped": skipped}


    gpu_pool = list(gpus) if gpus else [0]
    n_workers = max(1, len(gpu_pool))
    cache_lock = Lock()

    def _worker(
        item: tuple[str, list[str], Path, str], gpu: int
    ) -> tuple[str, dict, list[str]]:
        dataset_key, missing, config_path, alias = item
        dataset_dir = Path(data_root) / "train" / dataset_key
        results, failed = _run_dataset_sweep(
            dataset_key=dataset_key,
            missing=missing,
            config_path=config_path,
            alias=alias,
            work_key=work_key,
            gpu=gpu,
            log_dir=log_dir,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            dataset_dir=dataset_dir,
        )
        return dataset_key, results, failed

    print(
        f"[full_baseline] {len(todo)} datasets to sweep on GPUs={gpu_pool} "
        f"(epochs={epochs}, patience={patience}, batch_size={batch_size})"
    )

    futures: dict[Future, tuple[str, int]] = {}
    queue = list(todo)
    available_gpus = list(gpu_pool)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:

        while queue and available_gpus:
            item = queue.pop(0)
            gpu = available_gpus.pop(0)
            fut = pool.submit(_worker, item, gpu)
            futures[fut] = (item[0], gpu)

        while futures:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                dataset_key, gpu = futures.pop(fut)
                try:
                    ds_key, results, failed = fut.result()
                except Exception as exc:
                    print(
                        f"[full_baseline] WORKER CRASH {dataset_key} on GPU "
                        f"{gpu}: {exc}; see {log_dir}/{dataset_key}.log",
                        file=sys.stderr,
                    )
                    ds_key, results, failed = dataset_key, {}, []
                with cache_lock:
                    cache.setdefault(ds_key, {}).update(results)
                    trained.extend(f"{ds_key}/{m}" for m in results)
                    cache["_meta"]["n_pairs"] = sum(
                        len(v) for k, v in cache.items() if not k.startswith("_")
                    )
                    _write_json(cache_path, cache)
                msg = (
                    f"[full_baseline] DONE {ds_key} on GPU {gpu} "
                    f"({len(results)} models)"
                )
                if failed:
                    msg += f" | FAILED: {failed}"
                print(msg)
                available_gpus.append(gpu)

                if queue and available_gpus:
                    item = queue.pop(0)
                    next_gpu = available_gpus.pop(0)
                    fut2 = pool.submit(_worker, item, next_gpu)
                    futures[fut2] = (item[0], next_gpu)

    cache["_meta"]["n_pairs"] = sum(
        len(v) for k, v in cache.items() if not k.startswith("_")
    )
    _write_json(cache_path, cache)
    missing_pairs = [
        f"{dataset}/{model}"
        for dataset in datasets
        for model in models
        if model not in cache.get(dataset, {})
    ]
    if missing_pairs:
        raise RuntimeError(
            "Full baseline is incomplete; failed or missing pairs: "
            + ", ".join(missing_pairs)
        )
    return {"cache_path": str(cache_path), "trained": trained, "skipped": skipped}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--datasets", required=True, help="Comma-separated dataset keys"
    )
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=AGENT_ROOT / "data" / "GIFT-Eval",
    )
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    args = parser.parse_args()

    result = ensure_full_baseline(
        run_root=args.run_root,
        datasets=_split_csv(args.datasets),
        models=_split_csv(args.models),
        data_root=args.data_root,
        gpus=[int(x) for x in _split_csv(args.gpus)] if args.gpus else None,
        force=args.force,
        dry_run=args.dry_run,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
