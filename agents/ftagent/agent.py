from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .prompt import plan_hyperparameters

_LT_LIB_ROOT = Path(__file__).resolve().parents[2] / "lt_lib"
_LT_LIB_SRC = _LT_LIB_ROOT / "src"
if str(_LT_LIB_SRC) not in sys.path:
    sys.path.insert(0, str(_LT_LIB_SRC))

MAIN_FORECASTERS = [
    "Linear",
    "DLinear",
    "NLinear",
    "RLinear",
    "MixLinear",
    "TSMixer",
    "LightTS",
    "PatchMLP",
    "xPatch",
    "CMoS",
    "PatchTSMixer",
    "FITS",
    "CycleNet",
    "PaiFilter",
    "TexFilter",
    "TimeMixer",
    "TimeBridge",
    "TimeEmb",
    "Amplifier",
    "SparseTSF",
]
HELD_OUT_FORECASTERS = ["CrossLinear", "TimeBase", "FreqCycle"]
ALL_FORECASTERS = [*MAIN_FORECASTERS, *HELD_OUT_FORECASTERS]


def _resolve_array(path: Path, filename: str) -> Path:
    path = path.resolve()
    candidate = path / filename if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing array: {candidate}")
    return candidate


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _validate_windows(path: Path, meta: dict) -> None:
    windows = np.load(path, mmap_mode="r")
    expected = (
        int(meta["seq_len"]) + int(meta["pred_len"]),
        int(meta["C_eff"]),
    )
    if windows.ndim != 3 or windows.shape[1:] != expected:
        raise ValueError(
            f"{path} must have shape (N, {expected[0]}, {expected[1]}), "
            f"got {windows.shape}"
        )
    if windows.shape[0] < 2:
        raise ValueError(f"{path} must contain at least two windows")
    if windows.dtype != np.float32:
        raise ValueError(f"{path} must be float32, got {windows.dtype}")
    if not np.isfinite(windows).all():
        raise ValueError(f"{path} contains NaN or infinite values")


def _toml_value(value) -> str:
    return json.dumps(value)


def _write_config(
    path: Path,
    *,
    model: str,
    data_dir: Path,
    work_dir: Path,
    channels: int,
    seq_len: int,
    pred_len: int,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    patience: int,
    num_workers: int,
    scale_data: bool = True,
    overrides: dict | None = None,
) -> None:
    base_path = (_LT_LIB_ROOT / "configs" / "base.toml").resolve()
    model_path = (_LT_LIB_ROOT / "configs" / "models" / f"{model}.toml").resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model config: {model_path}")
    quote = json.dumps
    overrides = overrides or {"training": {}, "model_params": {}}
    model_overrides = "".join(
        f"{key} = {_toml_value(value)}\n"
        for key, value in overrides.get("model_params", {}).items()
    )
    training_overrides = overrides.get("training", {})
    selected_batch_size = int(training_overrides.get("batch_size", batch_size))
    optimizer_values = "".join(
        f"{key} = {_toml_value(training_overrides[key])}\n"
        for key in ("lr", "weight_decay")
        if key in training_overrides
    )
    optimizer_section = (
        f"[training.optimizer]\n{optimizer_values}" if optimizer_values else ""
    )
    text = f'''extends = [{quote(str(base_path))}, {quote(str(model_path))}]

[experiment]
description = "MetaCaster FTAgent"
random_seed = {seed}
work_dir = {quote(str(work_dir))}

[experiment.runtime]
device = {quote(device)}
use_multi_gpu = false
device_ids = [0]
amp = false
num_workers = {num_workers}

[task]
seq_len = {seq_len}
label_len = 0
pred_len = {pred_len}
features = "M"
inverse = false

[dataset]
name = "pre_split"
alias = "metacaster_synthetic"
root_path = {quote(str(data_dir))}
data_path = ""

[dataset.params]
val_ratio = 0.1111111111
scale = {str(scale_data).lower()}

[model.params]
enc_in = {channels}
dec_in = {channels}
c_out = {channels}
{model_overrides}
[training]
epochs = {epochs}
batch_size = {selected_batch_size}
patience = {patience}

{optimizer_section}
[evaluation]
metrics = ["mae", "mse"]
enable_profile = false
evaluate_test = false
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _GpuPool:
    def __init__(self, gpus: list[int], per_gpu: int) -> None:
        self._slots: queue.Queue[int] = queue.Queue()
        for gpu in gpus:
            for _ in range(per_gpu):
                self._slots.put(gpu)

    def acquire(self) -> int:
        return self._slots.get()

    def release(self, gpu: int) -> None:
        self._slots.put(gpu)


def _read_job_result(work_dir: Path, model: str) -> dict:
    performance = work_dir / "metacaster_synthetic" / model / "performance.csv"
    if not performance.is_file():
        raise FileNotFoundError(f"Missing training result: {performance}")
    with performance.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty training result: {performance}")
    row = rows[-1]
    checkpoint = Path(row["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    validation_mse = float(row["validation_mse"])
    if not np.isfinite(validation_mse):
        raise ValueError(f"Non-finite validation MSE for {model}: {validation_mse}")
    return {
        "validation_mse": validation_mse,
        "checkpoint": str(checkpoint.resolve()),
    }


def _run_training_attempt(
    *,
    model: str,
    config_path: Path,
    work_dir: Path,
    log_dir: Path,
    device: str,
    gpu_pool: _GpuPool | None,
    timeout: int,
) -> dict:
    gpu = gpu_pool.acquire() if gpu_pool is not None else None
    started = time.time()
    log_path = log_dir / f"{model}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "benchmark.cli",
                    "--config",
                    str(config_path),
                ],
                cwd=_LT_LIB_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                return {
                    "model": model,
                    "status": "failed",
                    "gpu": gpu,
                    "elapsed_seconds": time.time() - started,
                    "log": str(log_path),
                    "error": f"training timed out after {timeout}s",
                }
        if return_code != 0:
            return {
                "model": model,
                "status": "failed",
                "gpu": gpu,
                "elapsed_seconds": time.time() - started,
                "log": str(log_path),
                "error": f"training process exited with code {return_code}",
            }
        result = _read_job_result(work_dir, model)
        return {
            "model": model,
            "status": "ok",
            "gpu": gpu,
            "elapsed_seconds": time.time() - started,
            "config": str(config_path),
            "log": str(log_path),
            **result,
        }
    except Exception as exc:
        return {
            "model": model,
            "status": "failed",
            "gpu": gpu,
            "elapsed_seconds": time.time() - started,
            "log": str(log_path),
            "error": str(exc),
        }
    finally:
        if gpu_pool is not None and gpu is not None:
            gpu_pool.release(gpu)


def _normalization_spec(windows_path: Path) -> dict[str, list[float]]:
    windows = np.load(windows_path, mmap_mode="r")
    n_val = max(1, round(len(windows) / 9))
    channels = windows.shape[-1]
    training_values = np.asarray(windows[:-n_val]).reshape(-1, channels)
    mean = training_values.mean(axis=0, dtype=np.float64)
    scale = training_values.std(axis=0, dtype=np.float64)
    scale[scale == 0] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def _run_training_job(
    *, trial_id: str, hyperparameters: dict, retries: int, **kwargs
) -> dict:
    model = kwargs["model"]
    work_dir = kwargs["work_dir"]
    attempts = []
    for attempt in range(1, retries + 2):
        model_work_dir = work_dir / "metacaster_synthetic" / model
        if model_work_dir.exists():
            shutil.rmtree(model_work_dir)
        attempt_kwargs = dict(kwargs)
        attempt_kwargs["log_dir"] = kwargs["log_dir"] / f"attempt_{attempt}"
        result = _run_training_attempt(**attempt_kwargs)
        attempts.append(
            {
                "attempt": attempt,
                "status": result["status"],
                "error": result.get("error"),
                "log": result.get("log"),
            }
        )
        if result["status"] == "ok":
            result["attempts"] = attempts
            result["trial_id"] = trial_id
            result["hyperparameters"] = hyperparameters
            return result
    result["attempts"] = attempts
    result["trial_id"] = trial_id
    result["hyperparameters"] = hyperparameters
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_selected(config, checkpoint: Path) -> dict[str, float]:
    import torch
    from benchmark.registry import MODEL_REGISTRY
    from benchmark.registry.models import register_model_by_name
    from benchmark.runner.evaluator import evaluate
    from benchmark.runner.run_one import _build_device
    from data.provider import build_data_loader

    device = _build_device(config.experiment.runtime)
    dataset_params = config.dataset.params
    if hasattr(dataset_params, "model_dump"):
        dataset_params = dataset_params.model_dump()
    else:
        dataset_params = dict(dataset_params)
    test_set, test_loader = build_data_loader(
        config.dataset.name,
        config.dataset.root_path,
        config.dataset.data_path,
        (config.task.seq_len, config.task.label_len, config.task.pred_len),
        "test",
        config.task.features,
        dataset_params,
        config.training.batch_size,
        config.experiment.runtime.num_workers,
    )
    register_model_by_name(config.model.name)
    model_factory, params_schema = MODEL_REGISTRY.get(config.model.name)
    params = config.model.params
    if params_schema is not None:
        params = params_schema.model_validate(params).model_dump()
    model = model_factory(config, params).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    metrics, _ = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        label_len=config.task.label_len,
        pred_len=config.task.pred_len,
        features=config.task.features,
        inverse=config.task.inverse,
        dataset=test_set,
    )
    return {name: float(metrics[name]) for name in ("mse", "mae")}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train registered lightweight forecasters on MGAgent output, "
            "select Top-1 by validation MSE, and optionally evaluate it on test data."
        )
    )
    parser.add_argument("--synthetic", required=True, type=Path, help="dataset.npy or its directory")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing meta.json")
    parser.add_argument("--test", type=Path, default=None, help="Optional test.npy or its directory")
    parser.add_argument("--output", type=Path, default=Path("ft_output"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Forecasters to train (default: all 23 paper forecasters)",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Train only the 20 main forecasters, excluding the three held-out models",
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical GPU ids used when --device=cuda",
    )
    parser.add_argument(
        "--train-per-gpu",
        type=int,
        default=1,
        help="Maximum concurrent training processes per GPU",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel process count for CPU/MPS training",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-model training timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Automatic retries after a failed or timed-out model job",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow Top-1 selection when one or more requested models fail",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Ask FTAgent to plan and run model-specific hyperparameter trials",
    )
    parser.add_argument(
        "--search-trials",
        type=int,
        default=3,
        help="Maximum trials per model including its unchanged default configuration",
    )
    parser.add_argument(
        "--planner-model",
        default=os.getenv("FT_MODEL", "gpt-5.4"),
        help="OpenAI model used by FTAgent to plan the search",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    if args.search_trials < 1:
        raise ValueError("--search-trials must be at least 1")
    meta_path = args.input_dir.resolve() / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    synthetic = _resolve_array(args.synthetic, "dataset.npy")
    _validate_windows(synthetic, meta)

    output = args.output.resolve()
    for stale_name in (
        "selected_checkpoint.pth",
        "selected_config.toml",
        "model_spec.json",
        "training_results.json",
        "search_plan.json",
    ):
        (output / stale_name).unlink(missing_ok=True)
    data_dir = output / "data"
    work_dir = output / "work_dirs"
    config_dir = output / "configs"
    output.mkdir(parents=True, exist_ok=True)
    _link_or_copy(synthetic, data_dir / "train.npy")

    test_path = None
    if args.test is not None:
        test_path = _resolve_array(args.test, "test.npy")
        _validate_windows(test_path, meta)
        _link_or_copy(test_path, data_dir / "test.npy")

    models = list(args.models or (MAIN_FORECASTERS if args.main_only else ALL_FORECASTERS))
    if len(models) != len(set(models)):
        raise ValueError("--models contains duplicates")
    unknown = sorted(set(models) - set(ALL_FORECASTERS))
    if unknown:
        raise ValueError(f"Unknown forecasters: {', '.join(unknown)}")

    log_dir = output / "logs"
    for runtime_dir in (work_dir, config_dir, log_dir):
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)

    search_plan = plan_hyperparameters(
        models=models,
        metadata=meta,
        model_config_dir=_LT_LIB_ROOT / "configs" / "models",
        max_trials=args.search_trials if args.tune else 1,
        planner_model=args.planner_model,
    )
    (output / "search_plan.json").write_text(
        json.dumps(
            {
                "planner_model": args.planner_model if args.tune else None,
                "trials_per_model": args.search_trials if args.tune else 1,
                "models": search_plan,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    records: list[dict] = []
    jobs: list[tuple[str, str, Path, Path, dict]] = []
    for model in models:
        for trial_index, hyperparameters in enumerate(search_plan[model]):
            trial_id = f"{model}__trial_{trial_index:02d}"
            config_path = config_dir / f"{trial_id}.toml"
            trial_work_dir = work_dir / trial_id
            try:
                _write_config(
                    config_path,
                    model=model,
                    data_dir=data_dir,
                    work_dir=trial_work_dir,
                    channels=int(meta["C_eff"]),
                    seq_len=int(meta["seq_len"]),
                    pred_len=int(meta["pred_len"]),
                    seed=args.seed,
                    device=args.device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    num_workers=args.num_workers,
                    scale_data=not bool(meta.get("values_normalized", False)),
                    overrides=hyperparameters,
                )
                jobs.append(
                    (model, trial_id, config_path, trial_work_dir, hyperparameters)
                )
            except Exception as exc:
                records.append(
                    {
                        "model": model,
                        "trial_id": trial_id,
                        "hyperparameters": hyperparameters,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    gpu_pool = None
    if args.device == "cuda":
        gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        if not gpus:
            raise ValueError("--gpus must contain at least one GPU id")
        if args.train_per_gpu < 1:
            raise ValueError("--train-per-gpu must be at least 1")
        gpu_pool = _GpuPool(gpus, args.train_per_gpu)
        max_workers = len(gpus) * args.train_per_gpu
    else:
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        max_workers = args.workers

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_training_job,
                model=model,
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                config_path=config_path,
                work_dir=trial_work_dir,
                log_dir=log_dir / trial_id,
                device=args.device,
                gpu_pool=gpu_pool,
                timeout=args.timeout,
                retries=args.retries,
            ): trial_id
            for model, trial_id, config_path, trial_work_dir, hyperparameters in jobs
        }
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(
        key=lambda record: (
            models.index(record["model"]), record.get("trial_id", "")
        )
    )
    successful = [record for record in records if record["status"] == "ok"]
    successful_models = {record["model"] for record in successful}
    failed_models = [model for model in models if model not in successful_models]
    if failed_models and not args.allow_partial:
        (output / "training_results.json").write_text(
            json.dumps({"status": "failed", "models": records}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"All hyperparameter trials failed for {len(failed_models)} of "
            f"{len(models)} requested forecasters; inspect training_results.json "
            "and logs/, or pass --allow-partial."
        )
    if not successful:
        (output / "training_results.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            "No forecaster trained successfully; inspect training_results.json "
            "and per-model logs under logs/."
        )

    selected = min(
        successful,
        key=lambda record: (
            record["validation_mse"], record["model"], record["trial_id"]
        ),
    )
    validation_mse = selected["validation_mse"]
    selected_model = selected["model"]
    selected_checkpoint = output / "selected_checkpoint.pth"
    shutil.copy2(selected["checkpoint"], selected_checkpoint)
    selected_config_path = output / "selected_config.toml"
    shutil.copy2(selected["config"], selected_config_path)

    from benchmark.config import load_config

    loaded = load_config(str(selected_config_path))
    if len(loaded) != 1:
        raise RuntimeError(
            f"Expected one selected config for {selected_model}, got {len(loaded)}"
        )
    selected_config = loaded[0].config
    model_params = selected_config.model.params
    if hasattr(model_params, "model_dump"):
        model_params = model_params.model_dump()
    else:
        model_params = dict(model_params)
    model_spec_path = output / "model_spec.json"
    model_spec = {
        "schema_version": 1,
        "model": selected_model,
        "params": model_params,
        "task": {
            "seq_len": selected_config.task.seq_len,
            "pred_len": selected_config.task.pred_len,
            "channels": int(meta["C_eff"]),
            "features": selected_config.task.features,
        },
        "normalization": (
            {
                "mean": meta["normalization_mean"],
                "scale": meta["normalization_scale"],
            }
            if meta.get("values_normalized")
            else _normalization_spec(synthetic)
        ),
        "checkpoint": selected_checkpoint.name,
        "checkpoint_sha256": _sha256(selected_checkpoint),
        "hyperparameters": selected["hyperparameters"],
    }
    model_spec_path.write_text(json.dumps(model_spec, indent=2), encoding="utf-8")

    test_metrics = None
    test_error = None
    if test_path is not None:
        try:
            test_metrics = _evaluate_selected(selected_config, selected_checkpoint)
        except Exception as exc:
            test_error = str(exc)

    manifest = {
        "selection_metric": "validation_mse",
        "selected_model": selected_model,
        "selected_validation_mse": validation_mse,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": _sha256(selected_checkpoint),
        "selected_config": str(selected_config_path),
        "model_spec": str(model_spec_path),
        "selected_trial": selected["trial_id"],
        "selected_hyperparameters": selected["hyperparameters"],
        "hyperparameter_search": {
            "enabled": args.tune,
            "planner_model": args.planner_model if args.tune else None,
            "max_trials_per_model": args.search_trials if args.tune else 1,
            "plan": str(output / "search_plan.json"),
        },
        "test_metrics": test_metrics,
        "test_error": test_error,
        "parallelism": {
            "device": args.device,
            "max_workers": max_workers,
            "gpus": args.gpus if args.device == "cuda" else None,
            "train_per_gpu": args.train_per_gpu if args.device == "cuda" else None,
            "retries": args.retries,
        },
        "models": records,
    }
    (output / "training_results.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if test_error is not None:
        raise RuntimeError(f"Selected-model test evaluation failed: {test_error}")


if __name__ == "__main__":
    main()
