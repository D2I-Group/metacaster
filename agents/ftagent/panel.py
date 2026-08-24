from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Condition

AGENT_ROOT = Path(__file__).resolve().parents[2]
LT_LIB_ROOT = AGENT_ROOT / "lt_lib"


def _quote_list(values: list[str]) -> str:
    return "[\n" + "\n".join(f'    "{value}",' for value in values) + "\n]"


def _write_config(
    *,
    config_path: Path,
    run_id: str,
    round_n: int,
    dataset_key: str,
    model_name: str,
    dataset_dir: Path,
    channels: int,
    seq_len: int,
    pred_len: int,
    epochs: int,
    batch_size: int,
    patience: int,
) -> tuple[str, str]:
    alias = f"{run_id}__round_{round_n}__{dataset_key}__{model_name}"
    work_key = f"round_eval/{run_id}/round_{round_n}"
    model_path = f"../models/{model_name}.toml"
    config = f'''extends = ["../base.toml"]

[experiment]
description = "round_eval {run_id} round {round_n} {dataset_key}/{model_name}"
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
name = "pre_split"
alias = "{alias}"
root_path = "{dataset_dir.resolve()}"
data_path = ""

[dataset.params]
val_ratio = 0.1111111111
scale = false

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
models = {_quote_list([model_path])}
'''
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config, encoding="utf-8")
    return alias, work_key


def _read_latest_metric(performance_path: Path) -> dict[str, float]:
    import csv

    if not performance_path.exists():
        raise FileNotFoundError(f"missing performance.csv: {performance_path}")
    with performance_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty performance.csv: {performance_path}")
    return {"mse": float(rows[-1]["mse"]), "mae": float(rows[-1]["mae"])}


class _GpuPool:
    def __init__(self, gpus: list[int], per_gpu: int) -> None:
        self._slots = {gpu: per_gpu for gpu in gpus}
        self._condition = Condition()

    def acquire(self) -> int:
        with self._condition:
            while all(slots == 0 for slots in self._slots.values()):
                self._condition.wait()
            gpu = max(self._slots, key=self._slots.get)
            self._slots[gpu] -= 1
            return gpu

    def release(self, gpu: int) -> None:
        with self._condition:
            self._slots[gpu] += 1
            self._condition.notify()


def _train_one(
    *,
    run_id: str,
    round_n: int,
    dataset_key: str,
    model: str,
    dataset_dir: Path,
    channels: int,
    seq_len: int,
    pred_len: int,
    log_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    timeout_s: int,
    gpu_pool: _GpuPool,
    progress,
) -> dict:
    log_path = log_dir / f"train_{dataset_key}__{model}.log"
    config_path = (
        LT_LIB_ROOT
        / "configs"
        / "exp"
        / f"round_eval__{run_id}__round_{round_n}__{dataset_key}__{model}.toml"
    )
    alias, work_key = _write_config(
        config_path=config_path,
        run_id=run_id,
        round_n=round_n,
        dataset_key=dataset_key,
        model_name=model,
        dataset_dir=dataset_dir,
        channels=channels,
        seq_len=seq_len,
        pred_len=pred_len,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    gpu = gpu_pool.acquire()
    started = time.time()
    progress.event(phase="train", event="start", ds=dataset_key, model=model, gpu=gpu)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    performance_path = LT_LIB_ROOT / "work_dirs" / work_key / alias / model / "performance.csv"
    performance_path.unlink(missing_ok=True)
    return_code = -1
    error = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# {dataset_key}/{model} on GPU {gpu}\n")
            log.write(f"config: {config_path}\n")
            try:
                process = subprocess.run(
                    ["uv", "run", "metacaster-lt-lib", "--config", str(config_path)],
                    cwd=LT_LIB_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                )
                return_code = process.returncode
            except subprocess.TimeoutExpired as exc:
                error = f"timeout after {timeout_s}s"
                log.write(f"\n!!! TIMEOUT: {exc}\n")
                return_code = -2
            except Exception as exc:
                error = f"launcher crash: {exc}"
                log.write(f"\n!!! LAUNCHER CRASH: {exc}\n{traceback.format_exc()}\n")
                return_code = -3
    finally:
        gpu_pool.release(gpu)

    metric = None
    if return_code == 0:
        try:
            metric = _read_latest_metric(performance_path)
        except Exception as exc:
            error = f"missing performance.csv: {exc}"
    elif error is None:
        error = f"training process exited with code {return_code}"

    result = {
        "dataset": dataset_key,
        "model": model,
        "gpu": gpu,
        "rc": return_code,
        "elapsed": time.time() - started,
        "log_path": str(log_path),
    }
    if metric is not None:
        result.update(metric)
        result["ok"] = True
        result["failed"] = False
    else:
        result["ok"] = False
        result["failed"] = True
        result["error"] = error or f"rc={return_code} no metric"
    progress.event(
        phase="train",
        event="done",
        ds=dataset_key,
        model=model,
        gpu=gpu,
        ok=result["ok"],
        elapsed=result["elapsed"],
        mse=result.get("mse"),
        error=result.get("error"),
    )
    return result


def run_panel_training(
    *,
    run_id: str,
    round_n: int,
    pool_models: list[str],
    gen_results: dict[str, dict],
    data_root: Path,
    round_dir: Path,
    test_pool_root: Path,
    log_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    timeout_s: int,
    gpus: list[int],
    train_per_gpu: int,
    progress,
) -> list[dict]:
    del round_dir
    progress.event(
        phase="train",
        event="phase_start",
        n_panel=len(pool_models),
        n_ds=sum(1 for result in gen_results.values() if result["ok"]),
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_pool = _GpuPool(gpus, train_per_gpu)
    results = []
    tasks = []

    for dataset_key, generation_result in gen_results.items():
        if not generation_result["ok"]:
            for model in pool_models:
                results.append(
                    {
                        "dataset": dataset_key,
                        "model": model,
                        "ok": False,
                        "skipped": True,
                        "failed": False,
                        "error": f"generation failed: {generation_result.get('error')}",
                    }
                )
            continue

        try:
            source_dir = data_root / "train" / dataset_key
            metadata = json.loads((source_dir / "meta.json").read_text(encoding="utf-8"))
            test_path = (test_pool_root / f"{dataset_key}.npy").resolve()
            if not test_path.exists():
                raise FileNotFoundError(f"missing test pool {test_path}")
            dataset_dir = LT_LIB_ROOT / "dataset" / f"{run_id}__round_{round_n}__{dataset_key}"
            if dataset_dir.is_symlink() or dataset_dir.is_file():
                dataset_dir.unlink()
            elif dataset_dir.is_dir():
                shutil.rmtree(dataset_dir)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "train.npy").symlink_to(Path(generation_result["dataset_npy"]).resolve())
            (dataset_dir / "test.npy").symlink_to(test_path)
        except Exception as exc:
            for model in pool_models:
                results.append(
                    {
                        "dataset": dataset_key,
                        "model": model,
                        "ok": False,
                        "skipped": True,
                        "failed": False,
                        "error": f"setup failed: {exc!r}",
                    }
                )
            continue

        for model in pool_models:
            tasks.append(
                {
                    "dataset_key": dataset_key,
                    "model": model,
                    "dataset_dir": dataset_dir,
                    "channels": int(metadata["C_eff"]),
                    "seq_len": int(metadata["seq_len"]),
                    "pred_len": int(metadata["pred_len"]),
                }
            )

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, len(gpus) * train_per_gpu)) as pool:
        for task in tasks:
            future = pool.submit(
                _train_one,
                run_id=run_id,
                round_n=round_n,
                log_dir=log_dir,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                timeout_s=timeout_s,
                gpu_pool=gpu_pool,
                progress=progress,
                **task,
            )
            futures[future] = (task["dataset_key"], task["model"])

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                dataset_key, model = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "dataset": dataset_key,
                        "model": model,
                        "ok": False,
                        "skipped": False,
                        "failed": True,
                        "error": f"future crash: {exc}",
                    }
                results.append(result)

    progress.event(
        phase="train",
        event="phase_done",
        n_ok=sum(1 for result in results if result.get("ok")),
        n_failed=sum(1 for result in results if result.get("failed")),
        n_skipped=sum(1 for result in results if result.get("skipped")),
    )
    return results
