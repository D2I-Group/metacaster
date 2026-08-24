from __future__ import annotations

import json
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = """You are the FTAgent (FTAgent) of MetaCaster. Your task is to design a compact hyperparameter search for every registered lightweight forecaster before the training jobs are dispatched.

For each model, inspect its default configuration and the task metadata, then propose diverse, plausible alternatives that are likely to improve validation MSE. The host always includes the unchanged default configuration as trial 0, so do not repeat it.

You may adjust:
- training.optimizer.lr: positive float;
- training.optimizer.weight_decay: non-negative float;
- training.batch_size: positive integer;
- existing model.params entries, except enc_in, dec_in, and c_out.

Do not add unknown model parameters, change the architecture name, alter dataset paths, alter train/validation/test data, change the forecasting shape, or use test metrics. Keep coupled architectural parameters valid (for example divisibility constraints). Favor a small, informative search over a large Cartesian product.

Return JSON only, with this exact shape:
{"models":{"ModelName":[{"training":{"lr":0.001,"weight_decay":0.0001,"batch_size":32},"model_params":{"dropout":0.1}}]}}

Every requested model must appear. Each list must contain at most max_additional_trials entries. Empty lists are allowed when no safe adjustment is useful.
"""

_PROTECTED_MODEL_PARAMS = {"enc_in", "dec_in", "c_out"}
_ALLOWED_TRAINING = {"lr", "weight_decay", "batch_size"}


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("FTAgent did not return a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("FTAgent search plan must be a JSON object")
    return value


def _validate_scalar(value, *, name: str):
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        return value
    raise ValueError(f"Unsupported value for {name}: {value!r}")


def _validate_plan(raw: dict, defaults: dict[str, dict], max_trials: int) -> dict:
    raw_models = raw.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("FTAgent search plan is missing the models object")
    plan: dict[str, list[dict]] = {}
    for model, default in defaults.items():
        candidates = raw_models.get(model)
        if not isinstance(candidates, list):
            raise ValueError(f"FTAgent search plan is missing a list for {model}")
        clean_candidates = []
        known_params = set(default.get("model", {}).get("params", {}))
        known_params -= _PROTECTED_MODEL_PARAMS
        for index, candidate in enumerate(candidates[:max_trials]):
            if not isinstance(candidate, dict):
                raise ValueError(f"{model} candidate {index} must be an object")
            training = candidate.get("training", {})
            model_params = candidate.get("model_params", {})
            if not isinstance(training, dict) or not isinstance(model_params, dict):
                raise ValueError(f"{model} candidate {index} has invalid sections")
            unknown_training = set(training) - _ALLOWED_TRAINING
            unknown_params = set(model_params) - known_params
            if unknown_training or unknown_params:
                raise ValueError(
                    f"{model} candidate {index} contains unsupported keys: "
                    f"{sorted(unknown_training | unknown_params)}"
                )
            clean_training = {
                key: _validate_scalar(value, name=f"{model}.training.{key}")
                for key, value in training.items()
            }
            if "lr" in clean_training and float(clean_training["lr"]) <= 0:
                raise ValueError(f"{model} learning rate must be positive")
            if "weight_decay" in clean_training and float(
                clean_training["weight_decay"]
            ) < 0:
                raise ValueError(f"{model} weight decay must be non-negative")
            if "batch_size" in clean_training and int(
                clean_training["batch_size"]
            ) < 1:
                raise ValueError(f"{model} batch size must be positive")
            clean_candidates.append(
                {
                    "training": clean_training,
                    "model_params": {
                        key: _validate_scalar(value, name=f"{model}.{key}")
                        for key, value in model_params.items()
                    },
                }
            )
        plan[model] = [{"training": {}, "model_params": {}}, *clean_candidates]
    return plan


def plan_hyperparameters(
    *,
    models: list[str],
    metadata: dict,
    model_config_dir: Path,
    max_trials: int,
    planner_model: str,
) -> dict[str, list[dict]]:
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    defaults = {
        model: tomllib.loads(
            (model_config_dir / f"{model}.toml").read_text(encoding="utf-8")
        )
        for model in models
    }
    if max_trials == 1:
        return {
            model: [{"training": {}, "model_params": {}}] for model in models
        }
    base_defaults = tomllib.loads(
        (model_config_dir.parent / "base.toml").read_text(encoding="utf-8")
    )
    payload = {
        "task": {
            "seq_len": int(metadata["seq_len"]),
            "pred_len": int(metadata["pred_len"]),
            "channels": int(metadata["C_eff"]),
            "frequency": metadata.get("freq"),
        },
        "max_additional_trials": max_trials - 1,
        "training_defaults": base_defaults.get("training", {}),
        "model_defaults": defaults,
    }
    load_dotenv()
    response = OpenAI().responses.create(
        model=planner_model,
        instructions=SYSTEM_PROMPT,
        input=json.dumps(payload),
    )
    return _validate_plan(_extract_json(response.output_text), defaults, max_trials - 1)
