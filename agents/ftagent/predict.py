from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_LT_LIB_ROOT = Path(__file__).resolve().parents[2] / "lt_lib"
_LT_LIB_SRC = _LT_LIB_ROOT / "src"
if str(_LT_LIB_SRC) not in sys.path:
    sys.path.insert(0, str(_LT_LIB_SRC))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_input(path: Path, seq_len: int, channels: int) -> np.ndarray:
    array = np.load(path)
    if array.ndim == 2:
        array = array[None, ...]
    expected = (seq_len, channels)
    if array.ndim != 3 or array.shape[1:] != expected:
        raise ValueError(f"Input must have shape (N, {seq_len}, {channels}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Input contains NaN or infinite values")
    return array.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with an FTAgent-selected MetaCaster Forecaster"
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Lookback .npy array")
    parser.add_argument("--output", type=Path, default=Path("predictions.npy"))
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    import torch
    from benchmark.registry import MODEL_REGISTRY
    from benchmark.registry.models import register_model_by_name

    model_dir = args.model_dir.resolve()
    spec = json.loads((model_dir / "model_spec.json").read_text(encoding="utf-8"))
    checkpoint = model_dir / spec["checkpoint"]
    if _sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash does not match model_spec.json")

    task = spec["task"]
    seq_len = int(task["seq_len"])
    pred_len = int(task["pred_len"])
    channels = int(task["channels"])
    array = _load_input(args.input, seq_len, channels)
    mean = np.asarray(spec["normalization"]["mean"], dtype=np.float32)
    scale = np.asarray(spec["normalization"]["scale"], dtype=np.float32)
    if mean.shape != (channels,) or scale.shape != (channels,):
        raise ValueError(
            f"Normalization must contain {channels} channel values; "
            f"got mean={mean.shape}, scale={scale.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Normalization contains NaN or infinite values")
    if np.any(scale <= 0):
        raise ValueError("Normalization scale values must be positive")
    normalized = (array - mean) / scale

    register_model_by_name(spec["model"])
    factory, params_schema = MODEL_REGISTRY.get(spec["model"])
    params = spec["params"]
    if params_schema is not None:
        params = params_schema.model_validate(params).model_dump()
    config = SimpleNamespace(
        task=SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            label_len=0,
            features=task.get("features", "M"),
        )
    )
    device = torch.device(args.device)
    model = factory(config, params).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(normalized), args.batch_size):
            batch = torch.from_numpy(normalized[start : start + args.batch_size]).to(device)
            x_mark = torch.zeros((len(batch), seq_len, 6), device=device)
            decoder = torch.zeros((len(batch), pred_len, channels), device=device)
            y_mark = torch.zeros((len(batch), pred_len, 6), device=device)
            try:
                prediction = model(batch, x_mark, decoder, y_mark)
            except TypeError:
                prediction = model(batch)
            outputs.append(prediction[:, -pred_len:, :].cpu().numpy())
    result = np.concatenate(outputs, axis=0) * scale + mean
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result.astype(np.float32))
    print(f"Saved predictions {result.shape} to {args.output}")


if __name__ == "__main__":
    main()
