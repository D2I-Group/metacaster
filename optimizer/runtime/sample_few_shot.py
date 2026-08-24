
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N_FEWSHOT_TRAIN_RANGE = (10, 50)
AGENT_META_FIELDS = (
    "freq",
    "domain",
    "total_len",
    "seq_len",
    "pred_len",
    "C_eff",
    "n_target",
    "normalization_mean",
    "normalization_scale",
)


def _deterministic_seed(
    run_id: str, round_n: int, dataset_key: str, purpose: str = ""
) -> int:
    h = hashlib.sha256(f"{run_id}|{round_n}|{dataset_key}|{purpose}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0xFFFFFFFF


def sample_few_shot(
    *,
    dataset_dir: Path,
    output_dir: Path,
    run_id: str,
    round_n: int,
    n_fewshot: int | None = None,
    fewshot_range: tuple[int, int] = N_FEWSHOT_TRAIN_RANGE,
) -> dict:
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    dataset_key = meta["dataset_key"]
    total_len = int(meta["total_len"])
    n_train = int(meta["n_train"])
    n_val = int(meta.get("n_val", max(1, round(n_train / 9))))
    n_support_pool = n_train - n_val

    if n_fewshot is None:
        # The paper samples one K per optimization epoch and applies it to the
        # complete dataset batch, rather than drawing a different K per dataset.
        count_seed = _deterministic_seed(run_id, round_n, "shared", purpose="count")
        count_rng = np.random.default_rng(count_seed)
        lo, hi = fewshot_range
        n_fewshot = int(count_rng.integers(lo, hi + 1))

    n_fs = max(1, int(n_fewshot))
    if n_fs > n_support_pool:
        n_fs = n_support_pool


    sample_seed = _deterministic_seed(run_id, round_n, dataset_key, purpose="sample")
    sample_rng = np.random.default_rng(sample_seed)

    train_idx = np.load(dataset_dir / "train_idx.npy")
    if len(train_idx) != n_train:
        raise RuntimeError(
            f"meta.n_train={n_train} disagrees with train_idx length {len(train_idx)}"
        )
    rel_positions = np.sort(
        sample_rng.choice(n_support_pool, size=n_fs, replace=False)
    )

    raw = np.load(dataset_dir / "raw.npy", mmap_mode="r")
    if raw.ndim == 3:

        selected = train_idx[rel_positions]
        C = raw.shape[-1]
        windows = np.empty((n_fs, total_len, C), dtype=np.float32)
        for i, (s, p) in enumerate(selected):
            windows[i] = np.asarray(
                raw[int(s), int(p) : int(p) + total_len, :]
            )
        sample_idx_log = selected.tolist()
    else:

        abs_positions = train_idx[rel_positions].astype(np.int64)
        windows = np.stack(
            [np.asarray(raw[p : p + total_len]) for p in abs_positions], axis=0
        ).astype(np.float32)
        sample_idx_log = abs_positions.tolist()

    if meta.get("values_normalized"):
        mean = np.asarray(meta["normalization_mean"], dtype=np.float32)
        scale = np.asarray(meta["normalization_scale"], dtype=np.float32)
        windows = ((windows - mean) / scale).astype(np.float32)

    agent_dir = Path(output_dir) / "agent_view"
    agent_dir.mkdir(parents=True, exist_ok=True)
    np.save(agent_dir / "few_shot.npy", windows)

    agent_meta = {k: meta[k] for k in AGENT_META_FIELDS if k in meta}
    agent_meta["n_few_shot"] = int(windows.shape[0])
    # Match the paper protocol: generation size equals the authentic
    # train+validation window count recorded during dataset preparation.
    N_target = int(meta.get("n_target", n_train))
    agent_meta["n_target"] = N_target
    (agent_dir / "meta.json").write_text(json.dumps(agent_meta, indent=2))
    source_context = dataset_dir / "context.txt"
    if source_context.is_file():
        context = source_context.read_text(encoding="utf-8")
    else:
        context = (
            f"Time-series observations in the {meta.get('domain', 'unknown')} domain, "
            f"sampled at frequency {meta.get('freq', 'unknown')}, with "
            f"{meta.get('C_eff', windows.shape[-1])} channel(s).\n"
        )
    (agent_dir / "context.txt").write_text(context, encoding="utf-8")

    log = {
        "dataset_key": dataset_key,
        "run_id": run_id,
        "round_n": int(round_n),
        "n_few_shot": int(n_fs),
        "n_train": int(n_train),
        "fewshot_range": list(fewshot_range),
        "N_target": N_target,
        "expansion_factor": round(N_target / max(1, n_fs), 2),
        "sample_seed": int(sample_seed),
        "sample_idx_abs": sample_idx_log,
    }
    (Path(output_dir) / "sampling_log.json").write_text(json.dumps(log, indent=2))
    return log


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Path to data/GIFT-Eval/train/<dataset_key>/",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write agent_view/ (typically the per-dataset round dir)",
    )
    p.add_argument("--run-id", required=True, type=str, help="Optimizer run_id")
    p.add_argument("--round-n", required=True, type=int, help="Optimizer round index")
    p.add_argument(
        "--n-fewshot",
        type=int,
        default=None,
        help="Force a specific count (e.g. 30). Omit for random pick in --fewshot-range.",
    )
    p.add_argument(
        "--fewshot-range",
        type=int,
        nargs=2,
        default=list(N_FEWSHOT_TRAIN_RANGE),
        metavar=("LO", "HI"),
        help=f"Random count range when --n-fewshot is not set (default: {N_FEWSHOT_TRAIN_RANGE}).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    log = sample_few_shot(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        round_n=args.round_n,
        n_fewshot=args.n_fewshot,
        fewshot_range=tuple(args.fewshot_range),
    )

    print(json.dumps(log))


if __name__ == "__main__":
    main()
