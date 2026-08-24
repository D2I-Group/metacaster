from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate task-adaptive data and train a MetaCaster Forecaster"
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="Generate task-adaptive training windows",
        help="Task passed to MGAgent",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing few_shot.npy, meta.json, and optional context.txt",
    )
    parser.add_argument("--test", type=Path, default=None, help="Optional test.npy")
    parser.add_argument("--output", type=Path, default=Path("work_dir/metacaster"))
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Train only the listed Forecasters",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train all 23 Forecasters instead of the 20-model main pool",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Ask FTAgent to plan model-specific hyperparameter trials",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        help="Expose provider web search to MGAgent",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated dataset",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.models and args.all_models:
        raise ValueError("--models and --all-models cannot be used together")

    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    generated_dir = output / "generated"
    forecaster_dir = output / "forecaster"

    mg_command = [
        sys.executable,
        "-m",
        "agents.mgagent.agent",
        "--input-dir",
        str(input_dir),
        "--output",
        str(generated_dir),
    ]
    if args.enable_web_search:
        mg_command.append("--enable-web-search")
    if args.overwrite:
        mg_command.append("--overwrite")
    mg_command.append(args.task)
    subprocess.run(mg_command, cwd=_ROOT, check=True)

    ft_command = [
        "uv",
        "run",
        "--project",
        str(_ROOT / "lt_lib"),
        "python",
        "-m",
        "agents.ftagent.agent",
        "--synthetic",
        str(generated_dir / "dataset.npy"),
        "--input-dir",
        str(input_dir),
        "--gpus",
        args.gpus,
        "--output",
        str(forecaster_dir),
    ]
    if args.test is not None:
        ft_command.extend(["--test", str(args.test.resolve())])
    if args.models:
        ft_command.extend(["--models", *args.models])
    elif not args.all_models:
        ft_command.append("--main-only")
    if args.tune:
        ft_command.append("--tune")
    subprocess.run(ft_command, cwd=_ROOT, check=True)

    print(f"Selected Forecaster saved to {forecaster_dir}")


if __name__ == "__main__":
    main()
