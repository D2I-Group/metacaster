
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from generation.core.config import LLM_PROVIDER, MG_MAX_TURNS, MG_MODEL, build_client
from generation.core.loop import agent_loop
from generation.tools import build_tools
from generation.tools.skill import SkillLoader

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_DEFAULT_HARNESS_ROOT = Path(__file__).resolve().parents[1] / "harness"


def _expand_images(path: str) -> list[str]:
    p = Path(path)
    if p.is_dir():
        return sorted(
            str(f) for f in p.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES
        )
    return [path]


def _load_system_prompt(harness_root: Path) -> str:
    router_path = harness_root / "core" / "router.md"
    if not router_path.is_file():
        raise FileNotFoundError(
            f"Harness missing core/router.md under {harness_root}. "
            "The HPAgent's _copy_harness() should write this at "
            "run init; investigate the driver setup."
        )
    return router_path.read_text(encoding="utf-8")


def _validate_harness(harness_root: Path) -> None:
    if not harness_root.exists():
        raise FileNotFoundError(f"--harness-root path does not exist: {harness_root}")
    router_path = harness_root / "core" / "router.md"
    skills_path = harness_root / "skills"
    if not router_path.is_file():
        raise FileNotFoundError(
            f"Harness missing required file: {router_path} "
            "(driver _copy_harness should have written it)"
        )
    if not skills_path.is_dir():
        raise FileNotFoundError(f"Harness missing required directory: {skills_path}")


def _resolve_inputs(
    input_dir: Path, context_path: Path | None
) -> tuple[Path, Path, Path | None]:
    input_dir = input_dir.resolve()
    few_shot_path = input_dir / "few_shot.npy"
    meta_path = input_dir / "meta.json"
    if not few_shot_path.is_file():
        raise FileNotFoundError(f"Missing required few-shot tensor: {few_shot_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing required metadata: {meta_path}")
    if context_path is not None:
        resolved_context = context_path.resolve()
        if not resolved_context.is_file():
            raise FileNotFoundError(f"Missing explicit context file: {resolved_context}")
    else:
        resolved_context = input_dir / "context.txt"
        if not resolved_context.is_file():
            resolved_context = None
    return few_shot_path, meta_path, resolved_context


def _validate_generated_artifact(
    output_dir: Path, meta_path: Path, n_target: int | None
) -> Path:
    output_dir = output_dir.resolve()
    artifact = output_dir / "dataset.npy"
    if not artifact.is_file():
        raise FileNotFoundError(f"MGAgent did not create {artifact}")
    report_path = output_dir / "validation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"MGAgent did not create {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("MGAgent validation_report.json must declare passed=true")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    array = np.load(artifact, mmap_mode="r")
    expected_len = int(meta["seq_len"]) + int(meta["pred_len"])
    expected_channels = int(meta["C_eff"])
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D generated tensor, got shape {array.shape}")
    if array.shape[1:] != (expected_len, expected_channels):
        raise ValueError(
            "Generated tensor shape mismatch: "
            f"expected (*, {expected_len}, {expected_channels}), got {array.shape}"
        )
    if n_target is not None and array.shape[0] != n_target:
        raise ValueError(
            f"Expected {n_target} generated windows, got {array.shape[0]}"
        )
    if array.dtype != np.float32:
        raise ValueError(f"Expected float32 generated data, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("Generated tensor contains NaN or infinite values")
    return artifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time Series Dataset MGAgent",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task description (omit for interactive mode)",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing few_shot.npy, meta.json, and optional context.txt",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help="Optional context file override (default: <input-dir>/context.txt)",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=None,
        help="Require exactly this many generated windows",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        help="Expose provider web search (disabled by default for leakage safety)",
    )
    parser.add_argument(
        "--image",
        dest="images",
        action="append",
        default=[],
        metavar="PATH",
        help="Optional: attach reference images (file or directory, repeatable)",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="./work_dir",
        metavar="DIR",
        help="Directory where dataset.npy will be saved (default: ./work_dir)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing <output>/dataset.npy instead of failing",
    )
    parser.add_argument(
        "--harness-root",
        dest="harness_root",
        default=str(_DEFAULT_HARNESS_ROOT),
        metavar="PATH",
        help=(
            "Path to a harness directory containing core/router.md and skills/. "
            "Defaults to the bundled trained Harness; provide this option only "
            "to override it."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    existing_artifact = output_dir / "dataset.npy"
    if existing_artifact.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to reuse existing artifact: {existing_artifact}. "
            "Choose a new --output directory or pass --overwrite."
        )
    if args.overwrite:
        existing_artifact.unlink(missing_ok=True)

    images: list[str] = []
    for path in args.images:
        images.extend(_expand_images(path))

    few_shot_path, meta_path, context_path = _resolve_inputs(
        args.input_dir, args.context
    )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    n_target = args.n_target
    if n_target is None and metadata.get("n_target") is not None:
        n_target = int(metadata["n_target"])
    runtime_input = output_dir / "input"
    if runtime_input.exists():
        shutil.rmtree(runtime_input)
    runtime_input.mkdir(parents=True)
    shutil.copy2(few_shot_path, runtime_input / "few_shot.npy")
    shutil.copy2(meta_path, runtime_input / "meta.json")
    if context_path is not None:
        shutil.copy2(context_path, runtime_input / "context.txt")

    task = args.task
    if not task:
        print("Time Series MGAgent")
        print("Enter task description:")
        print(">>> ", end="", flush=True)
        task = input().strip()

    input_contract = (
        "\n\nInput directory: input\n"
        "Few-shot tensor: input/few_shot.npy\nMetadata: input/meta.json\n"
        f"Context: {'input/context.txt' if context_path else '(not supplied)'}\n"
    )
    if n_target is not None:
        input_contract += f"Generate exactly {n_target} windows.\n"
    task += input_contract

    harness_root = Path(args.harness_root).resolve()
    _validate_harness(harness_root)


    SYSTEM = _load_system_prompt(harness_root)
    skill_loader = SkillLoader(harness_root / "skills")
    tools, handlers = build_tools(
        skill_loader,
        workspace=output_dir,
        enable_web_search=args.enable_web_search,
    )


    format_kwargs = {
        "input_dir": "input",
        "output_dir": "{output_dir}",
        "skill_descriptions": skill_loader.get_descriptions(),
    }
    system_template = SYSTEM.format(**format_kwargs)


    if LLM_PROVIDER == "openai":
        result = agent_loop(
            task,
            system=system_template,
            images=images or None,
            client=build_client(),
            model=MG_MODEL,
            max_turns=MG_MAX_TURNS,
            tools=tools,
            tool_handlers=handlers,
            output_dir=args.output_dir,
        )
    elif LLM_PROVIDER == "anthropic":
        from generation.core.loop_anthropic import agent_loop as _anthropic_loop
        result = _anthropic_loop(
            task,
            system=system_template,
            images=images or None,
            model=MG_MODEL,
            max_turns=MG_MAX_TURNS,
            tools=tools,
            tool_handlers=handlers,
            output_dir=args.output_dir,
        )
    elif LLM_PROVIDER == "gemini":
        from generation.core.loop_gemini import agent_loop as _gemini_loop
        result = _gemini_loop(
            task,
            system=system_template,
            images=images or None,
            model=MG_MODEL,
            max_turns=MG_MAX_TURNS,
            tools=tools,
            tool_handlers=handlers,
            output_dir=args.output_dir,
        )
    elif LLM_PROVIDER == "vllm":
        from generation.core.loop_vllm import agent_loop as _vllm_loop
        result = _vllm_loop(
            task,
            system=system_template,
            images=images or None,
            model=MG_MODEL,
            max_turns=MG_MAX_TURNS,
            tools=tools,
            tool_handlers=handlers,
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. "
            "Expected one of: openai, anthropic, gemini, vllm."
        )

    artifact = _validate_generated_artifact(
        Path(args.output_dir), meta_path, n_target
    )
    print(f"\n{result}")
    print(f"Validated artifact: {artifact}")


if __name__ == "__main__":
    main()
