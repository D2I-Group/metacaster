
import argparse
import json
import sys
from pathlib import Path


def _brief_code(code: str) -> str:
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):

            if stripped.startswith("#"):
                return stripped[:100]
            continue

        return stripped[:100]
    return "(empty)"


def _brief_args(name: str, args: dict) -> str:
    if name == "run_python":
        return _brief_code(args.get("code", ""))
    if name in ("read_file", "read_image"):
        return args.get("path", args.get("file_path", "?"))
    if name == "write_file":
        return args.get("path", args.get("file_path", "?"))
    if name == "bash":
        cmd = args.get("command", "")
        return cmd[:100]
    if name == "load_skill":
        return args.get("skill_name", args.get("name", "?"))

    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)[:100]


def parse_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def summarize(entries: list[dict]) -> str:
    steps: list[str] = []
    tool_counts: dict[str, int] = {}
    turn = 0
    total_calls = 0


    prev_was_call = False

    for entry in entries:
        entry_type = entry.get("type")
        role = entry.get("role")


        if role == "user":
            content = entry.get("content", "")
            if isinstance(content, list):

                texts = [
                    p.get("text", "") for p in content if p.get("type") == "input_text"
                ]
                content = texts[0] if texts else ""
            if turn == 0:

                turn = 1
                preview = content[:120].replace("\n", " ")
                steps.append(f"[{turn:02d}] task: {preview}...")
            elif (
                "nudge" in content.lower()
                or "continue" in content.lower()
                or "not complete" in content.lower()
            ):
                turn += 1
                steps.append(f"[{turn:02d}] nudge")
            prev_was_call = False
            continue


        if entry_type == "function_call":
            name = entry.get("name", "?")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            total_calls += 1

            if not prev_was_call:
                turn += 1

            try:
                args = json.loads(entry.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}

            brief = _brief_args(name, args)
            steps.append(f"[{turn:02d}] {name}({brief})")
            prev_was_call = True
            continue


        if entry_type == "function_call_output":
            prev_was_call = False
            continue


        if entry_type == "message":
            content_parts = entry.get("content", [])
            texts = []
            for part in content_parts:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
            if texts:
                combined = " ".join(texts).strip()
                if combined:
                    preview = combined[:100].replace("\n", " ")
                    if not prev_was_call:
                        turn += 1
                    steps.append(f"[{turn:02d}] response: {preview}")
            prev_was_call = False
            continue


        if entry_type == "web_search_call":
            if not prev_was_call:
                turn += 1
            steps.append(f"[{turn:02d}] web_search")
            prev_was_call = False
            continue


    lines = [
        f"Turns:  {turn}",
        f"Calls:  {total_calls}",
        "",
        "Steps:",
    ]
    lines.extend(f"  {s}" for s in steps)

    if tool_counts:
        lines.append("")
        lines.append("Tool call counts:")
        for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")
        lines.append(f"  -- total: {total_calls}")

    return "\n".join(lines)


def process_file(jsonl_path: Path, stdout: bool = False) -> None:
    entries = parse_jsonl(jsonl_path)
    if not entries:
        print(f"  {jsonl_path.name}: empty, skipping")
        return

    summary = summarize(entries)

    run_label = (
        jsonl_path.parent.name if jsonl_path.stem == "conversation" else jsonl_path.stem
    )
    header = f"Run: {run_label}"

    if stdout:
        print(f"\n{'=' * 60}")
        print(header)
        print(summary)
    else:
        out_path = jsonl_path.with_name("summary.txt")
        out_path.write_text(f"{header}\n{summary}\n", encoding="utf-8")
        print(f"  {jsonl_path.name} -> {out_path.name}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "path",
        help="Path to a .jsonl file, a run log directory, or the .log root directory",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Print summaries to stdout instead of writing summary.txt files",
    )
    args = p.parse_args()

    path = Path(args.path)
    if path.is_file():
        process_file(path, stdout=args.stdout)
    elif path.is_dir():

        jsonl_files = sorted(path.glob("**/conversation.jsonl"))

        if not jsonl_files:
            jsonl_files = sorted(path.glob("**/*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in {path}")
            sys.exit(1)
        print(f"Summarizing {len(jsonl_files)} log(s)...")
        for f in jsonl_files:
            process_file(f, stdout=args.stdout)
    else:
        print(f"Not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
