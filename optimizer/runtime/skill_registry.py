
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_FRONTMATTER_FIELDS = ("name", "description", "version")
OPTIONAL_FRONTMATTER_FIELDS = ("applies_to_regimes", "applies_to_few_shot_signatures")


SKILL_MIN_BODY_LINES = 100
SKILL_TARGET_BODY_LINES = 150
SKILL_MIN_FUNCTIONS = 1


@dataclass
class SkillRecord:
    name: str
    description: str
    version: int | str
    applies_to_regimes: list[str] = field(default_factory=list)
    applies_to_few_shot_signatures: list[str] = field(default_factory=list)
    body: str = ""
    rel_path: str = ""
    errors: list[str] = field(default_factory=list)

    body_lines: int = 0
    n_python_blocks: int = 0
    n_functions: int = 0
    n_todo_marks: int = 0
    syntax_errors: list[str] = field(default_factory=list)
    quality_warnings: list[str] = field(default_factory=list)

    @property
    def health(self) -> str:
        if self.errors or self.syntax_errors:
            return "❌"
        if self.quality_warnings:
            return "⚠"
        if self.body_lines < SKILL_MIN_BODY_LINES:
            return "stub"
        return "✅"


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str, list[str]]:
    errors: list[str] = []
    if not text.startswith("---"):
        return {}, text, ["missing YAML frontmatter (file must start with '---')"]
    rest = text[3:]
    end = rest.find("\n---")
    if end < 0:
        return {}, text, ["unterminated YAML frontmatter (no closing '---')"]
    yaml_block = rest[:end]
    body = rest[end + 4:].lstrip("\n")

    try:
        import yaml
        meta = yaml.safe_load(yaml_block) or {}
        if not isinstance(meta, dict):
            errors.append("frontmatter is not a YAML mapping")
            meta = {}
    except Exception as exc:


        meta = {}
        cur_key: str | None = None
        for line in yaml_block.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("  - "):
                if cur_key is None:
                    errors.append(f"orphan list item: {line!r}")
                    continue
                meta.setdefault(cur_key, []).append(line[4:].strip().strip("'\""))
            elif ":" in line:
                k, _, v = line.partition(":")
                k = k.strip()
                v = v.strip()
                if v == "" or v.startswith("- "):
                    cur_key = k
                    if v.startswith("- "):
                        meta.setdefault(k, []).append(v[2:].strip().strip("'\""))
                    else:
                        meta.setdefault(k, [])
                else:
                    meta[k] = v.strip("'\"")
                    cur_key = None
        if not meta:
            errors.append(f"YAML parse failed and fallback empty: {exc}")

    return meta, body, errors


def _validate(meta: dict, body: str, file_rel: str) -> SkillRecord:
    errors: list[str] = []
    for k in REQUIRED_FRONTMATTER_FIELDS:
        if k not in meta:
            errors.append(f"missing required frontmatter field: {k}")
    name = str(meta.get("name", "")).strip()
    desc = str(meta.get("description", "")).strip()
    version = meta.get("version", "")
    regimes_raw = meta.get("applies_to_regimes") or []
    sigs_raw = meta.get("applies_to_few_shot_signatures") or []
    if not isinstance(regimes_raw, list):
        regimes_raw = [str(regimes_raw)]
    if not isinstance(sigs_raw, list):
        sigs_raw = [str(sigs_raw)]
    if not body.strip():
        errors.append("empty body")

    return SkillRecord(
        name=name,
        description=desc,
        version=version,
        applies_to_regimes=[str(x) for x in regimes_raw],
        applies_to_few_shot_signatures=[str(x) for x in sigs_raw],
        body=body,
        rel_path=file_rel,
        errors=errors,
    )


_PY_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_TODO_RE = re.compile(r"\bTODO\b|\bFIXME\b|NotImplementedError|raise\s+NotImplemented")


def _extract_python_blocks(body: str) -> list[str]:
    return [m.group(1) for m in _PY_BLOCK_RE.finditer(body)]


def validate_skill_quality(rec: SkillRecord) -> None:
    body = rec.body or ""
    rec.body_lines = body.count("\n") + (1 if body and not body.endswith("\n") else 0)

    blocks = _extract_python_blocks(body)
    rec.n_python_blocks = len(blocks)
    rec.n_todo_marks = sum(len(_TODO_RE.findall(b)) for b in blocks)

    n_funcs = 0
    syntax_errors: list[str] = []
    for i, code in enumerate(blocks):
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            syntax_errors.append(
                f"block#{i + 1} line {exc.lineno}: {exc.msg}"
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_funcs += 1
    rec.n_functions = n_funcs
    rec.syntax_errors = syntax_errors

    warnings: list[str] = []
    if rec.body_lines < SKILL_MIN_BODY_LINES:
        warnings.append(
            f"body too short ({rec.body_lines} lines, target ≥{SKILL_TARGET_BODY_LINES})"
        )
    elif rec.body_lines < SKILL_TARGET_BODY_LINES:
        warnings.append(
            f"body below target ({rec.body_lines} lines, target ≥{SKILL_TARGET_BODY_LINES})"
        )
    if rec.n_python_blocks == 0:
        warnings.append("no python code block — skills must contain executable code")
    if rec.n_functions < SKILL_MIN_FUNCTIONS:
        warnings.append(
            f"no top-level function (n_functions={rec.n_functions}); add at least one "
            "`def f(samples, n_target, seed) -> np.ndarray` style entry point"
        )
    if rec.n_todo_marks > 0:
        warnings.append(
            f"{rec.n_todo_marks} TODO/FIXME/NotImplementedError marker(s) — skill is unfinished"
        )
    rec.quality_warnings = warnings


def load_skills(harness_root: Path) -> list[SkillRecord]:
    harness_root = Path(harness_root)
    skills_dir = harness_root / "skills"
    out: list[SkillRecord] = []
    if not skills_dir.exists():
        return out
    for sd in sorted(skills_dir.iterdir()):
        if not sd.is_dir():
            continue
        skill_md = sd / "SKILL.md"
        if not skill_md.exists():
            out.append(SkillRecord(
                name=sd.name, description="", version="",
                rel_path=str(skill_md.relative_to(harness_root)),
                errors=[f"missing SKILL.md under skills/{sd.name}/"],
            ))
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
        except Exception as exc:
            out.append(SkillRecord(
                name=sd.name, description="", version="",
                rel_path=str(skill_md.relative_to(harness_root)),
                errors=[f"read failed: {exc}"],
            ))
            continue
        meta, body, fm_errors = _parse_frontmatter(text)
        rec = _validate(meta, body, str(skill_md.relative_to(harness_root)))
        rec.errors = fm_errors + rec.errors
        if not rec.name:
            rec.name = sd.name
        validate_skill_quality(rec)
        out.append(rec)
    return out


def render_manifest_table(skills: list[SkillRecord]) -> str:
    if not skills:
        return "_(no skills defined yet — round 0 must author at least one)_\n"
    lines = [
        "| name | health | lines | funcs | TODO | regimes | description |",
        "|---|:---:|---:|---:|---:|---|---|",
    ]
    for s in skills:
        regimes = ", ".join(s.applies_to_regimes) if s.applies_to_regimes else "_(any)_"
        lines.append(
            f"| {s.name} | {s.health} | {s.body_lines} | {s.n_functions} | "
            f"{s.n_todo_marks} | {regimes} | {s.description} |"
        )
    return "\n".join(lines) + "\n"


def render_health_report(skills: list[SkillRecord]) -> str:
    issue_blocks: list[str] = []
    for s in skills:
        bullets: list[str] = []
        for e in s.errors:
            bullets.append(f"- ❌ frontmatter: {e}")
        for e in s.syntax_errors:
            bullets.append(f"- ❌ syntax: {e}")
        for w in s.quality_warnings:
            bullets.append(f"- ⚠ quality: {w}")
        if bullets:
            issue_blocks.append(
                f"### `{s.rel_path}`\n" + "\n".join(bullets)
            )
    if not issue_blocks:
        return "_(all skills healthy)_\n"
    return (
        "Skill issues this round — fix before they erode generation quality:\n\n"
        + "\n\n".join(issue_blocks)
        + "\n"
    )


def render_full_bodies(skills: list[SkillRecord]) -> str:
    parts: list[str] = []
    for s in skills:
        if s.errors:
            parts.append(f"### `{s.rel_path}` ⚠ ERRORS: {'; '.join(s.errors)}\n")
            continue
        parts.append(
            f"### `{s.rel_path}` (name={s.name}, version={s.version})\n"
            f"_description: {s.description}_\n\n"
            f"```markdown\n{s.body}\n```\n"
        )
    return "\n".join(parts) if parts else "_(no skills)_\n"
