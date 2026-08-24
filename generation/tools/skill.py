
from pathlib import Path

import yaml

LOAD_SKILL_TOOL: dict = {
    "type": "function",
    "name": "load_skill",
    "description": (
        "Load the full instructions for an injectable skill. "
        "Call this before applying a skill you are not already familiar with."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name (must be one of the available skills).",
            }
        },
        "required": ["name"],
    },
}


def _make_schema(skill_names: list[str]) -> dict:
    schema = {
        "type": LOAD_SKILL_TOOL["type"],
        "name": LOAD_SKILL_TOOL["name"],
        "description": LOAD_SKILL_TOOL["description"],
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (must be one of the available skills).",
                }
            },
            "required": ["name"],
        },
    }
    if skill_names:
        schema["parameters"]["properties"]["name"]["enum"] = skill_names
    return schema


class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, dict] = {}
        self._scan()

    def _scan(self):
        if not self.skills_dir.exists():
            return
        for skill_dir in sorted(self.skills_dir.iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if skill_dir.is_dir() and skill_file.exists():
                raw = skill_file.read_text()
                meta, body = {}, raw
                if raw.startswith("---"):
                    parts = raw.split("---", 2)
                    if len(parts) >= 3:
                        meta = yaml.safe_load(parts[1]) or {}
                        body = parts[2].strip()
                self.skills[skill_dir.name] = {"meta": meta, "body": body}

    def get_descriptions(self) -> str:
        if not self.skills:
            return "  (no skills installed)"
        return "\n".join(
            f"  - {name}: {s['meta'].get('description', 'No description')}"
            for name, s in self.skills.items()
        )

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            available = ", ".join(self.skills) or "none"
            return f"Error: unknown skill '{name}'. Available: {available}"
        return f'<skill name="{name}">\n{skill["body"]}\n</skill>'

    def get_schema(self) -> dict:
        return _make_schema(list(self.skills))

    def handler(self, name: str) -> str:
        return self.get_content(name)
