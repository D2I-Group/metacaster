
from collections.abc import Callable
from functools import partial
from pathlib import Path

from . import read_file, read_image, run_python, write_file
from .skill import SkillLoader


def build_tools(
    skill_loader: SkillLoader,
    *,
    workspace: str | Path,
    enable_web_search: bool = False,
) -> tuple[list[dict], dict[str, Callable]]:
    tools: list[dict] = [
        read_file.SCHEMA,
        write_file.SCHEMA,
        read_image.SCHEMA,
        run_python.SCHEMA,
        skill_loader.get_schema(),
    ]
    if enable_web_search:
        tools.insert(0, {"type": "web_search"})
    handlers: dict[str, Callable] = {
        "read_file": partial(read_file.handler, root=workspace),
        "write_file": partial(write_file.handler, root=workspace),
        "read_image": partial(read_image.handler, root=workspace),
        "run_python": partial(run_python.handler, work_dir=workspace),
        "load_skill": skill_loader.handler,
    }
    return tools, handlers
