from pathlib import Path

from generation.core.image import load_image
from generation.tools._paths import resolve_within

SCHEMA = {
    "type": "function",
    "name": "read_image",
    "description": "Read an image from the agent workspace as a vision input.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative image path (PNG/JPEG/GIF/WEBP).",
            },
        },
        "required": ["path"],
    },
}


def handler(path: str, *, root: str | Path = "."):
    return load_image(str(resolve_within(path, root)))
