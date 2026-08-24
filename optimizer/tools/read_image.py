
from optimizer.core.image import load_image

SCHEMA = {
    "type": "function",
    "name": "read_image",
    "description": "Load image (PNG/JPG/GIF/WEBP) as vision input. Use for diagnostic plots.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

handler = load_image
