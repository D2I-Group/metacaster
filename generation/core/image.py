
import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_IMAGES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


@dataclass
class ImageResult:

    path: str
    data_uri: str
    size_kb: float

    def to_content_block(self) -> dict:
        return {"type": "input_image", "image_url": self.data_uri}

    def to_tool_output(self) -> list:
        return [
            {"type": "input_text", "text": str(self)},
            self.to_content_block(),
        ]

    def __str__(self) -> str:
        return f"Image loaded: {self.path} ({self.size_kb:.1f} KB)."


def load_image(path: str) -> "ImageResult | str":
    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    mime, _ = mimetypes.guess_type(str(p))
    if mime not in SUPPORTED_IMAGES:
        return f"Error: unsupported image type '{mime}'. Supported: {', '.join(SUPPORTED_IMAGES)}"
    raw = p.read_bytes()
    return ImageResult(
        path=str(p),
        data_uri=f"data:{mime};base64,{base64.b64encode(raw).decode()}",
        size_kb=round(len(raw) / 1024, 1),
    )


def build_content_with_images(text: str, image_paths: list[str]) -> list:
    content: list = [{"type": "input_text", "text": text}]
    for path in image_paths:
        result = load_image(path)
        if isinstance(result, ImageResult):
            content.append(result.to_content_block())
        else:
            content.append({"type": "input_text", "text": f"[image error: {result}]"})
    return content
