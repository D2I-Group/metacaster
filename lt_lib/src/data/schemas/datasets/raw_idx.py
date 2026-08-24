from pydantic import BaseModel


class RawIdxParameterConfig(BaseModel):
    """Parameters for the raw_idx dataset (raw + window-position idx storage)."""

    val_ratio: float = 1 / 9
    scale: bool = True
