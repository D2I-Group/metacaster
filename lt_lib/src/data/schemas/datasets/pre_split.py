from pydantic import BaseModel


class PreSplitParameterConfig(BaseModel):
    """Parameters for the pre_split dataset."""

    val_ratio: float = 1 / 9
    scale: bool = True
