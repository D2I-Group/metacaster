from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    cycle: int = 24
    seg_len: int = 48
    seg_stride: int = 24
    d_model: int = 256
    dropout: float = 0.1
    use_revin: bool = True
