from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    expansion_factor: int = 2
    e_layers: int = 2
    dropout: float = 0.1
    use_revin: bool = True
