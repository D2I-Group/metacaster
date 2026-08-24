"""Model registration for PatchTSMixer."""

from benchmark.registry import MODEL_REGISTRY
from models.patchtsmixer.model import Model
from models.patchtsmixer.schema import ModelParameterConfig


def register() -> None:
    """Register PatchTSMixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "PatchTSMixer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            d_model=params.get("d_model", 64),
            expansion_factor=params.get("expansion_factor", 2),
            e_layers=params.get("e_layers", 2),
            dropout=params.get("dropout", 0.1),
            use_revin=bool(params.get("use_revin", True)),
        ),
        ModelParameterConfig,
    )
