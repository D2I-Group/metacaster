"""Model registration for FreqCycle."""

from benchmark.registry import MODEL_REGISTRY
from models.freqcycle.model import Model
from models.freqcycle.schema import ModelParameterConfig


def register() -> None:
    """Register FreqCycle model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "FreqCycle",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cycle=params.get("cycle", 24),
            seg_len=params.get("seg_len", 48),
            seg_stride=params.get("seg_stride", 24),
            d_model=params.get("d_model", 256),
            dropout=params.get("dropout", 0.1),
            use_revin=bool(params.get("use_revin", True)),
        ),
        ModelParameterConfig,
    )
