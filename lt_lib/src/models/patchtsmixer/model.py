"""PatchTSMixer model (IBM, KDD'23).

Reference: "TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series
Forecasting" — https://arxiv.org/abs/2306.09364

Lightweight MLP-Mixer over (patch, channel, feature) axes operating on patched
input. Input tensors are patched along the time axis, embedded to d_model, then
passed through stacked Mixer blocks before a per-channel linear head projects
to the forecast horizon.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.revin import RevIN


class _MlpBlock(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = max(dim * expansion, 1)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MixerLayer(nn.Module):
    """One Mixer layer: patch-mix + feature-mix + channel-mix with residuals."""

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        d_model: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_patch = nn.LayerNorm(d_model)
        self.patch_mixer = _MlpBlock(num_patches, expansion, dropout)

        self.norm_feature = nn.LayerNorm(d_model)
        self.feature_mixer = _MlpBlock(d_model, expansion, dropout)

        self.norm_channel = nn.LayerNorm(d_model)
        self.channel_mixer = _MlpBlock(num_channels, expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, D)
        # patch mixing across N
        h = self.norm_patch(x)
        h = h.transpose(-1, -2)  # (B, C, D, N)
        h = self.patch_mixer(h)
        h = h.transpose(-1, -2)
        x = x + h

        # feature mixing across D
        x = x + self.feature_mixer(self.norm_feature(x))

        # channel mixing across C
        h = self.norm_channel(x)
        h = h.permute(0, 2, 3, 1)  # (B, N, D, C)
        h = self.channel_mixer(h)
        h = h.permute(0, 3, 1, 2)  # (B, C, N, D)
        x = x + h
        return x


class PatchTSMixerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int,
        stride: int,
        d_model: int,
        expansion_factor: int,
        e_layers: int,
        dropout: float,
        use_revin: bool,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        else:
            self.revin = None

        # Padding so patches cover the whole sequence (replicate last value).
        self.pad_len = (
            0
            if (seq_len - patch_len) % stride == 0
            else stride - ((seq_len - patch_len) % stride)
        )
        self.padder = nn.ReplicationPad1d((0, self.pad_len))
        self.num_patches = (seq_len + self.pad_len - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)

        self.layers = nn.ModuleList(
            [
                _MixerLayer(
                    num_patches=self.num_patches,
                    num_channels=enc_in,
                    d_model=d_model,
                    expansion=expansion_factor,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if self.revin is not None:
            x = self.revin(x, "norm")

        # (B, C, L) -> pad -> patch
        z = x.permute(0, 2, 1)
        z = self.padder(z)
        # unfold time axis into patches: (B, C, N, P)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = self.patch_embed(z)  # (B, C, N, D)

        for layer in self.layers:
            z = layer(z)

        y = self.head(z)  # (B, C, pred_len)
        y = y.permute(0, 2, 1)  # (B, pred_len, C)

        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int,
        stride: int,
        d_model: int,
        expansion_factor: int,
        e_layers: int,
        dropout: float,
        use_revin: bool,
    ) -> None:
        super().__init__()
        self.model = PatchTSMixerModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            expansion_factor=expansion_factor,
            e_layers=e_layers,
            dropout=dropout,
            use_revin=use_revin,
        )

    def forward(self, x, *args):
        return self.model(x)
