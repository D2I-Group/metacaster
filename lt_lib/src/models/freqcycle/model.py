"""FreqCycle model (AAAI'26).

Reference: "FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time
Series Forecasting" — https://arxiv.org/abs/2603.09661

No official code available at time of writing; this is a faithful
re-implementation following the architecture description in the paper.

Two modules:

* FECF (Filter-Enhanced Cycle Forecasting):
    - Learnable cyclical basis Q in R^(cycle, C).
    - Replicate Q to cover horizon, then apply learnable elementwise frequency
      filter: c' = iFFT(filter(FFT(c))).
* SFPL (Segmented Frequency-domain Pattern Learning):
    - Residual r = x - cycle(x).
    - Sliding-window segmentation with zero-padding -> s segments of length
      seg_len.
    - FFT per segment, softmax-weighted summation across segments, iFFT.
    - MLP FFN maps the aggregated time-domain residual signal to the forecast
      residual horizon.

The final forecast is filtered_cycle(horizon) + residual_prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.revin import RevIN


class _FECF(nn.Module):
    """Filter-enhanced cycle forecasting branch."""

    def __init__(self, cycle: int, enc_in: int, pred_len: int) -> None:
        super().__init__()
        self.cycle = cycle
        self.enc_in = enc_in
        self.pred_len = pred_len

        self.basis = nn.Parameter(torch.zeros(cycle, enc_in))
        # Filter operates on the rfft of the horizon-length signal.
        n_freq = pred_len // 2 + 1
        # Initialise close to identity (real=1, imag=0).
        self.filter_real = nn.Parameter(torch.ones(n_freq, enc_in))
        self.filter_imag = nn.Parameter(torch.zeros(n_freq, enc_in))

    def cycle_at(self, offset: torch.Tensor, length: int) -> torch.Tensor:
        # offset: (B,), length: int -> (B, length, C)
        idx = (
            offset.view(-1, 1) + torch.arange(length, device=offset.device).view(1, -1)
        ) % self.cycle
        return self.basis[idx]

    def filter_horizon(self, signal: torch.Tensor) -> torch.Tensor:
        # signal: (B, pred_len, C)
        spec = torch.fft.rfft(signal, n=self.pred_len, dim=1)
        weight = torch.complex(self.filter_real, self.filter_imag)
        spec = spec * weight.unsqueeze(0)
        return torch.fft.irfft(spec, n=self.pred_len, dim=1)


class _SFPL(nn.Module):
    """Segmented frequency-domain pattern learning branch."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        seg_len: int,
        seg_stride: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.seg_stride = seg_stride

        # Segments needed to cover seq_len with sliding window of seg_len/seg_stride.
        if seq_len <= seg_len:
            self.num_segments = 1
        else:
            self.num_segments = (seq_len - seg_len + seg_stride - 1) // seg_stride + 1
        self.padded_len = seg_len + (self.num_segments - 1) * seg_stride

        # Softmax weights over segments, learned per channel.
        self.seg_weight = nn.Parameter(torch.zeros(self.num_segments, enc_in))

        # MLP FFN: per-channel time-domain projection seg_len -> d_model -> pred_len.
        self.ffn = nn.Sequential(
            nn.Linear(seg_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # residual: (B, L, C)
        _b, _, _c = residual.shape
        # Zero-pad to padded_len.
        pad = self.padded_len - self.seq_len
        if pad > 0:
            residual = F.pad(residual, (0, 0, 0, pad))

        # (B, L', C) -> (B, C, L') -> unfold into (B, C, S, seg_len)
        z = residual.permute(0, 2, 1)
        segs = z.unfold(dimension=-1, size=self.seg_len, step=self.seg_stride)
        # (B, C, S, seg_len)

        # FFT per segment.
        spec = torch.fft.rfft(segs, n=self.seg_len, dim=-1)

        # Softmax across segments, per channel.
        # weight: (S, C) -> (1, C, S, 1)
        w = (
            torch.softmax(self.seg_weight, dim=0)
            .permute(1, 0)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        spec = (spec * w).sum(dim=2)  # (B, C, n_freq)

        # iFFT back to time domain (length seg_len).
        time_sig = torch.fft.irfft(spec, n=self.seg_len, dim=-1)  # (B, C, seg_len)

        # Per-channel MLP -> (B, C, pred_len)
        out = self.ffn(time_sig)
        return out.permute(0, 2, 1)  # (B, pred_len, C)


class FreqCycleModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cycle: int,
        seg_len: int,
        seg_stride: int,
        d_model: int,
        dropout: float,
        use_revin: bool,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cycle = cycle
        self.use_revin = use_revin

        self.revin = (
            RevIN(enc_in, affine=True, subtract_last=False) if use_revin else None
        )
        self.fecf = _FECF(cycle=cycle, enc_in=enc_in, pred_len=pred_len)
        self.sfpl = _SFPL(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            seg_len=min(seg_len, seq_len),
            seg_stride=max(1, min(seg_stride, seq_len)),
            d_model=d_model,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")

        cycle_in = self.fecf.cycle_at(cycle_index, self.seq_len)
        residual = x - cycle_in

        cycle_out = self.fecf.cycle_at(
            (cycle_index + self.seq_len) % self.cycle, self.pred_len
        )
        cycle_out = self.fecf.filter_horizon(cycle_out)

        residual_pred = self.sfpl(residual)

        y = cycle_out + residual_pred

        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cycle: int,
        seg_len: int,
        seg_stride: int,
        d_model: int,
        dropout: float,
        use_revin: bool,
    ) -> None:
        super().__init__()
        self.cycle = cycle
        self.model = FreqCycleModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            cycle=cycle,
            seg_len=seg_len,
            seg_stride=seg_stride,
            d_model=d_model,
            dropout=dropout,
            use_revin=use_revin,
        )

    def _resolve_cycle_index(self, x_time_stamp: torch.Tensor) -> torch.Tensor:
        # Match CycleNet's mark-based offset extraction.
        if self.cycle == 24:
            return x_time_stamp[:, 0, 3].to(torch.int64)
        if self.cycle == 7:
            return x_time_stamp[:, 0, 4].to(torch.int64)
        if self.cycle == 168:
            return (x_time_stamp[:, 0, 4] * 24 + x_time_stamp[:, 0, 3]).to(torch.int64)
        return x_time_stamp[:, 0, 3].to(torch.int64) % self.cycle

    def forward(self, x, x_time_stamp, *args):
        cycle_index = self._resolve_cycle_index(x_time_stamp)
        return self.model(x, cycle_index)
