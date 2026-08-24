---
name: generation_strategies
description: "Detailed algorithms for time-series data augmentation: decomposition bootstrap, Fourier, block bootstrap, regime-aware methods"
version: 2
---

# Time-Series Generation Strategies

## Overview

Given N_sample few-shot windows of shape (L, C), generate N_target synthetic windows
that faithfully reproduce the statistical and temporal properties of the originals.

The generated data trains downstream forecasting models, so temporal realism and
diversity matter more than perfect marginal statistics.

## Strategy Selection Guide

| Data characteristic | Recommended primary strategy | Why |
|---|---|---|
| General purpose | Decomposition Bootstrap | Preserves trend + residual structure |
| Strong periodicity (daily/weekly cycles) | Fourier-Informed Bootstrap | Preserves frequency content |
| Short sequences (L < 30) | Covariance-Based Sampling | Can model full temporal covariance |
| Sparse/intermittent (many zeros) | Regime-Aware Bootstrap | Preserves zero/active structure |
| High diversity across samples | Anchor Interpolation | Spans the distribution space |
| Very long sequences (L > 200) | Block Bootstrap + Perturbation | Computationally efficient |

Always combine a PRIMARY strategy (60-70% of samples) with a SECONDARY diversity
strategy (30-40%) to ensure good distributional coverage.

## Strategy 1: Decomposition Bootstrap (DEFAULT — use when unsure)

Decomposes each sample into trend + seasonal + residual, then recombines parts
from different samples with perturbation.

```python
import numpy as np

def decomposition_bootstrap(samples, n_target, seed=42):
    rng = np.random.default_rng(seed)
    n_samples, L, C = samples.shape

    # Adaptive kernel size based on window length
    kernel_size = max(3, min(L // 10, 25))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size

    # Decompose all samples: trend + residual
    trends = np.zeros_like(samples)
    residuals = np.zeros_like(samples)
    for i in range(n_samples):
        for c in range(C):
            padded = np.pad(samples[i, :, c], kernel_size // 2, mode='edge')
            trend = np.convolve(padded, kernel, mode='valid')[:L]
            trends[i, :, c] = trend
            residuals[i, :, c] = samples[i, :, c] - trend

    # Compute per-sample statistics for adaptive perturbation
    sample_means = samples.reshape(n_samples, -1).mean(axis=1)
    sample_stds = samples.reshape(n_samples, -1).std(axis=1)
    level_std = sample_means.std()  # how much levels vary across samples

    generated = np.empty((n_target, L, C), dtype=np.float32)
    for idx in range(n_target):
        # Pick different samples for trend and residuals
        trend_idx = rng.integers(0, n_samples)
        resid_idx = rng.integers(0, n_samples)

        # Perturb trend: scale around its mean + shift
        t = trends[trend_idx].copy()
        for c in range(C):
            t_mean = t[:, c].mean()
            scale = rng.normal(1.0, 0.08)
            shift = rng.normal(0, level_std * 0.3)
            t[:, c] = (t[:, c] - t_mean) * scale + t_mean + shift

        # Perturb residuals: scale slightly
        r = residuals[resid_idx].copy()
        resid_scale = rng.uniform(0.7, 1.3)
        r = r * resid_scale

        generated[idx] = t + r

    return generated
```

## Strategy 2: Fourier-Informed Bootstrap

Preserves frequency-domain structure. Best for data with clear periodic components.

```python
def fourier_bootstrap(samples, n_target, seed=42):
    rng = np.random.default_rng(seed)
    n_samples, L, C = samples.shape

    # Compute FFT for all samples
    ffts = np.fft.rfft(samples, axis=1)  # (n_samples, freq_bins, C)
    amplitudes = np.abs(ffts)
    phases = np.angle(ffts)

    # Identify dominant frequencies (top-k by mean amplitude)
    mean_amp = amplitudes.mean(axis=0)  # (freq_bins, C)
    n_freqs = ffts.shape[1]

    # Statistics for perturbation
    amp_mean = amplitudes.mean(axis=0)
    amp_std = amplitudes.std(axis=0) + 1e-10

    generated = np.empty((n_target, L, C), dtype=np.float32)
    for idx in range(n_target):
        # Pick a base sample for phase coherence
        base_idx = rng.integers(0, n_samples)

        # Amplitudes: sample from distribution around the mean
        # Use tighter perturbation for dominant frequencies
        new_amp = np.abs(rng.normal(amp_mean, amp_std * 0.25))

        # Phase: start from base sample's phases, add small noise
        # Keep low-frequency phases more stable (they control overall shape)
        phase_noise_scale = np.linspace(0.05, 0.3, n_freqs).reshape(-1, 1)
        phase_noise = rng.normal(0, phase_noise_scale, size=phases[base_idx].shape)
        new_phase = phases[base_idx] + phase_noise
        new_phase[0] = 0  # DC component phase = 0

        # Reconstruct
        new_fft = new_amp * np.exp(1j * new_phase)
        sample = np.fft.irfft(new_fft, n=L, axis=0)

        # Adjust mean and std to match a randomly selected original
        target_idx = rng.integers(0, n_samples)
        for c in range(C):
            orig_mean = samples[target_idx, :, c].mean()
            orig_std = samples[target_idx, :, c].std() + 1e-10
            gen_mean = sample[:, c].mean()
            gen_std = sample[:, c].std() + 1e-10
            sample[:, c] = (sample[:, c] - gen_mean) / gen_std * orig_std + orig_mean

        generated[idx] = sample

    return generated.astype(np.float32)
```

## Strategy 3: Anchor Interpolation (Diversity-Focused)

Creates new samples by interpolating between pairs of originals with temporal
alignment. Good for covering distribution space.

```python
def anchor_interpolation(samples, n_target, seed=42):
    rng = np.random.default_rng(seed)
    n_samples, L, C = samples.shape

    # Compute per-sample feature vectors for diversity-aware pairing
    sample_features = samples.reshape(n_samples, -1)
    sample_means = sample_features.mean(axis=1)

    # Sort samples by their mean level for structured interpolation
    sorted_idx = np.argsort(sample_means)

    generated = np.empty((n_target, L, C), dtype=np.float32)
    for idx in range(n_target):
        # Pick two samples — prefer diverse pairs
        i = rng.integers(0, n_samples)
        j = rng.integers(0, n_samples)

        # Beta distribution: U-shaped (0.3, 0.3) produces more extreme
        # values near 0 and 1, keeping most samples close to originals
        alpha = rng.beta(0.3, 0.3)

        # Interpolate
        mixed = alpha * samples[i] + (1 - alpha) * samples[j]

        # Add small calibrated noise to prevent exact duplicates
        noise_scale = samples.std(axis=(0, 1), keepdims=False) * 0.02  # (C,)
        noise = rng.normal(0, 1, size=(L, C)) * noise_scale
        mixed = mixed + noise

        generated[idx] = mixed

    return generated.astype(np.float32)
```

## Strategy 4: Block Bootstrap (for Long Sequences)

Preserves local temporal structure by resampling contiguous blocks.

```python
def block_bootstrap(samples, n_target, seed=42, block_fraction=0.2):
    rng = np.random.default_rng(seed)
    n_samples, L, C = samples.shape

    block_len = max(3, int(L * block_fraction))

    generated = np.empty((n_target, L, C), dtype=np.float32)
    for idx in range(n_target):
        # Start with a random base sample
        base_idx = rng.integers(0, n_samples)
        result = samples[base_idx].copy()

        # Replace 1-3 random blocks with blocks from other samples
        n_swaps = rng.integers(1, 4)
        for _ in range(n_swaps):
            # Source block from another sample
            src_idx = rng.integers(0, n_samples)
            src_start = rng.integers(0, L - block_len)
            dst_start = rng.integers(0, L - block_len)

            block = samples[src_idx, src_start:src_start + block_len].copy()

            # Level-adjust the block to match local context
            local_mean = result[max(0, dst_start - 5):dst_start + block_len + 5].mean(axis=0)
            block_mean = block.mean(axis=0)
            block = block - block_mean + local_mean

            # Smooth the transitions with cosine blending (5 timesteps)
            blend_len = min(5, block_len // 2)
            blend = np.linspace(0, 1, blend_len).reshape(-1, 1)

            result[dst_start:dst_start + blend_len] = (
                result[dst_start:dst_start + blend_len] * (1 - blend) +
                block[:blend_len] * blend
            )
            result[dst_start + blend_len:dst_start + block_len - blend_len] = (
                block[blend_len:block_len - blend_len]
            )
            end_start = dst_start + block_len - blend_len
            result[end_start:dst_start + block_len] = (
                block[block_len - blend_len:] * (1 - blend) +
                result[end_start:dst_start + block_len] * blend
            )

        # Small global perturbation
        scale = rng.normal(1.0, 0.03)
        shift = rng.normal(0, result.std() * 0.02)
        result = result * scale + shift

        generated[idx] = result

    return generated.astype(np.float32)
```

## Strategy 5: Covariance-Based Sampling (Short Sequences Only, L < 50)

Models the full temporal covariance structure. Only practical for short series.

```python
def covariance_sampling(samples, n_target, seed=42):
    rng = np.random.default_rng(seed)
    n_samples, L, C = samples.shape

    generated_channels = []
    for c in range(C):
        channel_data = samples[:, :, c]  # (n_samples, L)
        mean_ts = channel_data.mean(axis=0)
        centered = channel_data - mean_ts

        # Empirical covariance with regularization
        cov = np.cov(centered.T)  # (L, L)
        reg = np.eye(L) * max(1e-4 * np.trace(cov) / L, 1e-8)
        cov = cov + reg

        # Sample from multivariate normal
        new_samples = rng.multivariate_normal(mean_ts, cov, size=n_target)
        generated_channels.append(new_samples)

    # Stack channels: (n_target, L, C)
    result = np.stack(generated_channels, axis=2)

    # For multivariate: apply cross-channel correlation correction
    if C > 1:
        orig_flat = samples.reshape(n_samples, -1)
        gen_flat = result.reshape(n_target, -1)
        # Match marginal statistics per channel
        for c in range(C):
            orig_std = samples[:, :, c].std()
            gen_std = result[:, :, c].std() + 1e-10
            orig_mean = samples[:, :, c].mean()
            gen_mean = result[:, :, c].mean()
            result[:, :, c] = (result[:, :, c] - gen_mean) / gen_std * orig_std + orig_mean

    return result.astype(np.float32)
```

## Recommended Combinations

For most datasets, combine strategies for best results:

1. **Default mix**: 60% Decomposition Bootstrap + 40% Anchor Interpolation
2. **Periodic data**: 50% Fourier Bootstrap + 30% Decomposition Bootstrap + 20% Anchor Interpolation
3. **Short sequences (L<30)**: 50% Covariance Sampling + 30% Anchor Interpolation + 20% Decomposition Bootstrap
4. **Long sequences (L>200)**: 50% Block Bootstrap + 30% Decomposition Bootstrap + 20% Anchor Interpolation
5. **Sparse/intermittent**: Use Decomposition Bootstrap, but mask zeros separately (see below)
6. **High diversity**: 40% Anchor Interpolation + 40% Decomposition Bootstrap + 20% direct copies with noise

## Handling Special Cases

### Sparse/Intermittent Data (many zeros)
```python
# Generate in two stages:
# 1. Generate the "active" pattern (where values > 0)
# 2. Apply a zero mask sampled from the originals
zero_mask = (samples == 0)
zero_fraction = zero_mask.mean()
# After generation, randomly zero out ~zero_fraction of timesteps
# using patterns from original zero runs
```

### Bounded Data (e.g., percentages 0-100, non-negative counts)
```python
# After generation, clip to domain bounds
if value_range[0] is not None:
    generated = np.maximum(generated, value_range[0])
if value_range[1] is not None:
    generated = np.minimum(generated, value_range[1])
```

### Multivariate Cross-Channel Correlations
```python
# Compute cross-channel correlation from originals
# Then verify generated data has similar correlations
for i in range(C):
    for j in range(i+1, C):
        orig_corr = np.corrcoef(
            samples[:, :, i].ravel(), samples[:, :, j].ravel()
        )[0, 1]
        gen_corr = np.corrcoef(
            generated[:, :, i].ravel(), generated[:, :, j].ravel()
        )[0, 1]
        print(f"Ch {i}-{j}: orig={orig_corr:.3f} gen={gen_corr:.3f}")
```

## Post-Processing Checklist

After generation, always:
1. Verify shape is exactly (N_target, L, C).
2. Cast to float32.
3. Clip values to observed range (with 10% margin on each side).
4. If data is non-negative by nature, clip at 0.
5. Check for NaN/Inf — replace with channel mean if found.
6. Verify diversity: std of per-sample means should be 0.5x-2.0x of original.
7. Shuffle along axis 0 before saving.
