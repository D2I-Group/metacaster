---
name: regime_aware_bootstrap_suite
description: "Regime-aware bootstrap suite for periodic zero-locked, sparse intermittent, and continuous multivariate few-shot windows"
version: 3
applies_to_regimes:
  - periodic_zero_locked
  - intermittent_sparse
  - multichannel_bursty
  - continuous_seasonal
  - continuous_mixed
applies_to_few_shot_signatures:
  - "zero_fraction > 0.25 and nonnegative_fraction > 0.98 and best_candidate_lag_acf >= 0.75"
  - "zero_fraction > 0.60"
  - "n_channels > 1 and median_abs_channel_corr >= 0.25"
  - "best_candidate_lag_acf >= 0.45"
  - "C_eff == 1 and best_period <= 8 and zero_fraction < 0.05 and best_candidate_lag_acf >= 0.75"
  - "fallback when no narrower regime is a clear fit"
---

# Regime-Aware Bootstrap Suite

## Purpose

Use this skill when the few-shot tensor exhibits one of these observable regimes:

1. **Periodic zero-locked series** with repeated phase structure and long zero valleys.
2. **Intermittent sparse series** with many exact zeros and short active bursts.
3. **Continuous seasonal or mixed series** where shape preservation matters more than inventing new frequency content.
4. **Multichannel bursty series** where channels must move together.

The design addresses five recurring failure modes detected from the supplied windows: smearing exact-zero phases, overfilling inactive periods, collapsing whole-window amplitude diversity, breaking inter-channel correlation, and compressing short-cycle mean/std envelopes. It routes generation through five concrete strategies and a shared validation gate.

## Strategy Selection Guide

Compute the following few-shot diagnostics first:

- `zero_fraction`: fraction of values with absolute magnitude below `zero_eps`.
- `nonnegative_fraction`: fraction of values above `-zero_eps`.
- `best_candidate_lag_acf`: strongest mean autocorrelation among candidate periods.
- `best_period`: candidate lag achieving that strongest mean autocorrelation.
- `median_abs_channel_corr`: median absolute pairwise channel correlation over flattened windows.
- `sample_mean_cv`: std of per-window means divided by absolute mean of per-window means.

Then select a primary strategy:

| Regime | Trigger | Primary function | Why |
|---|---|---|---|
| `periodic_zero_locked` | `zero_fraction > 0.25`, `nonnegative_fraction > 0.98`, `best_candidate_lag_acf >= 0.75` | `phase_profile_bootstrap` | Preserves strong cycle shape and exact zero phases |
| `intermittent_sparse` | `zero_fraction > 0.60` and not periodic-zero-locked | `intermittent_mask_bootstrap` | Preserves zero runs and active bursts |
| `multichannel_bursty` | `C > 1` and `median_abs_channel_corr >= 0.25` | `correlation_block_mix` + minority `joint_envelope_block_mix` | Keeps strong cross-channel correlation while adding some joint-envelope diversity |
| `continuous_seasonal` | `best_candidate_lag_acf >= 0.45` and `best_period > 8` or zero-heavy | `phase_profile_bootstrap` + `correlation_block_mix` | Preserves periodic component without collapsing diversity |
| `continuous_seasonal` short-cycle subset | `C == 1`, `best_period <= 8`, `zero_fraction < 0.05`, `best_candidate_lag_acf >= 0.75` | `stratified_stat_phase_mix` + small `phase_profile_bootstrap` + very small `correlation_block_mix` tail | Preserves lag-4/6/8 structure while explicitly re-anchoring each generated window to the real mean/std envelope so the synthetic set does not collapse toward one middle amplitude band |
| `continuous_mixed` | fallback | `correlation_block_mix` | Conservative default for smooth real-valued windows |

## Recommended Combinations

- **Periodic zero-locked**: 100% `phase_profile_bootstrap`.
- **Intermittent sparse**: 100% `intermittent_mask_bootstrap`.
- **Continuous seasonal, short-cycle univariate** (`C == 1`, `best_period <= 8`, `zero_fraction < 0.05`): 85% `stratified_stat_phase_mix` + 10% `phase_profile_bootstrap` + 5% `correlation_block_mix`. Bias the stratified sampler toward tail mean/std bins and re-anchor each synthetic window to a sampled real envelope before the final repair step.
- **Continuous seasonal, other cases**: 60% `phase_profile_bootstrap` + 40% `correlation_block_mix`.
- **Multichannel bursty**: 80% `correlation_block_mix` + 20% `joint_envelope_block_mix` so the high-correlation backbone stays intact while a minority of samples widen joint level-vector coverage.
- **Continuous mixed**: 70% `correlation_block_mix` + 30% `phase_profile_bootstrap` with weak periodic blending.

Always finish with `repair_generated(...)`, then `validate_generated(...)`. If validation fails, switch to the next-more-conservative combination rather than adding large white noise.

## Executable Reference Implementation

```python
import numpy as np


def _safe_std(x, axis=None, keepdims=False):
    return np.std(x, axis=axis, keepdims=keepdims) + 1e-8


def _zero_eps(samples):
    return float(max(1e-8, 0.01 * np.std(samples)))


def _autocorr_lag(x, lag):
    x = np.asarray(x, dtype=np.float64)
    if lag <= 0 or lag >= x.shape[0]:
        return 0.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def _candidate_periods(L):
    base = [4, 6, 8, 12, 24, 48, 72, 96, 144, 168, 288]
    return [p for p in base if 2 < p < L - 2]


def _repeat_profile(profile, L):
    profile = np.asarray(profile)
    period = profile.shape[0]
    reps = int(np.ceil(L / period))
    if profile.ndim == 1:
        return np.tile(profile, reps)[:L]
    return np.tile(profile, (reps, 1))[:L]


def _phase_templates(samples, period):
    n_samples, L, C = samples.shape
    templates = np.zeros((n_samples, period, C), dtype=np.float32)
    for phase in range(period):
        idx = np.arange(phase, L, period)
        templates[:, phase, :] = samples[:, idx, :].mean(axis=1)
    return templates


def _flatten_pairwise_channel_corr(samples):
    n_samples, _, C = samples.shape
    if C <= 1:
        return 0.0
    corrs = []
    for a in range(C):
        xa = samples[:, :, a].reshape(n_samples, -1).reshape(-1)
        for b in range(a + 1, C):
            xb = samples[:, :, b].reshape(n_samples, -1).reshape(-1)
            if np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
                continue
            corr = np.corrcoef(xa, xb)[0, 1]
            if np.isfinite(corr):
                corrs.append(abs(float(corr)))
    if not corrs:
        return 0.0
    return float(np.median(corrs))


def infer_signature(samples):
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, L, C = samples.shape
    flat = samples.reshape(n_samples, -1)
    zero_eps = _zero_eps(samples)
    zero_fraction = float((np.abs(samples) <= zero_eps).mean())
    nonnegative_fraction = float((samples >= -zero_eps).mean())
    sample_means = flat.mean(axis=1)
    sample_mean_cv = float(sample_means.std() / (abs(sample_means.mean()) + 1e-8))
    candidates = _candidate_periods(L)
    scan_idx = np.arange(n_samples) if n_samples <= 64 else np.linspace(0, n_samples - 1, 64, dtype=int)
    acf_scores = {}
    for lag in candidates:
        vals = []
        for i in scan_idx:
            vals.append(_autocorr_lag(samples[i, :, 0], lag))
        acf_scores[lag] = float(np.mean(vals)) if vals else 0.0
    if acf_scores:
        best_period = max(acf_scores, key=acf_scores.get)
        best_candidate_lag_acf = float(acf_scores[best_period])
        medium_lags = [lag for lag in acf_scores if lag >= 24]
        if medium_lags:
            best_medium_period = max(medium_lags, key=lambda lag: acf_scores[lag])
            best_medium_lag_acf = float(acf_scores[best_medium_period])
        else:
            best_medium_period = best_period
            best_medium_lag_acf = best_candidate_lag_acf
    else:
        best_period = max(4, min(L // 4, 24))
        best_candidate_lag_acf = 0.0
        best_medium_period = best_period
        best_medium_lag_acf = 0.0
    median_abs_channel_corr = _flatten_pairwise_channel_corr(samples)

    if C == 1 and zero_fraction > 0.25 and nonnegative_fraction > 0.98 and best_medium_lag_acf >= 0.22:
        regime = "periodic_zero_locked"
    elif C > 1 and (median_abs_channel_corr >= 0.25 or sample_mean_cv >= 0.15):
        regime = "multichannel_bursty"
    elif zero_fraction > 0.60:
        regime = "intermittent_sparse"
    elif best_candidate_lag_acf >= 0.45 or best_medium_lag_acf >= 0.30:
        regime = "continuous_seasonal"
    else:
        regime = "continuous_mixed"

    return {
        "regime": regime,
        "n_samples": int(n_samples),
        "L": int(L),
        "C": int(C),
        "zero_fraction": zero_fraction,
        "nonnegative_fraction": nonnegative_fraction,
        "sample_mean_cv": sample_mean_cv,
        "best_period": int(best_period),
        "best_medium_period": int(best_medium_period),
        "best_candidate_lag_acf": best_candidate_lag_acf,
        "best_medium_lag_acf": best_medium_lag_acf,
        "median_abs_channel_corr": median_abs_channel_corr,
    }


def _block_bootstrap_from_pool(pool, L, rng, block_len):
    n_samples, pool_len, C = pool.shape
    out = np.zeros((L, C), dtype=np.float32)
    pos = 0
    max_start = max(1, pool_len - block_len + 1)
    while pos < L:
        take = min(block_len, L - pos)
        donor = int(rng.integers(0, n_samples))
        start = int(rng.integers(0, max_start))
        out[pos:pos + take] = pool[donor, start:start + take]
        pos += take
    return out


def repair_generated(generated, samples, max_iters=2):
    g = np.asarray(generated, dtype=np.float32).copy()
    samples = np.asarray(samples, dtype=np.float32)
    _, _, C = samples.shape
    lo = samples.min(axis=(0, 1))
    hi = samples.max(axis=(0, 1))
    orig_mean = samples.mean(axis=(0, 1))
    orig_std = samples.std(axis=(0, 1)) + 1e-8
    nonnegative = np.all(samples >= -_zero_eps(samples), axis=(0, 1))

    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    for _ in range(max_iters):
        for c in range(C):
            cur_mean = float(g[:, :, c].mean())
            cur_std = float(g[:, :, c].std()) + 1e-8
            g[:, :, c] = (g[:, :, c] - cur_mean) / cur_std * orig_std[c] + orig_mean[c]
            if nonnegative[c]:
                g[:, :, c] = np.maximum(g[:, :, c], 0.0)
            g[:, :, c] = np.clip(g[:, :, c], lo[c], hi[c])
    return g.astype(np.float32)


def phase_profile_bootstrap(samples, n_target, seed=42):
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, L, C = samples.shape
    sig = infer_signature(samples)
    period = int(sig["best_medium_period"] if sig["regime"] == "periodic_zero_locked" else sig["best_period"])
    block_len = max(4, min(L // 6, max(4, period // 2)))

    templates = _phase_templates(samples, period)
    repeated = np.stack([_repeat_profile(templates[i], L) for i in range(n_samples)], axis=0)
    residuals = samples - repeated
    global_template = templates.mean(axis=0)

    zero_eps = _zero_eps(samples)
    phase_zero_prob = np.zeros((period, C), dtype=np.float32)
    for phase in range(period):
        idx = np.arange(phase, L, period)
        phase_zero_prob[phase] = (np.abs(samples[:, idx, :]) <= zero_eps).mean(axis=(0, 1))
    source_zero_mask = np.abs(samples) <= zero_eps

    sample_means = samples.reshape(n_samples, -1).mean(axis=1)
    shift_scale = float(sample_means.std() * 0.25 + 1e-8)
    residual_scale = samples.std(axis=(0, 1), keepdims=False).reshape(1, C) * (0.01 if sig["regime"] == "periodic_zero_locked" else 0.04)

    generated = np.empty((n_target, L, C), dtype=np.float32)
    deterministic_zero = None
    phase_idx = np.arange(L) % period
    if sig["regime"] == "periodic_zero_locked":
        deterministic_zero = phase_zero_prob[phase_idx] >= 0.75
    final_zero_masks = np.zeros((n_target, L, C), dtype=bool) if deterministic_zero is not None else None

    for k in range(n_target):
        i = int(rng.integers(0, n_samples))
        j = int(rng.integers(0, n_samples))
        alpha = float(rng.uniform(0.35, 0.75))
        seasonal = alpha * repeated[i] + (1.0 - alpha) * repeated[j]
        seasonal = 0.70 * seasonal + 0.30 * _repeat_profile(global_template, L)

        resid = _block_bootstrap_from_pool(residuals, L, rng, block_len)
        sample = seasonal + (0.65 if sig["regime"] == "periodic_zero_locked" else 1.0) * resid

        channel_scale = rng.normal(loc=1.0, scale=0.03 if sig["regime"] == "periodic_zero_locked" else 0.05, size=(1, C)).astype(np.float32)
        sample_center = sample.mean(axis=0, keepdims=True)
        sample = (sample - sample_center) * channel_scale + sample_center
        sample = sample + rng.normal(0.0, 1.0, size=(L, C)).astype(np.float32) * residual_scale
        sample = sample + rng.normal(0.0, shift_scale, size=(1, C)).astype(np.float32)

        if deterministic_zero is not None:
            current_zero_mask = source_zero_mask[i] | deterministic_zero
            final_zero_masks[k] = current_zero_mask
            sample = np.where(current_zero_mask, 0.0, sample)
            sample = np.maximum(sample, 0.0)

        generated[k] = sample.astype(np.float32)

    generated = repair_generated(generated, samples)
    if final_zero_masks is not None:
        generated = np.where(final_zero_masks, 0.0, generated)
        generated = np.maximum(generated, 0.0)
    return generated.astype(np.float32)


def _quantile_groups(values, n_groups=6):
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n <= 0:
        return [np.array([], dtype=int)]
    order = np.argsort(values)
    n_groups = int(max(1, min(n_groups, n)))
    edges = np.linspace(0, n, n_groups + 1, dtype=int)
    groups = []
    for g in range(n_groups):
        group = order[edges[g]:edges[g + 1]]
        if group.size > 0:
            groups.append(group.astype(int))
    if not groups:
        groups = [order.astype(int)]
    return groups



def _group_membership(groups, n_items):
    membership = np.zeros(n_items, dtype=np.int32)
    for g, idx in enumerate(groups):
        membership[np.asarray(idx, dtype=int)] = g
    return membership



def _draw_group_member(rng, groups, extreme_prob=0.35):
    n_groups = len(groups)
    if n_groups <= 1:
        group_id = 0
    elif rng.random() < extreme_prob:
        group_id = 0 if rng.random() < 0.5 else n_groups - 1
    else:
        group_id = int(rng.integers(0, n_groups))
    members = np.asarray(groups[group_id], dtype=int)
    return group_id, int(members[int(rng.integers(0, len(members)))])



def _collect_neighbor_indices(groups, group_id, radius=1):
    picked = []
    lo = max(0, group_id - radius)
    hi = min(len(groups), group_id + radius + 1)
    for g in range(lo, hi):
        picked.extend(np.asarray(groups[g], dtype=int).tolist())
    if not picked:
        picked = np.asarray(groups[group_id], dtype=int).tolist()
    return np.asarray(picked, dtype=int)



def _samplewise_stat_reanchor(generated, samples, seed=42, mean_jitter_frac=0.03,
                              std_low=0.96, std_high=1.12, extreme_prob=0.60):
    """Match each generated window to a sampled real window envelope.

    This is intentionally window-wise rather than global: it preserves the
    temporal shape already synthesized for a sample, while restoring the
    real distribution of per-window mean and std that short-cycle traffic-like
    tensors need for downstream forecasting.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(generated, dtype=np.float32).copy()
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, _, C = samples.shape

    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1) + 1e-6
    aggregate_means = sample_means.mean(axis=1)
    mean_groups = _quantile_groups(aggregate_means, n_groups=min(7, n_samples))

    mean_jitter = sample_means.std(axis=0) * mean_jitter_frac + 1e-6
    lo = samples.min(axis=(0, 1))
    hi = samples.max(axis=(0, 1))
    nonnegative = np.all(samples >= -_zero_eps(samples), axis=(0, 1))

    for k in range(g.shape[0]):
        mean_gid, anchor_idx = _draw_group_member(rng, mean_groups, extreme_prob=extreme_prob)
        neighbor_idx = _collect_neighbor_indices(mean_groups, mean_gid, radius=1)
        partner_idx = int(neighbor_idx[int(rng.integers(0, len(neighbor_idx)))])

        target_mean = 0.82 * sample_means[anchor_idx] + 0.18 * sample_means[partner_idx]
        target_std = 0.82 * sample_stds[anchor_idx] + 0.18 * sample_stds[partner_idx]
        target_mean = target_mean + rng.normal(0.0, 1.0, size=C).astype(np.float32) * mean_jitter
        target_std = target_std * rng.uniform(std_low, std_high, size=C).astype(np.float32)

        cur_mean = g[k].mean(axis=0)
        cur_std = g[k].std(axis=0) + 1e-6
        g[k] = (g[k] - cur_mean.reshape(1, C)) / cur_std.reshape(1, C) * target_std.reshape(1, C) + target_mean.reshape(1, C)

        if np.any(nonnegative):
            g[k][:, nonnegative] = np.maximum(g[k][:, nonnegative], 0.0)
        g[k] = np.clip(g[k], lo.reshape(1, C), hi.reshape(1, C))

    return g.astype(np.float32)



def stratified_stat_phase_mix(samples, n_target, seed=42):
    """Return (n_target, L, C) float32 for short-cycle continuous seasonal tensors."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, L, C = samples.shape
    sig = infer_signature(samples)
    period = int(max(4, min(sig["best_period"], max(4, min(12, L // 4)))))
    block_len = int(max(2 * period, min(L // 6, max(period * 8, 2 * period))))

    templates = _phase_templates(samples, period)
    repeated = np.stack([_repeat_profile(templates[i], L) for i in range(n_samples)], axis=0)
    residuals = samples - repeated
    global_template = _repeat_profile(templates.mean(axis=0), L)

    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1) + 1e-6
    aggregate_means = sample_means.mean(axis=1)
    aggregate_stds = sample_stds.mean(axis=1)

    mean_groups = _quantile_groups(aggregate_means, n_groups=min(7, n_samples))
    std_groups = _quantile_groups(aggregate_stds, n_groups=min(7, n_samples))
    std_membership = _group_membership(std_groups, n_samples)

    mean_jitter_scale = sample_means.std(axis=0) * 0.035 + 1e-6
    std_scale_low = np.full(C, 0.96, dtype=np.float32)
    std_scale_high = np.full(C, 1.06, dtype=np.float32)
    if len(mean_groups) >= 2:
        std_scale_low[:] = 0.96
        std_scale_high[:] = 1.12

    global_lo = samples.min(axis=(0, 1))
    global_hi = samples.max(axis=(0, 1))
    nonnegative = np.all(samples >= -_zero_eps(samples), axis=(0, 1))
    fine_noise = samples.std(axis=(0, 1), keepdims=False).reshape(1, C) * 0.004

    generated = np.empty((n_target, L, C), dtype=np.float32)
    for k in range(n_target):
        mean_gid, target_idx = _draw_group_member(rng, mean_groups, extreme_prob=0.62)
        mean_neighborhood = _collect_neighbor_indices(mean_groups, mean_gid, radius=1)
        partner_idx = int(mean_neighborhood[int(rng.integers(0, len(mean_neighborhood)))])

        std_gid = int(std_membership[target_idx])
        std_neighborhood = _collect_neighbor_indices(std_groups, std_gid, radius=1)
        resid_pool = residuals[std_neighborhood]

        seasonal = 0.68 * repeated[target_idx] + 0.20 * repeated[partner_idx] + 0.12 * global_template
        resid = _block_bootstrap_from_pool(resid_pool, L, rng, block_len)
        sample = seasonal + resid

        insert_len = int(min(L, max(3 * period, min(14 * period, L // 3))))
        max_start = max(1, L - insert_len + 1)
        start = int(rng.integers(0, max_start))
        donor_block = samples[target_idx, start:start + insert_len].copy()
        local = sample[start:start + insert_len].copy()
        blend = float(rng.uniform(0.45, 0.65))
        sample[start:start + insert_len] = (1.0 - blend) * local + blend * donor_block

        target_mean = 0.82 * sample_means[target_idx] + 0.18 * sample_means[partner_idx]
        target_std = np.maximum(0.82 * sample_stds[target_idx] + 0.18 * sample_stds[partner_idx], 1e-6)
        target_mean = target_mean + rng.normal(0.0, 1.0, size=C).astype(np.float32) * mean_jitter_scale
        target_std = target_std * rng.uniform(std_scale_low, std_scale_high).astype(np.float32)

        cur_mean = sample.mean(axis=0)
        cur_std = sample.std(axis=0) + 1e-6
        sample = (sample - cur_mean.reshape(1, C)) / cur_std.reshape(1, C) * target_std.reshape(1, C) + target_mean.reshape(1, C)
        sample = sample + rng.normal(0.0, 1.0, size=(L, C)).astype(np.float32) * fine_noise

        cur_mean = sample.mean(axis=0)
        cur_std = sample.std(axis=0) + 1e-6
        sample = (sample - cur_mean.reshape(1, C)) / cur_std.reshape(1, C) * target_std.reshape(1, C) + target_mean.reshape(1, C)

        if np.any(nonnegative):
            sample[:, nonnegative] = np.maximum(sample[:, nonnegative], 0.0)
        sample = np.clip(sample, global_lo.reshape(1, C), global_hi.reshape(1, C))
        generated[k] = sample.astype(np.float32)

    generated = _samplewise_stat_reanchor(
        generated,
        samples,
        seed=seed + 17,
        mean_jitter_frac=0.03,
        std_low=0.96,
        std_high=1.12,
        extreme_prob=0.60,
    )
    return repair_generated(generated, samples, max_iters=1)



def intermittent_mask_bootstrap(samples, n_target, seed=42):
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, L, C = samples.shape
    zero_eps = _zero_eps(samples)
    zero_mask = np.abs(samples) <= zero_eps
    min_block = max(8, L // 24)
    max_block = max(min_block + 1, L // 6)
    amp_noise = samples.std(axis=(0, 1), keepdims=False).reshape(1, C) * 0.05
    sample_means = samples.reshape(n_samples, -1).mean(axis=1)

    generated = np.empty((n_target, L, C), dtype=np.float32)
    generated_masks = np.zeros((n_target, L, C), dtype=bool)
    for k in range(n_target):
        base_idx = int(rng.integers(0, n_samples))
        out = samples[base_idx].copy()
        out_mask = zero_mask[base_idx].copy()

        n_swaps = int(rng.integers(1, 5))
        for _ in range(n_swaps):
            donor_idx = int(rng.integers(0, n_samples))
            block_len = int(rng.integers(min_block, max_block))
            dst = int(rng.integers(0, max(1, L - block_len + 1)))
            src = int(rng.integers(0, max(1, L - block_len + 1)))
            out[dst:dst + block_len] = samples[donor_idx, src:src + block_len]
            out_mask[dst:dst + block_len] = zero_mask[donor_idx, src:src + block_len]

        active = ~out_mask
        mult = rng.lognormal(mean=0.0, sigma=0.12, size=(1, C)).astype(np.float32)
        out = np.where(active, np.maximum(out, 0.0) * mult, 0.0)
        out = np.where(active, out + rng.normal(0.0, 1.0, size=(L, C)).astype(np.float32) * amp_noise, 0.0)
        desired_mean = float(sample_means[int(rng.integers(0, n_samples))])
        current_mean = float(out.mean()) + 1e-8
        ratio = float(np.clip(desired_mean / current_mean, 0.50, 2.50))
        out = np.where(active, out * ratio, 0.0)
        out = np.maximum(out, 0.0)
        generated_masks[k] = ~active
        generated[k] = out.astype(np.float32)

    generated = repair_generated(generated, samples)
    generated = np.where(generated_masks, 0.0, generated)
    return generated.astype(np.float32)


def joint_envelope_block_mix(samples, n_target, seed=42):
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, L, C = samples.shape

    window_means = samples.mean(axis=1)
    window_stds = samples.std(axis=1) + 1e-6
    aggregate_levels = samples.reshape(n_samples, -1).mean(axis=1)
    level_shift_scale = float(aggregate_levels.std() * 0.12 + 1e-8)
    channel_global_std = samples.std(axis=(0, 1), keepdims=False).reshape(1, C)
    nonnegative = np.all(samples >= -_zero_eps(samples), axis=(0, 1))
    min_block = max(12, L // 18)
    max_block = max(min_block + 1, L // 4)

    mean_ac1 = np.zeros(C, dtype=np.float32)
    n_acf = min(64, n_samples)
    for c in range(C):
        vals = []
        for i in range(n_acf):
            vals.append(_autocorr_lag(samples[i, :, c], 1))
        mean_ac1[c] = float(np.mean(vals)) if vals else 0.0
    smooth_mix = np.clip((mean_ac1 - 0.70) / 0.25, 0.0, 1.0).reshape(1, C)

    generated = np.empty((n_target, L, C), dtype=np.float32)
    for k in range(n_target):
        i = int(rng.integers(0, n_samples))
        j = int(rng.integers(0, n_samples))
        alpha = float(rng.beta(0.8, 0.8))

        out = alpha * samples[i] + (1.0 - alpha) * samples[j]
        target_mean = alpha * window_means[i] + (1.0 - alpha) * window_means[j]
        target_std = np.maximum(alpha * window_stds[i] + (1.0 - alpha) * window_stds[j], 1e-6)

        current_mean = out.mean(axis=0)
        current_std = out.std(axis=0) + 1e-6
        vector_scale = np.clip(target_std / current_std, 0.85, 1.15)
        shared_scale = float(np.clip(np.mean(vector_scale), 0.85, 1.18))
        out = (out - current_mean.reshape(1, C)) * shared_scale + target_mean.reshape(1, C)
        out = (out - target_mean.reshape(1, C)) * (0.80 + 0.20 * vector_scale.reshape(1, C)) + target_mean.reshape(1, C)
        out = out + float(rng.normal(0.0, level_shift_scale))

        n_blocks = int(rng.integers(1, 4))
        for _ in range(n_blocks):
            donor_idx = int(rng.integers(0, n_samples))
            block_len = int(rng.integers(min_block, max_block))
            dst = int(rng.integers(0, max(1, L - block_len + 1)))
            src = int(rng.integers(0, max(1, L - block_len + 1)))
            donor_block = samples[donor_idx, src:src + block_len].copy()
            local = out[dst:dst + block_len].copy()
            donor_mean = donor_block.mean(axis=0, keepdims=True)
            donor_std = donor_block.std(axis=0, keepdims=True) + 1e-6
            local_mean = local.mean(axis=0, keepdims=True)
            local_std = local.std(axis=0, keepdims=True) + 1e-6
            donor_rescaled = (donor_block - donor_mean) / donor_std * local_std + local_mean
            blend = float(rng.uniform(0.35, 0.55))
            out[dst:dst + block_len] = (1.0 - blend) * local + blend * donor_rescaled

        shared_drive = rng.normal(0.0, 1.0, size=(L, 1)).astype(np.float32)
        for t in range(1, L):
            shared_drive[t] = 0.94 * shared_drive[t - 1] + 0.06 * shared_drive[t]
        out = out + shared_drive * (channel_global_std * 0.003)

        # Avoid smoothing high-ACF channels: it erases short-period detail and
        # empirically degrades downstream MSE on multichannel datasets.

        if np.any(nonnegative):
            out[:, nonnegative] = np.maximum(out[:, nonnegative], 0.0)
        generated[k] = out.astype(np.float32)

    return repair_generated(generated, samples)


def correlation_block_mix(samples, n_target, seed=42):
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, L, C = samples.shape
    min_block = max(8, L // 24)
    max_block = max(min_block + 1, L // 5)
    sample_means = samples.reshape(n_samples, -1).mean(axis=1)
    shift_scale = float(sample_means.std() * 0.30 + 1e-8)
    noise_scale = samples.std(axis=(0, 1), keepdims=False).reshape(1, C) * 0.02
    global_std = float(samples.std())

    generated = np.empty((n_target, L, C), dtype=np.float32)
    for k in range(n_target):
        i = int(rng.integers(0, n_samples))
        j = int(rng.integers(0, n_samples))
        alpha = float(rng.uniform(0.30, 0.70))
        out = alpha * samples[i] + (1.0 - alpha) * samples[j]
        n_blocks = int(rng.integers(1, 4))

        for _ in range(n_blocks):
            donor_idx = int(rng.integers(0, n_samples))
            block_len = int(rng.integers(min_block, max_block))
            dst = int(rng.integers(0, max(1, L - block_len + 1)))
            src = int(rng.integers(0, max(1, L - block_len + 1)))
            donor_block = samples[donor_idx, src:src + block_len].copy()
            out[dst:dst + block_len] = 0.50 * out[dst:dst + block_len] + 0.50 * donor_block

        center = out.mean(axis=0, keepdims=True)
        scalar_scale = float(rng.normal(loc=1.0, scale=0.06))
        out = (out - center) * scalar_scale + center
        out = out + float(rng.normal(0.0, shift_scale))
        shared_noise = rng.normal(0.0, 1.0, size=(L, 1)).astype(np.float32) * (0.01 * global_std)
        out = out + shared_noise + rng.normal(0.0, 1.0, size=(L, C)).astype(np.float32) * noise_scale
        generated[k] = out.astype(np.float32)

    return repair_generated(generated, samples)


def regime_aware_generate(samples, n_target, seed=42):
    sig = infer_signature(samples)
    regime = sig["regime"]

    if regime == "periodic_zero_locked":
        generated = phase_profile_bootstrap(samples, n_target, seed=seed)
    elif regime == "intermittent_sparse":
        generated = intermittent_mask_bootstrap(samples, n_target, seed=seed)
    elif regime == "multichannel_bursty":
        n_a = int(round(0.80 * n_target))
        part_a = correlation_block_mix(samples, n_a, seed=seed)
        part_b = joint_envelope_block_mix(samples, n_target - n_a, seed=seed + 1)
        generated = np.concatenate([part_a, part_b], axis=0)
    elif regime == "continuous_seasonal":
        if sig["C"] == 1 and sig["best_period"] <= 8 and sig["zero_fraction"] < 0.05 and sig["best_candidate_lag_acf"] >= 0.75:
            n_a = int(round(0.85 * n_target))
            n_b = int(round(0.10 * n_target))
            part_a = stratified_stat_phase_mix(samples, n_a, seed=seed)
            part_b = phase_profile_bootstrap(samples, n_b, seed=seed + 1)
            part_c = correlation_block_mix(samples, n_target - n_a - n_b, seed=seed + 2)
            generated = np.concatenate([part_a, part_b, part_c], axis=0)
        else:
            n_a = int(round(0.60 * n_target))
            part_a = phase_profile_bootstrap(samples, n_a, seed=seed)
            part_b = correlation_block_mix(samples, n_target - n_a, seed=seed + 1)
            generated = np.concatenate([part_a, part_b], axis=0)
    else:
        n_a = int(round(0.70 * n_target))
        part_a = correlation_block_mix(samples, n_a, seed=seed)
        part_b = phase_profile_bootstrap(samples, n_target - n_a, seed=seed + 1)
        generated = np.concatenate([part_a, part_b], axis=0)

    rng = np.random.default_rng(seed + 99)
    order = rng.permutation(generated.shape[0])
    generated = generated[order][:n_target]
    return generated.astype(np.float32)


def _mean_ac1(arr, n_limit):
    vals = []
    for i in range(min(n_limit, arr.shape[0])):
        x = arr[i].reshape(-1)
        if np.std(x) < 1e-12:
            continue
        vals.append(_autocorr_lag(x, 1))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def validate_generated(original, generated):
    original = np.asarray(original, dtype=np.float32)
    generated = np.asarray(generated)
    report = {"fail": [], "warn": [], "info": []}

    if generated.ndim != 3:
        report["fail"].append(f"generated ndim {generated.ndim} != 3")
    elif generated.shape[1:] != original.shape[1:]:
        report["fail"].append(f"shape tail mismatch: expected {original.shape[1:]}, got {generated.shape[1:]}")
    elif generated.shape[0] <= 0:
        report["fail"].append("generated has zero samples")
    else:
        report["info"].append(f"shape ok: {generated.shape}")

    if not np.isfinite(generated).all():
        report["fail"].append("generated contains NaN or Inf")
    else:
        report["info"].append("all values finite")

    if generated.dtype != np.float32:
        report["warn"].append(f"dtype is {generated.dtype}, expected float32")
    else:
        report["info"].append("dtype ok: float32")

    _, _, C = original.shape
    for c in range(C):
        o = original[:, :, c]
        g = generated[:, :, c]
        mean_rel = abs(float(g.mean()) - float(o.mean())) / (abs(float(o.mean())) + 1e-8)
        std_rel = abs(float(g.std()) - float(o.std())) / (abs(float(o.std())) + 1e-8)
        if mean_rel > 0.20:
            report["fail"].append(f"ch{c} mean drift {mean_rel:.1%} > 20%")
        elif mean_rel > 0.10:
            report["warn"].append(f"ch{c} mean drift {mean_rel:.1%} > 10%")
        if std_rel > 0.25:
            report["fail"].append(f"ch{c} std drift {std_rel:.1%} > 25%")
        elif std_rel > 0.15:
            report["warn"].append(f"ch{c} std drift {std_rel:.1%} > 15%")

    orig_ac1 = _mean_ac1(original, 64)
    gen_ac1 = _mean_ac1(generated.astype(np.float32), 128)
    ac1_gap = abs(gen_ac1 - orig_ac1)
    if ac1_gap > 0.15:
        report["fail"].append(f"lag1 acf gap {ac1_gap:.3f} > 0.15")
    elif ac1_gap > 0.08:
        report["warn"].append(f"lag1 acf gap {ac1_gap:.3f} > 0.08")
    else:
        report["info"].append(f"lag1 acf ok: orig={orig_ac1:.3f}, gen={gen_ac1:.3f}")

    orig_sample_means = original.reshape(original.shape[0], -1).mean(axis=1)
    gen_sample_means = generated.reshape(generated.shape[0], -1).mean(axis=1)
    div_ratio = float(gen_sample_means.std() / (orig_sample_means.std() + 1e-8))
    if div_ratio < 0.35 or div_ratio > 2.50:
        report["fail"].append(f"diversity ratio {div_ratio:.2f} outside [0.35, 2.50]")
    elif div_ratio < 0.50 or div_ratio > 2.00:
        report["warn"].append(f"diversity ratio {div_ratio:.2f} outside target [0.50, 2.00]")
    else:
        report["info"].append(f"diversity ratio ok: {div_ratio:.2f}")

    q_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    q_fail = 0
    q_warn = 0
    for q in q_levels:
        oq = float(np.quantile(orig_sample_means, q))
        gq = float(np.quantile(gen_sample_means, q))
        rel = abs(gq - oq) / (abs(oq) + 1e-8)
        if rel > 0.35:
            q_fail += 1
        elif rel > 0.20:
            q_warn += 1
    if q_fail >= 2:
        report["fail"].append(f"quantile mismatch: {q_fail}/5 quantiles exceed 35% error")
    elif q_warn >= 2:
        report["warn"].append(f"quantile drift: {q_warn}/5 quantiles exceed 20% error")
    else:
        report["info"].append("quantile match ok at q=[10,25,50,75,90]")

    orig_range = float(orig_sample_means.max() - orig_sample_means.min())
    gen_range = float(gen_sample_means.max() - gen_sample_means.min())
    range_ratio = gen_range / (orig_range + 1e-8)
    if range_ratio < 0.55:
        report["fail"].append(f"range coverage ratio {range_ratio:.2f} < 0.55")
    elif range_ratio < 0.70 or range_ratio > 1.60:
        report["warn"].append(f"range coverage ratio {range_ratio:.2f} outside [0.70, 1.60]")
    else:
        report["info"].append(f"range coverage ok: {range_ratio:.2f}")

    zero_eps = _zero_eps(original)
    orig_zero = float((np.abs(original) <= zero_eps).mean())
    gen_zero = float((np.abs(generated) <= zero_eps).mean())
    if orig_zero > 0.20:
        zero_gap = abs(gen_zero - orig_zero)
        if zero_gap > 0.10:
            report["fail"].append(f"zero fraction drift {zero_gap:.3f} > 0.10")
        elif zero_gap > 0.05:
            report["warn"].append(f"zero fraction drift {zero_gap:.3f} > 0.05")
        else:
            report["info"].append(f"zero fraction ok: orig={orig_zero:.3f}, gen={gen_zero:.3f}")

    if original.shape[2] > 1:
        orig_corr = _flatten_pairwise_channel_corr(original)
        gen_corr = _flatten_pairwise_channel_corr(generated.astype(np.float32))
        corr_gap = abs(gen_corr - orig_corr)
        if corr_gap > 0.20:
            report["fail"].append(f"channel corr gap {corr_gap:.3f} > 0.20")
        elif corr_gap > 0.12:
            report["warn"].append(f"channel corr gap {corr_gap:.3f} > 0.12")
        else:
            report["info"].append(f"channel corr ok: orig={orig_corr:.3f}, gen={gen_corr:.3f}")

    ok = len(report["fail"]) == 0
    return ok, report
```

## Execution Notes

1. Always call `infer_signature(samples)` first and log the returned regime.
2. For zero-heavy datasets, do not add large independent Gaussian noise after masking; it will immediately destroy sparsity and periodic zeros.
3. For multichannel datasets, keep the same block boundaries across channels. Do not mix channels independently.
4. `repair_generated(...)` is a narrow affine-and-clip repair step. It is safe for moderate mean/std drift, but it does not fix a broken regime choice.
5. If `validate_generated(...)` fails on zero-fraction drift or range coverage, regenerate with the regime-specific primary strategy rather than relaxing the thresholds.

## Validation Interpretation

Treat the following as red flags requiring regeneration:

- mean drift above 20% on any channel
- std drift above 25% on any channel
- lag-1 autocorrelation gap above 0.15
- diversity ratio outside 0.35 to 2.50
- two or more per-sample-mean quantiles with error above 35%
- range coverage below 0.55
- zero-fraction drift above 0.10 when the original windows are zero-heavy
- channel-correlation gap above 0.20 on multivariate datasets

If only warnings remain, keep the set that best preserves periodicity and sparsity while staying inside the original numeric domain.

## Preferred Runtime Pattern

1. Generate a candidate set with `regime_aware_generate(samples, n_target, seed)`.
2. Run `validate_generated(original=samples, generated=candidate)`.
3. If validation passes, write `validation_report.json` with top-level `passed: true`, `regime`, `strategy`, and the complete validation report, then save exactly once as float32.
4. If validation fails, write no final `dataset.npy`; regenerate and validate again:
   - for `periodic_zero_locked`, retry `phase_profile_bootstrap` with lower residual scale;
   - for `intermittent_sparse`, retry `intermittent_mask_bootstrap` with fewer swaps and no additive noise on zero positions;
   - for `multichannel_bursty`, retry with a smaller `joint_envelope_block_mix` proportion and use `correlation_block_mix` as the backbone whenever channel-correlation drift starts to rise;
   - for short-cycle `continuous_seasonal`, retry `stratified_stat_phase_mix` with stronger extreme-bin sampling before widening the `correlation_block_mix` share;
   - for `continuous_mixed`, retry `correlation_block_mix` with smaller block perturbations.
5. Never collapse to direct copies only; keep some block mixing so downstream forecasters see enough coverage.
