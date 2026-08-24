---
name: validation
description: "Validation suite with automated checks, visual diagnostics, and an iterative stat_match_rescale gate that converges through value-range clipping."
version: 5
---

# Validation Skill

Run this validation suite BEFORE saving dataset.npy. Both automated AND visual checks are required.

**IMPORTANT**: If validation finds CRITICAL issues or WARNINGS, you MUST fix
the generation and re-validate (up to 2 fix iterations). Do NOT save data
that has issues.

## Automated Validation Code

```python
import numpy as np

def validate_generated(original, generated, meta, output_dir):
    """
    Validate generated dataset against original samples.

    Args:
        original: np.ndarray (n_sample, L, C) - the few-shot input
        generated: np.ndarray (n_target, L, C) - the generated data
        meta: dict - from meta.json
        output_dir: str - where to save the report

    Returns:
        bool: True if all checks pass
    """
    issues = []
    warnings = []
    info = []

    n_target = meta.get("generation_target", 10000)
    value_range = meta.get("value_range", [None, None])
    C = original.shape[2]

    # 1. Shape check (CRITICAL)
    expected_L = original.shape[1]
    expected_C = original.shape[2]
    if generated.shape != (n_target, expected_L, expected_C):
        issues.append(
            f"SHAPE: expected ({n_target}, {expected_L}, {expected_C}), "
            f"got {generated.shape}"
        )
    else:
        info.append(f"Shape OK: {generated.shape}")

    # 2. NaN/Inf check (CRITICAL)
    n_nan = np.isnan(generated).sum()
    n_inf = np.isinf(generated).sum()
    if n_nan > 0:
        issues.append(f"Contains {n_nan} NaN values")
    if n_inf > 0:
        issues.append(f"Contains {n_inf} Inf values")
    if n_nan == 0 and n_inf == 0:
        info.append("No NaN/Inf values")

    # 3. dtype check
    if generated.dtype != np.float32:
        warnings.append(f"dtype: expected float32, got {generated.dtype}")
    else:
        info.append("dtype OK: float32")

    # 4. Per-channel statistics comparison
    for c in range(C):
        orig_ch = original[:, :, c]
        gen_ch = generated[:, :, c]

        # Mean comparison
        orig_mean = orig_ch.mean()
        gen_mean = gen_ch.mean()
        mean_rel_err = abs(gen_mean - orig_mean) / (abs(orig_mean) + 1e-8)
        if mean_rel_err > 0.3:
            issues.append(f"Ch{c} MEAN DRIFT: orig={orig_mean:.4f} gen={gen_mean:.4f} err={mean_rel_err:.1%}")
        elif mean_rel_err > 0.15:
            warnings.append(f"Ch{c} mean drift: orig={orig_mean:.4f} gen={gen_mean:.4f} err={mean_rel_err:.1%}")
        else:
            info.append(f"Ch{c} mean OK: orig={orig_mean:.4f} gen={gen_mean:.4f}")

        # Std comparison
        orig_std = orig_ch.std()
        gen_std = gen_ch.std()
        std_rel_err = abs(gen_std - orig_std) / (abs(orig_std) + 1e-8)
        if std_rel_err > 0.5:
            issues.append(f"Ch{c} STD MISMATCH: orig={orig_std:.4f} gen={gen_std:.4f} err={std_rel_err:.1%}")
        elif std_rel_err > 0.2:
            warnings.append(f"Ch{c} std drift: orig={orig_std:.4f} gen={gen_std:.4f} err={std_rel_err:.1%}")
        else:
            info.append(f"Ch{c} std OK: orig={orig_std:.4f} gen={gen_std:.4f}")

        # Value range
        gen_min, gen_max = gen_ch.min(), gen_ch.max()
        orig_min, orig_max = orig_ch.min(), orig_ch.max()
        margin = (orig_max - orig_min) * 0.15
        if gen_min < orig_min - margin or gen_max > orig_max + margin:
            warnings.append(
                f"Ch{c} range: gen=[{gen_min:.2f}, {gen_max:.2f}] "
                f"orig=[{orig_min:.2f}, {orig_max:.2f}]"
            )

    # 5. Autocorrelation check (lag-1, per channel)
    for c in range(C):
        orig_ac1 = []
        gen_ac1 = []
        for i in range(min(50, len(original))):
            s = original[i, :, c]
            if s.std() > 1e-10:
                ac = np.corrcoef(s[:-1], s[1:])[0, 1]
                if not np.isnan(ac):
                    orig_ac1.append(ac)
        for i in range(min(100, len(generated))):
            s = generated[i, :, c]
            if s.std() > 1e-10:
                ac = np.corrcoef(s[:-1], s[1:])[0, 1]
                if not np.isnan(ac):
                    gen_ac1.append(ac)

        if orig_ac1 and gen_ac1:
            orig_ac1_mean = np.mean(orig_ac1)
            gen_ac1_mean = np.mean(gen_ac1)
            ac_diff = abs(gen_ac1_mean - orig_ac1_mean)
            if ac_diff > 0.2:
                issues.append(f"Ch{c} AC1 MISMATCH: orig={orig_ac1_mean:.3f} gen={gen_ac1_mean:.3f}")
            else:
                info.append(f"Ch{c} AC1 OK: orig={orig_ac1_mean:.3f} gen={gen_ac1_mean:.3f}")

    # 6. Diversity check (per-sample means)
    sample_means_orig = original.reshape(len(original), -1).mean(axis=1)
    sample_means_gen = generated.reshape(len(generated), -1).mean(axis=1)
    div_ratio = sample_means_gen.std() / (sample_means_orig.std() + 1e-8)
    if div_ratio < 0.3:
        issues.append(f"LOW DIVERSITY: ratio={div_ratio:.2f} (generated too similar)")
    elif div_ratio > 3.0:
        issues.append(f"HIGH DIVERSITY: ratio={div_ratio:.2f} (generated too spread)")
    elif div_ratio < 0.5 or div_ratio > 2.0:
        warnings.append(f"Diversity ratio={div_ratio:.2f} (target: 0.5-2.0)")
    else:
        info.append(f"Diversity OK: ratio={div_ratio:.2f}")

    # 7. QUANTILE CHECK (most important for heavy-tailed data!)
    # Compare quantiles of per-sample means — catches distribution shape mismatches
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    orig_q = np.quantile(sample_means_orig, quantiles)
    gen_q = np.quantile(sample_means_gen, quantiles)
    info.append("Per-sample mean quantiles (orig → gen):")
    q_issues = 0
    for q, oq, gq in zip(quantiles, orig_q, gen_q):
        rel_err = abs(gq - oq) / (abs(oq) + 1e-8)
        status = "OK" if rel_err < 0.3 else "DRIFT"
        if rel_err >= 0.3:
            q_issues += 1
        info.append(f"  Q{q:.0%}: orig={oq:.2f} gen={gq:.2f} err={rel_err:.1%} {status}")
    if q_issues >= 3:
        issues.append(f"QUANTILE MISMATCH: {q_issues}/5 quantiles have >30% error — distribution shape is wrong")
    elif q_issues >= 2:
        warnings.append(f"Quantile drift: {q_issues}/5 quantiles have >30% error")

    # 8. Range coverage check — are min/max of per-sample means preserved?
    orig_range = sample_means_orig.max() - sample_means_orig.min()
    gen_range = sample_means_gen.max() - sample_means_gen.min()
    range_ratio = gen_range / (orig_range + 1e-8)
    info.append(f"Per-sample mean range: orig=[{sample_means_orig.min():.2f}, {sample_means_orig.max():.2f}] "
                f"gen=[{sample_means_gen.min():.2f}, {sample_means_gen.max():.2f}] ratio={range_ratio:.2f}")
    if range_ratio < 0.5:
        issues.append(f"RANGE COLLAPSED: generated per-sample means cover only {range_ratio:.0%} of original range. "
                       f"You need to generate more extreme high/low samples.")
    elif range_ratio < 0.7:
        warnings.append(f"Range shrinkage: ratio={range_ratio:.2f} — generated data missing some extreme values")

    # 9. Duplicate check (quick)
    n_check = min(200, len(generated))
    check_idx = np.random.choice(len(generated), n_check, replace=False)
    gen_flat = generated.reshape(len(generated), -1)
    n_dupes = 0
    for i in range(min(50, n_check)):
        for j in range(i+1, min(50, n_check)):
            if np.allclose(gen_flat[check_idx[i]], gen_flat[check_idx[j]], atol=1e-5):
                n_dupes += 1
    if n_dupes > 5:
        issues.append(f"DUPLICATES: {n_dupes} duplicate pairs found")
    else:
        info.append("No significant duplicates")

    # Build report
    lines = ["=" * 60, "VALIDATION REPORT", "=" * 60, ""]

    if issues:
        lines.append(f"CRITICAL ISSUES: {len(issues)}")
        for issue in issues:
            lines.append(f"  [FAIL] {issue}")
        lines.append("")

    if warnings:
        lines.append(f"WARNINGS: {len(warnings)}")
        for w in warnings:
            lines.append(f"  [WARN] {w}")
        lines.append("")

    if not issues:
        lines.append("ALL CRITICAL CHECKS PASSED")
        lines.append("")

    lines.append("Details:")
    for line in info:
        lines.append(f"  {line}")

    report = "\n".join(lines)
    print(report)

    with open(f"{output_dir}/validation_report.txt", "w") as f:
        f.write(report)

    return len(issues) == 0
```

## Visual Validation (REQUIRED)

After the automated checks, generate these plots and inspect them with `read_image`:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def visual_validation(original, generated, output_dir):
    """Generate comparison plots. Call read_image on each to verify quality."""

    rng = np.random.default_rng(123)
    C = original.shape[2]

    # Plot 1: Sample overlay comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Original samples
    idx_orig = rng.choice(len(original), min(8, len(original)), replace=False)
    for i in idx_orig:
        axes[0].plot(original[i, :, 0], alpha=0.5, linewidth=0.8)
    axes[0].set_title(f'Original Samples (n={len(original)})')
    axes[0].set_ylabel('Value')

    # Generated samples
    idx_gen = rng.choice(len(generated), min(8, len(generated)), replace=False)
    for i in idx_gen:
        axes[1].plot(generated[i, :, 0], alpha=0.5, linewidth=0.8)
    axes[1].set_title(f'Generated Samples (n={len(generated)})')
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Timestep')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/val_samples.png', dpi=100)
    plt.close()

    # Plot 2: Distribution of per-sample means
    orig_means = original.reshape(len(original), -1).mean(axis=1)
    gen_means = generated.reshape(len(generated), -1).mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(orig_means, bins=30, alpha=0.6, density=True, label='Original', color='blue')
    ax.hist(gen_means, bins=50, alpha=0.6, density=True, label='Generated', color='orange')
    ax.set_title('Distribution of Per-Sample Means')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/val_means_dist.png', dpi=100)
    plt.close()

    print(f"Plots saved to {output_dir}/val_samples.png and val_means_dist.png")
    print("Use read_image to inspect them.")
```

## Common Fixes

### Fix NaN values
```python
generated = np.nan_to_num(generated, nan=0.0)
```

### Fix value range
```python
margin = (orig_max - orig_min) * 0.05
generated = np.clip(generated, orig_min - margin, orig_max + margin)
```

### Fix mean/std drift
```python
# Per-channel correction
for c in range(C):
    gen_mean = generated[:, :, c].mean()
    gen_std = generated[:, :, c].std() + 1e-8
    orig_mean = original[:, :, c].mean()
    orig_std = original[:, :, c].std()
    generated[:, :, c] = (generated[:, :, c] - gen_mean) / gen_std * orig_std + orig_mean
```

### Stat-match rescale (use this for Phase 4.5 gate)

This is a SAFE rescale that aligns the aggregate mean/std to a target
(usually `meta.mean`/`meta.std`) while preserving per-sample temporal
structure. It rescales each sample by the SAME affine transform, so
per-sample shapes are unchanged — only the level and spread are shifted.

```python
def stat_match_rescale(generated, target_mean, target_std, value_range=(None, None),
                       max_iters=3, tol_mean=0.05, tol_std=0.08):
    """
    Align generated.mean()/std() to (target_mean, target_std) via iterative
    affine transforms. Preserves relative shape of every sample. Iteration
    handles the case where clipping to `value_range` after the first affine
    pushes the aggregate stats off-target (common for one-sided, non-negative
    data where large upscales get clipped at zero).

    Args:
        generated: np.ndarray (N, L, C)
        target_mean: float - desired aggregate mean (e.g. meta.mean)
        target_std:  float - desired aggregate std  (e.g. meta.std)
        value_range: (low, high) tuple for final clipping; None = no clip
        max_iters: int - max affine passes (default 3, usually converges in 1-2)
        tol_mean: float - stop when |mean_gap| <= tol_mean
        tol_std:  float - stop when |std_gap|  <= tol_std

    Returns:
        np.ndarray (N, L, C) float32
    """
    import numpy as np
    g = generated.astype(np.float32).copy()
    lo, hi = value_range if value_range is not None else (None, None)

    start_mean = float(g.mean())
    start_std  = float(g.std())
    print(f"stat_match_rescale start: mean={start_mean:.4f} std={start_std:.4f} (target mean={target_mean:.4f} std={target_std:.4f})")

    for it in range(max_iters):
        cur_mean = float(g.mean())
        cur_std  = float(g.std()) + 1e-8
        mean_gap = abs(cur_mean - target_mean) / (abs(target_mean) + 1e-8)
        std_gap  = abs(cur_std  - target_std)  / (abs(target_std)  + 1e-8)
        if mean_gap <= tol_mean and std_gap <= tol_std:
            print(f"stat_match_rescale iter {it}: converged (mean_gap={mean_gap:.1%}, std_gap={std_gap:.1%})")
            break
        scale = float(target_std) / cur_std
        g = (g - cur_mean) * scale + float(target_mean)
        if lo is not None:
            g = np.maximum(g, lo)
        if hi is not None:
            g = np.minimum(g, hi)
        post_mean = float(g.mean())
        post_std  = float(g.std())
        post_mean_gap = abs(post_mean - target_mean) / (abs(target_mean) + 1e-8)
        post_std_gap  = abs(post_std  - target_std)  / (abs(target_std)  + 1e-8)
        print(f"stat_match_rescale iter {it}: mean {cur_mean:.4f} -> {post_mean:.4f} (gap {post_mean_gap:.1%}), std {cur_std:.4f} -> {post_std:.4f} (gap {post_std_gap:.1%})")

    final_mean = float(g.mean())
    final_std  = float(g.std())
    final_mean_gap = abs(final_mean - target_mean) / (abs(target_mean) + 1e-8)
    final_std_gap  = abs(final_std  - target_std)  / (abs(target_std)  + 1e-8)
    print(f"stat_match_rescale final: mean={final_mean:.4f} (gap {final_mean_gap:.1%}), std={final_std:.4f} (gap {final_std_gap:.1%})")

    # If clipping prevented convergence, warn but don't fail
    if final_mean_gap > 0.15 or final_std_gap > 0.2:
        print(f"WARNING: residual gap after {max_iters} iters -- value_range clipping is the bottleneck. "
              f"Consider re-generating with a strategy that produces samples closer to target scale.")
    return g.astype(np.float32)
```

**When to use**: In Phase 4.5, after `validate_generated()` passes but
before saving. Compute `mean_gap = |gen_mean - target_mean| / |target_mean|`
and `std_gap = |gen_std - target_std| / |target_std|`. If either exceeds
the gate threshold (10% mean / 15% std), call `stat_match_rescale` with
the target coming from `meta.mean`/`meta.std` (preferred) or the sample
stats (fallback).

**What this rescale does NOT fix**: it won't repair a collapsed range
(low `range_ratio` in the quantile/range checks), low diversity, broken
autocorrelation, or a wrong sparsity pattern. Those need a re-generation.
The stat-match gate is a narrow safety net for the specific case where
the body of the distribution is shifted but the shape is correct.


### Fix low diversity
```python
# Do NOT just add noise — re-examine your generation strategy.
# Common causes: perturbation scales too small, or generating from a
# single prototype instead of using the full sample set.
```

### Fix autocorrelation mismatch
```python
# Low AC1 (too noisy): reduce noise/perturbation scale
# High AC1 (too smooth): increase residual perturbation or add calibrated noise
```
