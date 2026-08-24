
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PLOT_DPI = 110
N_OVERLAY_WINDOWS = 3
ACF_MAX_LAG = 200


def _safe_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:

        out = np.zeros(max_lag + 1)
        n = 0
        for c in range(x.shape[2]):
            for b in range(min(x.shape[0], 64)):
                v = x[b, :, c]
                v = v - v.mean()
                d = (v ** 2).sum()
                if d <= 0:
                    continue
                acf = np.correlate(v, v, mode="full")[len(v) - 1:]
                lag = min(max_lag + 1, len(acf))
                out[:lag] += acf[:lag] / d
                n += 1
        return out / max(n, 1)
    if x.ndim == 1:
        v = x - x.mean()
        d = (v ** 2).sum()
        if d <= 0:
            return np.zeros(max_lag + 1)
        acf = np.correlate(v, v, mode="full")[len(v) - 1:]
        lag = min(max_lag + 1, len(acf))
        out = np.zeros(max_lag + 1)
        out[:lag] = acf[:lag] / d
        return out
    raise ValueError(f"unsupported ndim={x.ndim}")


def _safe_fft_power(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:

        flat = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    elif x.ndim == 2:
        flat = x
    else:
        flat = x[None, :]
    flat = flat - flat.mean(axis=1, keepdims=True)
    n_use = min(flat.shape[0], 64)
    flat = flat[:n_use]
    spec = np.abs(np.fft.rfft(flat, axis=1)) ** 2
    return spec.mean(axis=0)


def _per_window_mse(real: np.ndarray, synth: np.ndarray) -> np.ndarray:
    n = min(real.shape[0], synth.shape[0], 200)
    rng = np.random.default_rng(0)
    idx_r = rng.choice(real.shape[0], size=n, replace=False)
    idx_s = rng.choice(synth.shape[0], size=n, replace=False)
    r = real[idx_r].astype(np.float64)
    s = synth[idx_s].astype(np.float64)
    return ((r - s) ** 2).reshape(n, -1).mean(axis=1)


def _plot_4panel(
    real: np.ndarray, synth: np.ndarray, title: str, out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)


    rng = np.random.default_rng(1)
    n_show = min(N_OVERLAY_WINDOWS, real.shape[0], synth.shape[0])
    real_idx = rng.choice(real.shape[0], size=n_show, replace=False)
    synth_idx = rng.choice(synth.shape[0], size=n_show, replace=False)
    ax = axes[0, 0]
    for i, ri in enumerate(real_idx):
        v = real[ri]
        if v.ndim == 2:
            v = v[:, 0]
        ax.plot(v, color="C0", alpha=0.7, lw=0.9, label="real" if i == 0 else None)
    for i, si in enumerate(synth_idx):
        v = synth[si]
        if v.ndim == 2:
            v = v[:, 0]
        ax.plot(v, color="C3", alpha=0.6, lw=0.9, label="synth" if i == 0 else None)
    ax.set_title("time-domain overlay (channel 0)")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(_safe_acf(real, ACF_MAX_LAG), color="C0", lw=1.0, label="real")
    ax.plot(_safe_acf(synth, ACF_MAX_LAG), color="C3", lw=1.0, label="synth")
    ax.set_title("autocorrelation (lag 0..200)")
    ax.legend(fontsize=8)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.axhline(0, color="k", lw=0.4)

    ax = axes[1, 0]
    pr = _safe_fft_power(real)
    ps = _safe_fft_power(synth)
    eps = 1e-12
    ax.plot(10 * np.log10(pr + eps), color="C0", lw=1.0, label="real")
    ax.plot(10 * np.log10(ps + eps), color="C3", lw=1.0, label="synth")
    ax.set_title("log power spectrum")
    ax.legend(fontsize=8)
    ax.set_xlabel("freq bin")
    ax.set_ylabel("dB")

    ax = axes[1, 1]
    try:
        mses = _per_window_mse(real, synth)
        ax.hist(mses, bins=40, color="C2")
        ax.set_title(f"per-window pair MSE (n={len(mses)})")
        ax.set_xlabel("MSE")
        ax.set_ylabel("count")
    except Exception as exc:
        ax.text(0.5, 0.5, f"MSE hist failed: {exc}", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def _plot_cross_round_hinge(
    run_root: Path,
    round_n: int,
    out_path: Path,
    current_per_pair: list[dict] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    rows: list[dict] = []
    for n in range(round_n):
        rd = run_root / "rounds" / f"round_{n}"
        for fname in ("round_eval_result.json", "summary.json"):
            p = rd / fname
            if p.exists():
                try:
                    s = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for r in s.get("per_pair") or []:
                    if r.get("hinge") is None:
                        continue
                    rows.append({
                        "round": n,
                        "dataset": r["dataset"],
                        "model": r["model"],
                        "hinge": float(r["hinge"]),
                        "audit_only": bool(r.get("audit_only", False)),
                    })
                break


    for r in current_per_pair or []:
        if r.get("hinge") is None:
            continue
        rows.append({
            "round": round_n,
            "dataset": r["dataset"],
            "model": r["model"],
            "hinge": float(r["hinge"]),
            "audit_only": bool(r.get("audit_only", False)),
        })

    if not rows:
        return

    by_ds: dict[str, dict[str, dict[int, tuple[float, bool]]]] = {}
    for r in rows:
        d = by_ds.setdefault(r["dataset"], {}).setdefault(r["model"], {})
        d[r["round"]] = (r["hinge"], r["audit_only"])

    n_ds = len(by_ds)
    cols = 2
    rows_g = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows_g, cols, figsize=(12, 3 * rows_g), constrained_layout=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (ds, models) in enumerate(sorted(by_ds.items())):
        ax = axes[i]
        for m_idx, (model, hist) in enumerate(sorted(models.items())):
            xs = sorted(hist.keys())
            ys = [hist[x][0] for x in xs]
            audits = [hist[x][1] for x in xs]
            style = "--" if any(audits) else "-"
            ax.plot(xs, ys, style, marker="o", lw=1.2,
                    color=f"C{m_idx}", label=f"{model}{'*' if any(audits) else ''}")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("round")
        ax.set_ylabel("hinge")
        ax.axhline(0, color="k", lw=0.3)
        ax.legend(fontsize=7)

    for j in range(n_ds, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("hinge trajectory by (dataset, LT-Lib Forecaster) — dashed = audit_only",
                 fontsize=12)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def _plot_dist_metric_trajectory(
    run_root: Path,
    round_n: int,
    out_path: Path,
    current_per_dataset_dist: dict[str, dict] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    METRIC_KEYS = ["mmd", "kl", "wasserstein", "coverage",
                   "density", "c2st", "acf_mae", "spectral_corr"]

    def _aggregate(pdd: dict | None) -> dict[str, float]:
        out: dict[str, float] = {}
        if not pdd:
            return out
        for k in METRIC_KEYS:
            vals = []
            for ds_metrics in pdd.values():
                if not isinstance(ds_metrics, dict):
                    continue
                v = ds_metrics.get(k)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(float(v))
            if vals:
                out[k] = sum(vals) / len(vals)
        return out

    series: dict[str, list[tuple[int, float]]] = {k: [] for k in METRIC_KEYS}

    for n in range(round_n):
        rd = run_root / "rounds" / f"round_{n}"
        for fname in ("round_eval_result.json", "summary.json"):
            p = rd / fname
            if p.exists():
                try:
                    s = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                agg = _aggregate(s.get("per_dataset_dist") or {})
                for k, v in agg.items():
                    series[k].append((n, v))
                break

    cur_agg = _aggregate(current_per_dataset_dist or {})
    for k, v in cur_agg.items():
        series[k].append((round_n, v))

    if not any(series.values()):
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), constrained_layout=True)
    for i, k in enumerate(METRIC_KEYS):
        ax = axes[i // 4, i % 4]
        if series[k]:
            xs, ys = zip(*series[k], strict=True)
            ax.plot(xs, ys, marker="o", lw=1.4, color=f"C{i}")
            ax.set_title(k, fontsize=10)
            ax.set_xlabel("round")
            ax.set_ylabel("mean across ds")
        else:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(k, fontsize=10)

    fig.suptitle("distribution-metric trajectory (mean across train pool)",
                 fontsize=12)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def generate_round_plots(
    *,
    round_dir: Path,
    datasets: list[str],
    train_pool_root: Path,
    gen_results: dict[str, dict],
    per_pair: list[dict],
    per_dataset_dist: dict[str, dict],
    run_root: Path,
    round_n: int,
) -> dict[str, str]:
    round_dir = Path(round_dir)
    out: dict[str, str] = {}


    for ds in datasets:
        gr = gen_results.get(ds, {}) or {}
        if not gr.get("ok"):
            continue
        synth_path = gr.get("dataset_npy")
        real_path = train_pool_root / f"{ds}.npy"
        if not synth_path or not Path(synth_path).exists() or not real_path.exists():
            continue
        try:
            real = np.load(real_path)
            synth = np.load(synth_path)
            out_path = round_dir / ds / "diag_4panel.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _plot_4panel(real, synth, title=f"{ds} — round {round_n}",
                         out_path=out_path)
            out[f"diag_4panel_{ds}"] = str(out_path.relative_to(run_root))
        except Exception:
            continue


    try:
        out_path = round_dir / "cross_round_hinge.png"
        _plot_cross_round_hinge(run_root=run_root, round_n=round_n,
                                out_path=out_path,
                                current_per_pair=per_pair)
        if out_path.exists():
            out["cross_round_hinge"] = str(out_path.relative_to(run_root))
    except Exception:
        pass


    try:
        out_path = round_dir / "dist_metric_trajectory.png"
        _plot_dist_metric_trajectory(run_root=run_root, round_n=round_n,
                                     out_path=out_path,
                                     current_per_dataset_dist=per_dataset_dist)
        if out_path.exists():
            out["dist_metric_trajectory"] = str(out_path.relative_to(run_root))
    except Exception:
        pass

    return out
