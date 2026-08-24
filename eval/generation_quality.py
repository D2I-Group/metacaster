
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf


def extract_features(windows: np.ndarray, max_samples: int = 5000) -> np.ndarray:
    N = len(windows)
    if max_samples < N:
        idx = np.random.choice(N, size=max_samples, replace=False)
        windows = windows[idx]
    return windows.reshape(len(windows), -1)


def compute_mmd(
    X: np.ndarray, Y: np.ndarray, bandwidths: list[float] | None = None
) -> float:
    if bandwidths is None:

        combined = np.vstack([X[:500], Y[:500]])
        dists = pairwise_distances(combined, metric="sqeuclidean")
        median_dist = np.median(dists[dists > 0])
        bandwidths = [median_dist * s for s in [0.5, 1.0, 2.0]]

    mmd_total = 0.0
    for bw in bandwidths:
        gamma = 1.0 / (2.0 * bw)
        K_XX = np.exp(-gamma * pairwise_distances(X, X, metric="sqeuclidean"))
        K_YY = np.exp(-gamma * pairwise_distances(Y, Y, metric="sqeuclidean"))
        K_XY = np.exp(-gamma * pairwise_distances(X, Y, metric="sqeuclidean"))
        mmd_total += K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return float(mmd_total / len(bandwidths))


def compute_kl_divergence(X: np.ndarray, Y: np.ndarray, bins: int = 100) -> float:
    n_features = X.shape[1]

    feat_idx = np.random.choice(n_features, size=min(50, n_features), replace=False)
    kl_values = []
    for f in feat_idx:
        combined = np.concatenate([X[:, f], Y[:, f]])
        bin_edges = np.histogram_bin_edges(combined, bins=bins)
        p, _ = np.histogram(X[:, f], bins=bin_edges, density=True)
        q, _ = np.histogram(Y[:, f], bins=bin_edges, density=True)

        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        kl = float(np.sum(p * np.log(p / q)))
        kl_values.append(kl)
    return float(np.mean(kl_values))


def compute_wasserstein(X: np.ndarray, Y: np.ndarray) -> float:
    n_features = X.shape[1]
    feat_idx = np.random.choice(n_features, size=min(50, n_features), replace=False)
    w_values = []
    for f in feat_idx:
        w = stats.wasserstein_distance(X[:, f], Y[:, f])
        w_values.append(w)
    return float(np.mean(w_values))


def compute_coverage(real_feat: np.ndarray, gen_feat: np.ndarray, k: int = 5) -> float:

    real_real_dist = pairwise_distances(real_feat)
    np.fill_diagonal(real_real_dist, np.inf)
    real_knn_dist = np.sort(real_real_dist, axis=1)[:, k - 1]


    real_gen_dist = pairwise_distances(real_feat, gen_feat)
    real_nn_gen_dist = real_gen_dist.min(axis=1)


    covered = (real_nn_gen_dist <= real_knn_dist).mean()
    return float(covered)


def compute_density(real_feat: np.ndarray, gen_feat: np.ndarray, k: int = 5) -> float:
    real_real_dist = pairwise_distances(real_feat)
    np.fill_diagonal(real_real_dist, np.inf)
    real_knn_dist = np.sort(real_real_dist, axis=1)[:, k - 1]

    real_gen_dist = pairwise_distances(real_feat, gen_feat)

    within = (real_gen_dist <= real_knn_dist[:, None]).sum(axis=1)
    return float(within.mean() / k)


def compute_c2st(real_feat: np.ndarray, gen_feat: np.ndarray) -> float:
    n_real, n_gen = len(real_feat), len(gen_feat)
    X = np.vstack([real_feat, gen_feat])
    y = np.concatenate([np.ones(n_real), np.zeros(n_gen)])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")
    return float(scores.mean())


def compute_stat_fidelity(real: np.ndarray, gen: np.ndarray) -> dict:

    real_flat = real.reshape(len(real), -1)
    gen_flat = gen.reshape(len(gen), -1)


    real_mean, gen_mean = real_flat.mean(), gen_flat.mean()
    real_std, gen_std = real_flat.std(), gen_flat.std()

    mean_err = abs(gen_mean - real_mean) / (abs(real_mean) + 1e-8)
    std_err = abs(gen_std - real_std) / (abs(real_std) + 1e-8)


    ks_stat, _ = stats.ks_2samp(real_flat.ravel()[:100000], gen_flat.ravel()[:100000])


    n_lags = 20
    C = real.shape[2]
    acf_diffs = []
    real_acfs = []
    gen_acfs = []
    for c in range(C):

        real_acf_samples = []
        for i in range(min(200, len(real))):
            try:
                a = acf(real[i, :, c], nlags=n_lags, fft=True)
                real_acf_samples.append(a)
            except Exception:
                pass
        gen_acf_samples = []
        for i in range(min(200, len(gen))):
            try:
                a = acf(gen[i, :, c], nlags=n_lags, fft=True)
                gen_acf_samples.append(a)
            except Exception:
                pass
        if real_acf_samples and gen_acf_samples:
            real_mean_acf = np.mean(real_acf_samples, axis=0)
            gen_mean_acf = np.mean(gen_acf_samples, axis=0)
            acf_diffs.append(np.mean(np.abs(real_mean_acf - gen_mean_acf)))
            real_acfs.append(real_mean_acf)
            gen_acfs.append(gen_mean_acf)

    acf_mae = float(np.mean(acf_diffs)) if acf_diffs else float("nan")


    spec_corrs = []
    for c in range(C):
        real_fft = np.abs(np.fft.rfft(real[: min(200, len(real)), :, c], axis=1)).mean(
            axis=0
        )
        gen_fft = np.abs(np.fft.rfft(gen[: min(200, len(gen)), :, c], axis=1)).mean(
            axis=0
        )
        if len(real_fft) > 1:
            corr = np.corrcoef(real_fft, gen_fft)[0, 1]
            spec_corrs.append(corr)
    spec_corr = float(np.mean(spec_corrs)) if spec_corrs else float("nan")


    real_per_sample_mean = real_flat.mean(axis=1)
    gen_per_sample_mean = gen_flat.mean(axis=1)
    real_diversity = real_per_sample_mean.std()
    gen_diversity = gen_per_sample_mean.std()
    diversity_ratio = gen_diversity / (real_diversity + 1e-8)

    return {
        "mean_real": round(float(real_mean), 4),
        "mean_gen": round(float(gen_mean), 4),
        "mean_rel_err": round(float(mean_err), 4),
        "std_real": round(float(real_std), 4),
        "std_gen": round(float(gen_std), 4),
        "std_rel_err": round(float(std_err), 4),
        "ks_stat": round(float(ks_stat), 4),
        "acf_mae": round(acf_mae, 4),
        "spectral_corr": round(spec_corr, 4),
        "diversity_ratio": round(float(diversity_ratio), 4),
        "_real_acfs": real_acfs,
        "_gen_acfs": gen_acfs,
    }


def plot_tsne(
    full: np.ndarray,
    gen: np.ndarray,
    sample: np.ndarray | None,
    output_path: Path,
    max_per_set: int = 1000,
    perplexity: int = 30,
    dataset_name: str = "",
) -> None:
    rng = np.random.default_rng(42)

    def subsample(arr, n):
        if len(arr) > n:
            idx = rng.choice(len(arr), size=n, replace=False)
            return arr[idx]
        return arr

    full_sub = subsample(full, max_per_set).reshape(min(len(full), max_per_set), -1)
    gen_sub = subsample(gen, max_per_set).reshape(min(len(gen), max_per_set), -1)

    arrays = [full_sub, gen_sub]
    labels = ["Full train", "Generated"]
    colors = ["#4477AA", "#EE7733"]
    sizes = [30, 40]
    markers = ["o", "o"]
    alphas = [0.7, 0.7]

    if sample is not None:
        sample_flat = sample.reshape(len(sample), -1)
        arrays.append(sample_flat)
        labels.append("Sample (few-shot)")
        colors.append("#CC3333")
        sizes.append(120)
        markers.append("x")
        alphas.append(1.0)

    combined = np.vstack(arrays)

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(combined) - 1),
        random_state=42,
        max_iter=1000,
    )
    embedded = tsne.fit_transform(combined_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 0
    for arr, label, color, size, marker, alpha in zip(
        arrays, labels, colors, sizes, markers, alphas, strict=False
    ):
        n = len(arr)
        if marker == "x":
            ax.scatter(
                embedded[offset : offset + n, 0],
                embedded[offset : offset + n, 1],
                c=color,
                s=size,
                alpha=alpha,
                label=label,
                marker=marker,
                linewidths=2.5,
            )
        else:
            ax.scatter(
                embedded[offset : offset + n, 0],
                embedded[offset : offset + n, 1],
                c=color,
                s=size,
                alpha=alpha,
                label=label,
                marker=marker,
                edgecolors="none",
            )
        offset += n

    ax.legend(fontsize=12, markerscale=1.2)
    title = (
        f"t-SNE: {dataset_name}" if dataset_name else "t-SNE: Generated vs Full Train"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_histogram(
    real: np.ndarray, gen: np.ndarray, output_path: Path, dataset_name: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    real_vals = real.ravel()
    gen_vals = gen.ravel()


    if len(real_vals) > 500000:
        real_vals = np.random.choice(real_vals, 500000, replace=False)
    if len(gen_vals) > 500000:
        gen_vals = np.random.choice(gen_vals, 500000, replace=False)

    bins = np.histogram_bin_edges(np.concatenate([real_vals, gen_vals]), bins=100)
    ax.hist(
        real_vals, bins=bins, alpha=0.5, density=True, color="#1F77B4", label="Real"
    )
    ax.hist(
        gen_vals, bins=bins, alpha=0.5, density=True, color="#FF8C00", label="Generated"
    )
    ax.legend(fontsize=12)
    title = (
        f"Value Distribution: {dataset_name}"
        if dataset_name
        else "Value Distribution: Real vs Generated"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_acf_comparison(
    real_acfs: list, gen_acfs: list, output_path: Path, dataset_name: str = ""
) -> None:
    n_channels = len(real_acfs)
    fig, axes = plt.subplots(1, n_channels, figsize=(6 * n_channels, 5), squeeze=False)

    for c in range(n_channels):
        ax = axes[0, c]
        lags = np.arange(len(real_acfs[c]))
        ax.plot(lags, real_acfs[c], "b-o", markersize=3, label="Real", alpha=0.8)
        ax.plot(lags, gen_acfs[c], "r-s", markersize=3, label="Generated", alpha=0.8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        if n_channels > 1:
            ch_title = (
                f"{dataset_name} — Channel {c}" if dataset_name else f"Channel {c}"
            )
        else:
            ch_title = f"ACF: {dataset_name}" if dataset_name else "ACF Comparison"
        ax.set_title(ch_title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_run(
    full_path: str,
    gen_path: str,
    sample_path: str | None,
    output_dir: str,
    max_dist_samples: int = 2000,
    dataset_name: str = "",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("  Loading data...")
    real = np.load(full_path)
    gen = np.load(gen_path)
    sample = np.load(sample_path) if sample_path else None

    print(f"    Real: {real.shape}, Generated: {gen.shape}")
    if sample is not None:
        print(f"    Sample: {sample.shape}")


    if real.shape[1:] != gen.shape[1:]:
        print(f"  WARNING: shape mismatch real={real.shape[1:]} gen={gen.shape[1:]}")

        min_l = min(real.shape[1], gen.shape[1])
        min_c = min(real.shape[2], gen.shape[2])
        real = real[:, :min_l, :min_c]
        gen = gen[:, :min_l, :min_c]


    print("  Computing distribution metrics...")
    real_feat = extract_features(real, max_dist_samples)
    gen_feat = extract_features(gen, max_dist_samples)

    mmd = compute_mmd(real_feat[:1000], gen_feat[:1000])
    kl = compute_kl_divergence(real_feat, gen_feat)
    wass = compute_wasserstein(real_feat, gen_feat)
    coverage = compute_coverage(real_feat[:1000], gen_feat[:1000])
    density = compute_density(real_feat[:1000], gen_feat[:1000])
    c2st = compute_c2st(real_feat, gen_feat)

    print(f"    MMD={mmd:.4f} KL={kl:.4f} Wass={wass:.4f}")
    print(f"    Coverage={coverage:.4f} Density={density:.4f} C2ST={c2st:.4f}")


    print("  Computing statistical fidelity...")
    stat = compute_stat_fidelity(real, gen)
    real_acfs = stat.pop("_real_acfs")
    gen_acfs = stat.pop("_gen_acfs")

    print(f"    Mean err={stat['mean_rel_err']:.4f} Std err={stat['std_rel_err']:.4f}")
    print(f"    KS={stat['ks_stat']:.4f} ACF_MAE={stat['acf_mae']:.4f}")
    print(
        f"    Spectral_corr={stat['spectral_corr']:.4f} Diversity={stat['diversity_ratio']:.4f}"
    )


    print("  Generating visualizations...")
    plot_tsne(real, gen, sample, out / "tsne.png", dataset_name=dataset_name)
    plot_histogram(real, gen, out / "histogram.png", dataset_name=dataset_name)
    if real_acfs and gen_acfs:
        plot_acf_comparison(
            real_acfs, gen_acfs, out / "acf.png", dataset_name=dataset_name
        )


    metrics = {
        "dataset": dataset_name,
        "mmd": round(mmd, 6),
        "kl_divergence": round(kl, 6),
        "wasserstein": round(wass, 6),
        "coverage": round(coverage, 4),
        "density": round(density, 4),
        "c2st": round(c2st, 4),
        **stat,
        "n_real": int(real.shape[0]),
        "n_gen": int(gen.shape[0]),
        "window_shape": list(real.shape[1:]),
    }


    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Saved to: {out}/")

    return metrics


def evaluate_manifest(manifest_path: str) -> None:
    manifest = json.loads(Path(manifest_path).read_text())
    all_rows = []

    for ds_key, ds_info in manifest.items():
        data_path = Path(ds_info["data_path"])
        full_path = str(data_path / "full" / "samples.npy")
        sample_path = str(data_path / "sample" / "samples.npy")

        if not Path(full_path).exists():
            print(f"Skipping {ds_key}: full data not found")
            continue

        for run_info in ds_info.get("runs", []):
            run_id = run_info["run_id"]
            round_num = run_info.get("round", "?")


            gen_path = None

            manifest_dir = Path(manifest_path).parent
            candidate = manifest_dir / ds_key / run_id / "dataset.npy"
            if candidate.exists():
                gen_path = str(candidate)
            else:
                print(
                    f"  Skipping {ds_key}/{run_id}: dataset.npy not found at {candidate}"
                )
                continue

            print(f"\n{'=' * 60}")
            print(f"Evaluating: {ds_key} round={round_num} run_id={run_id}")
            print(f"{'=' * 60}")


            eval_out = manifest_dir / ds_key / run_id / "eval"
            try:
                metrics = evaluate_run(
                    full_path=full_path,
                    gen_path=gen_path,
                    sample_path=sample_path,
                    output_dir=str(eval_out),
                    dataset_name=ds_key,
                )
                row = {
                    "dataset": ds_key,
                    "run_id": run_id,
                    "round": round_num,
                    **metrics,
                }
                all_rows.append(row)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue


    if all_rows:
        summary_path = Path(manifest_path).parent / "summary.csv"
        fieldnames = list(all_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSummary CSV: {summary_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


    p.add_argument("--full", default=None, help="Path to full/samples.npy")
    p.add_argument("--generated", default=None, help="Path to generated dataset.npy")
    p.add_argument(
        "--sample", default=None, help="Path to sample/samples.npy (optional)"
    )


    p.add_argument(
        "--manifest", default=None, help="Path to manifest.json for batch eval"
    )


    p.add_argument(
        "--output",
        default="eval_output",
        help="Output directory (single-run mode only)",
    )
    p.add_argument("--dataset_name", default="", help="Dataset name for plot titles")

    args = p.parse_args()

    np.random.seed(42)

    if args.manifest:
        evaluate_manifest(args.manifest)
    elif args.full and args.generated:
        evaluate_run(
            full_path=args.full,
            gen_path=args.generated,
            sample_path=args.sample,
            output_dir=args.output,
            dataset_name=args.dataset_name,
        )
    else:
        p.error(
            "Specify --manifest for batch mode, or --full + --generated for single run"
        )


if __name__ == "__main__":
    main()
