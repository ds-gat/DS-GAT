"""Plotting helpers for the DS-GAT analyses (used by analysis_dsgat3.py --aggregate):
confidence distributions (Fig. 3), posterior calibration, and robustness to
confidence noise (Fig. 2)."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
})

DATASET_COLORS = {'ppi5k': '#2166AC', 'cn15k': '#D6604D', 'nl27k': '#4DAC26'}
DATASET_LABELS = {'ppi5k': 'PPI5K',   'cn15k': 'CN15K',   'nl27k': 'NL27K'}
DATASETS       = ['ppi5k', 'cn15k', 'nl27k']
N_BINS         = 20
BUCKET_EDGES   = np.linspace(0, 1, 6)
BUCKET_LABELS  = ['0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']


def _bin_stats(conf_obs, values_per_seed, bin_edges):
    """
    Returns: centers, mean_per_bin, std_per_bin.
    values_per_seed: list of arrays [n_edges], one per seed.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    per_seed = []
    for vals in values_per_seed:
        row = []
        for i in range(len(bin_edges) - 1):
            mask = (conf_obs >= bin_edges[i]) & (conf_obs < bin_edges[i + 1])
            row.append(float(vals[mask].mean()) if mask.sum() > 0 else np.nan)
        per_seed.append(row)
    arr = np.array(per_seed)           # [n_seeds, n_bins]
    return centers, np.nanmean(arr, 0), np.nanstd(arr, 0)


def _agg_robustness(seed_rob_list, noise_levels):
    """Returns mean_arr, std_arr across seeds."""
    mat = np.array([[d[s] for s in noise_levels] for d in seed_rob_list])
    return mat.mean(0), (mat.std(0) if len(mat) > 1 else np.zeros(len(noise_levels)))


def _save(fig, path, paper_dir=None, dpi=200):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1A — Confidence distributions  (data-driven, one seed is enough)
# ═════════════════════════════════════════════════════════════════════════════

def plot_confidence_distributions(conf_dict, output_dir, paper_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle('Confidence Score Distributions Across Datasets', fontsize=13, y=1.02)
    stats_rows = []

    for ax, ds in zip(axes, DATASETS):
        if ds not in conf_dict:
            ax.set_visible(False)
            continue
        c   = conf_dict[ds]
        col = DATASET_COLORS[ds]

        ax.hist(c, bins=40, density=True, alpha=0.35, color=col, edgecolor='none')
        kde_x = np.linspace(0, 1, 300)
        try:
            kde = stats.gaussian_kde(c, bw_method='scott')
            ax.plot(kde_x, kde(kde_x), color=col, linewidth=2)
        except Exception:
            pass
        ax.axvline(np.mean(c),   color=col,   linestyle='--', linewidth=1.2,
                   label=f'mean={np.mean(c):.3f}')
        ax.axvline(np.median(c), color='gray', linestyle=':',  linewidth=1.0,
                   label=f'median={np.median(c):.3f}')

        skew_val = stats.skew(c)
        pct_high = (c > 0.8).mean()
        pct_low  = (c < 0.2).mean()
        std_val  = np.std(c)
        ax.text(0.97, 0.97,
                f'std={std_val:.3f}\nskew={skew_val:.2f}\n>0.8: {pct_high:.1%}\n<0.2: {pct_low:.1%}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='lightgray', alpha=0.8))
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xlabel('Confidence score')
        ax.set_ylabel('Density' if ds == 'ppi5k' else '')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

        q1, q3 = np.percentile(c, [25, 75])
        stats_rows.append({
            'dataset': DATASET_LABELS[ds], 'n_edges': len(c),
            'mean': np.mean(c), 'std': std_val, 'median': np.median(c),
            'iqr': q3 - q1, 'pct_high': pct_high, 'pct_low': pct_low,
            'skewness': skew_val, 'kurtosis': stats.kurtosis(c),
        })

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'confidence_distributions.png'), paper_dir)

    print("\n=== Confidence Distribution Statistics ===")
    print(f"{'Dataset':<10} {'N edges':>8} {'Mean':>7} {'Std':>7} {'Median':>8} "
          f"{'IQR':>7} {'>0.8':>7} {'<0.2':>7} {'Skew':>7} {'Kurt':>7}")
    print("-" * 82)
    for r in stats_rows:
        print(f"{r['dataset']:<10} {r['n_edges']:>8} {r['mean']:>7.3f} "
              f"{r['std']:>7.3f} {r['median']:>8.3f} {r['iqr']:>7.3f} "
              f"{r['pct_high']:>7.3f} {r['pct_low']:>7.3f} "
              f"{r['skewness']:>7.3f} {r['kurtosis']:>7.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1B — Bayesian calibration  (ribbons = mean ± std across seeds)
# ═════════════════════════════════════════════════════════════════════════════

def plot_bayesian_calibration_multi(seed_bayesian, output_dir, paper_dir=None):
    """
    seed_bayesian[ds] = list of (conf_obs, mu_layers, var_layers) per seed.
    One figure per layer: rows=datasets, cols=(obs vs mu ribbon, obs vs sigma2 ribbon, bucket bars).
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets   = [ds for ds in DATASETS if ds in seed_bayesian and seed_bayesian[ds]]
    if not datasets:
        return
    num_layers = len(seed_bayesian[datasets[0]][0][1])
    bin_edges  = np.linspace(0, 1, N_BINS + 1)

    for layer_idx in range(num_layers):
        fig, axes = plt.subplots(len(datasets), 3, figsize=(14, 4.5 * len(datasets)))
        fig.suptitle(
            f'Bayesian Stream Calibration — Layer {layer_idx + 1}  '
            f'(mean ± std, N={len(seed_bayesian[datasets[0]])} seeds)',
            fontsize=14, y=1.01)
        if len(datasets) == 1:
            axes = axes[np.newaxis, :]

        for row, ds in enumerate(datasets):
            seed_data = seed_bayesian[ds]
            conf_obs  = seed_data[0][0]           # same across seeds
            col       = DATASET_COLORS[ds]
            n_seeds   = len(seed_data)

            mu_list  = [sd[1][layer_idx] for sd in seed_data]   # list of [n_edges]
            var_list = [sd[2][layer_idx] for sd in seed_data]

            mu_mean  = np.stack(mu_list,  0).mean(0)
            var_mean = np.stack(var_list, 0).mean(0)

            # ── Col 0: obs vs mu (binned ribbon) ──────────────────────
            ax = axes[row, 0]
            centers, mu_b_m, mu_b_s = _bin_stats(conf_obs, mu_list, bin_edges)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='$y=x$')
            ax.plot(centers, mu_b_m, '-o', color=col, linewidth=2, markersize=4,
                    label=f'mean (N={n_seeds})')
            ax.fill_between(centers, mu_b_m - mu_b_s, mu_b_m + mu_b_s,
                            alpha=0.2, color=col, label='± std')
            corr_mu = np.corrcoef(conf_obs, mu_mean)[0, 1]
            ax.set_xlabel('Observed $c_{ij}$')
            ax.set_ylabel(r'Learned $\mu_{ij}$')
            ax.set_title(f'{DATASET_LABELS[ds]}\nObs vs $\\mu$  ($\\rho$={corr_mu:.3f})')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(fontsize=8)

            # ── Col 1: obs vs sigma^2 (binned ribbon) ─────────────────
            ax = axes[row, 1]
            centers, var_b_m, var_b_s = _bin_stats(conf_obs, var_list, bin_edges)
            ax.plot(centers, var_b_m, '-o', color=col, linewidth=2, markersize=4,
                    label=f'mean (N={n_seeds})')
            ax.fill_between(centers,
                            np.maximum(var_b_m - var_b_s, 0),
                            var_b_m + var_b_s,
                            alpha=0.2, color=col, label='± std')
            corr_var = np.corrcoef(conf_obs, var_mean)[0, 1]
            verdict  = '✓ higher unc. for low-conf' if corr_var < -0.05 else '△ weak polarity'
            ax.set_xlabel('Observed $c_{ij}$')
            ax.set_ylabel(r'Learned $\sigma^2_{ij}$')
            ax.set_title(f'Conf vs $\\sigma^2$  ($\\rho$={corr_var:.3f})\n{verdict}')
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8)

            # ── Col 2: mean sigma^2 per bucket with error bars ─────────
            ax = axes[row, 2]
            per_seed_buckets = []
            for vals in var_list:
                row_b = []
                for i in range(len(BUCKET_EDGES) - 1):
                    mask = (conf_obs >= BUCKET_EDGES[i]) & (conf_obs < BUCKET_EDGES[i + 1])
                    row_b.append(float(vals[mask].mean()) if mask.sum() > 0 else np.nan)
                per_seed_buckets.append(row_b)
            psb   = np.array(per_seed_buckets)          # [n_seeds, 5]
            bmean = np.nanmean(psb, 0)
            bstd  = np.nanstd(psb,  0) if n_seeds > 1 else np.zeros(5)
            x     = np.arange(len(BUCKET_LABELS))
            bars  = ax.bar(x, bmean, color=col, alpha=0.75, edgecolor='white')
            ax.errorbar(x, bmean, yerr=bstd, fmt='none', color='black',
                        capsize=4, linewidth=1.2)
            counts = [(conf_obs >= BUCKET_EDGES[i]) & (conf_obs < BUCKET_EDGES[i+1])
                      for i in range(5)]
            for bar, cnt_mask, bm in zip(bars, counts, bmean):
                ax.text(bar.get_x() + bar.get_width() / 2, bm * 1.05,
                        f'n={cnt_mask.sum()}', ha='center', va='bottom', fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(BUCKET_LABELS, rotation=30, fontsize=8)
            ax.set_xlabel('Confidence bucket')
            ax.set_ylabel(r'Mean $\sigma^2$ ± std')
            ax.set_title('Uncertainty by Confidence Bucket')

        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'bayesian_calibration_layer{layer_idx+1}.png'),
              paper_dir)

        print(f"\nLayer {layer_idx+1} Bayesian Summary (mean across {len(seed_bayesian[datasets[0]])} seeds):")
        for ds in datasets:
            sd_list  = seed_bayesian[ds]
            conf_obs = sd_list[0][0]
            mu_mean  = np.stack([sd[1][layer_idx] for sd in sd_list]).mean(0)
            var_mean = np.stack([sd[2][layer_idx] for sd in sd_list]).mean(0)
            print(f"  {DATASET_LABELS[ds]:6s} (N={len(sd_list)})  "
                  f"corr(obs,mu)={np.corrcoef(conf_obs, mu_mean)[0,1]:.4f}  "
                  f"corr(obs,sigma2)={np.corrcoef(conf_obs, var_mean)[0,1]:.4f}  "
                  f"mu=[{mu_mean.min():.3f},{mu_mean.max():.3f}]  "
                  f"sigma2_mean={var_mean.mean():.5f}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Dual-stream heatmaps  (mean ± std across seeds)
# ═════════════════════════════════════════════════════════════════════════════

def plot_stream_heatmaps_multi(seed_stream, output_dir, paper_dir=None):
    """
    seed_stream[ds] = list of stream_data (list of layer dicts) per seed.
    Heatmap color = mean corr; cell text = mean ± std.
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = [ds for ds in DATASETS if ds in seed_stream and seed_stream[ds]]
    if not datasets:
        print("[Analysis 2] No stream data available.")
        return
    num_layers = len(seed_stream[datasets[0]][0])

    for layer_idx in range(num_layers):
        n_all = max(len(seed_stream[ds]) for ds in datasets)
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
        if len(datasets) == 1:
            axes = [axes]
        fig.suptitle(
            f'Stream A vs Stream B Correlation — Layer {layer_idx + 1}  '
            f'(N={n_all} seeds)',
            fontsize=13, y=1.02)

        for ax, ds in zip(axes, datasets):
            C_list     = []
            corr_vals  = []
            for sd_list in seed_stream[ds]:
                layer = sd_list[layer_idx]
                alpha = layer.get('alpha')
                lam_w = layer.get('lam_w')
                if alpha is None or lam_w is None:
                    continue
                n   = min(len(alpha), len(lam_w))
                mat = np.stack([alpha[:n], lam_w[:n]], axis=0)
                C   = np.corrcoef(mat)
                C_list.append(C)
                corr_vals.append(C[0, 1])

            if not C_list:
                ax.set_visible(False)
                continue

            C_mean = np.mean(C_list, 0)
            C_std  = np.std(C_list,  0) if len(C_list) > 1 else np.zeros_like(C_mean)
            labels = ['Stream A\n$\\alpha_{ij}$',
                      'Stream B\n$\\lambda_r\\tilde{w}_{ij}$']

            im = ax.imshow(C_mean, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
            ax.set_xticks(range(2)); ax.set_yticks(range(2))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)

            cm = float(np.mean(corr_vals))
            cs = float(np.std(corr_vals)) if len(corr_vals) > 1 else 0.0
            ax.set_title(f'{DATASET_LABELS[ds]}\ncorr(A,B)={cm:.3f}±{cs:.3f}  (N={len(corr_vals)})')

            for i in range(2):
                for j in range(2):
                    cell = f'{C_mean[i,j]:.3f}\n±{C_std[i,j]:.3f}'
                    ax.text(j, i, cell, ha='center', va='center', fontsize=9,
                            color='white' if abs(C_mean[i, j]) > 0.5 else 'black',
                            fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            print(f"Layer {layer_idx+1} {DATASET_LABELS[ds]:6s}  "
                  f"corr(A,B) = {cm:.4f} ± {cs:.4f}  (N={len(corr_vals)})")

        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'stream_heatmap_layer{layer_idx+1}.png'),
              paper_dir)


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Robustness  (mean ± std ribbons across seeds)
# ═════════════════════════════════════════════════════════════════════════════

def plot_robustness_multi(seed_rob_dsgat, seed_rob_a4, output_dir, paper_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    datasets = [ds for ds in DATASETS
                if ds in seed_rob_dsgat and seed_rob_dsgat[ds]]
    if not datasets:
        return

    noise_levels = sorted(next(iter(seed_rob_dsgat.values()))[0].keys())
    fig, axes    = plt.subplots(1, len(datasets),
                                figsize=(5 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    n_dsgat = max(len(seed_rob_dsgat[ds]) for ds in datasets)
    n_a4    = max((len(seed_rob_a4.get(ds, [])) for ds in datasets), default=0)
    subtitle = f'N={n_dsgat} seeds DS-GAT'
    if n_a4:
        subtitle += f', N={n_a4} seeds w/o Bayesian'
    fig.suptitle(
        f'Robustness to Confidence Noise: DS-GAT vs. w/o Bayesian Stream\n({subtitle})',
        fontsize=13)

    for ax, ds in zip(axes, datasets):
        color = DATASET_COLORS[ds]

        mean_f, std_f = _agg_robustness(seed_rob_dsgat[ds], noise_levels)
        ax.plot(noise_levels, mean_f, 'o-', color=color,
                linewidth=2, markersize=6, label='DS-GAT')
        ax.fill_between(noise_levels, mean_f - std_f, mean_f + std_f,
                        alpha=0.15, color=color)

        if ds in seed_rob_a4 and seed_rob_a4[ds]:
            mean_a, std_a = _agg_robustness(seed_rob_a4[ds], noise_levels)
            ax.plot(noise_levels, mean_a, 's--', color=color,
                    linewidth=2, markersize=6, alpha=0.6,
                    label='w/o Uncert. Modeling')
            ax.fill_between(noise_levels, mean_a - std_a, mean_a + std_a,
                            alpha=0.10, color=color)

        ax.set_xlabel('Gaussian noise $\\sigma$')
        ax.set_ylabel('MRR' if ds == datasets[0] else '')
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'robustness_combined.png'), paper_dir)

    # Console: mean ± std per noise level
    print("\n=== Absolute MRR vs Noise (mean ± std across seeds) ===")
    hdr = f"{'Dataset':<8} {'Model':<22}" + \
          "".join(f"  σ={s:.2f}" for s in noise_levels)
    print(hdr); print("-" * len(hdr))
    for ds in datasets:
        mean_f, std_f = _agg_robustness(seed_rob_dsgat[ds], noise_levels)
        row = f"{DATASET_LABELS[ds]:<8} {'DS-GAT':<22}"
        for m, s in zip(mean_f, std_f):
            row += f"  {m:.4f}±{s:.4f}"
        print(row)
        if ds in seed_rob_a4 and seed_rob_a4[ds]:
            mean_a, std_a = _agg_robustness(seed_rob_a4[ds], noise_levels)
            row = f"{DATASET_LABELS[ds]:<8} {'w/o Uncert.':<22}"
            for m, s in zip(mean_a, std_a):
                row += f"  {m:.4f}±{s:.4f}"
            print(row)
        print()

    # Console: relative drop at sigma=0.3
    if 0.3 in noise_levels:
        sig_idx = noise_levels.index(0.3)
        print("=== Relative MRR drop (%) at σ=0.30 ===")
        for ds in datasets:
            mean_f, std_f = _agg_robustness(seed_rob_dsgat[ds], noise_levels)
            base  = mean_f[0]
            drop  = (base - mean_f[sig_idx]) / base * 100
            drop_s = std_f[sig_idx] / base * 100
            print(f"  DS-GAT  {DATASET_LABELS[ds]:6s}  {drop:.1f}% ± {drop_s:.1f}%")
            if ds in seed_rob_a4 and seed_rob_a4[ds]:
                mean_a, std_a = _agg_robustness(seed_rob_a4[ds], noise_levels)
                base_a  = mean_a[0]
                drop_a  = (base_a - mean_a[sig_idx]) / base_a * 100
                drop_as = std_a[sig_idx] / base_a * 100
                print(f"  w/o Unc {DATASET_LABELS[ds]:6s}  {drop_a:.1f}% ± {drop_as:.1f}%")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
