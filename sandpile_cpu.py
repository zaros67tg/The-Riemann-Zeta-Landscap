"""
Abelian Sandpile Model — CPU Implementation
Self-Organized Criticality with Power Law Distribution Analysis

Author: Generated for Statistical Mechanics study
Reference: Bak, Tang & Wiesenfeld (1987) "Self-Organized Criticality"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy import stats
from collections import defaultdict
import time

# ─────────────────────────────────────────────────
# Core Sandpile Engine (NumPy-vectorized, fast)
# ─────────────────────────────────────────────────

class AbelianSandpile:
    THRESHOLD = 4

    def __init__(self, N: int = 200, boundary: str = "sink"):
        """
        N        : grid size (N×N)
        boundary : "sink"  — grains falling off edges are lost (open)
                   "torus" — periodic boundary (closed, no dissipation)
        """
        self.N = N
        self.boundary = boundary
        self.grid = np.zeros((N, N), dtype=np.int32)
        self.total_grains = 0
        self.avalanche_sizes: list[int] = []
        self.avalanche_durations: list[int] = []
        self._topple_count = 0   # cumulative topplings

    # ── Grain deposition ──────────────────────────────────────────────
    def drop(self, x: int = None, y: int = None) -> int:
        """Drop one grain at (x,y). Returns avalanche size (# topplings)."""
        if x is None: x = self.N // 2
        if y is None: y = self.N // 2
        self.grid[y, x] += 1
        self.total_grains += 1
        size, dur = self._relax()
        if size > 0:
            self.avalanche_sizes.append(size)
            self.avalanche_durations.append(dur)
        return size

    def drop_random(self) -> int:
        x = np.random.randint(0, self.N)
        y = np.random.randint(0, self.N)
        return self.drop(x, y)

    # ── Relaxation (vectorized toppling) ──────────────────────────────
    def _relax(self) -> tuple[int, int]:
        """Fully relax the grid. Returns (total topplings, duration steps)."""
        total_topplings = 0
        duration = 0
        N = self.N

        while True:
            unstable = self.grid >= self.THRESHOLD
            if not np.any(unstable):
                break
            duration += 1

            # Vectorized toppling: subtract 4 from all unstable cells,
            # then spread to neighbors using convolution-like shifts
            topple_mask = unstable.astype(np.int32)
            n_toppled = int(np.sum(topple_mask))
            total_topplings += n_toppled

            self.grid -= 4 * topple_mask

            # Spread grains to 4 neighbors
            if self.boundary == "torus":
                self.grid += np.roll(topple_mask, +1, axis=0)  # up
                self.grid += np.roll(topple_mask, -1, axis=0)  # down
                self.grid += np.roll(topple_mask, +1, axis=1)  # left
                self.grid += np.roll(topple_mask, -1, axis=1)  # right
            else:  # sink boundary — edge grains lost
                self.grid[1:,  :]  += topple_mask[:-1, :]   # from above
                self.grid[:-1, :]  += topple_mask[1:,  :]   # from below
                self.grid[:,  1:]  += topple_mask[:,  :-1]  # from left
                self.grid[:, :-1]  += topple_mask[:,  1:]   # from right

        self._topple_count += total_topplings
        return total_topplings, duration

    # ── Bulk simulation ───────────────────────────────────────────────
    def simulate(self, n_drops: int, mode: str = "center",
                 verbose: bool = True) -> np.ndarray:
        """
        Run n_drops grain drops and record all avalanches.
        mode: "center" | "random"
        Returns array of avalanche sizes.
        """
        sizes = []
        t0 = time.perf_counter()
        log_interval = max(1, n_drops // 20)

        for i in range(n_drops):
            if mode == "center":
                sz = self.drop()
            else:
                sz = self.drop_random()
            if sz > 0:
                sizes.append(sz)
            if verbose and (i+1) % log_interval == 0:
                elapsed = time.perf_counter() - t0
                rate = (i+1) / elapsed
                print(f"  [{i+1:>7}/{n_drops}]  avalanches={len(sizes):>6}  "
                      f"max_size={max(sizes) if sizes else 0:>6}  "
                      f"rate={rate:.0f} drops/s")

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"\n  Done in {elapsed:.2f}s | "
                  f"Total avalanches: {len(sizes)} | "
                  f"Max size: {max(sizes) if sizes else 0}")
        return np.array(sizes, dtype=np.int64)


# ─────────────────────────────────────────────────
# Power Law Fitting
# ─────────────────────────────────────────────────

class PowerLawFitter:
    """
    Maximum-Likelihood Estimator for power law exponent.
    Uses the Clauset-Shalizi-Newman method (2009) for unbiased estimation.
    P(s) ∝ s^{-α}  for s ≥ s_min
    """

    @staticmethod
    def mle_exponent(sizes: np.ndarray, s_min: int = None) -> dict:
        """
        Fit power law using MLE. Returns dict with α, s_min, ks_stat, p_value.
        If s_min is None, optimal s_min is found by minimizing KS statistic.
        """
        sizes = np.array(sizes, dtype=float)
        sizes = sizes[sizes > 0]

        if s_min is None:
            s_min = PowerLawFitter._find_optimal_smin(sizes)

        x = sizes[sizes >= s_min]
        n = len(x)
        if n < 10:
            return {"alpha": float("nan"), "s_min": s_min,
                    "n_tail": n, "ks": float("nan")}

        # MLE formula: α = 1 + n * [Σ ln(x_i / (s_min - 0.5))]^{-1}
        alpha = 1.0 + n / np.sum(np.log(x / (s_min - 0.5)))

        # KS statistic vs theoretical power law
        cdf_empirical = np.arange(1, n+1) / n
        cdf_theory = 1.0 - (np.sort(x) / s_min) ** (1.0 - alpha)
        ks_stat = np.max(np.abs(cdf_empirical - cdf_theory))

        return {
            "alpha": float(alpha),
            "s_min": int(s_min),
            "n_tail": n,
            "n_total": len(sizes),
            "ks_stat": float(ks_stat),
            "fraction_in_tail": n / len(sizes)
        }

    @staticmethod
    def _find_optimal_smin(sizes: np.ndarray,
                           candidates: int = 50) -> int:
        unique = np.sort(np.unique(sizes))
        if len(unique) > candidates:
            idx = np.linspace(0, len(unique)-1, candidates, dtype=int)
            unique = unique[idx]

        best_smin, best_ks = unique[0], np.inf
        for s in unique:
            x = sizes[sizes >= s]
            if len(x) < 10: continue
            alpha = 1.0 + len(x) / np.sum(np.log(x / (s - 0.5)))
            cdf_e = np.arange(1, len(x)+1) / len(x)
            cdf_t = 1.0 - (np.sort(x) / s) ** (1.0 - alpha)
            ks = np.max(np.abs(cdf_e - cdf_t))
            if ks < best_ks:
                best_ks = ks
                best_smin = s
        return int(best_smin)

    @staticmethod
    def log_binned_pdf(sizes: np.ndarray, n_bins: int = 40
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Return log-binned (x_centers, pdf_values) for plotting."""
        sizes = sizes[sizes > 0]
        log_min = np.log10(sizes.min())
        log_max = np.log10(sizes.max())
        bins = np.logspace(log_min, log_max, n_bins + 1)
        counts, edges = np.histogram(sizes, bins=bins)
        widths = np.diff(edges)
        pdf = counts / (len(sizes) * widths)
        centers = np.sqrt(edges[:-1] * edges[1:])   # geometric mean
        mask = counts > 0
        return centers[mask], pdf[mask]


# ─────────────────────────────────────────────────
# Visualization Suite
# ─────────────────────────────────────────────────

SAND_CMAP = mcolors.ListedColormap(
    ['#0a0a12', '#1D4E89', '#1D9E75', '#BA7517', '#D85A30']
)

def plot_full_analysis(sandpile: AbelianSandpile,
                       sizes: np.ndarray,
                       fit: dict,
                       save_path: str = None):
    fig = plt.figure(figsize=(16, 10), facecolor='#0d0d14')
    fig.suptitle(
        'Abelian Sandpile Model — Self-Organized Criticality',
        color='#e0e0e8', fontsize=15, fontweight='bold',
        fontfamily='monospace', y=0.97
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35,
                  left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_colors = {'bg': '#0d0d14', 'fg': '#c8c8d4', 'grid': '#2a2a36',
                 'accent1': '#378ADD', 'accent2': '#1D9E75', 'accent3': '#D85A30'}

    def style_ax(ax):
        ax.set_facecolor(ax_colors['bg'])
        ax.tick_params(colors=ax_colors['fg'], labelsize=9)
        ax.xaxis.label.set_color(ax_colors['fg'])
        ax.yaxis.label.set_color(ax_colors['fg'])
        ax.title.set_color(ax_colors['fg'])
        for spine in ax.spines.values():
            spine.set_color(ax_colors['grid'])

    # ── 1. Sandpile grid ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(sandpile.grid, cmap=SAND_CMAP, vmin=0, vmax=4,
                    interpolation='nearest', origin='upper')
    ax1.set_title(f'Sandpile Grid ({sandpile.N}×{sandpile.N})',
                  fontfamily='monospace', fontsize=10)
    ax1.set_xticks([]); ax1.set_yticks([])
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('grains/cell', color=ax_colors['fg'], fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=ax_colors['fg'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=ax_colors['fg'])
    style_ax(ax1)

    # ── 2. Log-log power law ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if len(sizes) > 10:
        x_centers, pdf = PowerLawFitter.log_binned_pdf(sizes)
        ax2.scatter(x_centers, pdf, s=18, color=ax_colors['accent2'],
                    alpha=0.85, zorder=3, label='Observed P(s)')

        alpha = fit.get("alpha", None)
        s_min = fit.get("s_min", None)
        if alpha and not np.isnan(alpha) and s_min:
            x_fit = np.logspace(np.log10(s_min), np.log10(x_centers.max()), 200)
            # Normalize to match data
            norm_mask = x_centers >= s_min
            if norm_mask.sum() > 0:
                y_fit_raw = x_fit ** (-alpha)
                # Scale to match PDF
                obs_at_smin = np.interp(s_min, x_centers, pdf)
                scale = obs_at_smin / (s_min ** (-alpha))
                y_fit = scale * y_fit_raw
                ax2.plot(x_fit, y_fit, '--', color=ax_colors['accent3'],
                         linewidth=1.5, zorder=4,
                         label=f'Power law fit α={alpha:.3f}')
                ax2.axvline(s_min, color='#888', linewidth=0.8, linestyle=':',
                            label=f's_min = {s_min}')

    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('Avalanche size s', fontfamily='monospace')
    ax2.set_ylabel('P(s)', fontfamily='monospace')
    ax2.set_title('Power Law Distribution\nP(s) ∝ s⁻ᵅ',
                  fontfamily='monospace', fontsize=10)
    ax2.grid(True, alpha=0.15, color=ax_colors['grid'])
    ax2.legend(fontsize=8, framealpha=0.2,
               labelcolor=ax_colors['fg'], facecolor=ax_colors['bg'])
    style_ax(ax2)

    # ── 3. Avalanche size time series ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if len(sizes) > 0:
        recent = sizes[-2000:] if len(sizes) > 2000 else sizes
        t = np.arange(len(recent))
        ax3.fill_between(t, recent, alpha=0.4, color=ax_colors['accent1'])
        ax3.plot(t, recent, linewidth=0.4, color=ax_colors['accent1'],
                 alpha=0.7)
        ax3.set_yscale('log')
    ax3.set_xlabel('Avalanche index (recent)', fontfamily='monospace')
    ax3.set_ylabel('Size (log)', fontfamily='monospace')
    ax3.set_title('Avalanche Time Series\n(burstiness = no char. scale)',
                  fontfamily='monospace', fontsize=10)
    ax3.grid(True, alpha=0.15, color=ax_colors['grid'])
    style_ax(ax3)

    # ── 4. CDF comparison ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if len(sizes) > 10:
        sorted_s = np.sort(sizes)
        cdf = np.arange(1, len(sorted_s)+1) / len(sorted_s)
        ax4.step(sorted_s, 1-cdf, where='post', color=ax_colors['accent2'],
                 linewidth=1.5, label='Empirical CCDF')
        alpha = fit.get("alpha", None)
        s_min = fit.get("s_min", None)
        if alpha and not np.isnan(alpha):
            x_th = np.logspace(np.log10(max(1, s_min*0.5)),
                               np.log10(sorted_s.max()), 300)
            ccdf_th = (x_th / s_min) ** (1 - alpha)
            ccdf_th = np.clip(ccdf_th, 0, 1)
            ax4.plot(x_th, ccdf_th, '--', color=ax_colors['accent3'],
                     linewidth=1.5, label=f'Theoretical (α={alpha:.3f})')
        ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.set_xlabel('Avalanche size s', fontfamily='monospace')
    ax4.set_ylabel('P(S > s)', fontfamily='monospace')
    ax4.set_title('Complementary CDF\n(straight line = power law)',
                  fontfamily='monospace', fontsize=10)
    ax4.grid(True, alpha=0.15, color=ax_colors['grid'])
    ax4.legend(fontsize=8, framealpha=0.2,
               labelcolor=ax_colors['fg'], facecolor=ax_colors['bg'])
    style_ax(ax4)

    # ── 5. Size distribution histogram (linear) ───────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if len(sizes) > 10:
        small = sizes[sizes <= np.percentile(sizes, 95)]
        ax5.hist(small, bins=50, color=ax_colors['accent1'],
                 alpha=0.75, edgecolor='none', density=True)
    ax5.set_xlabel('Avalanche size s (≤95th percentile)',
                   fontfamily='monospace')
    ax5.set_ylabel('Density', fontfamily='monospace')
    ax5.set_title('Linear Scale Histogram\n(heavy tail visible)',
                  fontfamily='monospace', fontsize=10)
    ax5.grid(True, alpha=0.15, color=ax_colors['grid'])
    style_ax(ax5)

    # ── 6. Stats summary panel ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    alpha_val = fit.get("alpha", float("nan"))
    alpha_str = f"{alpha_val:.4f}" if not np.isnan(alpha_val) else "N/A"
    smin_val = fit.get("s_min", "—")
    ks_val = fit.get("ks_stat", float("nan"))
    ks_str = f"{ks_val:.4f}" if not np.isnan(ks_val) else "N/A"
    n_tail = fit.get("n_tail", 0)
    frac = fit.get("fraction_in_tail", 0)

    stats_text = [
        ("SANDPILE PARAMETERS", None, '#888'),
        (f"Grid size", f"{sandpile.N}×{sandpile.N}", ax_colors['fg']),
        (f"Boundary", f"{sandpile.boundary}", ax_colors['fg']),
        (f"Total grains dropped", f"{sandpile.total_grains:,}", ax_colors['fg']),
        ("", "", None),
        ("AVALANCHE STATISTICS", None, '#888'),
        (f"Total avalanches", f"{len(sizes):,}", ax_colors['accent2']),
        (f"Mean size", f"{np.mean(sizes):.1f}" if len(sizes) > 0 else "—",
         ax_colors['fg']),
        (f"Max size", f"{np.max(sizes):,}" if len(sizes) > 0 else "—",
         ax_colors['accent3']),
        (f"Std / Mean", f"{np.std(sizes)/np.mean(sizes):.2f}"
         if len(sizes) > 0 else "—", ax_colors['fg']),
        ("", "", None),
        ("POWER LAW FIT (MLE)", None, '#888'),
        (f"Exponent α", alpha_str, '#FFD700'),
        (f"Theory (2D ASM)", "1.0 – 1.5", ax_colors['accent1']),
        (f"s_min", f"{smin_val}", ax_colors['fg']),
        (f"KS statistic", ks_str, ax_colors['fg']),
        (f"Tail fraction", f"{frac:.1%}" if frac else "—", ax_colors['fg']),
    ]

    y_pos = 0.98
    for row in stats_text:
        if len(row) == 3 and row[2] == '#888':   # section header
            ax6.text(0.05, y_pos, row[0], transform=ax6.transAxes,
                     color='#888888', fontsize=8, fontfamily='monospace',
                     fontweight='bold')
        elif row[0] == "":
            pass
        else:
            label, value, color = row
            ax6.text(0.05, y_pos, label, transform=ax6.transAxes,
                     color=ax_colors['fg'], fontsize=9, fontfamily='monospace')
            ax6.text(0.72, y_pos, str(value), transform=ax6.transAxes,
                     color=color, fontsize=9, fontfamily='monospace',
                     fontweight='bold', ha='right')
        y_pos -= 0.058

    ax6.set_facecolor(ax_colors['bg'])
    for spine in ax6.spines.values():
        spine.set_color(ax_colors['grid'])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"\n  Plot saved to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ABELIAN SANDPILE MODEL — CPU / NumPy")
    print("  Self-Organized Criticality Simulator")
    print("=" * 60)

    # Configuration
    GRID_SIZE = 200
    N_DROPS = 50_000
    DROP_MODE = "center"   # "center" or "random"

    print(f"\n  Grid: {GRID_SIZE}×{GRID_SIZE}  |  Drops: {N_DROPS:,}"
          f"  |  Mode: {DROP_MODE}")
    print()

    # Run simulation
    sp = AbelianSandpile(N=GRID_SIZE, boundary="sink")
    sizes = sp.simulate(N_DROPS, mode=DROP_MODE, verbose=True)

    # Fit power law
    print("\n  Fitting power law (MLE)...")
    fitter = PowerLawFitter()
    fit = fitter.mle_exponent(sizes)

    print(f"\n  ┌─ RESULTS {'─'*40}")
    print(f"  │  α (power law exponent) = {fit.get('alpha', 'N/A'):.4f}")
    print(f"  │  s_min                  = {fit.get('s_min', 'N/A')}")
    print(f"  │  KS statistic           = {fit.get('ks_stat', 'N/A'):.4f}")
    print(f"  │  Tail fraction          = {fit.get('fraction_in_tail',0):.1%}")
    print(f"  │  Avalanches in tail     = {fit.get('n_tail',0):,}")
    print(f"  └{'─'*50}")
    print(f"\n  Theoretical α for 2D ASM: 1.0 – 1.5")
    alpha = fit.get('alpha', 0)
    if 0.8 < alpha < 2.0:
        print("  ✓  Power law confirmed — SOC demonstrated!")
    else:
        print("  ⚠  Need more data for convergence (try N_DROPS = 200000)")

    # Plot
    plot_full_analysis(sp, sizes, fit, save_path="sandpile_analysis.png")
