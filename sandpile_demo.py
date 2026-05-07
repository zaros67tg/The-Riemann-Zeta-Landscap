"""
Abelian Sandpile — Quick Demo Script
Run this to generate a plot quickly without the full simulation time.
Designed for professor presentations and quick demos.

Usage:
  python sandpile_demo.py
  python sandpile_demo.py --drops 100000 --grid 300
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from sandpile_cpu import AbelianSandpile, PowerLawFitter, SAND_CMAP


def demo_fractal_patterns(N: int = 100, n_drops: int = 20000):
    """
    Demonstrate fractal patterns at criticality.
    Drops all grains in center — produces the classic 'cathedral' pattern.
    """
    print(f"\n  Generating fractal pattern ({N}×{N}, {n_drops:,} drops)...")
    sp = AbelianSandpile(N=N, boundary="sink")

    snapshots = []
    checkpoints = [500, 2000, 5000, n_drops]

    for i in range(n_drops):
        sp.drop()  # always center
        if (i+1) in checkpoints:
            snapshots.append((i+1, sp.grid.copy()))
            print(f"    checkpoint: {i+1:,} drops")

    fig, axes = plt.subplots(1, len(snapshots), figsize=(14, 4),
                             facecolor='#0d0d14')
    fig.suptitle('Fractal Emergence — Abelian Sandpile (center drops)',
                 color='#e0e0e8', fontsize=13, fontfamily='monospace')

    for ax, (n, grid) in zip(axes, snapshots):
        im = ax.imshow(grid, cmap=SAND_CMAP, vmin=0, vmax=3,
                       interpolation='nearest')
        ax.set_title(f'{n:,} grains', color='#a0a0b0',
                     fontfamily='monospace', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#0d0d14')
        for spine in ax.spines.values():
            spine.set_color('#2a2a36')

    plt.tight_layout(pad=0.5)
    plt.savefig('sandpile_fractal.png', dpi=150, bbox_inches='tight',
                facecolor='#0d0d14')
    print("  Saved: sandpile_fractal.png")
    plt.show()
    return sp


def demo_power_law(N: int = 150, n_drops: int = 30000):
    """
    Demonstrate power law distribution with increasing data.
    Shows how the fit improves as more avalanches accumulate.
    """
    print(f"\n  Power law convergence demo ({N}×{N}, {n_drops:,} drops)...")
    sp = AbelianSandpile(N=N, boundary="sink")
    fitter = PowerLawFitter()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor='#0d0d14')
    fig.suptitle('Power Law Convergence — More Data → Better Fit',
                 color='#e0e0e8', fontsize=13, fontfamily='monospace')

    checkpoints = [1000, 3000, 8000, 15000, n_drops, n_drops]
    flat_axes = axes.flat

    for ax, target_drops in zip(flat_axes, checkpoints):
        current = len(sp.avalanche_sizes)
        needed = target_drops - sp.total_grains
        if needed > 0:
            _ = sp.simulate(needed, verbose=False)

        sizes = np.array(sp.avalanche_sizes)
        ax.set_facecolor('#0d0d14')
        for spine in ax.spines.values():
            spine.set_color('#2a2a36')
        ax.tick_params(colors='#a0a0b0', labelsize=8)
        ax.xaxis.label.set_color('#a0a0b0')
        ax.yaxis.label.set_color('#a0a0b0')

        if len(sizes) < 10:
            ax.text(0.5, 0.5, 'not enough data', ha='center', va='center',
                    color='#666', transform=ax.transAxes)
            continue

        x_c, pdf = fitter.log_binned_pdf(sizes)
        ax.scatter(x_c, pdf, s=14, color='#1D9E75', alpha=0.8, zorder=3)

        fit = fitter.mle_exponent(sizes)
        alpha = fit.get("alpha", np.nan)
        s_min = fit.get("s_min", 1)

        if not np.isnan(alpha) and s_min > 0:
            norm_mask = x_c >= s_min
            if norm_mask.sum() > 1:
                obs_at_smin = np.interp(s_min, x_c, pdf)
                x_fit = np.logspace(np.log10(s_min), np.log10(x_c.max()), 100)
                scale = obs_at_smin / (s_min ** (-alpha))
                ax.plot(x_fit, scale * x_fit**(-alpha), '--',
                        color='#D85A30', lw=1.5,
                        label=f'α = {alpha:.3f}')
                ax.legend(fontsize=8, framealpha=0.2, facecolor='#0d0d14',
                          labelcolor='#e0e0e8')

        ax.set_xscale('log'); ax.set_yscale('log')
        n_tot = len(sizes)
        ax.set_title(f'n={n_tot:,} avalanches',
                     color='#a0a0b0', fontfamily='monospace', fontsize=9)
        ax.grid(True, alpha=0.12, color='#2a2a36')
        ax.set_xlabel('s', fontfamily='monospace', fontsize=8)
        ax.set_ylabel('P(s)', fontfamily='monospace', fontsize=8)

    plt.tight_layout()
    plt.savefig('sandpile_powerlaw_convergence.png', dpi=150,
                bbox_inches='tight', facecolor='#0d0d14')
    print("  Saved: sandpile_powerlaw_convergence.png")
    plt.show()
    return sp


def demo_animation(N: int = 60, n_frames: int = 300,
                   save_gif: bool = False):
    """
    Create animation of sandpile evolving.
    WARNING: set save_gif=True to export .gif (slow!)
    """
    print(f"\n  Animation demo ({N}×{N})...")
    sp = AbelianSandpile(N=N, boundary="sink")

    # Pre-warm so we see interesting patterns quickly
    sp.simulate(3000, verbose=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    facecolor='#0d0d14')
    fig.suptitle('Abelian Sandpile — Live Evolution',
                 color='#e0e0e8', fontsize=12, fontfamily='monospace')

    im = ax1.imshow(sp.grid, cmap=SAND_CMAP, vmin=0, vmax=3,
                    interpolation='nearest', animated=True)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title('Grid state', color='#a0a0b0',
                  fontfamily='monospace', fontsize=9)
    ax1.set_facecolor('#0d0d14')

    history_sizes = [1]  # start with 1 to avoid log(0)
    line, = ax2.plot([1], [1], color='#378ADD', linewidth=0.8, alpha=0.8)
    ax2.set_yscale('log'); ax2.set_xscale('log')
    ax2.set_xlabel('Avalanche size', color='#a0a0b0',
                   fontfamily='monospace', fontsize=8)
    ax2.set_ylabel('Count', color='#a0a0b0',
                   fontfamily='monospace', fontsize=8)
    ax2.set_title('Running size distribution', color='#a0a0b0',
                  fontfamily='monospace', fontsize=9)
    ax2.set_facecolor('#0d0d14')
    ax2.tick_params(colors='#a0a0b0', labelsize=8)
    ax2.grid(True, alpha=0.12, color='#2a2a36')
    for spine in ax2.spines.values(): spine.set_color('#2a2a36')
    for spine in ax1.spines.values(): spine.set_color('#2a2a36')

    def update(frame):
        for _ in range(5):
            sp.drop()
        im.set_data(sp.grid)
        sizes = np.array(sp.avalanche_sizes) if sp.avalanche_sizes else np.array([1])
        if len(sizes) > 5:
            vals, counts = np.unique(sizes, return_counts=True)
            line.set_data(vals, counts)
            ax2.set_xlim(0.5, vals.max() * 2)
            ax2.set_ylim(0.5, counts.max() * 2)
        return im, line

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=50, blit=True)

    if save_gif:
        print("  Saving animation... (this takes a moment)")
        ani.save('sandpile_animation.gif', writer='pillow', fps=20)
        print("  Saved: sandpile_animation.gif")
    else:
        plt.show()
    return sp


def demo_universality(sizes: int = 20000):
    """
    Demonstrate universality: center vs random drops give same exponent.
    This is the hallmark of SOC — the attractor is independent of driving.
    """
    print(f"\n  Universality demo ({sizes:,} drops each)...")

    results = {}
    for mode in ["center", "random"]:
        sp = AbelianSandpile(N=150, boundary="sink")
        s = sp.simulate(sizes, mode=mode, verbose=False)
        fit = PowerLawFitter.mle_exponent(s)
        results[mode] = {"sizes": s, "fit": fit}
        alpha = fit.get("alpha", np.nan)
        print(f"    mode={mode:>8}  α={alpha:.4f}  n_av={len(s):,}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor='#0d0d14')
    fig.suptitle('Universality — Center vs Random Driving (same α)',
                 color='#e0e0e8', fontsize=12, fontfamily='monospace')
    colors = ['#378ADD', '#1D9E75']

    fitter = PowerLawFitter()
    for ax, (mode, data), col in zip(axes, results.items(), colors):
        s = data["sizes"]
        fit = data["fit"]
        ax.set_facecolor('#0d0d14')
        for spine in ax.spines.values(): spine.set_color('#2a2a36')
        ax.tick_params(colors='#a0a0b0', labelsize=8)

        x_c, pdf = fitter.log_binned_pdf(s)
        ax.scatter(x_c, pdf, s=16, color=col, alpha=0.85, zorder=3)

        alpha = fit.get("alpha", np.nan)
        s_min = fit.get("s_min", 1)
        if not np.isnan(alpha):
            obs = np.interp(s_min, x_c, pdf)
            x_f = np.logspace(np.log10(s_min), np.log10(x_c.max()), 200)
            scale = obs / (s_min**(-alpha))
            ax.plot(x_f, scale * x_f**(-alpha), '--', color='#D85A30',
                    lw=1.5, label=f'α = {alpha:.4f}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'Mode: {mode} | α = {alpha:.4f}',
                     color='#a0a0b0', fontfamily='monospace', fontsize=10)
        ax.legend(fontsize=9, framealpha=0.2, facecolor='#0d0d14',
                  labelcolor='#e0e0e8')
        ax.grid(True, alpha=0.12, color='#2a2a36')
        ax.set_xlabel('Avalanche size s', fontfamily='monospace', fontsize=8)
        ax.set_ylabel('P(s)', fontfamily='monospace', fontsize=8)

    plt.tight_layout()
    plt.savefig('sandpile_universality.png', dpi=150, bbox_inches='tight',
                facecolor='#0d0d14')
    print("  Saved: sandpile_universality.png")
    plt.show()


# ─────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abelian Sandpile Demo")
    parser.add_argument("--drops",  type=int, default=30000,
                        help="Number of grain drops (default: 30000)")
    parser.add_argument("--grid",   type=int, default=150,
                        help="Grid size N (default: 150)")
    parser.add_argument("--demo",   type=str, default="all",
                        choices=["fractal", "powerlaw", "animation",
                                 "universality", "all"],
                        help="Which demo to run")
    parser.add_argument("--gif",    action="store_true",
                        help="Save animation as GIF")
    args = parser.parse_args()

    print("=" * 60)
    print("  ABELIAN SANDPILE — DEMO SUITE")
    print("  Self-Organized Criticality")
    print("=" * 60)

    if args.demo in ("fractal", "all"):
        demo_fractal_patterns(N=min(args.grid, 120),
                              n_drops=min(args.drops, 20000))

    if args.demo in ("powerlaw", "all"):
        demo_power_law(N=args.grid, n_drops=args.drops)

    if args.demo in ("universality", "all"):
        demo_universality(sizes=min(args.drops, 25000))

    if args.demo in ("animation", "all"):
        demo_animation(N=min(args.grid, 80), save_gif=args.gif)

    print("\n  All demos complete!")
