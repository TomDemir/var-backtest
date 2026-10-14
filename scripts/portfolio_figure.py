"""Portfolio figure: SPY left tail, real returns vs the Gaussian assumption.

Every number comes from data/spy.csv and from varbt's own estimators, then is
cross-checked against results/summary.json (written by the CI backtest). If
the check fails the script stops: no figure from numbers the project did not
produce.

Usage:
    python scripts/download_data.py --tickers SPY
    python scripts/portfolio_figure.py            # -> montecarlo-var.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from varbt import estimators as est

BG, INK, AXIS, GRID, ACCENT = "#0d0d0d", "#f1f1f1", "#888888", "#1a1a1a", "#007BFF"
ALPHA, N_SIMS, SEED = 0.99, 10_000, 42


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/spy.csv")
    p.add_argument("--summary", default="results/summary.json")
    p.add_argument("--out", default="montecarlo-var.png")
    p.add_argument("--tolerance", type=float, default=5e-4,
                   help="max |difference| vs summary.json (Yahoo revises prices slightly per download)")
    a = p.parse_args()

    if not Path(a.data).exists():
        sys.exit(f"{a.data} missing: run `python scripts/download_data.py --tickers SPY` first")
    px = pd.read_csv(a.data, index_col="date", parse_dates=True)["close"].sort_index()
    r = np.log(px / px.shift(1)).dropna().to_numpy()

    # Full-sample, in-sample estimates: the same three estimators as the backtest.
    hist = est.historical(r, ALPHA)
    gauss = est.gaussian(r, ALPHA)
    mc = est.monte_carlo(r, ALPHA, np.random.default_rng(SEED), N_SIMS)
    mu, sigma = r.mean(), r.std(ddof=1)
    sims = mu + sigma * np.random.default_rng(SEED).standard_normal(N_SIMS)  # the 10,000 draws behind `mc`

    # Cross-check against the CI run. Refuse to plot numbers the project did not produce.
    ref = json.loads(Path(a.summary).read_text())["assets"]["SPY"]["full_sample"]
    for name, mine, theirs in (("historical", hist.var, ref["hist_var99"]),
                               ("gaussian", gauss.var, ref["gauss_var99"])):
        if abs(mine - theirs) > a.tolerance:
            sys.exit(f"{name} VaR {mine:.5f} differs from summary.json {theirs:.5f}: "
                     "re-run the backtest on this file first")

    gap_bps = (hist.var - gauss.var) * 1e4
    n_below = int((r < -gauss.var).sum())
    n_exp = len(r) * (1 - ALPHA)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })
    fig, ax = plt.subplots(figsize=(10, 5.625))  # 1600 x 900 px at dpi 160

    lo = min(r.min(), -hist.es) - 0.004
    hi = -0.008
    bins = np.linspace(lo, hi, 36)
    xs = np.linspace(lo, hi, 400)

    ax.hist(r[r <= hi], bins=bins, density=False, color=ACCENT, alpha=0.45,
            edgecolor=BG, linewidth=1.5, label=f"Real SPY daily returns ({len(r):,} days)")
    width = bins[1] - bins[0]
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma) * len(r) * width, color=INK, lw=2,
            label="Gaussian fit (same mean and volatility)")
    ax.hist(sims[sims <= hi], bins=bins, weights=np.full((sims <= hi).sum(), len(r) / N_SIMS),
            histtype="step", color=AXIS, lw=1.5, linestyle=":",
            label=f"Monte Carlo, {N_SIMS:,} Gaussian draws (rescaled)")

    ymax = ax.get_ylim()[1]
    lines = (  # (x, colour, style, label, side): historical label left of its line, others right
        (-hist.var, ACCENT, "-", f"Historical  {hist.var:.2%}", "left", 0.82),
        (-gauss.var, INK, "--", f"Gaussian  {gauss.var:.2%}", "right", 0.82),
        (-mc.var, AXIS, ":", f"Monte Carlo  {mc.var:.2%}", "right", 0.74),
    )
    for x, c, ls, txt, side, yf in lines:
        ax.axvline(x, color=c, lw=2, linestyle=ls)
        dx = -0.0006 if side == "left" else 0.0006
        x_txt = max(-gauss.var, -mc.var) if side == "right" else x
        ax.text(x_txt + dx, ymax * yf, txt, color=c, fontsize=10, va="top",
                ha="right" if side == "left" else "left")
    y_arrow = ymax * 0.62
    ax.annotate("", xy=(-hist.var, y_arrow), xytext=(-gauss.var, y_arrow),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.2})
    ax.text((-hist.var - gauss.var) / 2, y_arrow * 1.04, f"{gap_bps:.0f} bps",
            ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")

    ax.text(0.01, 0.97,
            f"99% one-day VaR, in-sample, SPY {pd.Timestamp(px.index[1]).year} to {px.index[-1].year}\n"
            f"Days with a loss beyond the Gaussian VaR: {n_below} (the model expects {n_exp:.1f})",
            transform=ax.transAxes, va="top", color=INK, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("daily log return (left tail only)")
    ax.set_ylabel("number of days")
    ax.set_xlim(lo, hi)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False, fontsize=9.5,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)

    print(f"historical {hist.var:.4%}  gaussian {gauss.var:.4%}  monte carlo {mc.var:.4%}  "
          f"gap {gap_bps:.1f} bps  days beyond gaussian VaR {n_below} vs {n_exp:.1f} expected")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
