"""Figures for the README. Palette: validated categorical + status steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
RETURNS = "#b9b8b2"
SERIES = {"historical": "#2a78d6", "fhs": "#eb6834"}
GOOD, CRITICAL = "#0ca30c", "#d03b3b"


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })


def hero(fc: pd.DataFrame, ticker: str, path: Path) -> None:
    """Unconditional vs conditional 99% VaR, same days, same scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from .backtest import christoffersen_ind

    _style(plt)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, sharey=True)
    titles = {"historical": "Historical simulation (equal weights, 250 days)",
              "fhs": "Filtered historical simulation (EWMA-scaled)"}
    for ax, name in zip(axes, ("historical", "fhs")):
        c = SERIES[name]
        hit = fc[f"{name}_hit"].to_numpy()
        _, p = christoffersen_ind(hit)
        ax.plot(fc.index, fc["ret"], lw=0.6, color=RETURNS)
        ax.plot(fc.index, -fc[f"{name}_var"], lw=2, color=c)
        ax.scatter(fc.index[hit], fc["ret"][hit], s=36, color=c, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.set_title(titles[name], loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.text(0.995, 0.04,
                f"{hit.sum()} exceptions (expected {len(hit) * 0.01:.0f})   "
                f"independence test p = {p:.3f}" if np.isfinite(p) else f"{hit.sum()} exceptions",
                transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "fc": SURFACE, "ec": GRID})
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("daily log return")
    axes[0].text(0.005, 0.96, "line = minus 99% VaR forecast   dots = days the loss exceeded it",
                 transform=axes[0].transAxes, va="top", color=INK_2, fontsize=8.5)
    fig.suptitle(f"{ticker}: one-day 99% VaR, out of sample", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scorecard(res: dict, path: Path, label: dict) -> None:
    """Estimator x asset grid: conditional coverage verdict at 99% and 95%."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    tickers = list(res)
    names = list(label)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, alpha in zip(axes, (0.99, 0.95)):
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i, name in enumerate(names):
            for j, t in enumerate(tickers):
                s = next(x for x in res[t]["results"] if x["estimator"] == name and x["alpha"] == alpha)
                ok = np.isfinite(s["cc_p"]) and s["cc_p"] >= 0.05
                col = GOOD if ok else CRITICAL
                ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=col, alpha=0.14, lw=0))
                ax.text(j + 0.5, i + 0.38, ("✓ pass" if ok else "✗ fail"), ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=col)
                ax.text(j + 0.5, i + 0.68, f"{s['exceptions']} / {s['expected']:.0f}",
                        ha="center", va="center", fontsize=8.5, color=INK_2)
        ax.set_xlim(0, len(tickers))
        ax.set_ylim(len(names), 0)
        ax.set_xticks(np.arange(len(tickers)) + 0.5, tickers)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_yticks(np.arange(len(names)) + 0.5, [label[n] for n in names])
        ax.set_title(f"{alpha:.0%} VaR", fontsize=11, fontweight="bold", color=INK, pad=22)
    fig.text(0.01, 0.01, "Christoffersen conditional coverage test at 5%. "
             "Numbers: exceptions / expected over the out-of-sample period.",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
