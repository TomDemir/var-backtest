"""Run the rolling VaR backtest and write results/.

Usage: python scripts/run_backtest.py --data data/spy.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from varbt.backtest import rolling_forecasts, summarise  # noqa: E402


def load_log_returns(path: str) -> pd.Series:
    px = pd.read_csv(path, index_col="date", parse_dates=True)["close"].sort_index()
    if px.isna().any() or (px <= 0).any():
        raise ValueError("prices must be positive and complete")
    return np.log(px / px.shift(1)).dropna()


def fmt(x: float, spec: str) -> str:
    return "n/a" if not np.isfinite(x) else format(x, spec)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/spy.csv")
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    r = load_log_returns(a.data)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    sha = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()

    all_rows, forecasts = [], {}
    for alpha in (0.95, 0.99):
        fc = rolling_forecasts(r, alpha, a.window, a.seed, a.sims)
        fc.to_csv(out / f"forecasts_{int(alpha*100)}.csv")
        forecasts[alpha] = fc
        all_rows += summarise(fc, alpha)

    meta = {
        "data_file": a.data, "sha256": sha,
        "first_return": str(r.index[0].date()), "last_return": str(r.index[-1].date()),
        "n_returns": len(r), "window": a.window, "seed": a.seed, "n_sims": a.sims,
        "test_start": str(forecasts[0.99].index[0].date()),
    }
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "results": [asdict(s) for s in all_rows]}, indent=2)
    )

    lines = [
        f"Data: `{a.data}` sha256 `{sha[:12]}`, returns {meta['first_return']} to "
        f"{meta['last_return']} (n={meta['n_returns']}). Rolling window {a.window} days, "
        f"out-of-sample test from {meta['test_start']}, seed {a.seed}, {a.sims} MC draws/day.",
        "",
        "| Estimator | Level | Days | Exceptions (obs / exp) | Rate obs / exp | Kupiec p | Christoffersen p | Cond. coverage p | ES ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in all_rows:
        lines.append(
            f"| {s.estimator} | {s.alpha:.0%} | {s.n_days} | {s.exceptions} / {s.expected:.1f} | "
            f"{s.rate_obs:.2%} / {s.rate_exp:.0%} | {fmt(s.kupiec_p, '.3f')} | "
            f"{fmt(s.christ_p, '.3f')} | {fmt(s.cc_p, '.3f')} | {fmt(s.es_ratio, '.2f')} |"
        )
    table = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(table)
    print(table)

    # Keep the README's results block identical to the latest run.
    readme = Path(__file__).resolve().parents[1] / "README.md"
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a.out == "results" and readme.exists():
        txt = readme.read_text()
        if start in txt and end in txt:
            head, rest = txt.split(start, 1)
            _, tail = rest.split(end, 1)
            readme.write_text(f"{head}{start}\n{table}{end}{tail}")
            print("README results block updated")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping chart")
        return
    fc = forecasts[0.99]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(fc.index, fc["ret"], lw=0.6, color="0.55", label="daily log return")
    for name, c in (("historical", "C0"), ("gaussian", "C3")):
        ax.plot(fc.index, -fc[f"{name}_var"], lw=1.1, color=c, label=f"-VaR 99% {name}")
        hit = fc[f"{name}_hit"]
        ax.scatter(fc.index[hit], fc["ret"][hit], s=14, color=c, zorder=3)
    ax.set_ylabel("log return")
    ax.set_title("SPY: 99% one-day VaR, 250-day rolling window (dots = exceptions)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "var99_backtest.png", dpi=150)
    print(f"chart -> {out / 'var99_backtest.png'}")


if __name__ == "__main__":
    main()
