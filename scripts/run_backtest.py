"""Run the rolling VaR backtest on every asset in data/ and write results/.

Usage: python scripts/run_backtest.py [--tickers SPY QQQ TLT GLD]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from varbt.backtest import rolling_forecasts, summarise
from varbt.estimators import ESTIMATORS, RISKMETRICS_LAMBDA

ROOT = Path(__file__).resolve().parents[1]
LABEL = {
    "historical": "Historical",
    "gaussian": "Gaussian",
    "monte_carlo": "Monte Carlo (GBM)",
    "ewma": "EWMA (RiskMetrics)",
    "fhs": "Filtered historical",
}


def load_log_returns(path: Path) -> pd.Series:
    px = pd.read_csv(path, index_col="date", parse_dates=True)["close"].sort_index()
    if px.isna().any() or (px <= 0).any():
        raise ValueError(f"{path}: prices must be positive and complete")
    return np.log(px / px.shift(1)).dropna()


def pv(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def mark(p: float) -> str:
    return "n/a" if not np.isfinite(p) else ("pass" if p >= 0.05 else "**fail**")


def full_sample(r: pd.Series) -> dict:
    """In-sample descriptive statistics. NOT a backtest."""
    x = r.to_numpy()
    h99 = -np.quantile(x, 0.01)
    g99 = -(x.mean() + stats.norm.ppf(0.01) * x.std(ddof=1))
    return {
        "excess_kurtosis": float(stats.kurtosis(x)),
        "skewness": float(stats.skew(x)),
        "hist_var99": float(h99),
        "gauss_var99": float(g99),
        "gap_bps": float((h99 - g99) * 1e4),
    }


def main_table(rows: list[dict], alpha: float) -> list[str]:
    head = (
        "| Estimator | Exceptions (obs / exp) | Kupiec p | Christoffersen p | Cond. cov. p "
        + ("| Basel zone " if alpha == 0.99 else "")
        + "| ES ratio | McNeil-Frey p | Mean VaR |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for s in rows:
        if s["alpha"] != alpha:
            continue
        lines.append(
            f"| {LABEL[s['estimator']]} | {s['exceptions']} / {s['expected']:.1f} | {pv(s['kupiec_p'])} | "
            f"{pv(s['christ_p'])} | {pv(s['cc_p'])} "
            + (f"| {s['basel']} " if alpha == 0.99 else "")
            + f"| {s['es_ratio']:.2f} | {pv(s['mf_p'])} | {s['mean_var']:.2%} |"
        )
    return lines


def robustness_table(res: dict, alpha: float) -> list[str]:
    tickers = list(res)
    head = "| Estimator | " + " | ".join(tickers) + " | Passes (of " + str(len(tickers)) + ") |"
    lines = [head, "|" + "---|" * (len(tickers) + 2)]
    for name in ESTIMATORS:
        cells, passes = [], 0
        for t in tickers:
            s = next(x for x in res[t]["results"] if x["estimator"] == name and x["alpha"] == alpha)
            ok = np.isfinite(s["cc_p"]) and s["cc_p"] >= 0.05
            passes += ok
            cells.append(f"{s['exceptions']}/{s['expected']:.0f} · {mark(s['cc_p'])}")
        lines.append(f"| {LABEL[name]} | " + " | ".join(cells) + f" | {passes} |")
    return lines


def inject(block: str, key: str) -> None:
    readme = ROOT / "README.md"
    start, end = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
    txt = readme.read_text()
    if start in txt and end in txt:
        head, rest = txt.split(start, 1)
        _, tail = rest.split(end, 1)
        readme.write_text(f"{head}{start}\n{block}{end}{tail}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "TLT", "GLD"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--out", default="results")
    p.add_argument("--no-readme", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, fc99 = {}, {}
    for t in a.tickers:
        path = Path(a.data_dir) / f"{t.lower()}.csv"
        r = load_log_returns(path)
        rows = []
        for alpha in (0.95, 0.99):
            fc = rolling_forecasts(r, alpha, a.window, a.seed, a.sims)
            fc.to_csv(out / f"forecasts_{t.lower()}_{int(alpha * 100)}.csv")
            rows += summarise(fc, alpha)
            if alpha == 0.99:
                fc99[t] = fc
        res[t] = {
            "meta": {
                "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "first_return": str(r.index[0].date()), "last_return": str(r.index[-1].date()),
                "n_returns": len(r), "test_start": str(fc99[t].index[0].date()),
                "test_days": len(fc99[t]),
            },
            "full_sample": full_sample(r),
            "results": [asdict(s) for s in rows],
        }
        print(f"{t}: done")

    config = {"window": a.window, "seed": a.seed, "n_sims": a.sims, "ewma_lambda": RISKMETRICS_LAMBDA}
    (out / "summary.json").write_text(json.dumps({"config": config, "assets": res}, indent=2))

    main_t = a.tickers[0]
    m = res[main_t]["meta"]
    fs = res[main_t]["full_sample"]
    rows = res[main_t]["results"]
    block = [
        f"**{main_t}**, returns {m['first_return']} to {m['last_return']}, "
        f"{m['test_days']} out-of-sample days from {m['test_start']} "
        f"(file sha256 `{m['sha256'][:12]}`). Window {a.window} days, EWMA lambda "
        f"{RISKMETRICS_LAMBDA}, seed {a.seed}, {a.sims:,} Monte Carlo draws per day.",
        "",
        "**99% VaR**",
        "",
        *main_table(rows, 0.99),
        "",
        "**95% VaR**",
        "",
        *main_table(rows, 0.95),
        "",
        f"In-sample, for reference only: excess kurtosis {fs['excess_kurtosis']:.1f}, "
        f"skewness {fs['skewness']:.2f}; the 99% historical quantile is "
        f"{fs['gap_bps']:.0f} bps beyond the Gaussian one.",
        "",
        "**Robustness, 99% VaR on four asset classes** "
        "(exceptions / expected · conditional coverage at 5%)",
        "",
        *robustness_table(res, 0.99),
        "",
        "**Same, 95% VaR**",
        "",
        *robustness_table(res, 0.95),
    ]
    text = "\n".join(block) + "\n"
    (out / "summary.md").write_text(text)
    print(text)
    if not a.no_readme and a.out == "results":
        inject(text, "RESULTS")

    from varbt.plots import hero, scorecard
    hero(fc99[main_t], main_t, out / "hero.png")
    scorecard(res, out / "scorecard.png", LABEL)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
