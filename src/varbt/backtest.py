"""Rolling out-of-sample VaR backtest with Kupiec and Christoffersen tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import estimators as est


def rolling_forecasts(
    log_returns: pd.Series,
    alpha: float,
    window: int = 250,
    seed: int = 42,
    n_sims: int = 10_000,
) -> pd.DataFrame:
    """Forecast day t using returns t-window .. t-1 only (strictly past data).

    Returns one row per forecast day with the realised return, each
    estimator's VaR / ES and an exception flag (realised loss > VaR).
    """
    r = log_returns.to_numpy(dtype=float)
    n = len(r)
    if n <= window:
        raise ValueError(f"need more than {window} returns, got {n}")

    # One independent, reproducible generator per forecast day.
    children = np.random.SeedSequence(seed).spawn(n - window)

    rows = []
    for k, t in enumerate(range(window, n)):
        past = r[t - window : t]  # excludes r[t]: no look-ahead
        fc = {
            "historical": est.historical(past, alpha),
            "gaussian": est.gaussian(past, alpha),
            "monte_carlo": est.monte_carlo(
                past, alpha, np.random.default_rng(children[k]), n_sims
            ),
        }
        row = {"date": log_returns.index[t], "ret": r[t]}
        for name, f in fc.items():
            row[f"{name}_var"] = f.var
            row[f"{name}_es"] = f.es
            row[f"{name}_hit"] = bool(-r[t] > f.var)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def _xlogy(x: float, y: float) -> float:
    """x * log(y) with the convention 0 * log(0) = 0."""
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures LR test. Returns (LR, p-value)."""
    hits = np.asarray(hits, dtype=bool)
    n, x = hits.size, int(hits.sum())
    p = 1.0 - alpha
    pi = x / n
    ll_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(stats.chi2.sf(lr, df=1))


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence LR test (first-order Markov).

    Returns (LR, p-value). p-value is NaN when the test is undefined
    (no exception, or no transition out of an exception day).
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    if n01 + n11 == 0 or n10 + n11 == 0:
        return float("nan"), float("nan")
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = (
        _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0)
        + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    )
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(stats.chi2.sf(lr, df=1))


@dataclass(frozen=True)
class BacktestSummary:
    estimator: str
    alpha: float
    n_days: int
    exceptions: int
    expected: float
    rate_obs: float
    rate_exp: float
    kupiec_lr: float
    kupiec_p: float
    christ_lr: float
    christ_p: float
    cc_p: float
    es_ratio: float  # mean realised loss on exception days / mean predicted ES


def summarise(fc: pd.DataFrame, alpha: float) -> list[BacktestSummary]:
    out = []
    for name in est.ESTIMATORS:
        hits = fc[f"{name}_hit"].to_numpy()
        n, x = hits.size, int(hits.sum())
        k_lr, k_p = kupiec_pof(hits, alpha)
        c_lr, c_p = christoffersen_ind(hits)
        cc_p = (
            float(stats.chi2.sf(k_lr + c_lr, df=2)) if np.isfinite(c_lr) else float("nan")
        )
        if x > 0:
            realised = (-fc.loc[fc[f"{name}_hit"], "ret"]).mean()
            predicted = fc.loc[fc[f"{name}_hit"], f"{name}_es"].mean()
            es_ratio = float(realised / predicted)
        else:
            es_ratio = float("nan")
        out.append(
            BacktestSummary(
                estimator=name, alpha=alpha, n_days=n, exceptions=x,
                expected=n * (1 - alpha), rate_obs=x / n, rate_exp=1 - alpha,
                kupiec_lr=k_lr, kupiec_p=k_p, christ_lr=c_lr, christ_p=c_p,
                cc_p=cc_p, es_ratio=es_ratio,
            )
        )
    return out
