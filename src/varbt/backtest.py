"""Rolling out-of-sample backtest: coverage, independence, ES and Basel tests."""

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
    """Forecast day t from returns t-window .. t-1 only (strictly past data).

    One row per forecast day: realised return, then for each estimator its
    VaR, ES, volatility scale and exception flag (realised loss > VaR).
    """
    r = log_returns.to_numpy(dtype=float)
    n = len(r)
    if n <= window:
        raise ValueError(f"need more than {window} returns, got {n}")

    sig = est.ewma_sigma(r)  # sig[t] depends on r[:t] only
    if np.isnan(sig[:window]).sum() >= window:
        raise ValueError("window too short for the EWMA burn-in")

    children = np.random.SeedSequence(seed).spawn(n - window)  # one stream per day

    rows = []
    for k, t in enumerate(range(window, n)):
        past = r[t - window : t]
        past_sig = sig[t - window : t]
        ok = ~np.isnan(past_sig)
        fc = {
            "historical": est.historical(past, alpha),
            "gaussian": est.gaussian(past, alpha),
            "monte_carlo": est.monte_carlo(past, alpha, np.random.default_rng(children[k]), n_sims),
            "ewma": est.ewma(sig[t], alpha),
            "fhs": est.fhs(past[ok], past_sig[ok], sig[t], alpha),
        }
        row = {"date": log_returns.index[t], "ret": r[t]}
        for name, f in fc.items():
            row[f"{name}_var"] = f.var
            row[f"{name}_es"] = f.es
            row[f"{name}_scale"] = f.scale
            row[f"{name}_hit"] = bool(-r[t] > f.var)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------- tests

def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures LR test. Returns (LR, p-value)."""
    hits = np.asarray(hits, dtype=bool)
    n, x = hits.size, int(hits.sum())
    p, pi = 1.0 - alpha, x / n
    lr = -2.0 * (
        _xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - pi) - _xlogy(x, pi)
    )
    return float(lr), float(stats.chi2.sf(lr, df=1))


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) first-order Markov independence LR test.

    NaN when undefined (no exception, or no day following an exception).
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    if n01 + n11 == 0 or n10 + n11 == 0:
        return float("nan"), float("nan")
    pi0, pi1 = n01 / (n00 + n01), n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(stats.chi2.sf(lr, df=1))


def basel_zone(n_days: int, exceptions: int, alpha: float = 0.99) -> str:
    """Basel traffic light, generalised from 250 days to n days.

    Cumulative binomial probability of observing at most `exceptions`
    under a correct model: < 95% green, < 99.99% yellow, else red.
    For n = 250 this reproduces the regulatory 0-4 / 5-9 / 10+ bands.
    """
    cdf = stats.binom.cdf(exceptions, n_days, 1.0 - alpha)
    if cdf < 0.95:
        return "green"
    return "yellow" if cdf < 0.9999 else "red"


def mcneil_frey(
    losses: np.ndarray, es: np.ndarray, scale: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """McNeil and Frey (2000) exceedance-residual test for ES.

    On exception days, e = (loss - ES) / scale should have mean zero if
    ES is right. One-sided bootstrap test of H1: mean > 0 (ES too low).
    Returns (mean residual, p-value); NaN with fewer than 3 exceptions.
    """
    e = (np.asarray(losses) - np.asarray(es)) / np.asarray(scale)
    if e.size < 3:
        return float("nan"), float("nan")
    obs = e.mean()
    centred = e - obs
    rng = np.random.default_rng(seed)
    boot = rng.choice(centred, size=(n_boot, e.size), replace=True).mean(axis=1)
    return float(obs), float((np.sum(boot >= obs) + 1) / (n_boot + 1))


# ------------------------------------------------------------------- summary

@dataclass(frozen=True)
class BacktestSummary:
    estimator: str
    alpha: float
    n_days: int
    exceptions: int
    expected: float
    rate_obs: float
    kupiec_p: float
    christ_p: float
    cc_p: float
    es_ratio: float  # mean realised loss / mean predicted ES, on exception days
    mf_resid: float
    mf_p: float
    basel: str
    mean_var: float  # average VaR level: the capital cost of the model


def summarise(fc: pd.DataFrame, alpha: float) -> list[BacktestSummary]:
    out = []
    for name in est.ESTIMATORS:
        hit = fc[f"{name}_hit"].to_numpy()
        n, x = hit.size, int(hit.sum())
        k_lr, k_p = kupiec_pof(hit, alpha)
        c_lr, c_p = christoffersen_ind(hit)
        cc_p = float(stats.chi2.sf(k_lr + c_lr, df=2)) if np.isfinite(c_lr) else float("nan")
        tail = fc[hit]
        losses = -tail["ret"].to_numpy()
        es_ratio = float(losses.mean() / tail[f"{name}_es"].mean()) if x else float("nan")
        mf_r, mf_p = mcneil_frey(losses, tail[f"{name}_es"].to_numpy(), tail[f"{name}_scale"].to_numpy())
        out.append(
            BacktestSummary(
                estimator=name, alpha=alpha, n_days=n, exceptions=x, expected=n * (1 - alpha),
                rate_obs=x / n, kupiec_p=k_p, christ_p=c_p, cc_p=cc_p, es_ratio=es_ratio,
                mf_resid=mf_r, mf_p=mf_p,
                basel=basel_zone(n, x, alpha) if alpha == 0.99 else "",
                mean_var=float(fc[f"{name}_var"].mean()),
            )
        )
    return out
