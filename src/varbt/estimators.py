"""One-day Value-at-Risk and Expected Shortfall estimators.

Conventions
-----------
* Input: daily log returns from the estimation window only.
* Output: VaR and ES as POSITIVE losses in log-return units.
  A 99% VaR of 0.02 means the one-day log return is expected to fall
  below -0.02 on 1% of days.
* Every estimator also returns a `scale`: its own volatility forecast,
  used to standardise tail residuals in the McNeil-Frey ES test.
* An estimator never sees data outside the window it is given. The
  no-look-ahead guarantee is enforced (and tested) in backtest.py.

Two families
------------
Unconditional (equal weight on the 250 past days): historical, gaussian,
monte_carlo. Conditional (recent days weigh more): ewma, fhs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

RISKMETRICS_LAMBDA = 0.94  # fixed a priori (J.P. Morgan, 1996), never tuned on the test set


@dataclass(frozen=True)
class RiskEstimate:
    var: float
    es: float
    scale: float


def _empirical(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    var = float(np.quantile(losses, alpha))
    return var, float(losses[losses >= var].mean())


# ---------------------------------------------------------------- unconditional

def historical(window: np.ndarray, alpha: float) -> RiskEstimate:
    """Empirical quantile of the window. No distributional assumption."""
    w = np.asarray(window, dtype=float)
    var, es = _empirical(-w, alpha)
    return RiskEstimate(var, es, float(w.std(ddof=1)))


def gaussian(window: np.ndarray, alpha: float) -> RiskEstimate:
    """Variance-covariance VaR: log returns i.i.d. N(mu, sigma^2)."""
    w = np.asarray(window, dtype=float)
    mu, sigma = w.mean(), w.std(ddof=1)
    z = stats.norm.ppf(alpha)
    var = -mu + sigma * z
    es = -mu + sigma * stats.norm.pdf(z) / (1.0 - alpha)
    return RiskEstimate(float(var), float(es), float(sigma))


def monte_carlo(
    window: np.ndarray, alpha: float, rng: np.random.Generator, n_sims: int = 10_000
) -> RiskEstimate:
    """One-day GBM Monte Carlo.

    mu is estimated on LOG returns, so it is already the log drift; the
    original script subtracted sigma^2/2 a second time. With Gaussian shocks
    and a one-day horizon this targets the same quantity as `gaussian`, plus
    simulation noise: it is kept as a check on the simulation code.
    """
    w = np.asarray(window, dtype=float)
    mu, sigma = w.mean(), w.std(ddof=1)
    var, es = _empirical(-(mu + sigma * rng.standard_normal(n_sims)), alpha)
    return RiskEstimate(var, es, float(sigma))


# ------------------------------------------------------------------ conditional

def ewma_sigma(returns: np.ndarray, lam: float = RISKMETRICS_LAMBDA, burn_in: int = 20) -> np.ndarray:
    """Causal EWMA volatility: out[t] uses returns strictly before t.

    sigma^2[t] = lam * sigma^2[t-1] + (1 - lam) * r[t-1]^2, zero mean
    (RiskMetrics convention). Seeded with the mean square of the first
    `burn_in` returns; out[:burn_in] is NaN and never used by the backtest.
    """
    r = np.asarray(returns, dtype=float)
    out = np.full(r.size, np.nan)
    var = float(np.mean(r[:burn_in] ** 2))
    for t in range(burn_in, r.size):
        out[t] = np.sqrt(var)
        var = lam * var + (1.0 - lam) * r[t] ** 2
    return out


def ewma(sigma_next: float, alpha: float) -> RiskEstimate:
    """RiskMetrics: N(0, sigma_ewma^2) for tomorrow."""
    z = stats.norm.ppf(alpha)
    return RiskEstimate(
        float(sigma_next * z), float(sigma_next * stats.norm.pdf(z) / (1.0 - alpha)), float(sigma_next)
    )


def fhs(window: np.ndarray, window_sigma: np.ndarray, sigma_next: float, alpha: float) -> RiskEstimate:
    """Filtered historical simulation (Barone-Adesi et al., 1999).

    Devolatilise each past return by its own EWMA forecast, take the
    empirical tail of the standardised residuals, rescale by tomorrow's
    forecast. Keeps the fat tails of the data AND reacts to volatility.
    """
    z = np.asarray(window, dtype=float) / np.asarray(window_sigma, dtype=float)
    var, es = _empirical(-z * sigma_next, alpha)
    return RiskEstimate(var, es, float(sigma_next))


ESTIMATORS = ("historical", "gaussian", "monte_carlo", "ewma", "fhs")
