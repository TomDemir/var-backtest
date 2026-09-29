"""One-day Value-at-Risk and Expected Shortfall estimators.

Conventions
-----------
* Input: a 1-D array of daily log returns (the estimation window only).
* Output: VaR and ES as POSITIVE losses in log-return units.
  A VaR of 0.02 at alpha=0.99 means: the one-day log return is expected
  to fall below -0.02 on 1% of days.
* An estimator never sees anything outside the window it is given.
  The no-look-ahead guarantee is enforced in backtest.py, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RiskEstimate:
    var: float
    es: float


def historical(window: np.ndarray, alpha: float) -> RiskEstimate:
    """Empirical quantile of the window. No distributional assumption."""
    losses = -np.asarray(window, dtype=float)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return RiskEstimate(var=var, es=float(tail.mean()))


def gaussian(window: np.ndarray, alpha: float) -> RiskEstimate:
    """Variance-covariance VaR: log returns assumed i.i.d. N(mu, sigma^2)."""
    w = np.asarray(window, dtype=float)
    mu, sigma = w.mean(), w.std(ddof=1)
    z = stats.norm.ppf(alpha)
    var = -mu + sigma * z
    es = -mu + sigma * stats.norm.pdf(z) / (1.0 - alpha)
    return RiskEstimate(var=float(var), es=float(es))


def monte_carlo(
    window: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    n_sims: int = 10_000,
) -> RiskEstimate:
    """GBM Monte Carlo, one-day horizon.

    mu is estimated on LOG returns, so it is already the log drift
    (mu_arith - sigma^2/2). Subtracting sigma^2/2 again would bias the
    drift downward; the original script did exactly that.

    Note: with Gaussian shocks and a one-day horizon this estimator
    targets the same quantity as `gaussian`, plus sampling noise.
    """
    w = np.asarray(window, dtype=float)
    mu, sigma = w.mean(), w.std(ddof=1)
    sim_log_returns = mu + sigma * rng.standard_normal(n_sims)
    return historical(sim_log_returns, alpha)


ESTIMATORS = ("historical", "gaussian", "monte_carlo")
