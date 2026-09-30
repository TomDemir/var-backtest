**SPY**, returns 2021-09-28 to 2026-09-25, 1004 out-of-sample days from 2022-09-26 (file sha256 `5d6cc5f8bb23`). Window 250 days, EWMA lambda 0.94, seed 42, 10,000 Monte Carlo draws per day.

**99% VaR**

| Estimator | Exceptions (obs / exp) | Kupiec p | Christoffersen p | Cond. cov. p | Basel zone | ES ratio | McNeil-Frey p | Mean VaR |
|---|---|---|---|---|---|---|---|---|
| Historical | 11 / 10.0 | 0.764 | 0.004 | 0.016 | green | 1.24 | 0.010 | 2.73% |
| Gaussian | 12 / 10.0 | 0.546 | 0.006 | 0.020 | green | 1.45 | 0.003 | 2.40% |
| Monte Carlo (GBM) | 14 / 10.0 | 0.236 | 0.013 | 0.022 | green | 1.36 | 0.006 | 2.39% |
| EWMA (RiskMetrics) | 16 / 10.0 | 0.082 | 0.252 | 0.114 | yellow | 1.28 | 0.004 | 2.16% |
| Filtered historical | 13 / 10.0 | 0.369 | 0.156 | 0.245 | green | 1.04 | 0.109 | 2.65% |

**95% VaR**

| Estimator | Exceptions (obs / exp) | Kupiec p | Christoffersen p | Cond. cov. p | ES ratio | McNeil-Frey p | Mean VaR |
|---|---|---|---|---|---|---|---|
| Historical | 38 / 50.2 | 0.065 | 0.014 | 0.009 | 1.09 | 0.088 | 1.69% |
| Gaussian | 40 / 50.2 | 0.126 | 0.091 | 0.074 | 1.23 | 0.005 | 1.68% |
| Monte Carlo (GBM) | 41 / 50.2 | 0.169 | 0.107 | 0.106 | 1.23 | 0.005 | 1.68% |
| EWMA (RiskMetrics) | 55 / 50.2 | 0.493 | 0.566 | 0.671 | 1.13 | 0.008 | 1.53% |
| Filtered historical | 49 / 50.2 | 0.862 | 0.691 | 0.910 | 0.99 | 0.414 | 1.67% |

In-sample, for reference only: excess kurtosis 7.9, skewness 0.15; the 99% historical quantile is 51 bps beyond the Gaussian one.

**Robustness, 99% VaR on four asset classes** (exceptions / expected · conditional coverage at 5%)

| Estimator | SPY | QQQ | TLT | GLD | Passes (of 4) |
|---|---|---|---|---|---|
| Historical | 11/10 · **fail** | 14/10 · pass | 9/10 · pass | 17/10 · **fail** | 2 |
| Gaussian | 12/10 · **fail** | 20/10 · **fail** | 7/10 · pass | 22/10 · **fail** | 1 |
| Monte Carlo (GBM) | 14/10 · **fail** | 21/10 · **fail** | 7/10 · pass | 24/10 · **fail** | 1 |
| EWMA (RiskMetrics) | 16/10 · pass | 16/10 · pass | 16/10 · pass | 15/10 · pass | 4 |
| Filtered historical | 13/10 · pass | 12/10 · pass | 17/10 · pass | 12/10 · pass | 4 |

**Same, 95% VaR**

| Estimator | SPY | QQQ | TLT | GLD | Passes (of 4) |
|---|---|---|---|---|---|
| Historical | 38/50 · **fail** | 40/50 · pass | 40/50 · pass | 61/50 · pass | 3 |
| Gaussian | 40/50 · pass | 46/50 · pass | 38/50 · pass | 60/50 · pass | 4 |
| Monte Carlo (GBM) | 41/50 · pass | 47/50 · pass | 39/50 · pass | 58/50 · pass | 4 |
| EWMA (RiskMetrics) | 55/50 · pass | 61/50 · pass | 57/50 · pass | 42/50 · **fail** | 3 |
| Filtered historical | 49/50 · pass | 52/50 · pass | 53/50 · pass | 48/50 · pass | 4 |
