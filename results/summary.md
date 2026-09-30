Data: `data/spy.csv` sha256 `a43744bb621c`, returns 2021-09-28 to 2026-09-25 (n=1254). Rolling window 250 days, out-of-sample test from 2022-09-26, seed 42, 10000 MC draws/day.

| Estimator | Level | Days | Exceptions (obs / exp) | Rate obs / exp | Kupiec p | Christoffersen p | Cond. coverage p | ES ratio |
|---|---|---|---|---|---|---|---|---|
| historical | 95% | 1004 | 38 / 50.2 | 3.78% / 5% | 0.065 | 0.014 | 0.009 | 1.09 |
| gaussian | 95% | 1004 | 40 / 50.2 | 3.98% / 5% | 0.126 | 0.091 | 0.074 | 1.23 |
| monte_carlo | 95% | 1004 | 41 / 50.2 | 4.08% / 5% | 0.169 | 0.107 | 0.106 | 1.23 |
| historical | 99% | 1004 | 11 / 10.0 | 1.10% / 1% | 0.764 | 0.004 | 0.016 | 1.24 |
| gaussian | 99% | 1004 | 12 / 10.0 | 1.20% / 1% | 0.546 | 0.006 | 0.020 | 1.45 |
| monte_carlo | 99% | 1004 | 14 / 10.0 | 1.39% / 1% | 0.236 | 0.013 | 0.022 | 1.36 |

Full sample, in-sample (descriptive only): excess kurtosis 7.88, skewness 0.15; 99% VaR historical 2.97% vs Gaussian 2.47% (gap 51 bps).
