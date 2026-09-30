# One-day VaR backtest on SPY: historical, Gaussian, Monte Carlo

Three textbook one-day Value-at-Risk estimators, tested out of sample on five
years of SPY (S&P 500 ETF) daily data with the Kupiec coverage test and the
Christoffersen independence test. Expected Shortfall is reported alongside.

The point of the repo is the **backtest**, not the estimators: a VaR number is
only worth something if its exceptions arrive at the promised rate and do not
cluster.

## Problem

For a long position in SPY, estimate the one-day loss that should be exceeded
on only 5% (VaR 95%) or 1% (VaR 99%) of days, then check whether it was.

## Method

| Step | Choice |
|---|---|
| Returns | Daily log returns of adjusted closes |
| Estimation window | Rolling, 250 trading days, strictly before the forecast day |
| Estimators | **Historical** (empirical quantile); **Gaussian** (mean and std of the window, normal quantile); **Monte Carlo** (one-day GBM, 10,000 draws per day, seeded) |
| ES | Mean loss beyond VaR (historical, MC); closed form (Gaussian) |
| Exception | Realised loss on day t greater than the VaR forecast for day t |
| Coverage test | Kupiec (1995) proportion of failures, LR ~ chi2(1) |
| Independence test | Christoffersen (1998) first-order Markov, LR ~ chi2(1); conditional coverage = sum, chi2(2) |
| ES check | Ratio of mean realised loss on exception days to mean predicted ES on those days (descriptive, not a formal test) |

Reproducibility: fixed download dates, one `SeedSequence` child per forecast
day for the Monte Carlo draws, pinned requirements, and a SHA-256 of the input
file written to `results/summary.json`.

## Results

<!-- RESULTS:START -->
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
<!-- RESULTS:END -->

## Known limitations

* **The Monte Carlo estimator adds no information.** With Gaussian shocks and a
  one-day horizon it estimates the same quantile as the Gaussian estimator,
  plus simulation noise. It is kept as a check on the simulation code, not as
  an independent model.
* **Unconditional volatility.** All three estimators weight the 250 past days
  equally, so they react slowly to volatility regimes. Exception clustering is
  the expected symptom; the Christoffersen test is there to measure it.
* **Low power at 99%.** About 1,000 test days give roughly 10 expected
  exceptions at 99%. Neither test can separate a good model from a mediocre
  one with that few events, and the Markov independence test is sensitive to
  one or two consecutive exceptions.
* **Single asset, single period.** One ETF, one five-year window, no
  transaction costs, no position sizing.
* **Data source.** Yahoo Finance via `yfinance`, not an institutional feed.
  Adjusted prices can be revised retroactively.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py          # writes data/spy.csv (not committed)
python scripts/run_backtest.py           # writes results/, updates this README
python -m unittest discover tests        # 13 tests
```

Data is downloaded locally rather than committed: `yfinance` states that the
Yahoo Finance API is intended for personal use, so the raw prices are not
redistributed here.

## Tests

`tests/test_varbt.py` checks: Gaussian VaR and ES against closed form;
historical converging to Gaussian on Gaussian data; ES > VaR; seeded Monte
Carlo reproducibility; no double Itô correction in the simulated drift;
**no look-ahead** (corrupting day t and later leaves the day-t forecast
unchanged); Kupiec against a hand-computed value (10 exceptions in 250 days at
99%, LR = 12.96); Christoffersen on clustered and spread exceptions; and
coverage close to nominal when the Gaussian model is true.

## Layout

```
src/varbt/estimators.py   VaR / ES estimators
src/varbt/backtest.py     rolling forecasts, Kupiec, Christoffersen
scripts/download_data.py  data download (yfinance)
scripts/run_backtest.py   backtest runner, writes results/
tests/test_varbt.py       unit tests (unittest, also runs under pytest)
```
