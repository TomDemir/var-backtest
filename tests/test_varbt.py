import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from varbt import estimators as est  # noqa: E402
from varbt.backtest import christoffersen_ind, kupiec_pof, rolling_forecasts  # noqa: E402


def _series(r: np.ndarray) -> pd.Series:
    return pd.Series(r, index=pd.bdate_range("2020-01-01", periods=len(r)))


class TestEstimators(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).standard_normal(200_000) * 0.01

    def test_gaussian_matches_closed_form(self):
        f = est.gaussian(self.x, 0.99)
        self.assertAlmostEqual(f.var, 0.01 * stats.norm.ppf(0.99), delta=2e-4)
        self.assertAlmostEqual(f.es, 0.01 * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01, delta=2e-4)

    def test_historical_converges_to_gaussian_on_gaussian_data(self):
        h, g = est.historical(self.x, 0.99), est.gaussian(self.x, 0.99)
        self.assertAlmostEqual(h.var, g.var, delta=3e-4)
        self.assertAlmostEqual(h.es, g.es, delta=3e-4)

    def test_es_exceeds_var(self):
        for fn in (est.historical, est.gaussian):
            f = fn(self.x[:500], 0.95)
            self.assertGreater(f.es, f.var)

    def test_monte_carlo_seeded(self):
        w = self.x[:250]
        a = est.monte_carlo(w, 0.99, np.random.default_rng(7))
        b = est.monte_carlo(w, 0.99, np.random.default_rng(7))
        c = est.monte_carlo(w, 0.99, np.random.default_rng(8))
        self.assertEqual(a, b)
        self.assertNotEqual(a.var, c.var)

    def test_monte_carlo_has_no_double_ito_correction(self):
        # Drift of simulated log returns must equal the window's mean log return.
        w = 0.001 + 0.03 * np.random.default_rng(1).standard_normal(250)
        f = est.monte_carlo(w, 0.5, np.random.default_rng(2), n_sims=2_000_000)
        self.assertAlmostEqual(-f.var, w.mean(), delta=1e-4)


class TestNoLookAhead(unittest.TestCase):
    def test_forecast_ignores_same_day_and_future(self):
        r = np.random.default_rng(3).standard_normal(300) * 0.01
        base = rolling_forecasts(_series(r), 0.99, window=250, n_sims=1000)
        t = 260
        r2 = r.copy()
        r2[t:] = -0.5  # corrupt day t and everything after
        alt = rolling_forecasts(_series(r2), 0.99, window=250, n_sims=1000)
        k = t - 250
        cols = [c for c in base.columns if c.endswith("_var") or c.endswith("_es")]
        pd.testing.assert_series_equal(base.iloc[k][cols], alt.iloc[k][cols], check_names=False)


class TestKupiec(unittest.TestCase):
    def test_exact_expected_rate_gives_zero_lr(self):
        hits = np.zeros(1000, bool)
        hits[:10] = True
        lr, p = kupiec_pof(hits, 0.99)
        self.assertAlmostEqual(lr, 0.0, places=10)
        self.assertAlmostEqual(p, 1.0, places=10)

    def test_reference_value_10_of_250(self):
        # Hand-computed: 2 * [240 ln(.96/.99) + 10 ln(.04/.01)] = 12.9555
        hits = np.zeros(250, bool)
        hits[:10] = True
        lr, _ = kupiec_pof(hits, 0.99)
        self.assertAlmostEqual(lr, 12.9555, places=3)

    def test_zero_exceptions_is_finite(self):
        lr, p = kupiec_pof(np.zeros(250, bool), 0.99)
        self.assertTrue(np.isfinite(lr) and 0 < p < 1)


class TestChristoffersen(unittest.TestCase):
    def test_clustered_hits_rejected(self):
        hits = np.zeros(1000, bool)
        hits[500:510] = True  # ten consecutive exceptions
        _, p = christoffersen_ind(hits)
        self.assertLess(p, 0.01)

    def test_spread_hits_not_rejected(self):
        hits = np.zeros(1000, bool)
        hits[::100] = True
        _, p = christoffersen_ind(hits)
        self.assertGreater(p, 0.05)

    def test_no_hits_undefined(self):
        lr, p = christoffersen_ind(np.zeros(100, bool))
        self.assertTrue(np.isnan(lr) and np.isnan(p))


class TestCalibrationOnSyntheticIID(unittest.TestCase):
    def test_gaussian_var_calibrated_when_model_is_true(self):
        r = np.random.default_rng(5).standard_normal(3250) * 0.01
        fc = rolling_forecasts(_series(r), 0.95, window=250, n_sims=2000)
        rate = fc["gaussian_hit"].mean()
        self.assertAlmostEqual(rate, 0.05, delta=0.012)


if __name__ == "__main__":
    unittest.main()
