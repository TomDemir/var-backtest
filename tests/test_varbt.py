import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from varbt import estimators as est  # noqa: E402
from varbt.backtest import (  # noqa: E402
    basel_zone, christoffersen_ind, kupiec_pof, mcneil_frey, rolling_forecasts,
)


def _series(r: np.ndarray) -> pd.Series:
    return pd.Series(r, index=pd.bdate_range("2020-01-01", periods=len(r)))


def _garch(n: int, seed: int) -> np.ndarray:
    """GARCH(1,1) with Student-t shocks: clustered, fat-tailed returns."""
    rng = np.random.default_rng(seed)
    w, a, b = 2e-6, 0.10, 0.88
    var, out = w / (1 - a - b), np.empty(n)
    for t in range(n):
        out[t] = np.sqrt(var) * rng.standard_t(6) / np.sqrt(6 / 4)
        var = w + a * out[t] ** 2 + b * var
    return out


class TestEstimators(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).standard_normal(200_000) * 0.01

    def test_gaussian_matches_closed_form(self):
        f = est.gaussian(self.x, 0.99)
        z = stats.norm.ppf(0.99)
        self.assertAlmostEqual(f.var, 0.01 * z, delta=2e-4)
        self.assertAlmostEqual(f.es, 0.01 * stats.norm.pdf(z) / 0.01, delta=2e-4)

    def test_historical_converges_to_gaussian_on_gaussian_data(self):
        h, g = est.historical(self.x, 0.99), est.gaussian(self.x, 0.99)
        self.assertAlmostEqual(h.var, g.var, delta=3e-4)
        self.assertAlmostEqual(h.es, g.es, delta=3e-4)

    def test_es_exceeds_var(self):
        for f in (est.historical(self.x[:500], 0.95), est.gaussian(self.x[:500], 0.95),
                  est.ewma(0.01, 0.95)):
            self.assertGreater(f.es, f.var)

    def test_monte_carlo_seeded(self):
        w = self.x[:250]
        a = est.monte_carlo(w, 0.99, np.random.default_rng(7))
        b = est.monte_carlo(w, 0.99, np.random.default_rng(7))
        c = est.monte_carlo(w, 0.99, np.random.default_rng(8))
        self.assertEqual(a, b)
        self.assertNotEqual(a.var, c.var)

    def test_monte_carlo_has_no_double_ito_correction(self):
        w = 0.001 + 0.03 * np.random.default_rng(1).standard_normal(250)
        f = est.monte_carlo(w, 0.5, np.random.default_rng(2), n_sims=2_000_000)
        self.assertAlmostEqual(-f.var, w.mean(), delta=1e-4)


class TestConditional(unittest.TestCase):
    def test_ewma_sigma_is_causal(self):
        r = np.random.default_rng(4).standard_normal(300) * 0.01
        s1 = est.ewma_sigma(r)
        r2 = r.copy()
        r2[150:] = 1.0
        s2 = est.ewma_sigma(r2)
        np.testing.assert_array_equal(s1[:151], s2[:151])  # sig[150] uses r[:150] only
        self.assertNotEqual(s1[151], s2[151])

    def test_ewma_recursion(self):
        r = np.random.default_rng(5).standard_normal(50) * 0.01
        s = est.ewma_sigma(r, lam=0.94, burn_in=20)
        self.assertAlmostEqual(s[21] ** 2, 0.94 * s[20] ** 2 + 0.06 * r[20] ** 2, places=15)

    def test_ewma_var_closed_form(self):
        self.assertAlmostEqual(est.ewma(0.02, 0.99).var, 0.02 * stats.norm.ppf(0.99), places=12)

    def test_fhs_equals_scaled_historical_under_constant_vol(self):
        w = np.random.default_rng(6).standard_normal(250) * 0.01
        f = est.fhs(w, np.full(250, 0.01), 0.03, 0.99)
        h = est.historical(w, 0.99)
        self.assertAlmostEqual(f.var, 3 * h.var, places=12)
        self.assertAlmostEqual(f.es, 3 * h.es, places=12)


class TestNoLookAhead(unittest.TestCase):
    def test_forecast_ignores_same_day_and_future(self):
        r = np.random.default_rng(3).standard_normal(300) * 0.01
        base = rolling_forecasts(_series(r), 0.99, window=250, n_sims=1000)
        t = 260
        r2 = r.copy()
        r2[t:] = -0.5  # corrupt day t and everything after
        alt = rolling_forecasts(_series(r2), 0.99, window=250, n_sims=1000)
        cols = [c for c in base.columns if c.endswith(("_var", "_es", "_scale"))]
        self.assertEqual(len(cols), 15)  # 5 estimators x 3 fields
        pd.testing.assert_series_equal(base.iloc[t - 250][cols], alt.iloc[t - 250][cols],
                                       check_names=False)


class TestKupiec(unittest.TestCase):
    def test_exact_expected_rate_gives_zero_lr(self):
        hits = np.zeros(1000, bool)
        hits[:10] = True
        lr, p = kupiec_pof(hits, 0.99)
        self.assertAlmostEqual(lr, 0.0, places=10)
        self.assertAlmostEqual(p, 1.0, places=10)

    def test_reference_value_10_of_250(self):
        # 2 * [240 ln(.96/.99) + 10 ln(.04/.01)] = 12.9555
        hits = np.zeros(250, bool)
        hits[:10] = True
        self.assertAlmostEqual(kupiec_pof(hits, 0.99)[0], 12.9555, places=3)

    def test_zero_exceptions_is_finite(self):
        lr, p = kupiec_pof(np.zeros(250, bool), 0.99)
        self.assertTrue(np.isfinite(lr) and 0 < p < 1)


class TestChristoffersen(unittest.TestCase):
    def test_clustered_hits_rejected(self):
        hits = np.zeros(1000, bool)
        hits[500:510] = True
        self.assertLess(christoffersen_ind(hits)[1], 0.01)

    def test_spread_hits_not_rejected(self):
        hits = np.zeros(1000, bool)
        hits[::100] = True
        self.assertGreater(christoffersen_ind(hits)[1], 0.05)

    def test_no_hits_undefined(self):
        lr, p = christoffersen_ind(np.zeros(100, bool))
        self.assertTrue(np.isnan(lr) and np.isnan(p))


class TestBaselAndES(unittest.TestCase):
    def test_basel_bands_at_250_days(self):
        # Regulatory bands: 0-4 green, 5-9 yellow, 10+ red.
        self.assertEqual([basel_zone(250, x) for x in (4, 5, 9, 10)],
                         ["green", "yellow", "yellow", "red"])

    def test_mcneil_frey_detects_understated_es(self):
        rng = np.random.default_rng(9)
        scale = np.full(40, 0.01)
        true_es = np.full(40, 0.03)
        losses = true_es + scale * (rng.standard_exponential(40) - 1.0)  # mean residual 0
        _, p_ok = mcneil_frey(losses, true_es, scale)
        _, p_bad = mcneil_frey(losses, 0.8 * true_es, scale)
        self.assertGreater(p_ok, 0.05)
        self.assertLess(p_bad, 0.01)

    def test_mcneil_frey_needs_three_points(self):
        self.assertTrue(np.isnan(mcneil_frey(np.ones(2), np.ones(2), np.ones(2))[1]))


class TestCalibration(unittest.TestCase):
    def test_gaussian_calibrated_when_model_is_true(self):
        r = np.random.default_rng(5).standard_normal(3250) * 0.01
        fc = rolling_forecasts(_series(r), 0.95, window=250, n_sims=2000)
        self.assertAlmostEqual(fc["gaussian_hit"].mean(), 0.05, delta=0.012)

    def test_conditional_models_beat_static_on_clustered_volatility(self):
        # On GARCH data, exceptions of the static model cluster; EWMA's should not.
        r = _garch(4250, seed=11)
        fc = rolling_forecasts(_series(r), 0.99, window=250, n_sims=500)
        p_static = christoffersen_ind(fc["gaussian_hit"].to_numpy())[1]
        p_ewma = christoffersen_ind(fc["ewma_hit"].to_numpy())[1]
        self.assertLess(p_static, 0.05)
        self.assertGreater(p_ewma, p_static)


if __name__ == "__main__":
    unittest.main()
