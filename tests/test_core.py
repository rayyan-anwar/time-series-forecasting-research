"""Tests: generator structure, forecaster maths, evaluation protocol (no leakage), intervals, service and API.

Author: Rayyan Anwar
"""
import unittest

import numpy as np

from app.data import KINDS, generate_series
from app.evaluation import (conformal_quantiles, evaluate_series, interval_coverage, mase_scale, metrics, origins,
                            paired_bootstrap_ratio, rolling_forecasts)
from app.models import FORECASTERS, ModelUnavailable
from app.service import InputError, forecast_series, forecast_synthetic


class TestData(unittest.TestCase):
    def test_deterministic_and_dates(self):
        a, m = generate_series("daily_weekly", seed=2)
        b, _ = generate_series("daily_weekly", seed=2)
        np.testing.assert_array_equal(a.values, b.values)
        self.assertEqual((m, len(a)), (7, 420))
        self.assertEqual(str(a.index.freq), "<Day>")
        self.assertFalse(np.array_equal(a.values, generate_series("daily_weekly", seed=3)[0].values))

    def test_structure_is_planted(self):
        y, _ = generate_series("daily_weekly", n=700, seed=0)
        detr = y.values - (100 + 0.15 * np.arange(700))
        dow = np.array([detr[i::7].mean() for i in range(7)])
        np.testing.assert_allclose(dow, [0, -8, -6, -4, 2, 14, 20], atol=1.2)
        s, _ = generate_series("level_shift", n=500, seed=1)
        self.assertAlmostEqual(s.values[400:].mean() - s.values[:300].mean(), 40, delta=3)
        a, _ = generate_series("ar1", n=3000, seed=0)
        self.assertAlmostEqual(float(np.corrcoef(a.values[:-1], a.values[1:])[0, 1]), 0.7, delta=0.05)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            generate_series("sine")
        with self.assertRaises(ValueError):
            generate_series("ar1", n=10)


class TestModels(unittest.TestCase):
    def test_all_models_return_finite_h_forecasts(self):
        y, m = generate_series("daily_weekly", seed=0)
        for name, f in FORECASTERS.items():
            out = f(y.values[:200], 14, m)
            self.assertEqual(out.shape, (14,), msg=name)
            self.assertTrue(np.isfinite(out).all(), msg=name)

    def test_benchmarks_hand_computed(self):
        y = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_allclose(FORECASTERS["naive"](y, 3, 1), [10, 10, 10])
        np.testing.assert_allclose(FORECASTERS["mean"](y, 2, 1), [5.5, 5.5])
        np.testing.assert_allclose(FORECASTERS["drift"](y, 3, 1), [11, 12, 13])
        np.testing.assert_allclose(FORECASTERS["seasonal_naive"](y, 5, 3), [8, 9, 10, 8, 9])
        np.testing.assert_allclose(FORECASTERS["moving_average"](y, 2, 3), [9, 9])

    def test_exponential_smoothing_on_known_signals(self):
        const = np.full(40, 7.0)
        np.testing.assert_allclose(FORECASTERS["ses"](const, 4, 1), 7.0, atol=1e-6)
        lin = 3.0 + 2.0 * np.arange(60)
        np.testing.assert_allclose(FORECASTERS["holt"](lin, 5, 1), 3 + 2 * np.arange(60, 65), atol=0.5)
        t = np.arange(96)
        seasonal = 50 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12)
        np.testing.assert_allclose(FORECASTERS["holt_winters"](seasonal, 12, 12),
                                   50 + 0.5 * np.arange(96, 108) + 10 * np.sin(2 * np.pi * np.arange(96, 108) / 12), atol=1.5)

    def test_invalid_and_unavailable(self):
        for bad in ([1.0, 2.0], [1.0, np.nan, 3.0, 4.0], np.zeros((3, 3))):
            with self.assertRaises(ValueError):
                FORECASTERS["naive"](bad, 3, 1)
        with self.assertRaises(ValueError):
            FORECASTERS["naive"]([1.0, 2.0, 3.0], 0, 1)
        with self.assertRaises(ModelUnavailable):
            FORECASTERS["holt_winters"](np.arange(30.0), 3, 1)
        with self.assertRaises(ModelUnavailable):
            FORECASTERS["holt_winters"](np.arange(10.0), 3, 7)
        with self.assertRaises(ModelUnavailable):
            FORECASTERS["ridge_lags"](np.arange(10.0), 3, 7)


class TestEvaluation(unittest.TestCase):
    def test_origins_are_valid_and_leave_room(self):
        o = origins(100, 10, 5, 50)
        self.assertEqual((o[0], o[-1]), (50, 90))
        self.assertTrue(all(t + 10 <= 100 for t in o))
        self.assertEqual(origins(60, 10, 50, 50), list(range(50, 51)))
        with self.assertRaises(ValueError):
            origins(55, 10, 3, 50)

    def test_metrics_hand_computed(self):
        fc, ac = np.array([[1., 2.], [3., 4.]]), np.array([[2., 2.], [3., 6.]])
        m = metrics(fc, ac, scale=0.5)
        self.assertAlmostEqual(m["mae"], 0.75)
        self.assertAlmostEqual(m["rmse"], np.sqrt((1 + 0 + 0 + 4) / 4))
        self.assertAlmostEqual(m["mase"], 1.5)
        self.assertAlmostEqual(m["bias"], 0.75)
        self.assertAlmostEqual(mase_scale(np.array([1., 3., 2., 6.]), 1), (2 + 1 + 4) / 3)
        self.assertAlmostEqual(mase_scale(np.array([1., 2., 3., 5., 7.]), 2), (2 + 3 + 4) / 3)

    def test_no_leakage_future_values_cannot_change_forecasts(self):
        y, m = generate_series("monthly_yearly", seed=1)
        y = y.values
        orgs = origins(len(y), 12, 4, 60)
        for model in FORECASTERS:
            try:
                base, _ = rolling_forecasts(y, m, 12, orgs, model)
            except ModelUnavailable:
                continue
            for k, t in enumerate(orgs):
                z = y.copy()
                z[t:] = -999.0
                alt, _ = rolling_forecasts(z, m, 12, [t], model)
                np.testing.assert_allclose(base[k], alt[0], err_msg=model)

    def test_conformal_quantile_and_coverage(self):
        errs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
        self.assertEqual(float(conformal_quantiles(errs, 0.8)[0]), 8.0)  # ceil(10*0.8) = 8th smallest
        rng = np.random.default_rng(0)
        fc = np.zeros((400, 3))
        ac = rng.normal(0, 1, (400, 3))
        cov = interval_coverage(fc, ac, 0.8)
        self.assertAlmostEqual(cov["empirical_coverage"], 0.8, delta=0.07)
        self.assertFalse(interval_coverage(fc[:3], ac[:3])["available"])

    def test_bootstrap_ratio_brackets_point_and_detects_better_model(self):
        rng = np.random.default_rng(1)
        good, bad = rng.normal(0, 1, (40, 5)), rng.normal(0, 2, (40, 5))
        p, lo, hi = paired_bootstrap_ratio(good, bad)
        self.assertLess(lo, p)
        self.assertGreater(hi, p)
        self.assertLess(hi, 1.0)

    def test_evaluate_series_structure(self):
        y, m = generate_series("daily_weekly", seed=0)
        r = evaluate_series(y.values, m, 14, models=["naive", "seasonal_naive", "holt_winters"], n_origins=6)
        self.assertEqual(set(r["models"]), {"naive", "seasonal_naive", "holt_winters"})
        self.assertEqual(len(r["models"]["naive"]["by_horizon_mae"]), 14)
        self.assertLess(r["models"]["seasonal_naive"]["mase"], r["models"]["naive"]["mase"])

    def test_scientific_expectations(self):
        hw = sn = 0.0
        for s in range(3):
            y, m = generate_series("daily_weekly", seed=s)
            r = evaluate_series(y.values, m, 14, models=["seasonal_naive", "holt_winters"], n_origins=8)
            hw += r["models"]["holt_winters"]["mase"] / 3
            sn += r["models"]["seasonal_naive"]["mase"] / 3
        self.assertLess(hw, sn)
        y, _ = generate_series("random_walk", seed=0)
        r = evaluate_series(y.values, 1, 10, models=["naive", "ridge_lags"], n_origins=8)
        self.assertGreater(r["models"]["ridge_lags"]["mase"], 0.8 * r["models"]["naive"]["mase"])


class TestService(unittest.TestCase):
    def test_forecast_series_output_and_interval(self):
        y, m = generate_series("monthly_yearly", seed=0)
        r = forecast_series(y.values, m, 12, "holt_winters")
        self.assertEqual(len(r["forecast"]), 12)
        self.assertTrue(all(lo <= f <= hi for lo, f, hi in zip(r["lower"], r["forecast"], r["upper"])))
        self.assertLess(r["backtest"]["mase"], r["backtest"]["reference_mase"])

    def test_synthetic_has_history_and_dates(self):
        r = forecast_synthetic("daily_weekly", 0, None, "seasonal_naive")
        self.assertEqual(len(r["forecast_dates"]), 14)
        self.assertEqual(len(r["history"]["values"]), 120)
        self.assertGreater(r["forecast_dates"][0], r["history"]["dates"][-1])

    def test_validation_errors(self):
        y = np.arange(100.0)
        bad = [dict(y=y[:10], m=1, h=3, model="naive"), dict(y=y, m=1, h=0, model="naive"), dict(y=y, m=1, h=3, model="arima"),
               dict(y=y, m=0, h=3, model="naive"), dict(y=y, m=1, h=3, model="naive", level=0.99),
               dict(y=np.append(y, np.nan), m=1, h=3, model="naive"), dict(y=y, m=7, h=3, model="holt_winters", level=0.8)]
        for kw in bad[:-1]:
            with self.assertRaises(InputError, msg=str(kw)[:40]):
                forecast_series(kw.pop("y"), kw.pop("m"), kw.pop("h"), kw.pop("model"), **kw)
        short = np.arange(30.0)
        with self.assertRaises(InputError):
            forecast_series(short, 20, 3, "holt_winters")
        with self.assertRaises(InputError):
            forecast_synthetic("sine", 0, 5, "naive")


try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    HAVE_API = True
except ImportError:
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "fastapi/httpx not installed - API tests are skipped, not passed")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.c = TestClient(app)

    def test_basics_and_forecast(self):
        self.assertEqual(self.c.get("/health").json()["author"], "Rayyan Anwar")
        self.assertIn("Rayyan Anwar", self.c.get("/").text)
        r = self.c.post("/api/forecast", json={"kind": "daily_weekly", "model": "seasonal_naive"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["forecast"]), 14)
        vals = [float(x) for x in np.arange(60.0)]
        r2 = self.c.post("/api/forecast", json={"values": vals, "horizon": 5, "model": "drift"})
        self.assertEqual(len(r2.json()["forecast"]), 5)

    def test_validation(self):
        for bad in ({}, {"kind": "daily_weekly", "values": [1.0] * 40, "horizon": 3}, {"values": [1.0] * 40},
                    {"kind": "x"}, {"kind": "ar1", "horizon": 100}, {"values": [1.0] * 10, "horizon": 3}):
            self.assertEqual(self.c.post("/api/forecast", json=bad).status_code, 422, msg=str(bad)[:40])


if __name__ == "__main__":
    unittest.main()
