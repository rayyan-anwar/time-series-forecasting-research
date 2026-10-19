"""Forecasters with a common interface: f(y, h, m) -> array of h forecasts. No future data can be used by construction.

Benchmarks (mean, naive, seasonal naive, drift, moving average), exponential smoothing (SES, Holt, additive Holt-Winters,
parameters fitted by SSE with L-BFGS-B), and level-normalised lag regression (ridge, gradient boosting).

Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


class ModelUnavailable(ValueError):
    """The model cannot be fitted on this much data / this season length."""


def _check(y, h):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError("series must be a finite 1-D array with at least 3 points")
    if h < 1:
        raise ValueError("h must be >= 1")
    return y


def mean_f(y, h, m):
    return np.full(h, _check(y, h).mean())


def naive(y, h, m):
    return np.full(h, _check(y, h)[-1])


def seasonal_naive(y, h, m):
    y = _check(y, h)
    if m < 2:
        return naive(y, h, m)
    if len(y) < m:
        raise ModelUnavailable("need at least one full season")
    return np.array([y[len(y) - m + (i % m)] for i in range(h)])


def drift(y, h, m):
    y = _check(y, h)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)


def moving_average(y, h, m):
    y = _check(y, h)
    w = m if m > 1 else 5
    return np.full(h, y[-min(w, len(y)):].mean())


def _fit(obj, x0, bounds):
    r = minimize(obj, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 60})
    return r.x


def _ses_sse(y, a):
    lvl, sse = y[0], 0.0
    for v in y[1:]:
        e = v - lvl
        sse += e * e
        lvl += a * e
    return sse, lvl


def ses(y, h, m):
    y = _check(y, h)
    a = float(_fit(lambda p: _ses_sse(y, p[0])[0], [0.3], [(0.01, 0.99)])[0])
    return np.full(h, _ses_sse(y, a)[1])


def _holt_run(y, a, b):
    lvl, tr, sse = y[0], y[1] - y[0], 0.0
    for v in y[1:]:
        e = v - (lvl + tr)
        sse += e * e
        new = lvl + tr + a * e
        tr = tr + a * b * e
        lvl = new
    return sse, lvl, tr


def holt(y, h, m):
    y = _check(y, h)
    a, b = _fit(lambda p: _holt_run(y, p[0], p[1])[0], [0.3, 0.1], [(0.01, 0.99), (0.01, 0.99)])
    _, lvl, tr = _holt_run(y, a, b)
    return lvl + tr * np.arange(1, h + 1)


def _hw_run(y, m, a, b, g):
    n = len(y)
    lvl = y[:m].mean()
    tr = (y[m:2 * m].mean() - y[:m].mean()) / m
    s = list(y[:m] - lvl)
    sse = 0.0
    for t in range(m, n):
        e = y[t] - (lvl + tr + s[t - m])
        sse += e * e
        new = lvl + tr + a * e
        tr = tr + a * b * e
        s.append(s[t - m] + g * (1 - a) * e)
        lvl = new
    return sse, lvl, tr, s


def holt_winters(y, h, m):
    y = _check(y, h)
    if m < 2:
        raise ModelUnavailable("Holt-Winters needs a season length m >= 2")
    if len(y) < 2 * m + 2:
        raise ModelUnavailable("Holt-Winters needs at least two full seasons")
    a, b, g = _fit(lambda p: _hw_run(y, m, *p)[0], [0.3, 0.05, 0.3], [(0.01, 0.99)] * 3)
    _, lvl, tr, s = _hw_run(y, m, a, b, g)
    n = len(y)
    return np.array([lvl + (i + 1) * tr + s[n - m + (i % m)] for i in range(h)])


def _lag_forecast(y, h, m, make):
    y = _check(y, h)
    L = min(max(2 * m if m > 1 else 5, 5), max(3, len(y) // 3))
    if len(y) < L + 12:
        raise ModelUnavailable("not enough history for lag regression")
    X, T = [], []
    for t in range(L, len(y)):
        w = y[t - L:t]
        X.append(w - w.mean())
        T.append(y[t] - w.mean())
    mdl = make().fit(np.array(X), np.array(T))
    hist, out = list(y), []
    for _ in range(h):
        w = np.array(hist[-L:])
        pred = float(mdl.predict((w - w.mean())[None, :])[0] + w.mean())
        out.append(pred)
        hist.append(pred)
    return np.array(out)


def ridge_lags(y, h, m):
    return _lag_forecast(y, h, m, lambda: Ridge(alpha=1.0))


def gbm_lags(y, h, m):
    return _lag_forecast(y, h, m, lambda: HistGradientBoostingRegressor(max_iter=80, learning_rate=0.1, max_depth=3, random_state=0))


FORECASTERS = {"mean": mean_f, "naive": naive, "seasonal_naive": seasonal_naive, "drift": drift, "moving_average": moving_average,
               "ses": ses, "holt": holt, "holt_winters": holt_winters, "ridge_lags": ridge_lags, "gbm_lags": gbm_lags}
