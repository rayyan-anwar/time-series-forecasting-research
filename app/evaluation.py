"""Rolling-origin (expanding window) evaluation with MAE/RMSE/MASE/sMAPE, conformal-style intervals and paired bootstrap.

Every forecast at origin t uses only y[:t]. Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np

from .models import FORECASTERS, ModelUnavailable


def origins(n: int, h: int, n_origins: int, min_train: int) -> list[int]:
    last = n - h
    if last < min_train or n_origins < 1:
        raise ValueError("series too short for the requested horizon and training size")
    k = min(n_origins, last - min_train + 1)
    return sorted({int(x) for x in np.linspace(min_train, last, k).round()})


def rolling_forecasts(y, m: int, h: int, orgs: list[int], model: str):
    """Returns (forecasts[origins, h], actuals[origins, h]) or raises ModelUnavailable."""
    f = FORECASTERS[model]
    fc, ac = [], []
    for t in orgs:
        fc.append(f(y[:t], h, m))
        ac.append(y[t:t + h])
    return np.array(fc), np.array(ac)


def mase_scale(y_train, m: int) -> float:
    lag = m if m > 1 and len(y_train) > m else 1
    d = np.abs(y_train[lag:] - y_train[:-lag])
    return float(d.mean()) if len(d) and d.mean() > 0 else 1.0


def metrics(fc, ac, scale: float) -> dict:
    e = ac - fc
    denom = (np.abs(ac) + np.abs(fc)) / 2
    smape = float(np.mean(np.where(denom > 0, np.abs(e) / np.where(denom > 0, denom, 1), 0)) * 100)
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "mase": float(np.abs(e).mean() / scale),
            "smape_pct": smape, "bias": float(e.mean())}


def conformal_quantiles(abs_err: np.ndarray, level: float = 0.8) -> np.ndarray:
    """Per-horizon split-conformal quantile of absolute errors from calibration origins (shape origins x h)."""
    n = abs_err.shape[0]
    k = min(n, int(np.ceil((n + 1) * level)))
    return np.sort(abs_err, axis=0)[k - 1]


def interval_coverage(fc, ac, level: float = 0.8) -> dict:
    """Calibrate on the first half of origins, test coverage on the second half."""
    n = fc.shape[0]
    if n < 4:
        return {"available": False}
    cal, test = slice(0, n // 2), slice(n // 2, n)
    q = conformal_quantiles(np.abs(ac[cal] - fc[cal]), level)
    hit = np.abs(ac[test] - fc[test]) <= q
    return {"available": True, "nominal": level, "empirical_coverage": float(hit.mean()), "calibration_origins": int(n // 2),
            "test_origins": int(n - n // 2), "mean_half_width": float(q.mean())}


def paired_bootstrap_ratio(err_a, err_b, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Ratio of mean absolute error of A to B, resampling forecast origins (rows). Returns (point, lo, hi)."""
    a, b = np.abs(err_a).mean(axis=1), np.abs(err_b).mean(axis=1)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        vals.append(a[i].mean() / max(b[i].mean(), 1e-12))
    return float(a.mean() / max(b.mean(), 1e-12)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def evaluate_series(y, m: int, h: int, models=None, n_origins: int = 10, min_train: int | None = None) -> dict:
    y = np.asarray(y, float)
    min_train = min_train or max(3 * m, 30, len(y) // 2)
    orgs = origins(len(y), h, n_origins, min_train)
    scale = mase_scale(y[:orgs[0]], m)
    out, errs = {}, {}
    for name in models or FORECASTERS:
        try:
            fc, ac = rolling_forecasts(y, m, h, orgs, name)
        except ModelUnavailable as exc:
            out[name] = {"available": False, "reason": str(exc)}
            continue
        out[name] = {"available": True, **metrics(fc, ac, scale), "by_horizon_mae": np.abs(ac - fc).mean(axis=0).tolist(),
                     "interval80": interval_coverage(fc, ac, 0.8)}
        errs[name] = ac - fc
    return {"origins": orgs, "horizon": h, "season": m, "mase_scale": scale, "models": out, "_errors": errs}
