"""Forecast a user-supplied or synthetic series with calibrated (conformal) intervals. Author: Rayyan Anwar"""
from __future__ import annotations

import numpy as np

from .data import KINDS, generate_series
from .evaluation import conformal_quantiles, metrics, origins, mase_scale, rolling_forecasts
from .models import FORECASTERS, ModelUnavailable


class InputError(ValueError):
    """Invalid user input."""


def forecast_series(y, m: int, h: int, model: str, level: float = 0.8) -> dict:
    y = np.asarray(y, float)
    if model not in FORECASTERS:
        raise InputError(f"model must be one of {list(FORECASTERS)}")
    if not 1 <= h <= 60:
        raise InputError("horizon must be in [1, 60]")
    if not 1 <= m <= 365:
        raise InputError("season must be in [1, 365]")
    if not 0.5 <= level <= 0.95:
        raise InputError("level must be in [0.5, 0.95]")
    if len(y) < max(30, h + 12) or not np.isfinite(y).all():
        raise InputError("need at least max(30, horizon + 12) finite values")
    try:
        fc = FORECASTERS[model](y, h, m)
    except ModelUnavailable as exc:
        raise InputError(f"{model} is unavailable for this series: {exc}") from None
    out = {"model": model, "season": m, "horizon": h, "forecast": fc.round(6).tolist(), "level": level}
    try:
        orgs = origins(len(y), h, 6, max(2 * m + 2, 20, len(y) // 2))
        f, a = rolling_forecasts(y, m, h, orgs, model)
        bn, an = rolling_forecasts(y, m, h, orgs, "seasonal_naive" if m > 1 and len(y) > 2 * m else "naive")
        scale = mase_scale(y[:orgs[0]], m)
        q = conformal_quantiles(np.abs(a - f), level)
        out["lower"], out["upper"] = (fc - q).round(6).tolist(), (fc + q).round(6).tolist()
        out["backtest"] = {**metrics(f, a, scale), "origins": len(orgs),
                           "reference_mase": metrics(bn, an, scale)["mase"],
                           "note": "MASE below the reference means better than seasonal naive (naive if no season) on past origins."}
        out["interval_note"] = f"Conformal-style {level:.0%} band from {len(orgs)} past forecast origins; coverage is approximate with so few origins."
    except (ValueError, ModelUnavailable):
        out["lower"] = out["upper"] = None
        out["interval_note"] = "Too little history to calibrate an interval."
    return out


def forecast_synthetic(kind: str, seed: int, h: int | None, model: str, level: float = 0.8) -> dict:
    if kind not in KINDS:
        raise InputError(f"kind must be one of {list(KINDS)}")
    y, m = generate_series(kind, seed=seed)
    h = h or KINDS[kind][2]
    r = forecast_series(y.values, m, h, model, level)
    freq = KINDS[kind][3]
    import pandas as pd
    future = pd.date_range(y.index[-1], periods=h + 1, freq=freq)[1:]
    r["history"] = {"dates": [d.strftime("%Y-%m-%d") for d in y.index[-120:]], "values": y.values[-120:].round(4).tolist()}
    r["forecast_dates"] = [d.strftime("%Y-%m-%d") for d in future]
    return r
