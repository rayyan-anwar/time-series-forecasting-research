"""Seeded synthetic series with known structure (trend, seasonality, noise, level shift, random walk, AR(1)).

SYNTHETIC DATA ONLY. Real series have holidays, structural breaks and revisions that these do not capture.
Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KINDS = {  # kind -> (default length, season length m, default horizon, frequency)
    "daily_weekly": (420, 7, 14, "D"),
    "monthly_yearly": (120, 12, 12, "MS"),
    "random_walk": (400, 1, 10, "D"),
    "ar1": (400, 1, 10, "D"),
    "level_shift": (420, 7, 14, "D"),
}
WEEK = np.array([0, -8, -6, -4, 2, 14, 20], dtype=float)


def generate_series(kind: str, n: int | None = None, seed: int = 0) -> tuple[pd.Series, int]:
    if kind not in KINDS:
        raise ValueError(f"unknown kind '{kind}'; choose from {list(KINDS)}")
    n0, m, _, freq = KINDS[kind]
    n = n or n0
    if not 40 <= n <= 5000:
        raise ValueError("n must be in [40, 5000]")
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if kind == "daily_weekly":
        y = 100 + 0.15 * t + WEEK[t % 7] + rng.normal(0, 5, n)
    elif kind == "monthly_yearly":
        y = 200 + 1.2 * t + 30 * np.sin(2 * np.pi * t / 12) + 10 * np.cos(4 * np.pi * t / 12) + rng.normal(0, 8, n)
    elif kind == "random_walk":
        y = 50 + np.cumsum(rng.normal(0, 1, n))
    elif kind == "ar1":
        e = rng.normal(0, 5, n)
        y = np.empty(n)
        y[0] = 100
        for i in range(1, n):
            y[i] = 100 + 0.7 * (y[i - 1] - 100) + e[i]
    else:  # level_shift
        y = 100 + WEEK[t % 7] + 40 * (t >= int(0.7 * n)) + rng.normal(0, 5, n)
    start = "2015-01-01" if freq == "MS" else "2023-01-01"
    return pd.Series(y, index=pd.date_range(start, periods=n, freq=freq), name=kind), m
