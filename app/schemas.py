"""Pydantic request models. Author: Rayyan Anwar"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Model = Literal["mean", "naive", "seasonal_naive", "drift", "moving_average", "ses", "holt", "holt_winters", "ridge_lags", "gbm_lags"]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["daily_weekly", "monthly_yearly", "random_walk", "ar1", "level_shift"] | None = None
    seed: int = Field(0, ge=0, le=10_000)
    values: list[float] | None = Field(None, min_length=30, max_length=2000)
    season: int = Field(1, ge=1, le=365)
    horizon: int | None = Field(None, ge=1, le=60)
    model: Model = "holt_winters"
    level: float = Field(0.8, ge=0.5, le=0.95)

    @model_validator(mode="after")
    def one_source(self):
        if (self.kind is None) == (self.values is None):
            raise ValueError("provide exactly one of 'kind' or 'values'")
        if self.values is not None and self.horizon is None:
            raise ValueError("horizon is required with 'values'")
        return self
