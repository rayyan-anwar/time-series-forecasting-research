# Evaluation

Author: Rayyan Anwar

## Research questions
1. When do seasonal or trend-aware statistical models and lag regression beat naive benchmarks, and by how much?
2. Do any models beat the naive forecast on a pure random walk (they should not)?
3. Are conformal-style intervals calibrated?

## Committed run (5 seeds x 12 rolling origins per series kind; `results/RESULTS.md`)
MASE, mean over seeds (1.0 = seasonal naive; naive for series without a season):

| Series | seasonal naive | naive | Holt-Winters | ridge lags | GBM lags | best |
|---|---:|---:|---:|---:|---:|---|
| daily_weekly | 1.01 | 2.33 | **0.75** | 0.82 | 0.80 | Holt-Winters |
| monthly_yearly | 1.01 | 1.82 | **0.46** | 0.55 | 0.55 | Holt-Winters |
| level_shift | 1.16 | 2.38 | **0.92** | 1.12 | 1.05 | Holt-Winters |
| random_walk | 2.25 (= naive) | 2.25 | n/a | 2.21 | 2.42 | none |
| ar1 | 1.56 (= naive) | 1.56 | n/a | 1.40 | 1.50 | historical mean (1.27) |

## Findings
- On seasonal series Holt-Winters cut MAE by 21% to 54% versus seasonal naive; paired bootstrap ratios: 0.74 (0.70 to 0.78) daily, 0.46 (0.43 to 0.50) monthly, 0.79 (0.72 to 0.88) level shift. The lag-regression models are close behind; GBM and ridge are within about 20% of Holt-Winters (7% to 9% on the daily series).
- **On the random walk nothing beats naive**: the nominally best model (ridge lags, ratio 0.99, CI 0.94 to 1.03) is statistically indistinguishable from naive. This is the expected sanity check and guards against over-reading small gains.
- On the stationary AR(1) series the historical mean wins (MAE ratio 0.82 vs naive, CI 0.70 to 0.97), because forecasts should revert to the mean; naive-type methods carry noise forward.
- Plain Holt (linear trend) is unstable on seasonal series (MASE 2.3 to 2.7 with large variance) and moving-average/SES ignore seasonality, so they trail the seasonal baseline.
- Leakage check passed: corrupting all values after each origin leaves every model's forecasts unchanged.
- 80% intervals: out-of-sample coverage was 0.84 to 0.97 for the better models (conservative, because calibration uses only about 6 origins and a finite-sample correction) and 0.75 to 0.81 for naive-type models on AR(1). The mean benchmark under-covers (0.57 to 0.77) on non-stationary series (trend, level shift, random walk), as it should.

## Caveats
- Synthetic series with simple planted structure favour additive ETS; real data have holidays, outliers, changing seasonality and structural breaks.
- Five seeds and 12 overlapping origins give noisy estimates; bootstrap intervals resample origins only, not series.
- Hyper-parameters (lag length, GBM size) were fixed, not tuned; no ARIMA/Prophet/neural baselines were run, so no claim covers them.
- Forecast accuracy on synthetic data says nothing about the future of any real business.
