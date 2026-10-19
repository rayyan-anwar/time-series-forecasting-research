# Model Card

Author: Rayyan Anwar

**Models:** historical mean, naive, seasonal naive, drift, moving average, simple exponential smoothing, Holt, additive Holt-Winters, ridge and gradient-boosting regression on level-normalised lags.

**Intended use:** learning and reproducible comparison of forecasting methods under a leakage-free rolling-origin protocol. **Not for:** financial, inventory, staffing or safety decisions without independent validation on your own data.

**Data:** seeded synthetic series generated in `app/data.py`; user-supplied numbers are processed in memory and not stored.

**Metrics:** MAE, RMSE, MASE, sMAPE, bias; interval coverage out of sample.

**Limitations and risks:** point forecasts and intervals describe the past error distribution and break under regime changes; intervals are approximate with few origins; additive seasonality only; no exogenous variables, holidays or hierarchies; no probabilistic models; forecasts can look authoritative while being wrong, so keep a human in the loop and monitor errors.
