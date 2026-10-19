# Time-Series Forecasting Research

**Author: Rayyan Anwar** — AI/ML Engineer | Researcher | Technology Manager | Software Developer

A leakage-free study of forecasting methods: naive and seasonal-naive benchmarks, exponential smoothing (SES, Holt, additive Holt-Winters fitted from scratch), and ridge / gradient-boosting regression on level-normalised lags. It uses **chronological rolling-origin validation**, reports MAE, RMSE, MASE and sMAPE, bootstraps the comparison against the right benchmark, and checks that prediction intervals actually cover.

> Synthetic series only. The valuable parts are the protocol and the sanity checks (for example, no model beats naive on a random walk), not the absolute numbers.

## Results (committed run, 5 seeds x 12 origins, `results/RESULTS.md`)
| Series | Best model | MASE (best) | MASE seasonal naive / naive | MAE ratio vs reference (95% CI) |
|---|---|---:|---:|---|
| daily, weekly season | Holt-Winters | 0.75 | 1.01 / 2.33 | 0.74 (0.70 to 0.78) |
| monthly, yearly season | Holt-Winters | 0.46 | 1.01 / 1.82 | 0.46 (0.43 to 0.50) |
| level shift, weekly season | Holt-Winters | 0.92 | 1.16 / 2.38 | 0.79 (0.72 to 0.88) |
| random walk | none (ridge 2.21 vs naive 2.25) | n/a | 2.25 | 0.99 (0.94 to 1.03) |
| stationary AR(1) | historical mean | 1.27 | 1.56 | 0.82 (0.70 to 0.97) |

Seasonal models help a lot; lag regression is competitive; nothing beats naive on a random walk; 80% intervals were conservative (0.84 to 0.97 coverage) for the better models. Details and caveats: [docs/EVALUATION.md](docs/EVALUATION.md).

## Implemented vs optional
| Feature | Status |
|---|---|
| 10 forecasters, rolling-origin evaluation, MASE/MAE/RMSE/sMAPE, leakage test | Implemented, tested |
| Per-horizon conformal-style intervals with out-of-sample coverage check | Implemented, tested |
| Paired bootstrap of model vs benchmark | Implemented, tested |
| FastAPI service + browser UI (synthetic or pasted series) | Implemented; API tests written but **not executed in the authoring sandbox** |
| ARIMA/ETS auto-selection, Prophet, neural forecasters, exogenous variables, real datasets | Not implemented |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/evaluate.py
uvicorn app.main:app --port 8019
```
`curl -s localhost:8019/api/forecast -H 'content-type: application/json' -d '{"kind":"monthly_yearly","model":"holt_winters"}'`

Endpoints: `GET /health`, `/api/info`, `/api/results`; `POST /api/forecast`; docs at `/docs`.

## Documentation
[Architecture](docs/ARCHITECTURE.md) · [Evaluation](docs/EVALUATION.md) · [Model card](docs/MODEL_CARD.md) · [Deployment](docs/DEPLOYMENT.md)

## Testing status
17 tests executed and passing (`python -m unittest`); 2 API tests skipped in the authoring sandbox (FastAPI/httpx unavailable). Docker not built; Ruff not run; UI JavaScript syntax-checked only.

## Known limitations
Synthetic data; fixed untuned hyper-parameters; additive seasonality only; few origins make intervals approximate; no exogenous variables or external baselines; no authentication.

## GitHub repository metadata
- **Description:** Leakage-free rolling-origin forecasting study: naive, ETS (from scratch) and lag-regression models with MASE, bootstrap comparisons and conformal intervals.
- **Topics:** `time-series`, `forecasting`, `holt-winters`, `conformal-prediction`, `backtesting`, `reproducible-research`, `fastapi`, `python`

## License
MIT © Rayyan Anwar. See [LICENSE](LICENSE).
