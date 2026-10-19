# Time-Series Forecasting Research

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F172A,50:2563EB,100:06B6D4&height=220&section=header&text=Time-Series%20Forecasting&fontSize=38&fontColor=FFFFFF&animation=fadeIn&fontAlignY=36&desc=Leakage-Free%20Evaluation%20%7C%20Reproducible%20Research&descAlignY=58&descSize=15" width="100%" alt="Animated Time-Series Forecasting banner"/>
</p>

<p align="center">
  <strong>Rayyan Anwar</strong><br/>
  AI/ML Engineer | Researcher | Technology Manager | Software Developer
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Forecasting-Time%20Series-2563EB?style=for-the-badge" alt="Time series forecasting"/>
  <img src="https://img.shields.io/badge/Evaluation-Rolling%20Origin-0891B2?style=for-the-badge" alt="Rolling-origin evaluation"/>
  <img src="https://img.shields.io/badge/Research-Reproducible-16A34A?style=for-the-badge" alt="Reproducible research"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT license"/>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-research-results">Results</a> •
  <a href="#-methodology">Methodology</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

## 🔬 Overview

A reproducible research project investigating how classical statistical forecasting and machine-learning methods perform under **chronological, leakage-free evaluation**.

The study compares naive and seasonal-naive benchmarks, exponential-smoothing models implemented from scratch, and lag-based regression models. It emphasizes sound validation, meaningful baselines, uncertainty estimation, and sanity checks rather than impressive-looking metrics alone.

> **Research principle:** Synthetic series only. The value of this project lies in its evaluation protocol and diagnostic checks—not in claiming that synthetic-data results automatically generalize to real-world forecasting.

### Research objectives

- Compare statistical and machine-learning forecasters against appropriate baselines.
- Prevent future information from leaking into training or model selection.
- Evaluate performance using multiple complementary metrics.
- Quantify model-versus-benchmark differences with paired bootstrap confidence intervals.
- Check whether nominal prediction-interval coverage matches observed coverage.
- Test expected behavior on random walks, seasonal series, level shifts, and stationary processes.

## 📊 Research Results

<sub>Committed experiment: 5 seeds × 12 rolling origins. See <code>results/RESULTS.md</code> and <code>docs/EVALUATION.md</code> for definitions, aggregation details, and caveats.</sub>

| Synthetic series | Best model | Best MASE | Seasonal naive MASE | Naive MASE | MAE ratio vs. reference (95% CI) |
|---|---|---:|---:|---:|---|
| Daily, weekly seasonality | Holt-Winters | 0.75 | 1.01 | 2.33 | 0.74 (0.70–0.78) |
| Monthly, yearly seasonality | Holt-Winters | 0.46 | 1.01 | 1.82 | 0.46 (0.43–0.50) |
| Level shift, weekly seasonality | Holt-Winters | 0.92 | 1.16 | 2.38 | 0.79 (0.72–0.88) |
| Random walk | None; ridge is close to naive | N/A | N/A | 2.25 | 0.99 (0.94–1.03) |
| Stationary AR(1) | Historical mean | 1.27 | 1.56 | Not reported | 0.82 (0.70–0.97) |

### What the results suggest

- **Seasonality matters:** Holt-Winters substantially improves on naive forecasting for the tested seasonal series.
- **Benchmarks matter:** Seasonal-naive forecasting is a strong reference when recurring seasonal patterns exist.
- **Random walks resist prediction:** Ridge achieves MASE 2.21 versus 2.25 for naive, with a confidence interval for the MAE ratio that includes 1. This does not establish a reliable improvement over naive.
- **No universal winner:** Performance depends on the data-generating process and the benchmark used.
- **Intervals need validation:** The better-performing models achieved observed coverage of approximately 0.84–0.97 for nominal 80% prediction intervals. These intervals were generally conservative in the committed experiment.

<p align="center">
  <img src="https://img.shields.io/badge/Random%20seeds-5-2563EB?style=flat-square" alt="Five random seeds"/>
  <img src="https://img.shields.io/badge/Rolling%20origins-12-0891B2?style=flat-square" alt="Twelve rolling origins"/>
  <img src="https://img.shields.io/badge/Benchmark-Required-16A34A?style=flat-square" alt="Benchmark required"/>
  <img src="https://img.shields.io/badge/Intervals-Coverage%20Checked-7C3AED?style=flat-square" alt="Interval coverage checked"/>
</p>

## 🧠 Methodology

### Forecasting models

The research evaluates 10 forecasters, including the following model families:

| Model family | Methods | Purpose |
|---|---|---|
| Baselines | Naive, seasonal naive | Establish reference performance |
| Exponential smoothing | SES, Holt, additive Holt-Winters | Model level, trend, and seasonality |
| Lag regression | Ridge regression, gradient boosting | Learn from level-normalized historical lags |
| Additional forecasters | See implementation and evaluation documentation | Complete the 10-model comparison |

The historical-mean model is also evaluated where appropriate. Consult the committed implementation and results for the exact model inventory and each series' eligible comparisons.

### Leakage-free evaluation

1. Generate or load a time series.
2. Split observations chronologically at each rolling origin.
3. Fit models using training observations only.
4. Select models without accessing the corresponding future test observations.
5. Produce forecasts and prediction intervals for the evaluation horizon.
6. Compare forecasts against the appropriate naive or seasonal-naive benchmark.
7. Aggregate results across origins and seeds.
8. Use paired bootstrap comparisons and empirical interval-coverage checks.

### Evaluation metrics

- **MAE:** Mean absolute forecast error.
- **RMSE:** Root mean squared error, which penalizes larger errors more heavily.
- **MASE:** Mean absolute scaled error relative to an in-sample naive or seasonal-naive scale.
- **sMAPE:** Symmetric mean absolute percentage error.
- **MAE ratio:** Model error divided by reference-model error; values below 1 indicate lower error.
- **Bootstrap confidence intervals:** Quantify uncertainty in paired model-versus-reference comparisons.
- **Interval coverage:** Measures how often observations fall inside their prediction intervals.

Metric definitions, reference selection, and aggregation procedures should be read alongside the committed evaluation report.

## 🧪 Implemented vs. Optional

| Capability | Status |
|---|---|
| Ten forecasting methods | Implemented |
| Chronological rolling-origin validation | Implemented and tested |
| MAE, RMSE, MASE, and sMAPE | Implemented |
| Leakage checks | Implemented and tested |
| Per-horizon conformal-style prediction intervals | Implemented and tested |
| Out-of-sample interval-coverage evaluation | Implemented and tested |
| Paired bootstrap model-versus-benchmark comparison | Implemented and tested |
| FastAPI service and browser interface | Implemented |
| Synthetic and pasted time-series workflows | Implemented |
| API tests | Written, but not executed in the authoring sandbox |
| Automatic ARIMA/ETS selection | Not implemented |
| Prophet and neural forecasting models | Not implemented |
| Exogenous variables and external datasets | Not implemented |

## 🖥️ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/rayyan-anwar/time-series-forecasting-research.git
cd time-series-forecasting-research
```

If your actual GitHub repository uses a different name, replace the repository URL accordingly.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux/macOS**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install development dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 4. Run the tests

```bash
python -m pytest -q
```

### 5. Reproduce the evaluation

```bash
python scripts/evaluate.py
```

Review the generated or committed evaluation artifacts in `results/`.

### 6. Launch the application

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8019
```

Open the browser at:

- Dashboard: http://127.0.0.1:8019
- API documentation: http://127.0.0.1:8019/docs
- Health check: http://127.0.0.1:8019/health

## 🔌 API Example

Generate a forecast for a monthly series with yearly seasonality:

```bash
curl -s http://127.0.0.1:8019/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"kind":"monthly_yearly","model":"holt_winters"}'
```

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/api/info` | Project and model information |
| `GET` | `/api/results` | Access committed research results |
| `POST` | `/api/forecast` | Generate a forecast |
| `GET` | `/docs` | Interactive API documentation |

## 📚 Documentation

- [Architecture](docs/ARCHITECTURE.md) — application design and components.
- [Evaluation](docs/EVALUATION.md) — experimental protocol, metrics, comparisons, and caveats.
- [Model Card](docs/MODEL_CARD.md) — intended use, limitations, and responsible interpretation.
- [Deployment](docs/DEPLOYMENT.md) — environment and deployment instructions.
- [Committed Results](results/RESULTS.md) — experiment results and reported metrics.

## ✅ Testing and Verification

Reported project status:

- **17 tests** executed and passing with `python -m unittest`.
- **2 API tests** skipped in the authoring sandbox because FastAPI/httpx were unavailable there.
- Docker image build: not performed.
- Ruff linting: not performed.
- Browser-interface JavaScript: syntax-checked only.

These checks have different levels of assurance. Passing unit tests does not mean every deployment path, API integration, or browser interaction has been fully verified.

## ⚠️ Limitations

- Experiments use synthetic data, not real business or scientific datasets.
- Hyperparameters are fixed and not comprehensively tuned.
- Seasonal modeling is additive; multiplicative seasonality is not implemented.
- A small number of rolling origins limits the precision of uncertainty estimates.
- Bootstrap confidence intervals depend on the pairing and resampling protocol.
- Conformal-style intervals are approximate and require more validation across different data-generating processes.
- External regressors, holidays, promotions, and structural drivers are not modeled.
- Results should not be interpreted as proof that one model will outperform another on unseen real-world datasets.

## 🗺️ Roadmap

- [ ] Evaluate on public real-world datasets.
- [ ] Add ARIMA and automatic ETS selection.
- [ ] Add optional Prophet and neural forecasting baselines.
- [ ] Support external regressors and holiday effects.
- [ ] Expand coverage calibration experiments across more origins and horizons.
- [ ] Add automated linting and continuous integration checks.
- [ ] Build and verify a reproducible Docker deployment.
- [ ] Execute API integration tests in a fully provisioned environment.

## 👨‍💻 Author

**Rayyan Anwar**  
AI/ML Engineer | Researcher | Technology Manager | Software Developer

- GitHub: [@rayyan-anwar](https://github.com/rayyan-anwar)
- Project: [Time-Series Forecasting Research](https://github.com/rayyan-anwar/time-series-forecasting-research)

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE).

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,50:2563EB,100:0F172A&height=120&section=footer&animation=fadeIn" width="100%" alt="Decorative animated footer"/>
  <sub>Built for transparent, reproducible, and scientifically grounded forecasting research.</sub>
</p>
