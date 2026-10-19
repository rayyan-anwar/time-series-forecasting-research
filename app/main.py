"""FastAPI application for Time-Series Forecasting Research. Author: Rayyan Anwar"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from . import __author__, __version__
from .data import KINDS
from .models import FORECASTERS
from .schemas import ForecastRequest
from .service import InputError, forecast_series, forecast_synthetic

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Time-Series Forecasting Research", version=__version__,
              description="Benchmark, exponential-smoothing and lag-regression forecasters with rolling-origin evaluation (synthetic data). "
                          "Author: Rayyan Anwar.", contact={"name": __author__}, license_info={"name": "MIT"})


@app.exception_handler(InputError)
async def on_input_error(_: Request, exc: InputError):
    return JSONResponse(status_code=422, content={"error": {"code": "invalid_input", "message": str(exc)}})


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "author": __author__}


@app.get("/api/info")
def info():
    return {"name": "Time-Series Forecasting Research", "author": __author__, "version": __version__,
            "models": list(FORECASTERS), "synthetic_kinds": {k: {"season": v[1], "default_horizon": v[2]} for k, v in KINDS.items()}}


@app.post("/api/forecast")
async def forecast(req: ForecastRequest):
    if req.kind:
        return await run_in_threadpool(forecast_synthetic, req.kind, req.seed, req.horizon, req.model, req.level)
    return await run_in_threadpool(forecast_series, req.values, req.season, req.horizon, req.model, req.level)


@app.get("/api/results")
def results():
    p = ROOT / "results" / "results.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="no results yet; run: python scripts/evaluate.py")
    return json.loads(p.read_text())


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "frontend" / "index.html")
