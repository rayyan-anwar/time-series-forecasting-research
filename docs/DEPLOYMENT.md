# Deployment

Author: Rayyan Anwar

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/evaluate.py          # about 15 s
uvicorn app.main:app --port 8019    # http://127.0.0.1:8019
```
## Docker
`docker compose up --build` (port 8019, health `/health`). Written but **not built or run** in the authoring sandbox; CI has a `docker build` job.
## Cloud
Any container host: build and push (your account), expose 8019 behind HTTPS, probe `/health`. Stateless, no secrets or storage. Requests are bounded (at most 2,000 values, horizon 60) but unauthenticated; add platform access control if public. Not deployed by the author.
## Screenshots
Forecast the monthly series with Holt-Winters and view the committed MASE table; capture into `docs/screenshots/` (none bundled).
