# Multi-stage Dockerfile: builds the React frontend and serves it from FastAPI.
#
# Stage 1: Build the frontend with Node 20.
# Stage 2: Install the Python backend, copy the static frontend, and run uvicorn.

# ---------- Stage 1 ----------
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2 ----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# System deps (libpq-dev only needed if running against Postgres with psycopg)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY example.env /app/example.env
COPY --from=frontend-build /app/frontend/dist /app/frontend_dist

# A simple wrapper that mounts the static dist at "/" alongside the API.
COPY <<'PY' /app/server.py
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import sys
sys.path.insert(0, "/app/backend")
from app.main import app as api_app  # type: ignore

DIST = Path("/app/frontend_dist")
if DIST.exists():
    api_app.mount("/", StaticFiles(directory=str(DIST), html=True), name="ui")

app = api_app
PY

ENV BACKEND_PORT=8000 BACKEND_HOST=0.0.0.0
EXPOSE 8000
CMD ["uvicorn", "server:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8000"]
