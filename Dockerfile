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
    # SPA deep-link fallback. `StaticFiles(html=True)` serves index.html for a
    # DIRECTORY request ("/"), but a client-side route like "/track-record"
    # looks for a FILE of that name, doesn't find one, and 404s with FastAPI's
    # {"detail":"Not Found"}. In-app navigation worked (React Router never hit
    # the server), so this only broke the cases nobody clicks through to:
    # typing a URL, refreshing, opening a bookmark, following a shared link.
    #
    # Starlette matches in registration order, and this is registered before
    # the StaticFiles mount, so it takes precedence for GET — which is why it
    # serves real files itself rather than delegating. The mount stays for
    # other methods (HEAD, and a 405 on anything else).
    #
    # It deliberately excludes the API prefixes: a bad /api/... path must keep
    # returning a JSON 404 rather than 200 with an HTML body, which would turn
    # every client-side fetch typo into an unparseable response.
    #
    # `is_relative_to` is the containment check — without it, "/../../etc/passwd"
    # would resolve outside the dist directory and serve arbitrary files.
    from fastapi.responses import FileResponse, JSONResponse

    _INDEX = DIST / "index.html"
    _API_PREFIXES = ("api/", "health", "docs", "redoc", "openapi.json")

    @api_app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str):
        if full_path.startswith(_API_PREFIXES):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (DIST / full_path).resolve()
        # Serve a real asset when one exists; containment check keeps
        # "../"-style paths from escaping the dist directory.
        if full_path and candidate.is_file() and candidate.is_relative_to(DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(_INDEX)

    api_app.mount("/", StaticFiles(directory=str(DIST), html=True), name="ui")

app = api_app
PY

ENV BACKEND_PORT=8000 BACKEND_HOST=0.0.0.0

# Cap glibc's per-thread arena count. This process runs uvicorn plus, on
# the worker service, the regen thread and a 10-thread APScheduler pool;
# glibc would otherwise open up to 8 arenas per core, each retaining its
# own high-water mark, so one transient spike permanently inflates RSS.
# Render enforces its memory limit on RSS, so that inflation is the
# difference between a restart and no restart. Set here as the image
# default; render.yaml sets it per-service too so it survives anyone
# running the image with an overridden env.
ENV MALLOC_ARENA_MAX=2

EXPOSE 8000
CMD ["uvicorn", "server:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8000"]
