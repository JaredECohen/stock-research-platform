"""Pytest config.

Two responsibilities:

1. Ensure the `backend/` directory is on `sys.path` so `app.*` imports work
   regardless of where pytest is launched from.
2. Force `ENABLE_LIVE_DATA=false` for the test session, BEFORE the app's
   `Settings` is constructed. This makes test runs deterministic and zero-
   cost even when the developer has live provider keys in their `.env`
   (or `config.env` ships with `ENABLE_LIVE_DATA=true` as the production
   default). Tests should never hit real provider APIs unless they
   explicitly opt in via mocking — this guard prevents accidental spend.
"""
import os
import sys
from pathlib import Path

# Set BEFORE any `app.config` import so pydantic-settings sees it as the
# winning value over both config.env and .env.
os.environ.setdefault("ENABLE_LIVE_DATA", "false")
os.environ.setdefault("USE_DEMO_DATA", "true")

sys.path.insert(0, str(Path(__file__).resolve().parent))
