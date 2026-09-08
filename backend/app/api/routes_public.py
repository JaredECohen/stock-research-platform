"""Public samples + analytics intake (FEAT-002) — STUB.

Slice S3 fills this module with `GET /api/public/samples`,
`GET /api/public/samples/{ticker}` and `POST /api/public/events`. It
exists now, empty, so `main.py` can include the router unconditionally
and `import app.main` does not depend on which slice merged first.

Contract for S3 (from the plan, §3.1): reads `public_samples` rows only;
imports nothing from `app.agents`, `app.providers` or
`app.services.data_service` — anonymous traffic must never be able to
cause LLM or provider spend. The policy table in `auth/policy.py`
already classifies all three paths as public.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
