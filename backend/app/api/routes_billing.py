"""Stripe billing (FEAT-002) — STUB.

Slice S4 fills this module with `POST /api/billing/checkout`, `/portal`,
`/reconcile`, the signature-verified `POST /api/billing/webhook`
(`@limiter.exempt`; classified public in `auth/policy.py` because the
signature, not a session, authenticates it) and the admin routes under
`/api/admin/billing/*` (covered by `admin_auth` by prefix). It exists
now, empty, so `main.py` can include the router unconditionally.

Contract for S4: `services/stripe_client.py` is a thin httpx client
(no SDK); a checkout redirect is never proof of payment — only a
webhook or an authenticated reconcile writes `subscriptions`, and
webhook processing never writes `users.trial_started_at/trial_ends_at`.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
