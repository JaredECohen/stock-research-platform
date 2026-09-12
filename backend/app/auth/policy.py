"""Which routes the login wall fronts, and how.

Default-deny: anything under `/api/` that is not named here is
`authenticated`, and `test_auth_policy.py` fails the build when an
OpenAPI path is NOT named here — a new route has to say what it is
before it ships. The admin surface is classified by delegating to
`admin_auth.is_protected`, so the two guards agree by construction: an
admin path is "not the customer middleware's business" (the admin
token, and only the admin token, opens it), and a customer JWT is never
consulted for it.

Levels:
  public         no token needed (marketing, health, config, samples,
                 the Stripe webhook — verified by signature in its route)
  authenticated  any signed-in user; `feature` may still meter it
  pro            signed-in AND plan == pro (middleware 402s otherwise)
  admin          bearer ADMIN_API_TOKEN, handled by `admin_auth`

`feature` names the `auth/features.py` entry the route's
`require_feature(...)` dependency charges against (slice S2 wires those).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..api import admin_auth

PUBLIC = "public"
AUTHENTICATED = "authenticated"
PRO = "pro"
ADMIN = "admin"


@dataclass(frozen=True)
class Policy:
    level: str
    feature: str | None = None
    note: str = ""

    @property
    def is_public(self) -> bool:
        return self.level == PUBLIC


_PUBLIC = Policy(PUBLIC)
_FREE = Policy(AUTHENTICATED)


def _pro(feature: str, note: str = "") -> Policy:
    return Policy(PRO, feature, note)


def _metered(feature: str, note: str = "") -> Policy:
    return Policy(AUTHENTICATED, feature, note)


# (method, path template, policy). Templates use FastAPI's `{param}`
# syntax; a param matches one path segment. Order matters only where a
# template could shadow another — none do today.
ROUTES: tuple[tuple[str, str, Policy], ...] = (
    # --- system / marketing ------------------------------------------------
    ("GET", "/health", _PUBLIC),
    ("GET", "/api/providers/status", _PUBLIC),
    ("GET", "/docs", _PUBLIC),
    ("GET", "/docs/oauth2-redirect", _PUBLIC),
    ("GET", "/redoc", _PUBLIC),
    ("GET", "/openapi.json", _PUBLIC),
    ("GET", "/api/public/config", _PUBLIC),
    ("GET", "/api/public/samples", _PUBLIC),
    ("GET", "/api/public/samples/{ticker}", _PUBLIC),
    ("POST", "/api/public/events", _PUBLIC),
    # Stripe calls this; the route verifies the `Stripe-Signature` itself.
    ("POST", "/api/billing/webhook", Policy(PUBLIC, note="stripe-signed")),
    # Browser telemetry — exempt from the admin token for the same reason.
    ("POST", "/api/admin/ui-log", _PUBLIC),
    # --- account / billing ---------------------------------------------------
    ("GET", "/api/me", _FREE),
    ("POST", "/api/me/bootstrap", _FREE),
    ("GET", "/api/me/usage", _FREE),
    ("POST", "/api/billing/checkout", _FREE),
    ("POST", "/api/billing/portal", _FREE),
    ("POST", "/api/billing/reconcile", _FREE),
    # --- stocks --------------------------------------------------------------
    ("GET", "/api/stocks", _FREE),
    ("GET", "/api/stocks/{ticker}", _FREE),
    ("GET", "/api/stocks/{ticker}/prices", _FREE),
    ("GET", "/api/stocks/{ticker}/memo", _metered("memo_view", "3 distinct tickers/month on Free")),
    ("GET", "/api/stocks/{ticker}/memory", _pro("memo_history")),
    ("GET", "/api/stocks/{ticker}/memos", _pro("memo_history")),
    ("POST", "/api/stocks/{ticker}/analyze", _metered("research_run", "charged only when a job is created")),
    ("GET", "/api/stocks/{ticker}/analyze/status", _FREE),
    # --- screener ------------------------------------------------------------
    ("GET", "/api/screener", _FREE),
    ("POST", "/api/screener/run", _FREE),
    ("POST", "/api/screener/custom", _FREE),
    ("POST", "/api/screener/nl", _pro("pm_chat", "one LLM call; counts as a chat turn")),
    # --- chat ----------------------------------------------------------------
    ("POST", "/api/chat", _metered("pm_chat")),
    # --- dcf / comps: Free follows memo ------------------------------------
    ("GET", "/api/dcf/{ticker}/default-assumptions", _metered("dcf")),
    ("GET", "/api/dcf/{ticker}/consensus", _metered("dcf")),
    ("GET", "/api/dcf/{ticker}/saved", _metered("dcf")),
    ("POST", "/api/dcf/{ticker}", _metered("dcf")),
    ("GET", "/api/comps/{ticker}", _metered("comps")),
    # --- pro-only surfaces ---------------------------------------------------
    ("POST", "/api/portfolio/build", _pro("portfolio")),
    ("GET", "/api/macro/series", _pro("macro")),
    ("POST", "/api/macro/analyze", _pro("pm_chat", "one LLM call; counts as a chat turn")),
    ("GET", "/api/data-catalog/meta", _pro("data_catalog")),
    ("GET", "/api/data-catalog/series", _pro("data_catalog")),
    ("GET", "/api/data-catalog/series/{series_id}", _pro("data_catalog")),
    ("GET", "/api/data-catalog/ticker/{ticker}/context", _pro("data_catalog")),
    ("GET", "/api/data-catalog/ticker/{ticker}/geography", _pro("data_catalog", "allow_llm forced off for customers")),
    ("GET", "/api/data-catalog/ticker/{ticker}/overlay/{name}", _pro("data_catalog")),
    # --- fundamentals explorer (FEAT-001) ------------------------------------
    # Catalog and series are DB reads shaped by the plan inside the route
    # (`features.shape`); commentary is the one LLM call and is metered.
    ("GET", "/api/fundamentals/catalog", _FREE),
    ("POST", "/api/fundamentals/series", _FREE),
    ("POST", "/api/fundamentals/commentary", _metered("chart_commentary", "Free 5 / Pro 100 a month; 2 in flight")),
    # --- product features that live under the admin prefix ------------------
    # These are the `admin_auth.EXEMPT_PREFIXES` (browser-called, no admin
    # token). They are customer routes in everything but path.
    ("GET", "/api/admin/track-record", _pro("track_record")),
    ("POST", "/api/admin/evaluate-outcomes", _pro("track_record", "global 1/10min limit")),
    ("GET", "/api/admin/dcf-versions/{ticker}", _pro("memo_history")),
    ("GET", "/api/admin/lopsidedness-audit", _pro("track_record", "no UI caller today")),
    # --- fundamental scorecard (Phase 6) ------------------------------------
    # DB reads only; the worker computes. Listed BEFORE the `{ticker}`
    # template so `spec` / `evaluation` / `export` are not read as tickers
    # (the first matching row wins). The export row is documentation: with
    # `SCORECARD_EXPORT_TOKEN` set `lookup` classifies it public and the
    # route enforces that token itself (see `_scorecard_export_policy`).
    ("GET", "/api/scorecard", _pro("scorecard")),
    ("GET", "/api/scorecard/spec", _pro("scorecard")),
    ("GET", "/api/scorecard/evaluation", _pro("scorecard")),
    ("GET", "/api/scorecard/export", _pro("scorecard", "public when SCORECARD_EXPORT_TOKEN is set; the route checks it")),
    ("GET", "/api/scorecard/{ticker}", _pro("scorecard")),
)

SCORECARD_EXPORT_PATH = "/api/scorecard/export"

# --- FEAT-003 Industry Analysis -----------------------------------------
# These four surfaces do not have a fixed level: `latest` follows
# `INDUSTRY_ANALYSIS_ACCESS` (owner decision 1, default `public`), while
# history, changes and the cross-industry snapshot are Pro. The tiers live
# in `services/industry_report_store.access_policy()` — the same function
# the routes' own dependency and the chat tool read — so the login wall and
# the route seam cannot disagree about a surface. Resolved per call, like
# the scorecard export, because the setting is an env var and tests flip it.
#
# `/taxonomy` is always public: it carries the access policy itself, so the
# UI can explain a gate rather than render a 401.
INDUSTRY_SURFACES: tuple[tuple[str, str, str], ...] = (
    ("GET", "/api/industries/taxonomy", "latest"),
    ("GET", "/api/industries/snapshot", "pm_chat"),
    ("GET", "/api/industries/{code}/report", "latest"),
    ("GET", "/api/industries/{code}/companies", "latest"),
    ("GET", "/api/industries/{code}/history", "history"),
    ("GET", "/api/industries/{code}/changes", "changes"),
)
INDUSTRY_TAXONOMY_PATH = "/api/industries/taxonomy"
INDUSTRY_FEATURE = "industry_analysis"


def _industry_policy(surface: str, path: str) -> Policy:
    from ..services.industry_report_store import PUBLIC as TIER_PUBLIC
    from ..services.industry_report_store import surface_tier

    if path == INDUSTRY_TAXONOMY_PATH:
        return Policy(PUBLIC, note="always answers; carries the access policy so the UI can explain the gate")
    if surface_tier(surface) == TIER_PUBLIC:
        return Policy(PUBLIC, note=f"industry {surface} surface is public while INDUSTRY_ANALYSIS_ACCESS=public")
    return _pro(INDUSTRY_FEATURE, f"industry {surface} surface")


def _scorecard_export_policy() -> Policy:
    """The export has two audiences. With `SCORECARD_EXPORT_TOKEN` unset it
    is a Pro read like its siblings. With the token set it is handed to a
    downstream system that has neither a customer login nor the admin
    token, so the customer wall must let it through and the route enforces
    the bearer itself (`routes_scorecard._check_export_token`). Resolved
    per call because tests flip the setting."""
    from ..config import settings
    if settings.scorecard_export_token:
        return Policy(PUBLIC, note="SCORECARD_EXPORT_TOKEN set — the route enforces its own bearer")
    return _pro("scorecard", "Pro read; set SCORECARD_EXPORT_TOKEN for a token-gated export")


def _compile(template: str) -> re.Pattern[str]:
    pattern = re.sub(r"\{[^/}]+\}", r"[^/]+", re.escape(template).replace(r"\{", "{").replace(r"\}", "}"))
    return re.compile(f"^{pattern}$")


_COMPILED: tuple[tuple[str, re.Pattern[str], Policy], ...] = tuple(
    (method.upper(), _compile(template), policy) for method, template, policy in ROUTES
)

_INDUSTRY_COMPILED: tuple[tuple[str, re.Pattern[str], str], ...] = tuple(
    (method.upper(), _compile(template), surface) for method, template, surface in INDUSTRY_SURFACES
)

_ADMIN = Policy(ADMIN, note="admin_auth owns this prefix")
_DEFAULT_DENY = Policy(AUTHENTICATED, note="unclassified /api route — default deny")


def lookup(method: str, path: str) -> tuple[Policy, bool]:
    """(policy, explicit). `explicit` is False when the answer came from a
    fallback rule rather than the table — the coverage test wants every
    real route to be explicit; the middleware only wants the policy."""
    m = method.upper()
    if m == "OPTIONS":
        # CORS preflights carry no credentials by design and never reach a
        # handler; challenging them just breaks the browser.
        return _PUBLIC, True
    if m == "GET" and path == SCORECARD_EXPORT_PATH:
        return _scorecard_export_policy(), True
    for rm, rx, surface in _INDUSTRY_COMPILED:
        if rm == m and rx.match(path):
            return _industry_policy(surface, path), True
    for rm, rx, policy in _COMPILED:
        if rm == m and rx.match(path):
            return policy, True
    if admin_auth.is_protected(m, path):
        return _ADMIN, True
    if path == "/api" or path.startswith("/api/"):
        return _DEFAULT_DENY, False
    # Anything else is the SPA / static files (served by the Dockerfile's
    # server.py in production, never by this app), and the marketing site.
    return _PUBLIC, True


def classify(method: str, path: str) -> Policy:
    return lookup(method, path)[0]


def is_public(method: str, path: str) -> bool:
    return classify(method, path).is_public


def templated_to_concrete(path_template: str) -> str:
    """Turn an OpenAPI path template into a concrete path the matcher can
    classify: `/api/stocks/{ticker}/memo` → `/api/stocks/X/memo`."""
    return re.sub(r"\{[^/}]+\}", "X", path_template)
