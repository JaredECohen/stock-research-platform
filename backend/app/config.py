"""Application configuration loaded from environment variables.

A single Settings object is constructed at import time. Modules that need
runtime config import `settings` from here. All values are safe defaults so
the application boots even with a completely empty environment.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_env_files() -> list[str]:
    """Resolve env files in load order.

    Order matters — pydantic-settings applies LATER files on top of earlier
    ones. The intended precedence is:
        1. config.env   ← committed defaults (model assignments, feature
                          flags, runtime tuning). In git.
        2. .env         ← gitignored secrets + per-deployment overrides
                          (API keys, DATABASE_URL, …).
        3. process env  ← OS environment wins over both (handled by
                          pydantic-settings automatically).

    Both files are searched at the repo root and inside `backend/` so the
    same code works regardless of where the process is launched from.
    Missing files are silently skipped by pydantic-settings.
    """
    here = Path(__file__).resolve()
    backend_dir = here.parent.parent  # backend/
    repo_root = here.parent.parent.parent  # repo root
    return [
        # Defaults / committed config — loaded first.
        str(repo_root / "config.env"),
        str(backend_dir / "config.env"),
        # Secrets / per-developer overrides — loaded second so they win.
        str(repo_root / ".env"),
        str(backend_dir / ".env"),
    ]


class Settings(BaseSettings):
    # Every credential-bearing field is declared `repr=False`. The repr is
    # rendered by things nobody reviews for secrets: pytest prints it for
    # any failing `assert settings.<attr> ...` (the "where True =
    # Settings(...).has_llm" line), and so do `--showlocals`, `%r` logging
    # and error reporters that capture frame locals. Only the repr changes;
    # values, env loading and model_dump are untouched.
    # test_config_secret_repr fails if a new *_key / *_token / *_secret /
    # *_salt / *_password / *_dsn field, or any *_url / *_uri field it does
    # not list as public, is added without it.
    model_config = SettingsConfigDict(
        env_file=tuple(_project_env_files()),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM — provider selection + per-provider config
    llm_provider: str = "auto"  # "auto" | "openai" | "anthropic"
    openai_api_key: str = Field(default="", repr=False)
    openai_strong_model: str = "gpt-5.5"
    openai_cheap_model: str = "gpt-4.1-mini"
    # Multi-agent role assignment (Phase 3+). The PM uses the existing Chat
    # Completions route; match the model already selected in config.env even
    # when that repository-level file is absent from the deployment image.
    openai_pm_model: str = "gpt-5.5"
    openai_sector_model: str = "gpt-5.4"
    openai_tool_model: str = "gpt-5.4"
    # Macro agent: GPT-5.4 default per the architecture spec; flip to Gemini
    # by setting OPENAI_MACRO_MODEL="" + GEMINI_API_KEY in the agent code path.
    openai_macro_model: str = "gpt-5.4"
    anthropic_api_key: str = Field(default="", repr=False)
    anthropic_strong_model: str = "claude-opus-4-8"
    anthropic_cheap_model: str = "claude-haiku-4-5"
    anthropic_critic_model: str = "claude-opus-4-8"
    # Gemini (Google GenAI) — used for news/social/long-doc analysts.
    # Two access paths:
    #   - Direct API: set GEMINI_API_KEY. Quick setup; generous free tier.
    #   - Vertex AI: set VERTEX_PROJECT_ID (+ optionally VERTEX_LOCATION,
    #     VERTEX_MODEL). Auth via Google Application Default Credentials —
    #     run `gcloud auth application-default login` locally, or set
    #     GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json in prod.
    # When both are configured, Vertex wins so production deployments
    # don't accidentally fall back to API-key auth.
    gemini_api_key: str = Field(default="", repr=False)
    # gemini-2.5-flash returns 404 "no longer available to new users" for a
    # key created after its retirement (production's key, 2026-09-25), and
    # "gemini-3.1-pro" was never a valid ID: only the -preview one exists.
    gemini_news_model: str = "gemini-3.5-flash-lite"
    gemini_social_model: str = "gemini-3.5-flash-lite"
    gemini_longdoc_model: str = "gemini-3.1-pro-preview"
    vertex_project_id: str = ""
    vertex_location: str = "us-central1"
    # When set, overrides the per-agent Gemini model envs across all Gemini
    # calls (news / social / longdoc) on the Vertex backend. Leave empty to
    # let each agent use its own GEMINI_*_MODEL.
    vertex_model: str = ""
    # Bounded provider failover (openai <-> anthropic). When the active
    # provider's breaker is open or a call fails, `llm.chat_json` /
    # `chat_text` try the *other* configured provider exactly once, on
    # that provider's own route default. One hop, no retry loop — the
    # point is to keep a memo run alive through a single-vendor outage,
    # not to hide a misconfiguration. Gemini is a specialist path and
    # never participates. `cooldown_seconds` is how long a failover keeps
    # `/api/providers/status` reporting `degraded`.
    llm_failover_enabled: bool = True
    llm_failover_cooldown_seconds: float = 600.0
    # Anthropic prompt caching. `_anthropic_chat` sends the system prompt as
    # a `cache_control` block and, where a call site declares a static prefix
    # via `llm_call_context(static_prefix_chars=…)`, splits the user content
    # at that boundary so the repeated part is read at ~0.1x. The prompt
    # bytes the model sees are identical either way; the flag exists so a
    # bad cache interaction can be switched off on Render without a deploy.
    llm_prompt_caching_enabled: bool = True
    # LLM attribution (owner, 2026-09-25). The guard on every LLM entry
    # point: strict raises on a call that names no registered action, warn
    # logs one WARNING per call site, off records nothing. APP_ENV=production
    # always runs warn (a strict raise inside the memo pipeline is swallowed
    # by safe_call and would silently turn every memo into stub findings;
    # attribution critique #10).
    llm_attribution_mode: str = "warn"
    # Kill switch for the per-call `app.llm.calls` INFO lines on success
    # only. Error/skip/failover/breaker WARNINGs and the DB rows stay.
    llm_call_log_enabled: bool = True

    @field_validator("llm_attribution_mode")
    @classmethod
    def _attribution_mode_known(cls, v: str) -> str:
        mode = str(v).strip().lower()
        if mode not in ("strict", "warn", "off"):
            raise ValueError("llm_attribution_mode must be 'strict', 'warn' or 'off'")
        return mode

    # Database
    database_url: str = Field(default="sqlite:///./marketmosaic.db", repr=False)

    # Feature flags
    use_demo_data: bool = True
    enable_live_data: bool = False
    enable_agent_critic: bool = True
    # Reported on /api/providers/status, but NO retrieval path reads it: the
    # filing and earnings analysts query `vector_store.search` first whenever
    # the ticker is known, and the filing analyst falls back to BM25
    # (`retrieval_service.search`) when that returns nothing, fails, or is
    # skipped for lack of a ticker. `false` therefore does not
    # mean semantic retrieval is off; a production audit read it that way.
    # Whether a real switch should exist is a retrieval-policy decision for
    # the owner.
    enable_vector_search: bool = False
    # Phase 3: route single_stock_analysis through the OpenAI Agents SDK
    # instead of the legacy hand-rolled graph. Default off so existing tests
    # keep using the deterministic legacy path.
    use_agents_sdk: bool = False
    # Phase 5: always-on monitoring loops (EDGAR, news, social, macro). Default
    # off in dev/test; flip on in prod via env.
    # Render injects RENDER_GIT_COMMIT into every service; surfacing it on
    # /health makes a deploy verifiable from outside (the deploy-visible canary).
    render_git_commit: str = ""
    enable_monitoring: bool = False
    # Theme 5: DB-backed memo-regen queue. The worker thread that drains
    # `regen_jobs` starts with the app; disable to run an API-only
    # replica that enqueues but never executes (e.g., when a dedicated
    # worker service owns execution). Poll interval is how long the
    # worker sleeps when the queue is empty; max queue age bounds how
    # stale a queued-but-never-claimed job can be before startup
    # recovery expires it instead of running it (guards against a
    # backlog of forgotten jobs burning LLM spend after downtime).
    enable_regen_worker: bool = True
    regen_worker_poll_seconds: float = 2.0
    regen_queue_max_age_minutes: int = 30
    # Long-term agent memory: the LEGACY file backend (filesystem markdown,
    # delta-triggered). `memory_dir` is the root; companies live at
    # <root>/companies/<TICKER>.md and sectors at <root>/sectors/<sector_slug>.md.
    # Render's container disk is ephemeral, so production keeps this false
    # (render.yaml); the Postgres learning ledger (W7, `learning_*` below)
    # replaces it there rather than reusing this flag with a new meaning —
    # ten call sites read it as "touch the markdown files".
    enable_long_term_memory: bool = True
    memory_dir: str = "./memory"
    # W7 learning ledger (owner decision 9, 2026-09-24): lessons are testable
    # hypotheses whose credibility is a Beta posterior over later, eligible
    # outcomes, so memory updates priors as evidence arrives instead of
    # being a fixed instruction. Off by default (CI, dev laptops): writes
    # happen only where render.yaml turns them on.
    learning_ledger_writes: bool = False
    # A CEILING on how far learned priors may reach prompts: off | shadow |
    # inject. The effective mode is min(this, the DB-held mode), and the DB
    # mode defaults to shadow (computed and audited, never injected), so
    # "inject" here only permits a later, gated admin promotion. "off" is
    # the emergency stop: no DB read, prompts byte-identical to today. A str
    # with a validator for the same reason as `rating_reconciliation_mode`.
    learning_mode_max: str = "off"
    # The nightly applicability judge (cheap route) spends at most this much
    # per night. Owner default adopted 2026-09-24: <= 20 calls, <= $0.25.
    learning_judge_max_calls_per_night: int = 20
    learning_judge_max_usd_per_night: float = 0.25

    @field_validator("learning_mode_max")
    @classmethod
    def _learning_mode_max_known(cls, v: str) -> str:
        mode = str(v).strip().lower()
        if mode not in ("off", "shadow", "inject"):
            raise ValueError("learning_mode_max must be 'off', 'shadow' or 'inject'")
        return mode
    # Wave 3C / 8A: drill-down "long-form" agent reports. The deterministic
    # markdown is always populated (cheap, no LLM); when this flag is on,
    # the deterministic body is enriched via a 1-2 paragraph LLM expansion
    # per agent. Default-on per MASTER_PLAN §6 (decision logged): the
    # ~$0.02-0.04 per-memo token cost is small relative to the user-visible
    # value; toggle off via env if smoke runtime spikes.
    enable_long_form_reports: bool = True
    # Wave 9: deep-research iterative PM↔sector dialog. After the parallel
    # fan-out, the PM critique step asks 0-3 follow-up questions and has the
    # targeted specialists re-run with the question as additional prompt
    # context. Hard cap on rounds + questions per round bounds cost.
    # Default-on — Wave 10 flipped this from off → on per
    # PRODUCT_DESIGN_REVIEW.md §1.2. Capped at 1 round / 2 questions
    # to keep memo cost increase to ~15-25%; raise the cap if the
    # platform tier supports it.
    enable_deep_research: bool = True
    deep_research_max_rounds: int = 1
    deep_research_max_questions_per_round: int = 2
    # When entry count crosses this cap, the oldest entries are condensed
    # into a "Historical context" block rather than discarded outright.
    memory_max_entries: int = 50
    memory_condense_batch: int = 10

    # Providers
    fmp_api_key: str = Field(default="", repr=False)
    alpha_vantage_api_key: str = Field(default="", repr=False)
    fred_api_key: str = Field(default="", repr=False)
    polygon_api_key: str = Field(default="", repr=False)
    tiingo_api_key: str = Field(default="", repr=False)
    finnhub_api_key: str = Field(default="", repr=False)
    intrinio_api_key: str = Field(default="", repr=False)
    nasdaq_data_link_api_key: str = Field(default="", repr=False)
    # Sector-overlay providers. All three work without a key against
    # public endpoints; supplying a key lifts the daily rate cap.
    eia_api_key: str = Field(default="", repr=False)
    bls_api_key: str = Field(default="", repr=False)
    census_api_key: str = Field(default="", repr=False)
    sec_user_agent: str = "MarketMosaic contact@example.com"
    # How old a stale `provider_cache` row may be and still be served when
    # the live provider misses, per capability, overriding the defaults in
    # `services.provider_cache.MAX_STALE_BY_CAPABILITY` without a deploy.
    # Format: "capability=seconds,capability=seconds" (a d/h/m suffix is
    # accepted, e.g. "quote=900,news=12h"). Bad entries are logged and
    # skipped rather than failing startup.
    provider_cache_max_stale: str = ""

    # App / server
    app_env: str = "development"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Per-IP rate limiting (slowapi). `memory://` works for a single
    # replica; switch to `redis://...` when scaling out so all instances
    # share counters. Set ENABLED=false in tests so TestClient runs
    # don't pollute production counters.
    rate_limit_enabled: bool = True
    rate_limit_storage_url: str = Field(default="memory://", repr=False)

    # Bearer token guarding the `/api/admin/*` ops surface (plus
    # `/api/seed-universe`, which lives in routes_admin but outside that
    # prefix). Empty means "unauthenticated", which is only tolerated
    # outside production — `admin_auth` fails CLOSED when `app_env` is
    # production and no token is set, because failing open is precisely
    # how this surface ended up publicly reachable.
    #
    # Set it in the Render dashboard (render.yaml carries `sync: false`),
    # never in the repo. Empty by default so dev and the test suite work
    # without ceremony.
    admin_api_token: str = Field(default="", repr=False)

    # PM rating blend (Option A). The final rating label is derived from a
    # weighted mix of the LLM's directional call and the deterministic
    # factor_pm_score: `final = w * llm_score + (1 - w) * factor_pm_score`,
    # where llm_score is the bucket center of the PM's rating_label
    # (Very Bullish=90 … Very Bearish=10). 0.0 reproduces pre-blend
    # behavior (factor score is dispositive); 1.0 gives the LLM full
    # authority. Clamped to [0.0, 1.0] at use site.
    llm_rating_weight: float = 0.4

    # W2b research-quality guards (owner decision 7, 2026-09-24).
    # `rating_reconciliation_mode` is 7(b)'s kill switch: "enforce" sets a
    # Bullish rating on overvalued valuation evidence (or the Bearish
    # mirror) to Neutral unless the PM stated a substantive reason;
    # "record" computes and stores the same reconciliation but publishes
    # the blended rating unchanged — an env flip on the worker instead of a
    # redeploy if the rule misbehaves. Record mode leaves published output
    # untouched by 7(b): an unenforced ("downgraded" but not applied)
    # divergence carries no confidence cap either, because the
    # `divergence_unreviewed` cap is only for a reason 7(b) ACCEPTED without
    # a live critic's review. Anything else fails validation at
    # boot rather than silently meaning one of the two. A `str` with a
    # validator, not a `Literal`: the image-defaults test loads this module
    # outside `sys.modules`, where a postponed `Literal` annotation cannot
    # be resolved.
    rating_reconciliation_mode: str = "enforce"
    # 7(a) (number-to-source check, a later slice): False means untraceable
    # figures are flagged only, never withheld from supporting lists.
    number_check_withhold: bool = True

    @field_validator("rating_reconciliation_mode")
    @classmethod
    def _rating_reconciliation_mode_known(cls, v: str) -> str:
        mode = str(v).strip().lower()
        if mode not in ("enforce", "record"):
            raise ValueError("rating_reconciliation_mode must be 'enforce' or 'record'")
        return mode

    # Runtime
    cache_ttl_seconds: int = 3600
    max_agent_context_chars: int = 60000
    max_stocks_in_portfolio: int = 25
    default_stock_universe: str = "large_cap_demo"

    # ------------------------------------------------------------------
    # FEAT-002 — customer accounts, freemium trial, Pro subscriptions.
    # ------------------------------------------------------------------
    # Every flag defaults OFF so a deployment that sets nothing behaves
    # exactly as before this feature existed: no login wall, no meters,
    # no billing routes doing anything. The three flags are independent
    # and are meant to flip in this order (see docs/ops): AUTH_ENABLED
    # first with internal accounts, then USAGE_LIMITS_ENABLED, then
    # BILLING_ENABLED once Stripe test-mode has been exercised.
    #
    # Fail-closed rule: `auth_enabled` with no Clerk issuer/JWKS configured
    # makes `customer_auth_middleware` answer 503 `auth_unavailable` on every
    # non-public route rather than letting anyone through. Marketing/public
    # routes stay up. A half-configured login wall is not a login wall.
    auth_enabled: bool = False
    billing_enabled: bool = False
    usage_limits_enabled: bool = False
    # Clerk. The JWT template named "marketmosaic" carries `email` and
    # `email_verified` claims so the backend never calls Clerk's API per
    # request. RS256 only; keys come from `<frontend-api>/.well-known/
    # jwks.json`. `clerk_authorized_parties` is a comma-separated list of
    # origins the token's `azp` must match (PUBLIC_BASE_URL in prod).
    clerk_issuer: str = ""
    clerk_jwks_url: str = ""
    clerk_publishable_key: str = Field(default="", repr=False)
    clerk_authorized_parties: str = ""
    # Stripe. No SDK — `services/stripe_client.py` is a thin httpx client.
    # Secrets are read only there and in the webhook verifier; never log.
    stripe_secret_key: str = Field(default="", repr=False)
    stripe_webhook_secret: str = Field(default="", repr=False)
    stripe_price_pro_monthly: str = ""
    stripe_price_pro_annual: str = ""
    stripe_portal_configuration_id: str = ""
    # Checkout return URLs + Clerk allowed origin. Empty means "not set",
    # which billing treats as unconfigured.
    public_base_url: str = ""
    # Curated public samples shown to logged-out visitors (S3 builds them
    # in the worker; the public routes only read). Comma-separated.
    sample_tickers: str = "NVDA,COST,JPM"
    # Card-less Pro trial length, started once per verified email.
    trial_days: int = 7
    # How long a `past_due` subscription keeps Pro while the card is fixed.
    grace_days: int = 7
    # Per-environment overrides for `auth/features.py` allowances and
    # `auth/ratelimit.py` scopes, so a number can change without a deploy.
    # JSON objects; bad JSON is logged and ignored (defaults apply).
    #   ENTITLEMENT_OVERRIDES_JSON='{"pm_chat": {"free": 5, "pro": 500}}'
    #   RATE_LIMIT_OVERRIDES_JSON='{"research": "5/hour"}'
    entitlement_overrides_json: str = "{}"
    rate_limit_overrides_json: str = "{}"
    # Whether GET /api/stocks/{t}/memo may run the agent graph inside the
    # request. None (the default) resolves to `not auth_enabled`: today's
    # behaviour with the login wall off, worker-only generation with it
    # on. Not symmetric: `false` forces worker-only generation even with
    # the wall off, but `true` under the wall is deliberately ignored by
    # GET /memo and POST /chat (it would skip memo_view metering and run
    # LLM work inside a customer request). Set explicitly only to force
    # worker-only mode for an experiment.
    memo_inline_generation: bool | None = None
    # Legal pages ship as clearly labelled drafts until the owner records
    # the review date here (any non-empty value flips the banner off).
    legal_reviewed_at: str = ""
    # Salt for the bootstrap IP / user-agent hashes on `users`. Those
    # hashes exist so repeated trial creation from one source is visible
    # later; salting keeps them from being a rainbow-table lookup of the
    # visitor's IP. Rotate to invalidate. Empty salt still hashes (dev).
    abuse_hash_salt: str = Field(default="", repr=False)
    # How many proxies sit between the internet and uvicorn — i.e. how
    # many trailing `X-Forwarded-For` entries were written by infrastructure
    # we trust. Render is exactly one hop, so the caller is the LAST entry
    # (the one Render appended); anything left of it came from the client
    # and proves nothing. Read by `rate_limit.client_ip`, which keys every
    # per-IP ceiling and the bootstrap IP hash. 0 ignores the header and
    # uses the socket peer: right for a directly exposed dev server, wrong
    # behind any proxy (every caller then looks like the proxy).
    trusted_proxy_hops: int = 1

    # ------------------------------------------------------------------
    # FEAT-001 — Fundamentals Explorer.
    # ------------------------------------------------------------------
    # `enable_fundamentals_explorer` is the rollback switch: off, and
    # `main.create_app` mounts neither fundamentals router, so the page
    # sees 404s and nothing else changes. `fundamentals_anon_commentary`
    # governs the commentary endpoint while the login wall is OFF: the
    # default (False) answers an anonymous visitor with the deterministic
    # degraded shape — no LLM call, nothing charged — because DEVPLAN's
    # logged-out surface has no commentary generation and a spoofable
    # session header is not a meter. Under AUTH_ENABLED the flag is moot:
    # the `chart_commentary` feature meters signed-in users.
    # `fundamentals_commentary_model` pins the commentary model; empty
    # means the cheap route's provider default.
    enable_fundamentals_explorer: bool = True
    fundamentals_anon_commentary: bool = False
    fundamentals_commentary_model: str = ""

    # ------------------------------------------------------------------
    # Phase 6 — Fundamental Factor Scorecard (versioned, point-in-time).
    # ------------------------------------------------------------------
    # Three independent kill switches. `enable_scorecard` governs reads and
    # the memo/PM integration; `enable_scorecard_loop` governs the worker's
    # daily scoring loop (rows stay readable when it is off);
    # `enable_scorecard_disagreement_regen` is the ONLY path by which the
    # feature can spend LLM money, so it defaults off and is capped per day.
    enable_scorecard: bool = True
    enable_scorecard_loop: bool = True
    enable_scorecard_disagreement_regen: bool = False
    scorecard_disagreement_regen_daily_cap: int = 3
    # Point-in-time lag rule, used when neither the provider nor a stored
    # filing supplies the filing date: a figure for a period ending on D is
    # treated as knowable on D + lag. 75 days for annual rows is
    # deliberately conservative — large accelerated filers have 60 days for
    # a 10-K, everyone else 75/90 — so the scorecard errs toward "not yet
    # known" rather than toward lookahead. 45 days for quarterly rows is
    # the widest 10-Q deadline (no quarterly ingestion in fs-v1, kept so
    # the rule is complete for rows that carry `fiscal_quarter`).
    scorecard_pit_lag_annual_days: int = 75
    scorecard_pit_lag_quarter_days: int = 45
    # Cross-sectional normalisation. The curated universe is ~170 names
    # (sp500.json starter list), so thresholds are sized for 100–600 names
    # and nothing below hardcodes the count: sector-neutral z needs at
    # least `scorecard_min_sector_n` names in a sector (else universe z
    # with a note); evaluation legs are quintiles with at least
    # `scorecard_min_leg_n` names each (else the month is skipped and
    # reported).
    scorecard_min_sector_n: int = 5
    scorecard_min_leg_n: int = 15
    scorecard_winsor_pct: float = 0.025
    scorecard_min_coverage: float = 0.6
    # Retention: daily rows are a convenience and age out; month-end rows
    # are the evaluation's sample and are kept forever.
    scorecard_daily_retention_days: int = 45
    scorecard_backfill_months: int = 60
    # Disagreement thresholds, in percentile points between the memo's
    # rating bucket centre and the scorecard percentile.
    scorecard_disagreement_material: float = 40.0
    scorecard_disagreement_watch: float = 25.0
    # Bearer token for `GET /api/scorecard/export`. Separate from
    # `admin_api_token` so the export can be handed to a downstream system
    # without granting the ops surface. Empty (default) means the export
    # follows the same policy as the other scorecard reads.
    scorecard_export_token: str = Field(default="", repr=False)

    # ------------------------------------------------------------------
    # FEAT-003 — GICS Industry Group analysts and weekly Industry Analysis.
    # ------------------------------------------------------------------
    # Every setting for the feature lives here (slice 1 owns the file) so the
    # parallel slices — knowledge/analysts, analytics, jobs, API, frontend —
    # read one block instead of each adding its own. Defaults keep the
    # feature dark: no memo routing change, no worker loops enqueueing
    # reports. `enable_industry_reports` follows the web/worker split in
    # render.yaml (false on web, true on the worker); the taxonomy registry
    # and the daily classification audit are DB-only and always available.
    #
    # `gics_taxonomy_version` names the taxonomy the deployment expects. It
    # is the key the importer uses for the bundled knowledge JSON and what
    # `gics_registry.ensure_taxonomy` activates on a fresh database; the
    # ACTIVE version is always read from the database (two processes share
    # nothing else), never from this value.
    gics_taxonomy_version: str = "gics-2026-04"
    # Display policy (owner decision 2026-09-24: our own labels publicly, no
    # licensed names or codes). `internal_labels` names a node by
    # `industry_labels`; `codes_and_names` is for internal/admin rendering
    # only and logs a warning. Public API bodies are projected through
    # `industry_labels.project_public` whatever this says — licensing is
    # not a toggle — so the setting only decides `gics_registry.display()`.
    gics_display_mode: str = "internal_labels"
    # Memo pipeline: when on, a mapped company's memo runs the Industry
    # Group Analyst (one per memo) alongside the sector analyst. Off until a
    # production A/B is read; the sector analyst stays primary either way.
    enable_industry_analyst_routing: bool = False
    # Worker: the weekly report loop + the report-job drainer. Web keeps it
    # off (reads only); page views never generate.
    enable_industry_reports: bool = False
    # Report generation caps. Two bounded LLM calls per report; the third
    # job attempt runs deterministic so a week never ends blank.
    industry_report_max_llm_calls: int = 2
    industry_report_max_attempts: int = 3
    industry_reports_max_jobs_per_run: int = 40
    # Publication: Sunday 06:30 UTC (after sector_digest_loop 05:30) with the
    # as-of set to the prior Friday's close; period_key is that Friday's
    # ISO week. Owner decision 3 — defaults recorded in the setup doc.
    industry_reports_cron_dow: str = "sun"
    industry_reports_cron_hour: int = 6
    industry_reports_cron_minute: int = 30
    industry_reports_as_of_weekday: int = 4  # Monday=0 … Friday=4
    # Analytics: sample floor below which a group reports
    # `insufficient_sample` (a labelled state, not an error); the weekly
    # price warm-up budget (provider fetches per run, lowest-coverage groups
    # first) so first-run coverage is honest rather than mostly `no_prices`;
    # the benchmark set (owner decision 2 — no ETF/index-vendor series).
    industry_stats_min_sample: int = 3
    industry_price_warmup_budget: int = 150
    industry_benchmarks: str = "universe_ew,sector_ew,KFR.MKT_RF.D"
    # PM context: the company's own group plus at most this many groups
    # linked by dependency edges; the cross-industry block is capped at
    # 2,000 characters by the renderer.
    industry_pm_max_linked_groups: int = 2
    # Access (owner decision 1): `public` for the latest report; history,
    # changes-since-prior and PM chat integration are `pro` when
    # AUTH_ENABLED. The taxonomy endpoint always answers with the policy.
    industry_analysis_access: str = "public"
    # A report older than this (by as-of) is flagged stale on the read API,
    # as is one whose latest refresh attempt failed.
    industry_report_stale_after_days: int = 10
    # Owner decision 4: auto-publish after validation. The review-queue
    # status (`pending_review`) exists; no publish endpoint is built while
    # this stays False.
    industry_reports_require_review: bool = False
    # Owner decision 1 (2026-09-24): template editions are stored for audit
    # only and never displayed, so a week in which a group produced no
    # validated analyst edition leaves that group "not updated". The weekly
    # health verdict is the share of finished groups in that state: with 25
    # groups, 0.10 tolerates two and calls three or more an unhealthy week
    # (FIX-002: "a run where more than 1-2 groups land deterministic is a
    # failed run"). A rate rather than a count, so a taxonomy revision
    # needs no code change.
    industry_report_not_updated_unhealthy_rate: float = 0.10
    # Output-token cap for the writer batch that carries `outlook`. Declared
    # here with its sibling (this block has one owner per deploy wave); the
    # forecast-assumption slice is the consumer.
    industry_report_outlook_max_tokens: int = 4000
    # The deployed worker's drainer runs the legacy-edition reclassification
    # (template rows -> audit_only, flags re-derived) once, recorded by a
    # ledger row, so nobody needs a production shell for it. Reads never
    # depend on it; this is the switch that stops the automatic run.
    industry_reclassify_legacy_editions: bool = True

    @property
    def industry_benchmarks_list(self) -> list[str]:
        return [b.strip() for b in self.industry_benchmarks.split(",") if b.strip()]

    @property
    def auth_configured(self) -> bool:
        """Enough Clerk config to verify a token at all."""
        return bool(self.clerk_issuer) and bool(self.clerk_jwks_url)

    @property
    def billing_configured(self) -> bool:
        return bool(self.stripe_secret_key) and bool(self.stripe_webhook_secret)

    @property
    def clerk_authorized_parties_list(self) -> list[str]:
        return [p.strip() for p in self.clerk_authorized_parties.split(",") if p.strip()]

    @property
    def sample_tickers_list(self) -> list[str]:
        return [t.strip().upper() for t in self.sample_tickers.split(",") if t.strip()]

    @property
    def legal_reviewed(self) -> bool:
        return bool(self.legal_reviewed_at.strip())

    @property
    def memo_inline_generation_effective(self) -> bool:
        if self.memo_inline_generation is None:
            return not self.auth_enabled
        return bool(self.memo_inline_generation)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_gemini(self) -> bool:
        """True when ANY Gemini access path is configured (direct API or Vertex)."""
        return bool(self.gemini_api_key) or self.has_vertex

    @property
    def has_vertex(self) -> bool:
        """True when Vertex AI backend is configured. Auth handled by ADC —
        set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default
        login`. Vertex wins over direct API when both are set."""
        return bool(self.vertex_project_id)

    @property
    def has_llm(self) -> bool:
        return self.has_openai or self.has_anthropic

    @property
    def active_llm_provider(self) -> str:
        """Resolve the effective provider, honoring LLM_PROVIDER + key presence."""
        choice = (self.llm_provider or "auto").lower()
        if choice == "anthropic" and self.has_anthropic:
            return "anthropic"
        if choice == "openai" and self.has_openai:
            return "openai"
        # auto: prefer Anthropic if configured, else OpenAI
        if self.has_anthropic:
            return "anthropic"
        if self.has_openai:
            return "openai"
        return "none"

    @property
    def llm_enabled(self) -> bool:
        return self.has_llm and not self.use_demo_data_only

    @property
    def use_demo_data_only(self) -> bool:
        # If demo data flag is explicit and live data disabled, force demo
        return self.use_demo_data and not self.enable_live_data


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
