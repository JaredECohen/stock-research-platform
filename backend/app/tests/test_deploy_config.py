"""Static checks on render.yaml against the Dockerfile's actual layout.

These exist because of a real production failure: the worker service
shipped with `dockerCommand: python -m app.worker`, which exits 1 with
"No module named 'app'". The image's WORKDIR is /app while the backend
is copied to /app/backend, so the `app` package is not importable from
the default working directory. It was only ever tested from the
backend/ directory locally, where it works, so nothing caught it until
Render emailed about the crash.

Deploy config is code that runs exactly once per deploy, in an
environment nothing else reproduces. Cheap static assertions are the
only practical guard short of building the image in CI.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RENDER_YAML = REPO_ROOT / "render.yaml"
DOCKERFILE = REPO_ROOT / "Dockerfile"

pytestmark = pytest.mark.skipif(
    not RENDER_YAML.exists() or not DOCKERFILE.exists(),
    reason="deploy config not present (e.g. installed package, not a checkout)",
)


def _services() -> list[dict]:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(RENDER_YAML.read_text())["services"]


def _backend_dest() -> str:
    """Where the Dockerfile copies the backend to, e.g. /app/backend."""
    m = re.search(r"^COPY\s+backend\s+(\S+)", DOCKERFILE.read_text(), re.MULTILINE)
    assert m, "Dockerfile no longer has a `COPY backend <dest>` line"
    return m.group(1)


def test_worker_command_can_import_the_app_package():
    """`python -m app.worker` must resolve the `app` package.

    Two acceptable mechanisms: PYTHONPATH pointing at the backend (what
    we use), or a command that cd's there. Either satisfies the import;
    neither present is the production failure this test exists for.
    """
    backend_dest = _backend_dest()
    workers = [s for s in _services() if s.get("type") == "worker"]
    assert workers, "no worker service defined in render.yaml"
    for svc in workers:
        cmd = svc.get("dockerCommand", "")
        assert "app.worker" in cmd, f"{svc['name']}: unexpected worker command {cmd!r}"
        env = {e["key"]: e.get("value") for e in svc.get("envVars", []) if "value" in e}
        on_pythonpath = backend_dest in (env.get("PYTHONPATH") or "")
        assert on_pythonpath or backend_dest in cmd, (
            f"{svc['name']}: neither PYTHONPATH nor dockerCommand {cmd!r} puts "
            f"{backend_dest} on sys.path, so the `app` package is not importable "
            f"and the service will exit 1 with ModuleNotFoundError"
        )


def test_worker_command_execs_so_sigterm_reaches_python():
    """A wrapping shell must hand off via `exec`.

    Without it the shell is PID 1 and absorbs the SIGTERM Render sends on
    deploy; Python never sees it, the graceful shutdown in app/worker.py
    never runs, and the worker is SIGKILLed mid-job instead.
    """
    for svc in _services():
        cmd = svc.get("dockerCommand", "")
        if not cmd.startswith(("sh ", "bash ", "/bin/sh ", "/bin/bash ")):
            continue
        assert " exec " in f" {cmd} ", (
            f"{svc['name']}: shell-wrapped dockerCommand {cmd!r} must `exec` "
            f"the process so it receives SIGTERM directly"
        )


def test_web_and_worker_own_disjoint_background_work():
    """Exactly one service may run the regen queue + monitoring loops.

    Both on would re-couple the memory profiles the split exists to
    separate; both off would silently stop memo regeneration entirely.
    """
    owners = []
    for svc in _services():
        env = {e["key"]: e.get("value") for e in svc.get("envVars", []) if "value" in e}
        if env.get("ENABLE_REGEN_WORKER") == "true" or env.get("ENABLE_MONITORING") == "true":
            owners.append(svc["name"])
    assert len(owners) == 1, (
        f"expected exactly one service running regen/monitoring, got {owners}"
    )


def test_exactly_one_service_owns_the_industry_report_queue():
    """FEAT-003: `ENABLE_INDUSTRY_REPORTS` must be true on one service.

    Both on: two processes run the Sunday loop and two drainers race the
    queue — duplicated provider spend, and the web service back on the
    hook for the worker's peak. Both off (or the key missing): the weekly
    loop records "disabled" forever and no Industry Analysis is ever
    generated, with nothing user-visible to say so.
    """
    values = {}
    for svc in _services():
        env = {e["key"]: e.get("value") for e in svc.get("envVars", []) if "value" in e}
        assert "ENABLE_INDUSTRY_REPORTS" in env, (
            f"{svc['name']} does not define ENABLE_INDUSTRY_REPORTS; the flag must be "
            f"explicit on every service, not inherited from a code default"
        )
        values[svc["name"]] = env["ENABLE_INDUSTRY_REPORTS"]
    owners = sorted(name for name, value in values.items() if value == "true")
    assert len(owners) == 1, (
        f"expected exactly one service with ENABLE_INDUSTRY_REPORTS=true, got {owners} "
        f"(all values: {values})"
    )
    web, worker = _web_and_worker()
    assert values[worker["name"]] == "true", "the worker owns report generation"
    assert values[web["name"]] == "false", "a page view must never generate a report"


def test_industry_analyst_routing_is_explicit_and_on_for_both_services():
    """Owner decision 2026-09-24: new memos route the Industry Group Analyst.

    BOTH services write memos — the worker drains the regen queue, and web
    runs them inline (GET /memo on a stale snapshot, POST /analyze?sync=true,
    chat) while FEAT-002 is dark — so one value per process would make a
    memo's roster depend on which process wrote it. Explicit on each, not
    inherited: the code default is false (CI, dev laptops), so a missing key
    silently turns routing off in production.
    """
    values = {}
    for svc in _services():
        env = {e["key"]: e.get("value") for e in svc.get("envVars", []) if "value" in e}
        assert "ENABLE_INDUSTRY_ANALYST_ROUTING" in env, (
            f"{svc['name']} does not define ENABLE_INDUSTRY_ANALYST_ROUTING; the flag must be "
            f"explicit on every service, not inherited from the code default (false)"
        )
        values[svc["name"]] = env["ENABLE_INDUSTRY_ANALYST_ROUTING"]
    web, worker = _web_and_worker()
    assert values == {web["name"]: "true", worker["name"]: "true"}, values


def test_nightly_live_suite_routes_like_production():
    """The nightly live suite mirrors the production memo path, so its memos
    route too; `test_regen_smoke` checks the routed read is an analyst's."""
    yaml = pytest.importorskip("yaml")
    workflow = REPO_ROOT / ".github" / "workflows" / "nightly-live.yml"
    steps = yaml.safe_load(workflow.read_text())["jobs"]["live-tests"]["steps"]
    live = [s for s in steps if "test_regen_smoke.py" in (s.get("run") or "")]
    assert len(live) == 1, "nightly-live.yml no longer runs test_regen_smoke.py in one step"
    assert live[0].get("env", {}).get("ENABLE_INDUSTRY_ANALYST_ROUTING") == "true"


def test_every_service_caps_malloc_arenas():
    """MALLOC_ARENA_MAX is the cheapest guard against RSS ratcheting in
    these thread-heavy processes; a new service silently omitting it
    would regress the 2026-08-12 memory work."""
    for svc in _services():
        env = {e["key"]: e.get("value") for e in svc.get("envVars", []) if "value" in e}
        assert env.get("MALLOC_ARENA_MAX"), f"{svc['name']} is missing MALLOC_ARENA_MAX"


# ---------------------------------------------------------------------------
# FEAT-002 — accounts / billing configuration lives on the web service only
# ---------------------------------------------------------------------------

FEAT_002_FLAGS = ("AUTH_ENABLED", "BILLING_ENABLED", "USAGE_LIMITS_ENABLED")
FEAT_002_SECRETS = (
    "CLERK_ISSUER", "CLERK_JWKS_URL", "CLERK_PUBLISHABLE_KEY", "CLERK_AUTHORIZED_PARTIES",
    "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "STRIPE_PRICE_PRO_MONTHLY",
    "STRIPE_PRICE_PRO_ANNUAL", "STRIPE_PORTAL_CONFIGURATION_ID", "PUBLIC_BASE_URL", "ABUSE_HASH_SALT",
)


def _web_and_worker():
    services = _services()
    web = [s for s in services if s.get("type") == "web"]
    workers = [s for s in services if s.get("type") == "worker"]
    assert len(web) == 1 and len(workers) == 1, "expected exactly one web and one worker service"
    return web[0], workers[0]


def test_feat_002_flags_ship_off_on_web():
    """The login wall, meters and billing flip on in the dashboard, in
    order, after the owner checklist — never by a deploy of this file."""
    web, _ = _web_and_worker()
    env = {e["key"]: e.get("value") for e in web.get("envVars", []) if "value" in e}
    for flag in FEAT_002_FLAGS:
        assert env.get(flag) == "false", f"web must define {flag}: \"false\" (got {env.get(flag)!r})"


def test_feat_002_secrets_are_dashboard_placeholders_on_web():
    """Present (so the dashboard shows them) but `sync: false` (so a value
    can never be committed here)."""
    web, _ = _web_and_worker()
    entries = {e["key"]: e for e in web.get("envVars", [])}
    for key in FEAT_002_SECRETS:
        assert key in entries, f"web is missing a placeholder for {key}"
        assert entries[key].get("sync") is False and "value" not in entries[key], (
            f"{key} must be `sync: false` with no value in render.yaml"
        )


def test_web_reads_the_caller_through_exactly_one_proxy():
    """Render fronts the web service with one proxy, so the socket peer is
    the proxy on every request. `rate_limit.client_ip` keys every per-IP
    ceiling (slowapi's, the DB limiter's ip: buckets, the trial-bootstrap
    cap and its IP hash) on the X-Forwarded-For entry TRUSTED_PROXY_HOPS
    from the end; without the pin those become site-wide limits. And
    FORWARDED_ALLOW_IPS="*" is not a substitute: uvicorn's always-trust
    mode takes the FIRST entry, which the client controls."""
    web, _ = _web_and_worker()
    env = {e["key"]: e.get("value") for e in web.get("envVars", []) if "value" in e}
    assert env.get("TRUSTED_PROXY_HOPS") == "1", "web must pin TRUSTED_PROXY_HOPS: \"1\" for Render's single proxy"
    assert env.get("FORWARDED_ALLOW_IPS") != "*", "always-trust proxy headers let the client pick its address"
    assert not re.search(r"--forwarded-allow-ips[=\"',\s]*\*", DOCKERFILE.read_text()), (
        "Dockerfile CMD must not run uvicorn with --forwarded-allow-ips '*'"
    )


def test_worker_never_sees_clerk_or_stripe():
    """The worker reads no JWT and no webhook; a Clerk/Stripe value there
    is secret surface with no consumer (and, for the SDK-free Stripe
    client, no code path)."""
    _, worker = _web_and_worker()
    keys = {e["key"] for e in worker.get("envVars", [])}
    leaked = sorted(k for k in keys if k.startswith(("CLERK_", "STRIPE_")))
    assert not leaked, f"worker defines {leaked}"
    for flag in FEAT_002_FLAGS:
        assert flag not in keys, f"{flag} is a web-only flag"
