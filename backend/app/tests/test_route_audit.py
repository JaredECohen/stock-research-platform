"""The FEAT-002 route inventory must not drift from the API.

`docs/economics/route-audit-2026-09.md` classifies every backend path
(called-by, external calls, auth today, proposed customer policy, rate
scope). The customer-policy layer is built from that table, so a route
added to the API without a row is a route nobody has decided the policy
for — default-deny will 401 it, but silently. Enumerating `app.openapi()`
here (the same source `test_admin_auth.py` uses, for the same reason: a
version-stable contract rather than `app.routes` introspection) turns
that omission into a CI failure that names the route.

The table is parsed, not hardcoded, so keeping the doc honest is the only
maintenance. When `auth/policy.py` lands (slice S1) this test should be
re-pointed at the policy table and the doc becomes the human-readable
mirror.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DOC = REPO_ROOT / "docs" / "economics" / "route-audit-2026-09.md"
COST_DOC = REPO_ROOT / "docs" / "economics" / "unit-costs-2026-09.md"
COST_SCRIPT = REPO_ROOT / "backend" / "scripts" / "audit_unit_costs.py"

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}

# Every policy cell must start with one of these — the vocabulary from the
# doc's legend. A row with a blank or free-text policy is as bad as no row.
_POLICY_TOKENS = ("public", "free", "pro", "metered:", "dcf", "comps", "admin")


def _openapi_routes() -> set[tuple[str, str]]:
    spec = app.openapi()["paths"]
    return {
        (method.upper(), path)
        for path, operations in spec.items()
        for method in operations
        if method.upper() not in ("HEAD", "OPTIONS")
    }


def _documented_routes() -> dict[tuple[str, str], list[str]]:
    """(METHOD, path) -> cells, from every markdown table row whose first
    cell is an HTTP method and second cell a backticked path. Header,
    separator and prose rows fall out naturally."""
    rows: dict[tuple[str, str], list[str]] = {}
    for line in AUDIT_DOC.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] not in _METHODS:
            continue
        m = re.fullmatch(r"`(/[^`]*)`", cells[1])
        if m is None:
            continue
        rows[(cells[0], m.group(1))] = cells
    return rows


def test_audit_doc_exists_and_parses():
    assert AUDIT_DOC.is_file(), f"missing {AUDIT_DOC}"
    docs = _documented_routes()
    assert len(docs) >= 50, (
        f"only {len(docs)} route rows parsed from {AUDIT_DOC.name} — the table "
        "format changed (expected `| METHOD | `/path` | ...`) or rows were lost"
    )


def test_every_openapi_route_is_in_the_audit_table():
    """The anti-drift assertion: add a route, classify it, or fail CI."""
    live = _openapi_routes()
    assert live, "no routes in the OpenAPI schema — the check is looking in the wrong place"
    documented = set(_documented_routes())
    missing = sorted(live - documented)
    assert not missing, (
        "routes in app.openapi() with no row in docs/economics/route-audit-2026-09.md "
        "(classify them: called-by, external calls, auth today, policy, rate scope):\n  "
        + "\n  ".join(f"{m} {p}" for m, p in missing)
    )


def test_every_documented_live_route_carries_a_policy():
    """A row exists but says nothing about the policy → still unclassified.

    Only rows for routes that exist today are checked; the doc also lists
    routes FEAT-002 will add, in a shorter table whose columns differ.
    """
    live = _openapi_routes()
    docs = _documented_routes()
    # Backend table columns: Method | Path | Called by | External | Auth today | Policy | Rate | Notes
    bad = []
    for key in sorted(live):
        cells = docs.get(key)
        if cells is None:
            continue  # reported by the drift test above
        if len(cells) < 7:
            bad.append(f"{key[0]} {key[1]}: expected 7+ cells, got {len(cells)}")
            continue
        policy = cells[5]
        if not policy or not policy.startswith(_POLICY_TOKENS):
            bad.append(f"{key[0]} {key[1]}: policy cell {policy!r} not in {_POLICY_TOKENS}")
        if not cells[6]:
            bad.append(f"{key[0]} {key[1]}: empty rate-scope cell")
    assert not bad, "audit rows without a usable classification:\n  " + "\n  ".join(bad)


def test_admin_routes_are_documented_as_admin_unless_browser_exempt():
    """The audit must agree with the guard that already ships: everything
    `admin_auth.is_protected` covers is `admin`, and only the browser-
    exempt product routes carry a customer policy. A row that disagrees
    would have S1 build a policy the middleware contradicts."""
    from app.api import admin_auth

    docs = _documented_routes()
    disagreements = []
    for (method, path), cells in docs.items():
        if (method, path) not in _openapi_routes():
            continue
        policy = cells[5]
        if admin_auth.is_protected(method, path):
            if policy != "admin":
                disagreements.append(f"{method} {path}: guarded by admin token but policy {policy!r}")
        elif admin_auth.is_exempt(method, path):
            if policy == "admin":
                disagreements.append(f"{method} {path}: browser-exempt but policy 'admin'")
    assert not disagreements, "\n  ".join(["audit vs admin_auth:"] + disagreements)


def _policy_cell(cells: list[str]) -> str:
    """The policy cell of a row from either table in the doc: the backend
    inventory (Method | Path | Called by | External | Auth today | Policy |
    Rate | Notes) or the shorter FEAT-002 additions table (Method | Path |
    External | Policy | Rate)."""
    return cells[5] if len(cells) >= 7 else cells[3]


def _expected_from_doc(policy: str) -> tuple[str, str | None]:
    """Doc vocabulary → (policy level, feature-or-None) as `auth/policy.py`
    would state it. Feature is None where the doc names none."""
    from app.auth import policy as pol

    text = policy.strip()
    head = re.split(r"[\s(,]", text, maxsplit=1)[0].lower()
    m = re.search(r"metered:(\w+)", text)
    if text.startswith("pro + metered:"):
        return pol.PRO, m.group(1)
    if head == "metered:" or text.startswith("metered:"):
        return pol.AUTHENTICATED, m.group(1) if m else None
    if head in ("dcf", "comps"):
        return pol.AUTHENTICATED, head
    if head == "public" or head == "stripe":
        return pol.PUBLIC, None
    if head == "free":
        return pol.AUTHENTICATED, None
    if head == "pro":
        return pol.PRO, None
    if head == "admin":
        return pol.ADMIN, None
    raise AssertionError(f"unrecognised policy cell {policy!r}")


def test_audit_doc_policy_column_agrees_with_auth_policy():
    """The doc is the human-readable mirror of `auth/policy.py`; the
    middleware enforces the table, not the doc. A row that says `free`
    for a route the table calls `pro` (or names a different meter) would
    have the pricing page promise something the backend refuses. Checked
    for every route that exists today, so the two cannot drift apart."""
    from app.auth import policy as pol

    docs = _documented_routes()
    live = _openapi_routes()
    disagreements = []
    for (method, path), cells in sorted(docs.items()):
        if (method, path) not in live:
            continue
        level, feature = _expected_from_doc(_policy_cell(cells))
        actual, explicit = pol.lookup(method, pol.templated_to_concrete(path))
        if not explicit:
            disagreements.append(f"{method} {path}: not named in auth/policy.py")
            continue
        if actual.level != level:
            disagreements.append(f"{method} {path}: doc {level!r} vs policy {actual.level!r}")
        if feature is not None and actual.feature != feature:
            disagreements.append(f"{method} {path}: doc meters {feature!r} vs policy {actual.feature!r}")
    assert not disagreements, "\n  ".join(["route-audit doc vs auth/policy.py:"] + disagreements)


def test_unit_cost_doc_has_no_invented_numbers():
    """§5 must stay 'NOT YET MEASURED' until the script's JSON exists.

    A measured row would come with a `unit-costs-measured-<date>.json`
    next to the doc; without one, any dollar figure in the results table
    is an estimate wearing a measurement's clothes.
    """
    text = COST_DOC.read_text()
    assert "NOT YET MEASURED" in text
    results = text.split("## 5. Results", 1)[1].split("## 6.", 1)[0]
    measured_json = list((REPO_ROOT / "docs" / "economics").glob("unit-costs-measured-*.json"))
    if not measured_json:
        assert not re.search(r"\$\s*\d", results.replace("$1.50", "").replace("$14.995", "")), (
            "unit-costs doc §5 contains a dollar figure but no measured JSON exists"
        )


def test_unit_cost_script_refuses_to_spend_by_default():
    """Without RUN_LIVE_TESTS=1 the script must exit 0 without importing the
    app (so a developer .env with live keys is never even read)."""
    env = {**os.environ}
    env.pop("RUN_LIVE_TESTS", None)
    out = subprocess.run(
        [sys.executable, str(COST_SCRIPT)],
        cwd=str(REPO_ROOT / "backend"),
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert out.returncode == 0, f"exit {out.returncode}: {out.stdout}\n{out.stderr}"
    assert "not running" in out.stdout
    assert "RUN_LIVE_TESTS" in out.stdout


def test_unit_cost_script_refuses_without_llm_keys():
    """RUN_LIVE_TESTS=1 alone is not enough: no key, no spend, exit 0.

    Blanks the same keys CI blanks; the gate is `settings.has_llm`, so a
    dev .env's VERTEX_PROJECT_ID (no credentials) must not count as a key —
    that exact case once let the script past an earlier `has_gemini` gate.
    """
    env = {**os.environ, "RUN_LIVE_TESTS": "1",
           "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": "",
           "GOOGLE_API_KEY": ""}
    out = subprocess.run(
        [sys.executable, str(COST_SCRIPT)],
        cwd=str(REPO_ROOT / "backend"),
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert out.returncode == 0, f"exit {out.returncode}: {out.stdout}\n{out.stderr}"
    assert "no LLM key configured" in out.stdout


# ---------------------------------------------------------------------------
# The measurement script's own invariants (pure functions + the swap)
# ---------------------------------------------------------------------------

def _load_cost_script():
    """Import `backend/scripts/audit_unit_costs.py` by path — `scripts/` is
    not a package. Module-level code only reads env and defines constants;
    nothing runs (and nothing can spend) without `main()`."""
    spec = importlib.util.spec_from_file_location("audit_unit_costs_under_test", COST_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("use_agents_sdk", [False, True])
def test_pm_chat_sample_never_generates_a_memo(monkeypatch, use_agents_sdk):
    """A `single_stock_analysis` chat turn must answer from the stored memo
    on BOTH inline entry points.

    Regression: the first draft swapped only `orchestrator.run_stock_memo`.
    With `USE_AGENTS_SDK=true` (the committed config.env default) the
    orchestrator resolves `sdk_runtime.run_stock_memo_via_sdk` at call
    time instead, which runs a full memo under its own uuid run_id — so a
    chat sample silently billed a 5-9 minute memo and reported only the
    intent call. Both entry points are replaced with sentinels that fail
    the test if reached; the stored-memo stand-in must be what runs.
    """
    from app.agents import orchestrator as orch_mod
    from app.agents import sdk_runtime
    from app.config import settings
    from app.services import memo_store

    mod = _load_cost_script()
    reached: list[str] = []

    def memo_ran(ticker: str, **_kw):
        reached.append(ticker)
        raise AssertionError("a pm_chat sample started a memo run")

    monkeypatch.setattr(orch_mod, "run_stock_memo", memo_ran)
    monkeypatch.setattr(sdk_runtime, "run_stock_memo_via_sdk", memo_ran)
    monkeypatch.setattr(settings, "use_agents_sdk", use_agents_sdk)
    # Deterministic routing into the inline-memo branch, no LLM and no
    # SDK chat agent involved (both would need keys).
    monkeypatch.setattr(orch_mod, "classify_intent",
                        lambda _m: ("single_stock_analysis", ["NVDA"], None))
    monkeypatch.setattr(orch_mod, "_is_conceptual_followup", lambda _m, _h: False)
    # No stored memo → the stand-in raises its own ValueError; a memo run
    # would have raised the sentinel's AssertionError instead.
    monkeypatch.setattr(memo_store, "latest_memo", lambda _t: None)

    go = mod._pm_chat(0, ["NVDA", "COST"])
    with pytest.raises(ValueError, match="inline generation disabled"):
        go("audit-test-run")

    assert reached == []
    # Restored even though the turn raised.
    assert orch_mod.run_stock_memo is memo_ran
    assert sdk_runtime.run_stock_memo_via_sdk is memo_ran


def test_free_dcf_is_not_an_allowance_term():
    """`POST /api/dcf/{t}` is an LLM call on every request with an
    assumptions body (`build_dcf` caches only `assumptions=None`) and is
    bounded by the `series` rate scope, not by the follows-memo rule. It
    must therefore live in the declared-assumption dict with a rate
    ceiling beside it, never in `ALLOWANCES` where `3 × C(dcf)` would
    read as a ceiling. The doc's Free formula must say the same."""
    mod = _load_cost_script()
    assert "dcf" not in mod.ALLOWANCES["free"]
    assert "comps" not in mod.ALLOWANCES["free"]
    assert mod.ASSUMED_UNMETERED["free"]["dcf"] > 0
    for plan in ("free", "pro"):
        for feat in mod.ASSUMED_UNMETERED[plan]:
            assert feat in mod.RATE_CEILING_PER_HOUR, f"{plan}.{feat} has no rate ceiling"
    # Metered / cache-bounded / assumed dicts must not double-count a feature.
    for plan in ("free", "pro"):
        kinds = [set(mod.ALLOWANCES[plan]), set(mod.CACHE_BOUNDED[plan]),
                 set(mod.ASSUMED_UNMETERED[plan])]
        assert not (kinds[0] & kinds[1]) and not (kinds[0] & kinds[2]) and not (kinds[1] & kinds[2])

    section = COST_DOC.read_text().split("## 3.", 1)[1].split("## 4.", 1)[0]
    free_part = section.split("**Free Explorer", 1)[1].split("**Pro,", 1)[0]
    # The fenced block is the formula; the prose around it may quote the
    # old wrong term while explaining why it was wrong.
    free_formula = free_part.split("```", 2)[1]
    assert "U_free_dcf" in free_formula
    assert not re.search(r"\b\d+·C\(dcf\)", free_formula), "Free dcf written as an allowance term"
    # The multiplier the doc states must be the one the script uses.
    assert f"U_free_dcf = {mod.ASSUMED_UNMETERED['free']['dcf']}" in free_formula


def test_worst_case_keeps_term_kinds_apart_and_never_zeroes_an_unmeasured_term():
    mod = _load_cost_script()

    def feat(mean):
        return {"mean_llm_cost_usd": mean}

    costs = {"memo_view": 0.0, "research_run": 2.0, "pm_chat": 0.01,
             "chart_commentary": 0.005, "dcf": 0.002, "comps": 0.003}
    wc = mod._worst_case({k: feat(v) for k, v in costs.items()})

    free = wc["free"]
    assert free["metered"] == {"memo_view": 0.0, "research_run": 2.0,
                               "pm_chat": 0.1, "chart_commentary": 0.025}
    assert free["cache_bounded"] == {"comps": round(6 * 0.003, 4)}
    assert free["assumed_unmetered"] == {"dcf": round(60 * 0.002, 4)}
    assert free["rate_ceiling_per_hour_usd"] == {"dcf": round(3600 * 0.002, 4)}
    expected_free = 2.0 + 0.1 + 0.025 + 6 * 0.003 + 60 * 0.002
    assert free["total_usd"] == round(expected_free, 4)
    assert free["unmeasured_terms"] == []
    assert wc["verdict"]["free_under_threshold"] is False  # research_run alone is $2

    # Pro has a term the script never samples → total unknown, not zero,
    # while the measured partial is still reported.
    pro = wc["pro"]
    assert pro["total_usd"] is None
    assert "portfolio_build" in pro["unmeasured_terms"]
    assert pro["total_measured_usd"] == round(
        20 * 2.0 + 300 * 0.01 + 100 * 0.005 + 200 * 0.002 + 20 * 0.003, 4)
    assert wc["verdict"]["pro_under_threshold"] is None

    # An unmeasured feature (no ok sample) makes the plan total None too.
    partial = {k: feat(v) for k, v in costs.items()}
    partial["chart_commentary"] = {"mean_llm_cost_usd": None}
    wc2 = mod._worst_case(partial)
    assert wc2["free"]["total_usd"] is None
    assert wc2["free"]["metered"]["chart_commentary"] is None
    assert "chart_commentary" in wc2["free"]["unmeasured_terms"]
    assert wc2["free"]["total_measured_usd"] == round(expected_free - 0.025, 4)
    assert wc2["verdict"]["free_under_threshold"] is None
