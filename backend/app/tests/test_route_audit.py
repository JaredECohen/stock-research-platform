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

import os
import re
import subprocess
import sys
from pathlib import Path

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
