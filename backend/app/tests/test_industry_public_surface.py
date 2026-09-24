"""Owner decision 2026-09-24 — no licensed taxonomy on any public surface.

MarketMosaic shows its OWN industry labels publicly: no third-party
classification brand, no taxonomy code, no registry name, on anything a
reader (or a model writing public prose) can see. Internally everything
stays keyed by code — the registry, the store, the admin surface — and the
public projection (`industry_labels.project_public`) sits between the two.

This file is the contract, walked over real bodies rather than asserted
field by field, so a field added next month is covered without anyone
remembering to add it:

* every public industry GET — taxonomy, report (latest and by number),
  history, changes, companies, snapshot (current and a legacy row) — and
  every refusal those routes answer (404 no_report / edition_withheld /
  unknown group, 422 bad_version), requested by slug AND by internal code;
* the portfolio build's industry exposure block;
* the PM's industry block and the chat tool's payload.

The world includes the worst case on purpose: a LEGACY analyst edition
(written before the rule, directly into the table, so no validator ever
saw it) whose prose, claim bases, facts, metadata, generation block,
errors and sources carry the brand, bracketed industry codes, the group's
code in "(dddd)" form, `mapping:dddd`-style bases, an industry registry
name and "Industry Group Analyst dddd". It is the served `latest`.

What "public-clean" means (`public_leaks`), for every string value AND
every dict key at every depth:

* no "gics", case-insensitive, anywhere — the brand, the internal key
  `gics-2026-04`, a module name that spells it — with letters on neither
  side ("Biologics" is a word, and a real dependency edge says it);
* no known 6/8-digit industry/sub-industry code as a token;
* no known 4-digit group code as "(dddd)" or "group dddd", and no string
  value or dict key EQUAL to a known 2/4-digit code (a bare code in a list
  or a keyed map) — years excepted where the code is year-shaped;
* no registry name: a value EQUAL to any registry node name under a
  naming key (`name`, `*_name`, `label`), a value equal to a multi-word
  industry/sub-industry name anywhere, and — after our own labels are
  blanked — no distinctive registry name (one with "&" or ",") at any
  level inside any string. Single common words ("Energy", "Software") and
  a sector's plain name used as ordinary English under a non-naming key
  (a source's `domain: "Health Care"`) are ordinary prose;
* `provider_industry` is exempt: it is the data provider's own string and
  is public by design (W1 §4.5).
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import CrossIndustrySnapshot, IndustryReport, IndustryReportJob, IndustryStatSnapshot
from app.services import gics_registry as reg
from app.services import industry_analytics as ia
from app.services import industry_classification as ic
from app.services import industry_labels as il
from app.services import industry_report_store as rs
from app.services import industry_report_worker as jobs
from app.tests.fixtures import industry_analyst_stub
from app.tests.gating_helpers import seed_demo_universe

AS_OF = datetime(2026, 9, 4, 21, 0)
PERIOD = "2026-W36"
LEGACY_PERIOD = "2026-W37"
LEGACY_SNAPSHOT_PERIOD = "2026-W30"
CUTOFF = AS_OF.date()


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------

_YEAR = re.compile(r"^(?:19|20)\d\d$")
_LONG_TOKEN = re.compile(r"(?<![\w$.,])(\d{8}|\d{6})(?![\w%]|[.,]\d)")
_NAME_KEYS_RE = re.compile(r"^(?:name|label|.*_name)$")
# A refusal that quotes an identifier ("industry group '45' not found"):
# the quote marks prove it is an identifier, not a count.
_QUOTED_CODE = re.compile(r"['\"](\d{2}|\d{4})['\"]")


class _Rules:
    def __init__(self) -> None:
        names = il.registry_names()
        labels = il.load()
        self.short = {c for c in names if len(c) in (2, 4)}
        self.groups = {c for c in names if len(c) == 4 and not _YEAR.match(c)}
        self.long = {c for c in names if len(c) in (6, 8)}
        self.all_names = {n for n in names.values() if n}
        self.long_multiword = {n for c, n in names.items() if len(c) in (6, 8) and n and " " in n}
        distinctive = sorted({n for n in names.values() if n and ("&" in n or "," in n)}, key=len, reverse=True)
        self.distinctive_re = re.compile(
            r"(?<![\w&])(?:" + "|".join(re.escape(n) for n in distinctive) + r")(?![\w])")
        ours = sorted({lab for lab, _ in (*labels.sectors.values(), *labels.groups.values())}, key=len, reverse=True)
        self.ours_re = re.compile("|".join(re.escape(lab) for lab in ours))
        self.group_forms = re.compile(
            r"\((" + "|".join(sorted(self.groups)) + r")\)|\bgroup\s+(" + "|".join(sorted(self.groups)) + r")(?!\d)"
            r"|Industry Group Analyst\s+\d{4}", re.I)


_RULES: _Rules | None = None


def _rules() -> _Rules:
    global _RULES
    if _RULES is None:
        _RULES = _Rules()
    return _RULES


def text_leaks(text: str) -> list[str]:
    """Why a piece of public TEXT (a string value, or a rendered block)
    would show the licensed taxonomy; empty when clean."""
    r = _rules()
    out: list[str] = []
    if il.has_brand(text):
        out.append("brand or internal key")
    if any(m.group(1) in r.long for m in _LONG_TOKEN.finditer(text)):
        out.append("industry/sub-industry code")
    if r.group_forms.search(text):
        out.append("group code form")
    if any(m.group(1) in r.short for m in _QUOTED_CODE.finditer(text)):
        out.append("quoted taxonomy code")
    if r.distinctive_re.search(r.ours_re.sub(" ", text)):
        out.append("distinctive registry name")
    return out


def public_leaks(obj: Any, path: str = "$", *, key: str = "") -> list[str]:
    """Every leak in a JSON body, as ``path: reason`` strings."""
    r = _rules()
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if il.has_brand(ks):
                found.append(f"{path}: key {ks!r} carries the brand")
            if ks in r.short:
                found.append(f"{path}: key {ks!r} is a taxonomy code")
            if ks in r.all_names:
                found.append(f"{path}: key {ks!r} is a registry name")
            if ks == "provider_industry":
                continue
            found.extend(public_leaks(v, f"{path}.{ks}", key=ks))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(public_leaks(v, f"{path}[{i}]", key=key))
    elif isinstance(obj, str):
        value = obj.strip()
        found.extend(f"{path}: {why} in {obj[:120]!r}" for why in text_leaks(obj))
        if value in r.short and not (_YEAR.match(value) and value not in r.groups):
            found.append(f"{path}: value {value!r} is a taxonomy code")
        if _NAME_KEYS_RE.match(key) and value in r.all_names:
            found.append(f"{path}: {key} {value!r} is a registry name")
        if value in r.long_multiword:
            found.append(f"{path}: value {value!r} is an industry/sub-industry registry name")
    return found


def test_the_walker_catches_what_it_is_for():
    """A clean result from a walker that sees nothing proves nothing: the
    unprojected shapes this file exists to keep off the page are all
    caught, and their projections are all clean."""
    names = il.registry_names()
    raw = {
        "code": "4530", "name": names["4530"], "sector_code": "45",
        "note": "per GICS", "taxonomy_version": "gics-2026-04",
        "text": "KPIs [453010] matter; Semis (4530) lead; group 4530; Industry Group Analyst 4530",
        "report_versions": {"4530": 2}, "missing_groups": ["4530"],
        "rows": [{"industry_code": "453010", "industry_name": names["453010"]}],
        "sub": names["45301010"],
    }
    leaks = public_leaks(raw)
    for needle in ("$.code", "$.name", "$.sector_code", "$.note", "$.taxonomy_version", "$.text",
                   "$.report_versions: key", "$.missing_groups[0]", "$.rows[0].industry_code",
                   "$.rows[0].industry_name", "$.sub"):
        assert any(leak.startswith(needle) for leak in leaks), (needle, leaks)
    assert public_leaks(il.project_public(raw)) == []
    # ...and ordinary prose is not a leak.
    assert public_leaks({"domain": "Health Care", "text": "Energy names rallied in 2030; top 10 banks",
                         "provider_industry": "Semiconductors", "label": il.label("4530")}) == []


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------


def _series(drift: float) -> list[dict[str, Any]]:
    return [
        {"date": (CUTOFF - timedelta(days=i)).isoformat(),
         "close": round(100.0 * (1 + drift * (400 - i) / 400), 4)}
        for i in range(400, -1, -1)
    ]


def _wipe(version_id: int) -> None:
    with SessionLocal() as db:
        for model in (IndustryReportJob, IndustryReport, IndustryStatSnapshot, CrossIndustrySnapshot):
            db.execute(delete(model).where(model.taxonomy_version_id == version_id))
        db.commit()


def _leaky(text: str, *, code: str, industry: str, industry_name: str) -> str:
    """Legacy analyst prose as the pre-rule prompt let a model write it."""
    return (f"{text} The GICS® industry group {code} ({code}) leads [{industry}, {industry}]; "
            f"{industry_name} names lag; Industry Group Analyst {code} notes group {code} breadth.")


@pytest.fixture(scope="module")
def world() -> Iterator[dict[str, Any]]:
    """One group with a real stub-analyst edition (v1, through the real
    worker path with prices), a LEGACY leaky analyst edition on top of it
    (v2, the served `latest`), an audit-only template (v3), a group whose
    only edition is a template, the real cross-industry snapshot and a
    legacy snapshot row whose group lists are bare codes."""
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    ic.classify_all(version=info)
    by_group = ic.constituents_by_group(version=info)
    code = ic.current_for(["NVDA"], version=info)["NVDA"]["industry_group_code"]
    floor = int(settings.industry_stats_min_sample)
    assert len(by_group.get(code, [])) >= floor, "the NVDA group must clear the sample floor"
    template_code = next(c for c, t in sorted(by_group.items()) if c != code and t)
    industry = reg.industries_of_group(code, version=info)[0]
    sub = next(s for s in reg.sub_industries_of(code, version=info)
               if s.name != reg.group(code, version=info).name and ("&" in s.name or "," in s.name))
    _wipe(info.id)

    prices = {t: _series(0.1 + 0.01 * i) for i, t in enumerate(sorted(by_group[code]))}
    groups = {code: by_group[code]}

    def loaders() -> ia.Loaders:
        return ia.Loaders(
            constituents_by_group=lambda version: dict(groups),
            companies=lambda ts: {t: {"company_name": f"{t} Co", "market_cap": 1.0e9, "is_active": True,
                                      "shares_outstanding": None, "last_price": None} for t in ts},
            metrics=lambda ts: {},
            cached_prices=lambda ts: {t: prices[t] for t in ts if t in prices},
            cached_price_tickers=lambda ts: {t for t in ts if t in prices},
            fetch_prices=lambda t: prices.get(t),
            market_factor=lambda: None,
            prior_stats=lambda c, version, key: None,
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ia, "default_loaders", loaders)
        industry_analyst_stub.install(mp)
        jobs.enqueue_period(PERIOD, [code], version=info, source="test")
        done = jobs.drain()
        assert [j["status"] for j in done] == ["succeeded", "succeeded"], done
    v1 = rs.latest_publishable(code, version=info)
    assert v1 is not None and v1["version"] == 1

    # v2 — the legacy analyst edition, written the way the pre-rule code
    # stored one: straight into the table, no validator, no projection.
    import copy
    payload = copy.deepcopy(v1["payload"])
    sections = payload["sections"]
    kw = {"code": code, "industry": industry.code, "industry_name": sub.name}
    for name in ("overview", "drivers", "outlook", "risks"):
        interp = sections[name]["interpretation"]
        interp["text"] = _leaky(interp["text"], **kw)
        interp["claims"].append({
            "type": "causal_inference", "text": _leaky("Legacy claim.", **kw),
            "basis": [f"mapping:{code}", f"mandate:{code}", f"gics:{industry.code}", f"industry:{industry.code}"],
            "falsifier": f"GICS {industry.code} {industry.name} reverses for two quarters.",
        })
    sections["outlook"]["interpretation"]["analyst_view"] = _leaky("Capacity tightens.", **kw)
    sections["overview"]["facts"]["legacy_note"] = f"per GICS {industry.code} {industry.name}"
    sections["metadata"]["facts"]["analyst"] = f"Industry Group Analyst {code}"
    with SessionLocal() as db:
        row = IndustryReport(
            taxonomy_version_id=info.id, industry_group_code=code, version=2, parent_report_id=v1["id"],
            period_key=LEGACY_PERIOD, as_of=AS_OF + timedelta(days=7), status="succeeded", is_latest_good=True,
            generation={"generation_mode": "llm", "model": "legacy-model",
                        "agent": f"Industry Group Analyst {code}", "taxonomy": "gics-2026-04"},
            degraded=[f"cross_industry:gics:{industry.code}"],
            errors=[f"themes: GICS lookup for {industry.code} failed"],
            source_manifest=[{"ref": "GICS structure workbook", "url": "https://www.msci.com/our-solutions/gics",
                              "note": f"{code} {reg.group(code, version=info).name}"}],
            stats_id=v1.get("stats_id"), generated_at=AS_OF + timedelta(days=9),
            payload=payload,
        )
        db.add(row)
        db.execute(IndustryReport.__table__.update().where(IndustryReport.id == v1["id"]).values(
            status="superseded", is_latest_good=False))
        db.commit()
    # v3 — an audit-only template: `?version=3` is an `edition_withheld` refusal.
    rs.save_report(code=code, period_key="2026-W38", as_of=AS_OF + timedelta(days=14), version=info,
                   generation={"generation_mode": "deterministic"},
                   payload={"sections": {"overview": {"facts": {}, "interpretation": {"text": "tpl"}}}})
    # A group whose only edition is a template: `no_report` with the count.
    rs.save_report(code=template_code, period_key=PERIOD, as_of=AS_OF, version=info,
                   generation={"generation_mode": "deterministic"},
                   payload={"sections": {"overview": {"facts": {}, "interpretation": {"text": "tpl"}}}})
    # A legacy snapshot row: bare-code group lists, internal key, names.
    name = reg.group(code, version=info).name
    with SessionLocal() as db:
        db.add(CrossIndustrySnapshot(
            taxonomy_version_id=info.id, period_key=LEGACY_SNAPSHOT_PERIOD, as_of=AS_OF - timedelta(days=42),
            schema_version=1, computed_at=AS_OF,
            payload={"taxonomy_version": info.version_key,
                     "groups": [{"code": code, "name": name, "status": "ok"}],
                     "missing_groups": [code], "insufficient_sample_groups": [code, template_code],
                     "spillovers": [{"id": "E1", "codes": [code, template_code], "label": f"{name} to peers",
                                     "moves": [{"code": code, "name": name}]}],
                     "major_events": [{"ticker": "NVDA", "industry_group_code": code}]},
            stats_ids=[], report_versions={code: 2, template_code: 1},
        ))
        db.commit()

    yield {"info": info, "code": code, "slug": il.slug(code), "template_code": template_code,
           "industry": industry, "sub": sub}
    _wipe(info.id)
    reg.activate_version(info.version_key)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _reads(w: dict[str, Any]) -> list[tuple[str, int]]:
    slug, code = w["slug"], w["code"]
    tslug = il.slug(w["template_code"])
    return [
        ("/api/industries/taxonomy", 200),
        (f"/api/industries/{slug}/report", 200),
        (f"/api/industries/{code}/report", 200),              # an old link's internal code
        (f"/api/industries/{slug}/report?version=1", 200),
        (f"/api/industries/{slug}/history", 200),
        (f"/api/industries/{slug}/changes", 200),
        (f"/api/industries/{slug}/changes?from=1&to=2", 200),
        (f"/api/industries/{slug}/companies", 200),
        (f"/api/industries/{code}/companies", 200),
        ("/api/industries/snapshot", 200),
        (f"/api/industries/snapshot?period_key={LEGACY_SNAPSHOT_PERIOD}", 200),
        # refusals are public bodies too
        (f"/api/industries/{tslug}/report", 404),               # no_report + withheld count
        (f"/api/industries/{w['template_code']}/report", 404),
        (f"/api/industries/{slug}/report?version=3", 404),      # edition_withheld
        (f"/api/industries/{slug}/report?version=99", 404),     # no such edition
        (f"/api/industries/{slug}/report?version=abc", 422),
        (f"/api/industries/{slug}/changes?to=3", 404),          # edition_withheld
        (f"/api/industries/{slug}/changes?from=99", 404),
        (f"/api/industries/{w['industry'].code}/report", 404),  # a 6-digit code is not a group
        (f"/api/industries/{code[:2]}/history", 404),           # nor is a sector
        (f"/api/industries/{il.slug(code[:2])}/companies", 404),
        ("/api/industries/no-such-group/report", 404),
        ("/api/industries/snapshot?period_key=2020-W01", 404),
    ]


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_no_gics_mark_or_internal_code_on_any_public_surface(world, client):
    problems: dict[str, list[str]] = {}
    for path, status in _reads(world):
        resp = client.get(path)
        assert resp.status_code == status, (path, resp.status_code, resp.text[:300])
        leaks = public_leaks(resp.json())
        if leaks:
            problems[path] = leaks[:8]
    assert problems == {}


def test_every_read_answers_with_the_slug_whichever_form_the_url_used(world, client):
    slug, code = world["slug"], world["code"]
    by_slug = client.get(f"/api/industries/{slug}/report").json()
    by_code = client.get(f"/api/industries/{code}/report").json()
    assert by_slug["code"] == by_code["code"] == slug
    assert by_slug["name"] == il.label(code) and by_slug["sector_code"] == il.slug(code[:2])
    assert by_slug["version"] == by_code["version"] == 2
    for suffix in ("companies", "history", "changes"):
        assert client.get(f"/api/industries/{slug}/{suffix}").json()["code"] == slug, suffix
    taxonomy = client.get("/api/industries/taxonomy").json()
    group = next(g for s in taxonomy["sectors"] for g in s["industry_groups"] if g["code"] == slug)
    assert group["name"] == il.label(code) and group["latest_report"]["version"] == 2
    assert taxonomy["taxonomy_version"]["key"] == il.PUBLIC_TAXONOMY_KEY


def test_the_legacy_edition_is_served_scrubbed_not_withheld(world, client):
    """The projection is the net for legacy prose: the edition is still
    served (it is a real analyst edition), with codes turned into slugs in
    its references and the brand, codes and registry names gone."""
    body = client.get(f"/api/industries/{world['slug']}/report").json()
    overview = body["payload"]["sections"]["overview"]["interpretation"]
    assert il.label(world["code"]) in overview["text"]
    legacy = next(c for c in overview["claims"] if c["type"] == "causal_inference" and "Legacy claim" in c["text"])
    assert f"mapping:{world['slug']}" in legacy["basis"] and f"mandate:{world['slug']}" in legacy["basis"]
    assert all(str(world["code"]) not in b and world["industry"].code not in b for b in legacy["basis"])
    assert body["payload"]["sections"]["metadata"]["facts"]["analyst"] == il.public_display_name(world["code"])
    assert body["attribution"] and not il.has_brand(body["attribution"])
    assert body["mapping_caveat"] == il.PUBLIC_MAPPING_CAVEAT


def test_the_no_report_refusal_names_the_group_by_label(world, client):
    detail = client.get(f"/api/industries/{il.slug(world['template_code'])}/report").json()["detail"]
    assert detail["code"] == "no_report" and detail["withheld_editions"] == 1
    assert detail["name"] == il.label(world["template_code"])
    assert detail["industry_group_code"] == il.slug(world["template_code"])
    assert il.label(world["template_code"]) in detail["message"]
    assert detail["taxonomy_version"] == il.PUBLIC_TAXONOMY_KEY


def test_portfolio_exposure_block_is_public(world, client):
    resp = client.post("/api/portfolio/build", json={"market_view": "soft landing with AI capex", "num_holdings": 8})
    assert resp.status_code == 200, resp.text
    block = resp.json()["industry_exposure"]
    assert block["status"] == "ok" and block["by_group"]
    assert public_leaks(block) == []


def test_pm_block_and_chat_payload_are_public(world):
    """The PM writes public memo prose from its block, and the chat tool's
    answer reaches a user verbatim: both are public surfaces. The legacy
    edition's leaky analyst view is the one quoted."""
    from app.agents import pm_context

    block = pm_context.industry_context_block(ticker="NVDA", tickers=["JPM", "ZZNOPE"])
    assert block and f"Industry group {il.label(world['code'])}" in block
    assert "Capacity tightens." in block
    assert text_leaks(block) == [], block
    codes = {c for c in il.registry_names() if len(c) == 4 and not _YEAR.match(c)}
    assert not [c for c in codes if re.search(rf"(?<!\d){c}(?!\d)", block)], block

    payload = pm_context.industry_context_payload(tickers=["NVDA", "JPM", "ZZNOPE"], code=world["slug"])
    assert payload["status"] == "ok" and payload["code"]["code"] == world["slug"]
    assert public_leaks(payload) == []
    by_code = pm_context.industry_context_payload(code=world["code"])
    assert by_code["code"] == payload["code"]
