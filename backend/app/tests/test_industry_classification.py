"""FEAT-003 slice 1 — company → GICS industry-group classification.

The classification is a derived crosswalk (we hold no GICS codes for any
company), so the tests pin the derivation rules rather than any
particular company's group: research-map precedence over the provider
alias, the explicit states, staleness detected by fingerprint and fixed
by superseding (never rewriting) the old row, bulk reads on the web
process, and the daily loop's honest note.

Companies inserted here carry a ``ZZ`` prefix and are removed on
teardown; the demo universe is classified by ticker list so unmappable
companies other suites insert never leak into the "zero missing" claim.
"""
from __future__ import annotations

import copy

import pytest
from sqlalchemy import delete, event, update

from app.database import SessionLocal, engine
from app.models import Company, CompanyIndustryClassification, TaxonomyVersion
from app.monitoring import industry_classification_loop as loop
from app.services import gics_registry as reg
from app.services import industry_classification as ic
from app.services import industry_knowledge as ik
from app.tests.fixtures.demo_dataset import COMPANY_PROFILES
from app.tests.gating_helpers import seed_demo_universe

DEMO_TICKERS = sorted(COMPANY_PROFILES)


@pytest.fixture(scope="module", autouse=True)
def _universe():
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    yield info
    reg.activate_version(info.version_key)


@pytest.fixture()
def company():
    """Insert throwaway companies; delete them and their rows afterwards."""
    made: list[str] = []

    def make(ticker: str, sector: str, industry: str, sub_industry: str | None = None) -> str:
        with SessionLocal() as db:
            db.merge(Company(
                ticker=ticker, company_name=f"{ticker} Test Co", exchange="TEST",
                sector=sector, industry=industry, sub_industry=sub_industry,
                universe_tier="data_only",
            ))
            db.commit()
        made.append(ticker)
        return ticker

    yield make
    with SessionLocal() as db:
        db.execute(delete(CompanyIndustryClassification).where(
            CompanyIndustryClassification.ticker.in_(made)))
        db.execute(delete(Company).where(Company.ticker.in_(made)))
        db.commit()


def _set_labels(ticker: str, *, industry: str) -> None:
    with SessionLocal() as db:
        db.execute(update(Company).where(Company.ticker == ticker).values(industry=industry))
        db.commit()


# --- normalisation and the alias map ----------------------------------------


def test_normalize_label_folds_case_punctuation_and_ampersands():
    assert ic.normalize_label("Software - Application ") == "software application"
    assert ic.normalize_label("Oil & Gas E&P") == "oil and gas e and p"
    assert ic.normalize_label("TECHNOLOGY") == ic.normalize_label("Technology") == "technology"
    assert ic.normalize_label("  Consumer   Cyclical ") == "consumer cyclical"
    assert ic.normalize_label(None) == "" and ic.normalize_label("") == ""


def test_every_alias_code_exists_in_the_active_taxonomy():
    """A typo in provider_aliases.json would silently turn a label into
    `missing`; the registry is the only authority on which codes exist."""
    aliases = ic.alias_map()
    assert aliases.version and aliases.as_of and aliases.provider == "fmp"
    for key, code in aliases.sectors.items():
        assert reg.node(code) is not None and reg.node(code).level == "sector", (key, code)
    for key, code in aliases.industries.items():
        node = reg.node(code)
        assert node is not None and node.level in ("industry_group", "industry"), (key, code)


def test_demo_fixture_labels_all_resolve_through_the_alias_map():
    aliases = ic.alias_map()
    for ticker, profile in COMPANY_PROFILES.items():
        assert ic.normalize_label(profile["sector"]) in aliases.sectors, (ticker, profile["sector"])
        assert ic.normalize_label(profile["industry"]) in aliases.industries, (ticker, profile["industry"])


# --- classify_all over the demo universe --------------------------------------


def test_demo_universe_classifies_with_no_missing_and_every_sector_mapped():
    summary = ic.classify_all(tickers=DEMO_TICKERS)
    assert summary["classified"] == len(DEMO_TICKERS)
    assert summary["counts"]["missing"] == 0
    assert summary["counts"]["fallback"] == 0
    assert summary["counts"]["mapped"] + summary["counts"]["conflict"] == len(DEMO_TICKERS)
    assert summary["unmapped_labels"] == []

    current = ic.current_for(DEMO_TICKERS)
    assert set(current) == set(DEMO_TICKERS)
    for sector in {p["sector"] for p in COMPANY_PROFILES.values()}:
        tickers = [t for t, p in COMPANY_PROFILES.items() if p["sector"] == sector]
        assert any(current[t]["state"] == "mapped" for t in tickers), sector
    for row in current.values():
        assert row["is_current"] is True
        assert row["industry_group_code"] == reg.group_code_for(row["industry_group_code"])
        assert row["mapping_caveat"] == reg.MAPPING_CAVEAT
        assert row["author"] and row["source_as_of"]

    # A second pass over identical inputs changes nothing.
    again = ic.classify_all(tickers=DEMO_TICKERS)
    assert again["inserted"] == 0 and again["reclassified"] == 0
    assert again["unchanged"] == len(DEMO_TICKERS)


def test_research_map_wins_and_records_its_provenance():
    ref = ik.security_reference("NVDA")
    assert ref is not None, "NVDA is in the map's security reference"
    # Forced so the row was written by THIS code, not left over in the
    # shared test database from an earlier revision with the same outcome.
    ic.classify_ticker("NVDA", force=True)
    row = ic.current_for(["NVDA"])["NVDA"]
    assert row["state"] == "mapped"
    assert row["source"] == "research_map" and row["method"] == "security_reference"
    assert row["sub_industry_code"] == ref["codes"][0]
    assert row["sub_industry_codes"] == ref["codes"]
    assert row["industry_group_code"] == ref["codes"][0][:4]
    assert row["industry_code"] == ref["codes"][0][:6]
    assert row["sector_code"] == ref["codes"][0][:2]
    assert row["author"] == f"Investment_Universe_163_Map.json@{ik.security_reference_as_of()}"
    assert row["source_as_of"] == ref["as_of"]
    assert row["evidence"]["research_map_source_id"] == ref["source_id"]
    assert row["evidence"]["provider_alias_agrees"] is True
    assert row["confidence"] == ic.CONFIDENCE[("research_map", "mapped")]


def test_security_reference_normalises_share_class_separators():
    """The map spells share classes with a hyphen; providers do not agree.
    Exact spelling first, then the separator swapped — never anything
    fuzzier (a separator-free symbol is a different ticker)."""
    hyphenated = next(
        (s for s in ik.load_industry_knowledge()["security_reference"] if "-" in s), None,
    )
    assert hyphenated is not None, "the map carries at least one share-class symbol"
    base, cls = hyphenated.split("-", 1)
    exact = ik.security_reference(hyphenated)
    assert exact is not None and exact["symbol"] == exact["matched_symbol"] == hyphenated
    for spelling in (f"{base}.{cls}", f"{base}/{cls}", f"{base} {cls}", f" {base.lower()}.{cls.lower()} "):
        ref = ik.security_reference(spelling)
        assert ref is not None, spelling
        assert ref["matched_symbol"] == hyphenated and ref["codes"] == exact["codes"]
        assert ref["symbol"] == spelling.strip().upper()
    assert ik.security_reference(f"{base}{cls}") is None  # BRKB is not BRK-B
    assert ik.security_reference("") is None
    plain = next(s for s in ik.load_industry_knowledge()["security_reference"] if "-" not in s)
    assert ik.security_reference(plain)["matched_symbol"] == plain


def test_research_map_matches_a_provider_spelled_share_class(company):
    hyphenated = next(s for s in ik.load_industry_knowledge()["security_reference"] if "-" in s)
    ref = ik.security_reference(hyphenated)
    provider_symbol = hyphenated.replace("-", ".")
    company(provider_symbol, sector="Zzz Sector", industry="No Such Industry")
    ic.classify_ticker(provider_symbol, force=True)
    row = ic.current_for([provider_symbol])[provider_symbol]
    assert row["state"] == "mapped" and row["source"] == "research_map"
    assert row["sub_industry_code"] == ref["codes"][0]
    assert row["evidence"]["research_map_symbol"] == hyphenated
    # An exact match carries no normalisation note.
    ic.classify_ticker("NVDA", force=True)
    assert "research_map_symbol" not in ic.current_for(["NVDA"])["NVDA"]["evidence"]


def test_provider_alias_is_the_fallback_when_the_map_has_no_entry():
    ticker = next(t for t in DEMO_TICKERS if ik.security_reference(t) is None)
    ic.classify_all(tickers=[ticker])
    row = ic.current_for([ticker])[ticker]
    code = ic.alias_map().industries[ic.normalize_label(COMPANY_PROFILES[ticker]["industry"])]
    assert row["state"] == "mapped"
    assert row["source"] == "provider_alias" and row["method"] == "alias_industry"
    assert row["industry_group_code"] == code[:4]
    assert row["sub_industry_code"] is None and row["sub_industry_codes"] == []
    assert row["author"] == f"provider_aliases.json@{ic.alias_map().version}"
    assert row["source_as_of"] == ic.alias_map().as_of
    assert row["evidence"]["alias_key"] == ic.normalize_label(COMPANY_PROFILES[ticker]["industry"])


# --- the explicit states -------------------------------------------------------


def test_sector_only_resolution_is_fallback_not_mapped(company):
    t = company("ZZFALL", "Technology", "No Such Provider Label")
    row = ic.classify_ticker(t)
    assert row is not None and row["changed"] is True
    assert row["state"] == "fallback" and row["source"] == "provider_alias"
    assert row["method"] == "alias_sector"
    assert row["sector_code"] == "45" and row["industry_group_code"] is None
    assert row["evidence"]["unmapped_industry_label"] == "No Such Provider Label"
    # A fallback row knows its sector, not its group: never a constituent.
    assert t not in ic.constituents("4510") and t not in ic.constituents("4530")


def test_unknown_labels_are_missing(company):
    t = company("ZZMISS", "Zzz Unknown Sector", "Qqq Unknown Industry")
    row = ic.classify_ticker(t)
    assert row["state"] == "missing" and row["source"] == "none"
    assert row["industry_group_code"] is None and row["sector_code"] is None
    assert row["confidence"] == 0.0
    assert row["evidence"]["unmapped_sector_label"] == "Zzz Unknown Sector"
    audit = ic.audit()
    assert t in audit["tickers_by_state"]["missing"]
    assert {"sector": "Zzz Unknown Sector", "industry": "Qqq Unknown Industry", "count": 1} in audit["unmapped_labels"]
    assert audit["mapping_caveat"] == reg.MAPPING_CAVEAT
    assert audit["alias_map"]["version"] == ic.alias_map().version


def test_conflict_when_map_and_alias_disagree_on_the_group(company, monkeypatch):
    t = company("ZZCONF", "Technology", "Software")
    alias_code = ic.alias_map().industries["software"]
    energy = next(s["code"] for s in ik.list_sub_industries("1010"))
    assert energy[:4] != alias_code[:4]
    real = ik.security_reference

    def fake(symbol):
        if str(symbol).upper() == t:
            return {"symbol": t, "codes": [energy], "as_of": "2026-09-10",
                    "source_id": "S29", "caveat": ik.SECURITY_REFERENCE_CAVEAT}
        return real(symbol)

    monkeypatch.setattr(ic.industry_knowledge, "security_reference", fake)
    row = ic.classify_ticker(t)
    assert row["state"] == "conflict"
    assert row["source"] == "research_map"  # the map wins routing
    assert row["industry_group_code"] == energy[:4]
    assert row["sub_industry_code"] == energy
    assert [c["source"] for c in row["candidates"]] == ["research_map", "provider_alias"]
    assert row["candidates"][1]["industry_group_code"] == alias_code[:4]
    assert "disagree" in row["evidence"]["conflict"]
    assert row["confidence"] == ic.CONFIDENCE[("research_map", "conflict")]
    # Conflict rows are constituents of the group the map chose.
    assert t in ic.constituents(energy[:4]) and t not in ic.constituents(alias_code[:4])
    assert t not in ic.constituents(energy[:4], states=("mapped",))


# --- staleness and history -------------------------------------------------------


def test_stale_rows_are_reclassified_by_superseding_and_history_survives():
    ticker = next(t for t in DEMO_TICKERS if ik.security_reference(t) is None)
    original = COMPANY_PROFILES[ticker]["industry"]
    ic.classify_all(tickers=[ticker])
    before = ic.current_for([ticker])[ticker]
    software = ic.alias_map().industries["software"]
    assert before["industry_group_code"] != software[:4]
    try:
        _set_labels(ticker, industry="Software")

        detect = ic.classify_all(tickers=[ticker], reclassify=False)
        assert detect["stale_detected"] == 1 and detect["stale_fixed"] == 0
        stale = ic.current_for([ticker])[ticker]
        assert stale["id"] == before["id"] and stale["state"] == "stale"
        assert stale["evidence"]["previous_state"] == "mapped"
        assert stale["evidence"]["stale_reason"] == "inputs_changed"
        assert detect["counts"]["stale"] == 1

        fixed = ic.classify_all(tickers=[ticker])
        assert fixed["stale_detected"] == 1 and fixed["stale_fixed"] == 1 and fixed["reclassified"] == 1
        assert fixed["changed"] == [{
            "ticker": ticker, "from": before["industry_group_code"], "to": software[:4],
            "from_state": "mapped", "to_state": "mapped",
        }]
        after = ic.current_for([ticker])[ticker]
        assert after["id"] != before["id"]
        assert after["industry_group_code"] == software[:4]
        assert after["source_industry"] == "Software"

        rows = ic.history(ticker)
        old = next(r for r in rows if r["id"] == before["id"])
        assert old["is_current"] is False and old["superseded_at"] is not None
        assert old["superseded_reason"] == "inputs_changed"
        assert old["state"] == "stale" and old["evidence"]["previous_state"] == "mapped"
        assert rows[-1]["id"] == after["id"] and rows[-1]["is_current"] is True
    finally:
        _set_labels(ticker, industry=original)
        ic.classify_all(tickers=[ticker])
    restored = ic.current_for([ticker])[ticker]
    assert restored["industry_group_code"] == before["industry_group_code"]
    assert [r["is_current"] for r in ic.history(ticker)].count(True) == 1


def test_classify_ticker_is_a_no_op_on_identical_inputs_and_force_supersedes():
    ic.classify_all(tickers=["MSFT"])
    same = ic.classify_ticker("MSFT")
    assert same["changed"] is False
    forced = ic.classify_ticker("MSFT", force=True)
    assert forced["changed"] is True and forced["id"] != same["id"]
    old = next(r for r in ic.history("MSFT") if r["id"] == same["id"])
    assert old["superseded_reason"] == "forced" and old["is_current"] is False
    assert ic.classify_ticker("msft ")["id"] == forced["id"]


def test_classify_ticker_returns_none_without_company_or_taxonomy(monkeypatch):
    assert ic.classify_ticker("ZZNOPE") is None
    assert ic.classify_ticker("") is None
    active = reg.active_version()
    with SessionLocal() as db:
        db.execute(update(TaxonomyVersion).values(is_active=False))
        db.commit()
    try:
        assert ic.classify_ticker("NVDA") is None
        with pytest.raises(reg.TaxonomyNotImported):
            ic.classify_all(tickers=["NVDA"])
    finally:
        reg.activate_version(active.version_key)


# --- reads used by the memo run and the analytics -------------------------------


def test_current_for_is_one_select():
    ic.classify_all(tickers=DEMO_TICKERS)
    statements: list[str] = []

    def spy(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", spy)
    try:
        rows = ic.current_for(DEMO_TICKERS)
    finally:
        event.remove(engine, "before_cursor_execute", spy)
    assert len(rows) == len(DEMO_TICKERS)
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1, statements
    assert ic.current_for([]) == {}
    assert ic.current_for(["ZZNOPE"]) == {}


def test_constituents_by_group_counts_come_from_rows():
    ic.classify_all(tickers=DEMO_TICKERS)
    by_group = ic.constituents_by_group()
    current = ic.current_for(DEMO_TICKERS)
    for ticker, row in current.items():
        assert ticker in by_group[row["industry_group_code"]]
    for code, tickers in by_group.items():
        assert reg.group(code).level == "industry_group"
        assert tickers == sorted(set(tickers))
        assert set(ic.constituents(code)) >= set(t for t in tickers if t in current)
    nvda = current["NVDA"]
    assert "NVDA" in ic.constituents(nvda["sub_industry_code"])
    assert "NVDA" in ic.constituents(nvda["industry_code"])
    assert ic.constituents("12") == [] and ic.constituents("abcd") == []


def test_inputs_fingerprint_tracks_labels_alias_and_map_editions(monkeypatch):
    a = ic.inputs_fingerprint("Technology", "Software", None)
    assert a == ic.inputs_fingerprint("Technology", "Software", None)
    assert a != ic.inputs_fingerprint("Technology", "Software ", None)
    monkeypatch.setattr(ic.industry_knowledge, "security_reference_as_of", lambda: "1999-01-01")
    assert a != ic.inputs_fingerprint("Technology", "Software", None)


# --- the on-demand hook ---------------------------------------------------------


def test_lazy_universe_hook_classifies_a_new_symbol_and_never_fails_the_request(monkeypatch):
    from app.api import routes_stocks

    seen: list[str] = []
    monkeypatch.setattr(routes_stocks, "ensure_company_in_universe", lambda t: {"ticker": t})
    monkeypatch.setattr(routes_stocks, "backfill_ticker", lambda t: None)
    monkeypatch.setattr(ic, "classify_ticker", lambda t, **kw: seen.append(t))
    assert routes_stocks._ensure_lazy_universe("zzhook") == "analyzed_on_demand"
    assert seen == ["ZZHOOK"]

    def boom(t, **kw):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(ic, "classify_ticker", boom)
    assert routes_stocks._ensure_lazy_universe("zzhook") == "analyzed_on_demand"


# --- the daily loop ---------------------------------------------------------------


@pytest.fixture()
def recorded(monkeypatch):
    calls: list[tuple[tuple, dict]] = []
    real = loop.record_run

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        real(*args, **kwargs)

    monkeypatch.setattr(loop, "record_run", spy)
    return calls


def _summary(**counts) -> dict:
    base = {"mapped": 0, "fallback": 0, "missing": 0, "conflict": 0, "stale": 0}
    base.update(counts)
    return {
        "taxonomy_version": "gics-test", "classified": sum(base.values()),
        "inserted": 1, "reclassified": 2, "stale_fixed": 1, "counts": base,
        "unmapped_labels": [
            {"sector": "Zzz", "industry": f"Label {i}", "count": 1} for i in range(7)
        ] if base["missing"] else [],
    }


def test_loop_reports_success_only_when_nothing_is_missing(recorded, monkeypatch):
    monkeypatch.setattr(loop.industry_classification, "classify_all",
                        lambda **kw: _summary(mapped=30, fallback=2, conflict=1))
    loop.run_once()
    (args, kwargs), = recorded
    assert args == (loop.LOOP_NAME,) and kwargs["success"] is True
    note = kwargs["note"]
    assert "classified=33" in note and "mapped=30" in note and "fallback=2" in note
    assert "missing=0" in note and "conflict=1" in note and "stale_fixed=1" in note
    assert "unmapped_labels" not in note

    recorded.clear()
    monkeypatch.setattr(loop.industry_classification, "classify_all",
                        lambda **kw: _summary(mapped=30, missing=2))
    loop.run_once()
    (args, kwargs), = recorded
    assert kwargs["success"] is False
    assert "missing=2" in kwargs["note"]
    assert "unmapped_labels=Zzz/Label 0×1" in kwargs["note"]
    assert "(+2 more)" in kwargs["note"]  # capped at 5 labels in the note


def test_loop_goes_red_on_taxonomy_drift(recorded, monkeypatch):
    """Regression: a regenerated knowledge JSON under the active key left
    the loop reporting success while the registry served the old nodes."""
    monkeypatch.setattr(loop.industry_classification, "classify_all",
                        lambda **kw: _summary(mapped=30))
    drift = {
        "kind": "same_key_changed_structure", "active_version_key": "gics-test",
        "bundled_version_key": "gics-test", "active_checksum": "a" * 64,
        "bundled_checksum": "b" * 64, "remedy": "re-import",
    }
    monkeypatch.setattr(loop.gics_registry, "bundled_drift", lambda active: drift)
    summary = loop.run_once()
    assert summary["taxonomy_drift"] is drift
    (args, kwargs), = recorded
    assert kwargs["success"] is False
    note = kwargs["note"]
    assert "missing=0" in note and "taxonomy_drift=1" in note
    assert f"bundled=gics-test@{'b' * 12}" in note
    assert f"active_checksum={'a' * 12}" in note and "drift_kind=same_key_changed_structure" in note

    recorded.clear()
    monkeypatch.setattr(loop.gics_registry, "bundled_drift", lambda active: None)
    assert loop.run_once()["taxonomy_drift"] is None
    (args, kwargs), = recorded
    assert kwargs["success"] is True and "taxonomy_drift=0" in kwargs["note"]


def test_loop_reads_drift_from_the_real_registry(recorded, monkeypatch):
    payload = copy.deepcopy(ik.load_industry_knowledge())
    payload["sectors"][0]["industry_groups"][0]["name"] += " (renamed)"
    monkeypatch.setattr(reg, "_load_payload", lambda path: payload)
    summary = loop.run_once()
    assert summary["taxonomy_drift"]["kind"] == "same_key_changed_structure"
    assert summary["taxonomy_drift"]["active_checksum"] == reg.active_version().checksum
    (args, kwargs), = recorded
    assert kwargs["success"] is False and "taxonomy_drift=1" in kwargs["note"]


def test_loop_records_failure_before_propagating(recorded, monkeypatch):
    def boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(loop.industry_classification, "classify_all", boom)
    with pytest.raises(RuntimeError):
        loop.run_once()
    (args, kwargs), = recorded
    assert kwargs["success"] is False and "error=RuntimeError" in kwargs["note"]


def test_loop_runs_against_the_real_registry(recorded):
    summary = loop.run_once()
    assert summary["taxonomy_version"] == reg.active_version().version_key
    assert summary["classified"] >= len(DEMO_TICKERS)
    (args, kwargs), = recorded
    assert f"taxonomy={summary['taxonomy_version']}" in kwargs["note"]
    assert "classified=" in kwargs["note"] and "missing=" in kwargs["note"]
    assert kwargs["success"] is (summary["counts"]["missing"] == 0)


def test_loop_is_registered_daily_after_history_backfill():
    from app.monitoring import KNOWN_LOOPS, history_backfill

    class Sched:
        def add_job(self, fn, trigger, **kw):
            self.kw = kw
            self.trigger = trigger

    s = Sched()
    loop.register(s)
    assert s.trigger == "cron" and s.kw["id"] == loop.LOOP_NAME
    assert (s.kw["hour"], s.kw["minute"]) == (loop.HOUR_UTC, loop.MINUTE_UTC)
    # After history_backfill has refreshed the company profiles it classifies.
    h = Sched()
    history_backfill.register(h)
    assert (loop.HOUR_UTC, loop.MINUTE_UTC) > (h.kw["hour"], h.kw["minute"])
    assert loop.LOOP_NAME in KNOWN_LOOPS and "industry_weekly_loop" in KNOWN_LOOPS
