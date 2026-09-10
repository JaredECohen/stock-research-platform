"""FEAT-003 slice 3 — the PM's cross-industry snapshot.

What can rot here: a group without stats rendered as zeros instead of
listed as missing, a PM block that outgrows its 2,000-character budget
as the registry grows, a dependency edge presented without its source or
as a correlation, more linked groups than the cap, and a period computed
twice producing two rows. Each has a test.

Stats rows are hand-built dicts in the shape ``industry_analytics.
stats_dict`` returns, injected through ``SnapshotLoaders``; the registry
supplies the group list so no count is ever typed here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import delete

from app.config import settings
from app.database import SessionLocal
from app.models import CrossIndustrySnapshot
from app.services import gics_registry as reg
from app.services import industry_classification as ic
from app.services import industry_snapshot as isn
from app.tests.fixtures.demo_dataset import COMPANY_PROFILES
from app.tests.gating_helpers import seed_demo_universe

AS_OF = datetime(2026, 9, 4, 21, 0)
PERIOD = "2026-W36"


@pytest.fixture(scope="module", autouse=True)
def _taxonomy():
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    ic.classify_all(tickers=sorted(COMPANY_PROFILES), version=info)
    yield info
    with SessionLocal() as db:
        db.execute(delete(CrossIndustrySnapshot).where(
            CrossIndustrySnapshot.taxonomy_version_id == info.id, CrossIndustrySnapshot.period_key == PERIOD,
        ))
        db.commit()
    reg.activate_version(info.version_key)


def _stats(code: str, *, sid: int, ret_1m: float | None, breadth: float | None = 0.6, rel: float | None = None,
           n: int = 5, status: str = "ok", tickers: tuple[str, ...] = ("T1", "T2")) -> dict[str, Any]:
    entry = {"equal_weight": ret_1m, "median": ret_1m, "market_cap_weight": ret_1m, "n": n}
    if ret_1m is None:
        entry["reason"] = "history_window"
    return {
        "id": sid, "code": code, "period_key": PERIOD, "as_of": AS_OF.isoformat(),
        "sample": {"n_with_prices": n, "n_constituents": n},
        "per_ticker": {t: {"last_close": 1.0} for t in tickers},
        "payload": {
            "status": status,
            "returns": {"1w": {"equal_weight": 0.01, "n": n}, "1m": entry, "ytd": {"equal_weight": None, "n": 0, "reason": "history_window"}},
            "benchmark_relative": {"universe_ew": {"1m": {"value": rel if rel is not None else ret_1m}},
                                   "sector_ew": {"1m": {"value": None, "reason": "missing"}},
                                   "KFR.MKT_RF.D": {"1m": {"value": None, "reason": "missing"}}},
            "breadth": {"1m": {"pct_positive": breadth, "n": n}},
            "dispersion": {"stdev": 0.02, "n": n},
            "valuation": {"ev_ebitda": {"median": 12.5, "n": n}, "pe_ttm": {"median": None, "reason": "no_metric_values"}},
            "fundamental_momentum": {"value": None, "reason": "no_prior_period"},
        },
    }


def _loaders(stats: dict[str, dict[str, Any]], *, macro: dict[str, Any] | None = None,
             events: list[dict[str, Any]] | None = None) -> isn.SnapshotLoaders:
    return isn.SnapshotLoaders(
        stats=lambda period_key, version: stats,
        macro=lambda: macro,
        events=lambda tickers, cutoff, days: [e for e in (events or []) if e["ticker"] in tickers],
        report_versions=lambda version: {code: 1 for code in stats},
    )


def _codes(n: int) -> list[str]:
    return [g.code for g in reg.industry_groups()][:n]


# --- compute ------------------------------------------------------------------


def test_groups_without_stats_are_listed_as_missing_with_counts_from_the_registry(_taxonomy):
    a, b = _codes(2)
    stats = {a: _stats(a, sid=1, ret_1m=0.08), b: _stats(b, sid=2, ret_1m=-0.03, n=2, status="insufficient_sample")}
    row = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(stats))
    groups = reg.industry_groups(version=_taxonomy)
    payload = row.payload
    assert payload["coverage"] == {"n_groups": len(groups), "n_with_stats": 2, "n_insufficient_sample": 1}
    missing = {m["code"] for m in payload["missing_groups"]}
    assert missing == {g.code for g in groups} - {a, b}
    assert all(m["reason"] == "no_stats_for_period" for m in payload["missing_groups"])
    by_code = {r["code"]: r for r in payload["groups"]}
    assert by_code[a]["ret_1m_ew"] == 0.08 and by_code[a]["regime_label"] == "broad leadership"
    assert by_code[b]["regime_label"] == "insufficient_sample"
    absent = next(iter(missing))
    assert by_code[absent]["status"] == "no_stats" and by_code[absent]["ret_1m_ew"] is None
    assert by_code[a]["reasons"] == {"ret_ytd_ew": "history_window", "rel_1m_vs_sector": "missing",
                                     "rel_1m_vs_market_factor": "missing", "val_median_pe_ttm": "no_metric_values",
                                     "fund_momentum": "no_prior_period"}
    assert payload["regime"]["macro_regime"] is None and payload["regime"]["reason"] == "no_macro_broadcast"
    assert row.stats_ids == [1, 2] and row.report_versions == {a: 1, b: 1}


def test_regime_labels_follow_the_stated_rules():
    assert isn.regime_label(None) == "no_stats"
    assert isn.regime_label(_stats("x", sid=1, ret_1m=0.05, rel=0.03, breadth=0.7)) == "broad leadership"
    assert isn.regime_label(_stats("x", sid=1, ret_1m=0.05, rel=0.03, breadth=0.4)) == "narrow leadership"
    assert isn.regime_label(_stats("x", sid=1, ret_1m=-0.05, rel=-0.03, breadth=0.3)) == "broad weakness"
    assert isn.regime_label(_stats("x", sid=1, ret_1m=-0.05, rel=-0.03, breadth=0.5)) == "lagging (mixed breadth)"
    assert isn.regime_label(_stats("x", sid=1, ret_1m=0.01, rel=0.0)) == "in line with universe"
    assert isn.regime_label(_stats("x", sid=1, ret_1m=None)) == "no_return_data"
    assert isn.regime_label(_stats("x", sid=1, ret_1m=0.0, status="insufficient_sample")) == "insufficient_sample"
    assert "leading" in isn.REGIME_RULES and "broad" in isn.REGIME_RULES


def test_spillovers_carry_their_source_and_a_signal_never_a_correlation(_taxonomy):
    graph = isn.dependency_graph()
    assert graph["available"] and graph["links"]
    assert all(link["codes"] for link in graph["links"])  # unlabelled edges never become links
    edge = next(link for link in graph["links"] if link["kind"] == "edge")
    rel = next(link for link in graph["links"] if link["kind"] == "relationship")
    assert edge["source"] == "atlas_dependencies_sheet"
    assert rel["source"] == "universe_map_cross_industry_relationships"
    moved = edge["codes"][0]
    stats = {moved: _stats(moved, sid=1, ret_1m=0.09)}
    row = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(stats))
    spill = {s["id"]: s for s in row.payload["spillovers"]}
    assert spill[edge["id"]]["signal"] == "active" and spill[edge["id"]]["source"] == edge["source"]
    assert spill[edge["id"]]["n_observed"] == 1
    assert "hypothes" in spill[edge["id"]]["status"].lower() or "relationship" in spill[edge["id"]]["status"].lower()
    assert "not estimated correlations" in row.payload["dependency_graph"]["caveat"]
    unobserved = [s for s in row.payload["spillovers"] if moved not in s["codes"]]
    assert all(s["signal"] == "unobserved" for s in unobserved)
    dormant = {moved: _stats(moved, sid=1, ret_1m=0.01)}
    row = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(dormant))
    assert {s["id"]: s for s in row.payload["spillovers"]}[edge["id"]]["signal"] == "dormant"


def test_events_are_counted_per_group_and_macro_regime_is_carried(_taxonomy):
    a, b = _codes(2)
    stats = {a: _stats(a, sid=1, ret_1m=0.02, tickers=("AAA", "BBB")), b: _stats(b, sid=2, ret_1m=0.0, tickers=("CCC",))}
    events = [
        {"ticker": "AAA", "event_type": "earnings", "event_date": "2026-09-10", "title": "Q3", "materiality": "high", "source": "t"},
        {"ticker": "CCC", "event_type": "conference", "event_date": "2026-09-12", "title": "c", "materiality": "low", "source": "t"},
        {"ticker": "ZZZ", "event_type": "earnings", "event_date": "2026-09-12", "title": "z", "materiality": "high", "source": "t"},
    ]
    macro = {"regime": "Late cycle", "favored_sectors": ["Energy"], "pressured_sectors": [], "as_of": "2026-09-04T20:00:00"}
    row = isn.compute_cross_snapshot(
        PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(stats, macro=macro, events=events),
    )
    by_code = {r["code"]: r for r in row.payload["groups"]}
    assert by_code[a]["events_14d"] == 1 and by_code[b]["events_14d"] == 1
    assert row.payload["n_events"] == 2  # ZZZ is not a covered constituent
    assert row.payload["major_events"][0]["ticker"] == "AAA" and row.payload["major_events"][0]["industry_group_code"] == a
    assert row.payload["regime"]["macro_regime"] == "Late cycle" and row.payload["regime"]["source"] == "macro_broadcast"


def test_persisting_the_same_period_twice_updates_one_row(_taxonomy):
    a = _codes(1)[0]
    first = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, loaders=_loaders({a: _stats(a, sid=1, ret_1m=0.02)}))
    second = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, loaders=_loaders({a: _stats(a, sid=7, ret_1m=0.05)}))
    assert first.id == second.id and second.stats_ids == [7]
    latest = isn.latest_snapshot(version=_taxonomy)
    assert latest["id"] == first.id and latest["payload"]["groups"][0]["code"]
    assert isn.snapshot_for_period(PERIOD, version=_taxonomy)["id"] == first.id


# --- render ------------------------------------------------------------------


def test_render_pm_block_stays_within_2000_chars_for_the_whole_registry(_taxonomy):
    groups = reg.industry_groups(version=_taxonomy)
    stats = {g.code: _stats(g.code, sid=i + 1, ret_1m=0.03 * (i % 5 - 2)) for i, g in enumerate(groups)}
    row = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(stats))
    block = isn.render_pm_block(row)
    assert len(block) <= isn.PM_BLOCK_MAX_CHARS
    assert block.startswith(f"Cross-industry snapshot {PERIOD}")
    assert f"{len(groups)}/{len(groups)} groups with stats" in block
    first_line = block.split("\n")[1]
    assert first_line.startswith(groups[0].code) or "|" in first_line
    # Over-generous budgets are clipped to the requested size, tight ones say what was dropped.
    tight = isn.render_pm_block(row, max_chars=700)
    assert len(tight) <= 700 and "omitted for length" in tight
    assert isn.render_pm_block(None) == ""


def test_render_shows_missing_evidence_as_n_a_and_names_missing_groups(_taxonomy):
    a, b = _codes(2)
    stats = {a: _stats(a, sid=1, ret_1m=None, breadth=None), b: _stats(b, sid=2, ret_1m=0.0, status="insufficient_sample")}
    row = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(stats))
    block = isn.render_pm_block(row)
    line_a = next(line for line in block.split("\n") if line.startswith(a))
    assert "n/a" in line_a and "0.0%" not in line_a.split("|")[1]
    line_b = next(line for line in block.split("\n") if line.startswith(b))
    assert "insufficient_sample" in line_b
    n_missing = len(reg.industry_groups(version=_taxonomy)) - 2
    assert f"No stats this period for {n_missing} group(s)" in block
    # Groups without stats are named in the tail, not padded into the table.
    absent = next(m["code"] for m in row.payload["missing_groups"])
    assert not any(line.startswith(absent + " ") for line in block.split("\n")[1:])


# --- relevance ------------------------------------------------------------------


def test_relevant_groups_return_own_groups_then_capped_linked_groups(_taxonomy, monkeypatch):
    nvda = ic.current_for(["NVDA"], version=_taxonomy)["NVDA"]["industry_group_code"]
    linked_all = isn.linked_group_codes(nvda)
    assert linked_all, "the fixture group should carry at least one dependency link"
    detail = isn.relevant_groups_detail(["NVDA", "ZZNOTATICKER"], cap=1, version=_taxonomy)
    assert detail["own"] == [nvda] and len(detail["linked"]) == 1
    assert detail["linked"][0]["code"] in linked_all and detail["linked"][0]["via"][0]["source"]
    assert detail["unmapped"] == [{"ticker": "ZZNOTATICKER", "state": "unclassified"}]
    assert isn.relevant_groups_for("NVDA", cap=0, version=_taxonomy) == [nvda]
    monkeypatch.setattr(settings, "industry_pm_max_linked_groups", 2)
    assert len(isn.relevant_groups_for(["NVDA"], version=_taxonomy)) == 1 + min(2, len(linked_all))
    # Portfolio-wide: the union of own groups, then linked groups not already owned.
    jpm = ic.current_for(["JPM"], version=_taxonomy)["JPM"]["industry_group_code"]
    both = isn.relevant_groups_detail(["NVDA", "JPM"], cap=2, version=_taxonomy)
    assert both["own"] == [nvda, jpm] and all(item["code"] not in (nvda, jpm) for item in both["linked"])
    assert both["n_linked_candidates"] >= len(both["linked"])


def test_group_rows_name_codes_the_snapshot_does_not_carry(_taxonomy):
    a = _codes(1)[0]
    snap = {"payload": {"groups": [{"code": a, "ret_1m_ew": 0.1}]}}
    assert isn.group_rows(snap, [a, "9999"]) == [{"code": a, "ret_1m_ew": 0.1}, {"code": "9999", "status": "not_in_snapshot"}]
