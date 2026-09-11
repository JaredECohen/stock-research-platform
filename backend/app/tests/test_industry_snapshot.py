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
             events: list[dict[str, Any]] | None = None,
             news: list[dict[str, Any]] | None = None) -> isn.SnapshotLoaders:
    return isn.SnapshotLoaders(
        stats=lambda period_key, version: stats,
        macro=lambda: macro,
        events=lambda tickers, cutoff, days: [e for e in (events or []) if e["ticker"] in tickers],
        news=lambda tickers, cutoff, days: [e for e in (news or []) if e["ticker"] in tickers],
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


def test_render_never_exceeds_its_budget_and_always_counts_what_it_dropped(_taxonomy):
    """The budget is a hard ceiling AND the block stays honest inside it.

    The failure this guards against is a block that fits by being cut
    mid-sentence while still reading as a complete table: every line the
    renderer emits must be a whole line it would have emitted with an
    unlimited budget, and any group line it left out must be counted in
    the closing note.
    """
    groups = reg.industry_groups(version=_taxonomy)
    with_stats = groups[: max(1, len(groups) - 3)]
    stats = {g.code: _stats(g.code, sid=i + 1, ret_1m=0.03 * (i % 5 - 2)) for i, g in enumerate(with_stats)}
    row = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(stats))
    whole = set(isn.render_pm_block(row, max_chars=100_000).split("\n"))
    n_lines = sum(1 for line in whole if line[:4].isdigit() and "|" in line)
    assert n_lines == len(with_stats)

    for budget in range(240, 2400, 11):
        block = isn.render_pm_block(row, max_chars=budget)
        assert len(block) <= budget, f"budget {budget} overrun: {len(block)}"
        shown = 0
        for line in block.split("\n"):
            if line.startswith("… omitted for length:") or line == "… truncated for length.":
                continue
            assert line in whole, f"budget {budget} emitted a partial line: {line!r}"
            if line[:4].isdigit() and "|" in line:
                shown += 1
        if shown < n_lines:
            assert "omitted for length" in block, f"budget {budget} dropped {n_lines - shown} line(s) silently"
            if "… truncated for length." not in block:
                note = next(line for line in block.split("\n") if line.startswith("… omitted for length:"))
                assert f"{n_lines - shown} of {n_lines} group line(s)" in note, note
        else:
            assert "group line(s)" not in block


def test_unlabelled_dependency_edges_are_counted_and_never_become_spillovers(_taxonomy):
    """An Atlas edge whose exposures resolve to no industry group is
    unlabelled: it names no groups, so it can say nothing about any. It
    must be dropped from the links AND counted, not silently absent and
    not attached to an arbitrary group. Counts come from the JSON.
    """
    import json

    raw = json.loads(isn.ATLAS_PATH.read_text(encoding="utf-8"))
    unlabelled_ids = {str(e.get("edge_id")) for e in raw["edges"] if not (e.get("industry_group_codes") or [])}
    assert unlabelled_ids, "fixture expectation: the Atlas carries at least one unresolved edge"

    graph = isn.dependency_graph()
    labelled = [link for link in graph["links"] if link["kind"] == "edge"]
    assert len(labelled) == len(raw["edges"]) - len(unlabelled_ids)
    assert graph["unlabelled_edges"] == len(unlabelled_ids)
    assert not (unlabelled_ids & {link["id"] for link in labelled})

    a = _codes(1)[0]
    row = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, persist=False,
                                     loaders=_loaders({a: _stats(a, sid=1, ret_1m=0.09)}))
    assert not (unlabelled_ids & {s["id"] for s in row.payload["spillovers"]})
    assert row.payload["dependency_graph"]["unlabelled_edges"] == len(unlabelled_ids)
    # Every edge that DID survive names at least one group, so its signal
    # is a statement about observed moves rather than an empty claim.
    assert all(s["codes"] for s in row.payload["spillovers"])


# --- event channels -----------------------------------------------------------


def test_both_event_channels_are_read_and_each_says_why_it_is_empty(_taxonomy):
    """The snapshot's major events come from the catalyst calendar AND the
    stored news. A channel that returns nothing must say so under
    ``events_sources`` — otherwise a short list reads as a quiet week when
    it may just be a channel nobody consulted."""
    a, b = _codes(2)
    stats = {a: _stats(a, sid=1, ret_1m=0.02, tickers=("AAA", "BBB")), b: _stats(b, sid=2, ret_1m=0.0, tickers=("CCC",))}
    events = [
        {"ticker": "AAA", "event_type": "earnings", "event_date": "2026-09-10", "title": "Q3",
         "materiality": "medium", "source": "fmp"},
    ]
    news = [
        {"ticker": "CCC", "event_type": "news", "event_date": "2026-09-02", "title": "Plant fire",
         "materiality": "high", "source": "news_agent", "url": "https://example.test/1"},
        {"ticker": "ZZZ", "event_type": "news", "event_date": "2026-09-02", "title": "Ignored",
         "materiality": "high", "source": "news_agent", "url": ""},
    ]
    row = isn.compute_cross_snapshot(
        PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(stats, events=events, news=news),
    )
    payload = row.payload
    assert payload["n_events"] == 1 and payload["n_news"] == 1  # ZZZ is not a covered constituent
    by_code = {r["code"]: r for r in payload["groups"]}
    assert by_code[a]["events_14d"] == 1 and by_code[a]["news_14d"] == 0
    assert by_code[b]["events_14d"] == 0 and by_code[b]["news_14d"] == 1
    # The high-materiality news item outranks the medium catalyst.
    assert [(e["ticker"], e["kind"]) for e in payload["major_events"]] == [("CCC", "news"), ("AAA", "catalyst")]
    sources = payload["events_sources"]
    assert sources["catalysts"]["n"] == 1 and sources["news"]["n"] == 1
    assert "never a live provider call" in sources["news"]["source"]
    assert "trailing" in sources["news"]["window"] and "forward" in sources["catalysts"]["window"]

    quiet = isn.compute_cross_snapshot(
        PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=_loaders(stats, events=events),
    )
    assert quiet.payload["events_sources"]["news"]["reason"] == "none_stored_in_window"
    assert quiet.payload["events_sources"]["catalysts"]["reason"] is None


def test_a_failing_event_channel_is_named_not_swallowed(_taxonomy):
    a = _codes(1)[0]
    stats = {a: _stats(a, sid=1, ret_1m=0.02, tickers=("AAA",))}

    def boom(tickers, cutoff, days):
        raise RuntimeError("news store down")

    loaders = _loaders(stats)
    loaders.news = boom
    row = isn.compute_cross_snapshot(PERIOD, AS_OF, version=_taxonomy, persist=False, loaders=loaders)
    assert row.payload["events_sources"]["news"] == {
        "n": 0, "reason": "read_failed:RuntimeError",
        "window": f"trailing {isn.EVENT_WINDOW_DAYS} days to the as-of",
        "source": "stored news_hot snapshots, material/breaking only (never a live provider call)",
    }
    assert row.payload["n_news"] == 0


def test_the_news_loader_reads_stored_alerts_only_and_filters_by_severity_and_window():
    """The real loader: material and breaking alerts the news agent already
    persisted, inside the trailing window, newest snapshot per ticker. A
    provider is never called — ``news_service`` is one HTTP call per
    ticker and a universe-wide weekly pass cannot loop one."""
    from datetime import date, timedelta

    from app.cache import cache_put

    cutoff = date(2026, 9, 4)
    ticker = "NEWSTEST"
    cache_put(
        f"news_hot:{ticker}", "news_hot",
        payload={"ticker": ticker, "alerts": [
            {"title": "Recall announced", "severity": "breaking", "published_at": "2026-09-02", "url": "u1",
             "source": "news_service"},
            {"title": "Analyst chatter", "severity": "routine", "published_at": "2026-09-02", "url": "u2"},
            {"title": "Old guidance cut", "severity": "material",
             "published_at": (cutoff - timedelta(days=isn.EVENT_WINDOW_DAYS + 5)).isoformat(), "url": "u3"},
        ]},
        generated_by="test", cost_tokens=0,
    )
    rows = isn._db_news([ticker, "NOSUCHTICKER"], cutoff, isn.EVENT_WINDOW_DAYS)
    assert [r["title"] for r in rows] == ["Recall announced"]
    assert rows[0] == {
        "ticker": ticker, "event_type": "news", "event_date": "2026-09-02", "title": "Recall announced",
        "materiality": "high", "source": "news_service", "url": "u1",
    }
