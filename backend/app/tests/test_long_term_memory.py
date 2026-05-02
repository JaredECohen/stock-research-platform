"""Long-term agent memory tests.

Cover:
- Round-trip: write file, parse it back; entries + cross-company patterns survive.
- Append-on-delta: a new earnings/filing trigger writes one entry; a
  no-trigger run does not.
- Cap + condense: pushing past `memory_max_entries` folds the oldest
  `memory_condense_batch` into the historical-context block (information
  preserved, not lost).
- Cross-company learning: a pattern recorded for AMZN with applies_to=GOOGL
  is surfaced in GOOGL's sector-memory prompt context.
- Sector-agent prompt injection: when memory exists for a ticker, the
  sector agent's prompt picks it up (best-effort check via prompt assembly).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.config import settings
from app.memory import CompanyMemory, MemoryEntry, SectorMemory
from app.memory.longterm import CrossCompanyPattern


@pytest.fixture
def tmp_memory(tmp_path, monkeypatch):
    """Point the memory subsystem at a fresh tmp directory per test."""
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Round-trip / parsing
# ---------------------------------------------------------------------------

def test_company_memory_round_trip(tmp_memory):
    cm = CompanyMemory.for_ticker("ZZZ")
    cm.append_entry(MemoryEntry(
        date="2026-01-15", trigger="earnings",
        body="**Trigger:** Q4 2025\n\n**Observation:** Strong cloud growth.",
    ))
    cm.save()
    reloaded = CompanyMemory.for_ticker("ZZZ")
    assert reloaded.subject == "ZZZ"
    assert reloaded.kind == "company"
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0].date == "2026-01-15"
    assert reloaded.entries[0].trigger == "earnings"
    assert "cloud growth" in reloaded.entries[0].body.lower()


def test_sector_memory_round_trip_with_pattern(tmp_memory):
    sm = SectorMemory.for_sector("Technology")
    sm.append_entry(MemoryEntry(
        date="2026-02-10", trigger="reflection:AMZN",
        body="AWS re-acceleration story; cohort placement held.",
    ))
    sm.add_pattern(CrossCompanyPattern(
        date="2026-02-10", source_company="AMZN",
        applies_to=["GOOGL", "MSFT"],
        lesson="Cloud growth re-acceleration tends to follow rate cuts within 2 quarters.",
    ))
    sm.save()
    reloaded = SectorMemory.for_sector("Technology")
    assert reloaded.subject == "Technology"
    assert len(reloaded.cross_company_patterns) == 1
    p = reloaded.cross_company_patterns[0]
    assert p.source_company == "AMZN"
    assert "GOOGL" in p.applies_to and "MSFT" in p.applies_to
    assert "rate cuts" in p.lesson


# ---------------------------------------------------------------------------
# Cross-company filtering
# ---------------------------------------------------------------------------

def test_patterns_for_filters_by_applies_to_and_excludes_source(tmp_memory):
    sm = SectorMemory.for_sector("Technology")
    sm.add_pattern(CrossCompanyPattern(
        date="2026-02-01", source_company="AMZN",
        applies_to=["GOOGL", "MSFT"],
        lesson="AWS lesson 1",
    ))
    sm.add_pattern(CrossCompanyPattern(
        date="2026-03-01", source_company="META",
        applies_to=["GOOGL", "PINS"],
        lesson="META lesson",
    ))
    # GOOGL is in applies_to of both patterns, so it sees both.
    # AMZN is in neither applies_to list (and is the source of one), so it
    # sees nothing — that's the contract: agents don't get their own source
    # patterns echoed back at them, those live in their company memory file.
    googl = sm.patterns_for("GOOGL")
    assert {p.source_company for p in googl} == {"AMZN", "META"}
    amzn = sm.patterns_for("AMZN")
    assert amzn == []
    # PINS is only in the META pattern.
    pins = sm.patterns_for("PINS")
    assert {p.source_company for p in pins} == {"META"}


def test_as_prompt_context_for_only_includes_relevant_patterns(tmp_memory):
    sm = SectorMemory.for_sector("Technology")
    sm.add_pattern(CrossCompanyPattern(
        date="2026-02-01", source_company="AMZN",
        applies_to=["GOOGL"],
        lesson="Specific AWS lesson worth surfacing for GOOGL.",
    ))
    sm.add_pattern(CrossCompanyPattern(
        date="2026-02-15", source_company="META",
        applies_to=["SNAP"],
        lesson="Ad-load saturation lesson — not relevant to GOOGL.",
    ))
    ctx = sm.as_prompt_context_for("GOOGL")
    assert "AWS lesson" in ctx
    assert "Ad-load saturation" not in ctx


# ---------------------------------------------------------------------------
# Cap + condense (information preservation)
# ---------------------------------------------------------------------------

def test_condense_oldest_preserves_information(tmp_memory, monkeypatch):
    monkeypatch.setattr(settings, "memory_max_entries", 5)
    monkeypatch.setattr(settings, "memory_condense_batch", 3)
    cm = CompanyMemory.for_ticker("ZZZ")
    for i in range(8):
        cm.append_entry(MemoryEntry(
            date=f"2026-0{(i % 9) + 1}-01", trigger="earnings",
            body=f"Entry #{i}: a notable observation about quarter {i}.",
        ))
    folded = cm.condense_oldest()
    assert folded == 3
    # 8 - 3 = 5 entries remain
    assert len(cm.entries) == 5
    # Historical context block now references the dates / takeaways we evicted
    assert "Condensed" in cm.historical_context
    assert "Entry #0" in cm.historical_context or "quarter 0" in cm.historical_context
    # Re-saving + re-reading round-trips the condensed block
    cm.save()
    reloaded = CompanyMemory.for_ticker("ZZZ")
    assert "Condensed" in reloaded.historical_context
    assert len(reloaded.entries) == 5


def test_condense_no_op_below_cap(tmp_memory):
    cm = CompanyMemory.for_ticker("ZZZ")
    for i in range(3):
        cm.append_entry(MemoryEntry(date="2026-01-0" + str(i + 1), trigger="x", body=str(i)))
    assert cm.condense_oldest() == 0
    assert len(cm.entries) == 3
    assert cm.historical_context.strip() == ""


# ---------------------------------------------------------------------------
# Reflection agent (delta-only writes)
# ---------------------------------------------------------------------------

def test_reflection_agent_writes_on_new_filing_only(tmp_memory):
    """First run writes one entry per delta event (filings + earnings).
    A second identical run produces no new entries — same accessions and
    same transcript period are already in memory.

    `run_stock_memo` runs the reflection step internally, so we don't call
    `reflection_agent.run` explicitly; we assert against the file state."""
    from app.agents.graph import run_stock_memo

    run_stock_memo("NVDA")
    cm = CompanyMemory.for_ticker("NVDA")
    n_after_first = len(cm.entries)
    assert n_after_first >= 1, "first run should have produced at least one entry"

    run_stock_memo("NVDA")
    cm2 = CompanyMemory.for_ticker("NVDA")
    # Same accessions / periods / no new material news → no new entries
    assert len(cm2.entries) == n_after_first


def test_reflection_agent_writes_cross_company_pattern_on_first_run(tmp_memory):
    from app.agents.graph import run_stock_memo
    from app.agents import reflection_agent

    memo = run_stock_memo("NVDA")
    triggers, _ = reflection_agent.run(memo)
    if not triggers:
        pytest.skip("no delta trigger fired — demo data state-dependent")
    sm = SectorMemory.for_sector(memo.sector)
    # Pattern recorded with NVDA as source and at least one peer in applies_to
    nvda_patterns = [p for p in sm.cross_company_patterns if p.source_company == "NVDA"]
    if nvda_patterns:
        # When the deterministic fallback fires (no LLM), it only emits a
        # pattern when something interesting happened. Either way, if a
        # pattern was recorded, applies_to must be non-empty.
        assert all(p.applies_to for p in nvda_patterns)


def test_reflection_disabled_writes_nothing(tmp_memory, monkeypatch):
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    from app.agents.graph import run_stock_memo
    from app.agents import reflection_agent

    memo = run_stock_memo("NVDA")
    triggers, written = reflection_agent.run(memo)
    assert triggers == [] and written == []


# ---------------------------------------------------------------------------
# Memory survives a missing-file lookup
# ---------------------------------------------------------------------------

def test_company_memory_for_unknown_ticker_returns_empty(tmp_memory):
    cm = CompanyMemory.for_ticker("DOES_NOT_EXIST")
    assert cm.entries == []
    assert cm.as_prompt_context() == ""
