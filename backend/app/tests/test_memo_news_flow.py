"""FIX-018 / slice G1 end to end: one news snapshot per memo run reaches the
sector analyst, the routed industry-group analyst, intake and the PM.

Before G1 the same `news_hot` row reached the sector analyst as 600
characters of JSON, the industry analyst not at all, intake never, and the
PM only nested in the sector entry of its Findings JSON with no instruction.

Each test runs a demo memo (no keys) with `llm.chat_json` replaced by a spy
that records every prompt and answers nothing, so every agent takes its
deterministic path and the prompts are exactly what a live model would get.
The ticker's `news_hot` rows are invalidated around every test so no other
test's memo reads them.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from app.agents import graph, intake, news_agent, news_context, prompts
from app.agents import industry_analysts as ia
from app.agents import llm as llm_mod
from app.agents.news_context import EMPTY_SECTOR_LINE
from app.agents.source_ledger import SourceLedger
from app.cache import cache_put, invalidate
from app.config import settings
from app.services import gics_registry as reg
from app.services import industry_classification as ic
from app.tests.gating_helpers import seed_demo_universe

TICKER = "JPM"
MARKER = "JPMorgan wins MARKER-7Q9 custody mandate"
SUMMARY_MARKER = "SUMMARY-7Q9 the mandate covers 1.2 trillion in assets"


@pytest.fixture(scope="module", autouse=True)
def _universe():
    seed_demo_universe()
    assert reg.ensure_taxonomy(activate=True) is not None
    ic.classify_all(tickers=[TICKER])


@pytest.fixture(autouse=True)
def _news_rows():
    ia.clear_cache()
    invalidate(f"news_hot:{TICKER}", "news_hot")
    cache_put(f"news_hot:{TICKER}", "news_hot", payload={"ticker": TICKER, "alerts": [
        {"ticker": TICKER, "sector": None, "title": MARKER, "summary": SUMMARY_MARKER,
         "url": "https://www.reuters.com/business/jpm-custody", "severity": "material",
         "published_at": None, "source": "news_service"},
    ]}, generated_by="test", ttl_seconds=4 * 3600)
    yield
    invalidate(f"news_hot:{TICKER}", "news_hot")
    ia.clear_cache()


def _spy(monkeypatch) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {"sector": [], "industry": [], "pm": [], "other": []}

    def spy(prompt: str, **kw: Any) -> None:
        if prompt.startswith(prompts.PM_SYNTHESIS_PROMPT):
            seen["pm"].append(prompt)
        elif prompt.startswith("You are a sector analyst"):
            seen["sector"].append(prompt)
        elif llm_mod.current_call_context().get("agent_name") == ia.AGENT_NAME:
            seen["industry"].append(prompt)
        else:
            seen["other"].append(prompt)
        return None

    monkeypatch.setattr(llm_mod, "chat_json", spy)
    return seen


def test_one_news_snapshot_reaches_sector_industry_and_pm(monkeypatch):
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    seen = _spy(monkeypatch)
    memo = graph.run_stock_memo(TICKER)

    (sector,) = seen["sector"]
    slot = sector.split("You will also be handed", 1)[0]
    assert MARKER in slot and EMPTY_SECTOR_LINE not in slot
    (pm,) = seen["pm"]
    head, findings = pm.split("\n\nFindings:\n", 1)
    assert head.index(f"## Recent news for {TICKER}") > len(prompts.PM_SYNTHESIS_PROMPT)
    # The rendered item line. (Long-term memory may also quote an earlier
    # run's headline elsewhere in a prompt; the block line is what counts.)
    (line,) = [ln for ln in head.splitlines() if ln.startswith("1. [material]") and MARKER in ln]
    assert SUMMARY_MARKER in line
    # All three read the same rendered item line.
    assert line in slot
    (industry,) = seen["industry"]
    assert industry.index("Ratios (observed): ") < industry.index(line)
    # The duplicate read is gone from the JSON copy; the stored memo keeps it.
    assert "pending_news_alerts" not in findings and MARKER not in findings
    stored = memo.sector_agent_view.data["pending_news_alerts"]
    assert [a["title"] for a in stored] == [MARKER]


def test_skipped_sector_still_leaves_the_pm_news_block(monkeypatch):
    """Intake may skip the sector analyst (debate off). The PM's block comes
    from the run's context, not from the sector finding, so the PM still
    reads the news."""
    seen = _spy(monkeypatch)
    monkeypatch.setattr(intake, "run_intake",
                        lambda *a, **k: intake.IntakeDecision(skipped={"sector"}, rationale="r"))
    graph.run_stock_memo(TICKER)
    assert seen["sector"] == []
    (pm,) = seen["pm"]
    head = pm.split("\n\nFindings:\n", 1)[0]
    assert f"## Recent news for {TICKER}" in head
    assert any(MARKER in ln and ln.startswith("1. [material]") for ln in head.splitlines())


def test_the_memo_hands_intake_the_run_news(monkeypatch):
    """REGRESSION: graph called `run_intake` with no news, so intake was
    always told there was none."""
    _spy(monkeypatch)
    handed: list[Any] = []

    def spy_intake(profile, news_alerts=None, **kw):
        handed.append(news_alerts)
        return intake.IntakeDecision()

    monkeypatch.setattr(intake, "run_intake", spy_intake)
    graph.run_stock_memo(TICKER)
    (alerts,) = handed
    assert [a["title"] for a in alerts] == [MARKER]


def test_load_failure_gives_empty_context_and_consistent_ledger(monkeypatch):
    """A failed read is an empty context, not a sector analyst reading the
    cache on its own: every reader then agrees there is no news, and the
    ledger holds no news the models were not shown (news critique)."""
    seen = _spy(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("news read failed")

    monkeypatch.setattr(news_context, "load_for_memo", boom)
    registered: list[tuple[str, str]] = []
    real_register = SourceLedger.register

    def recording(self, kind, ref, obj, **kw):
        registered.append((kind, ref))
        return real_register(self, kind, ref, obj, **kw)

    monkeypatch.setattr(SourceLedger, "register", recording)
    memo = graph.run_stock_memo(TICKER)
    (sector,) = seen["sector"]
    # The template's news slot (long-term memory, further down, may quote an
    # earlier run's headlines; that is memory, not this run's news read).
    slot = sector.split("You will also be handed", 1)[0]
    assert EMPTY_SECTOR_LINE in slot and MARKER not in slot and "<news>" not in sector
    (pm,) = seen["pm"]
    assert "## Recent news" not in pm and "<news>" not in pm
    assert not [r for r in registered if r[0] == "news"]
    assert memo.sector_agent_view.data["pending_news_alerts"] == []


def test_gather_registers_the_news_once(monkeypatch):
    _spy(monkeypatch)
    registered: list[tuple[str, str, Any]] = []
    real_register = SourceLedger.register

    def recording(self, kind, ref, obj, **kw):
        registered.append((kind, ref, obj))
        return real_register(self, kind, ref, obj, **kw)

    monkeypatch.setattr(SourceLedger, "register", recording)
    graph.run_stock_memo(TICKER)
    news = [(ref, obj) for kind, ref, obj in registered if kind == "news"]
    # Once, under the ref the block tells the PM to cite, with the shown text.
    assert news == [(f"news_alerts:{TICKER}", {"items": [{"title": MARKER, "summary": SUMMARY_MARKER}]})]


_REFS_HEADER = "## Source refs (for forecast_assumptions basis_ref)\n"


def _refs(pm_prompt: str) -> list[str]:
    head = pm_prompt.split("\n\nFindings:\n", 1)[0]
    return head.split(_REFS_HEADER, 1)[1].split("\n", 1)[0].split(", ")


def test_pm_refs_offer_news_only_when_there_is_news(monkeypatch):
    """Declared change (G1 review): before G1 the sector analyst registered
    `news_alerts:{T}` even for an empty alert list, so every no-news PM
    prompt offered it as a `basis_ref` and an assumption citing it resolved
    against nothing. The ref is now offered only with news behind it; this
    is the one byte change to a no-news PM prompt."""
    seen = _spy(monkeypatch)
    graph.run_stock_memo(TICKER)
    (pm,) = seen["pm"]
    assert f"news_alerts:{TICKER}" in _refs(pm)

    invalidate(f"news_hot:{TICKER}", "news_hot")
    seen = _spy(monkeypatch)
    graph.run_stock_memo(TICKER)
    (pm,) = seen["pm"]
    assert "## Recent news" not in pm
    refs = _refs(pm)
    assert refs and f"news_alerts:{TICKER}" not in refs


def test_backtest_memo_reads_and_fetches_no_news(monkeypatch):
    """REGRESSION guard (G1 review): the gather stage must hand
    `load_for_memo` the run's `as_of_date`. Without it a backtest reads the
    live cache under an `:asof:` subject, finds nothing, and (on a live
    deploy) makes a grounded fetch that feeds today's headlines into a
    historical memo: look-ahead plus spend. The fetch switch is forced on
    here so dropping the argument is visible in demo mode."""
    seen = _spy(monkeypatch)
    monkeypatch.setattr(news_context, "_should_fetch", lambda: True)
    fetched: list[str] = []
    monkeypatch.setattr(news_agent, "run", lambda ticker, **kw: fetched.append(ticker))
    graph.run_stock_memo(TICKER, as_of_date=date.today() - timedelta(days=120))
    assert fetched == []
    (sector,) = seen["sector"]
    slot = sector.split("You will also be handed", 1)[0]
    assert EMPTY_SECTOR_LINE in slot and MARKER not in slot
    (pm,) = seen["pm"]
    assert "## Recent news" not in pm and "<news>" not in pm
