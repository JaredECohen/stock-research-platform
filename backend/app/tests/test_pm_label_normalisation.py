"""L6 (TradingAgents lessons, 2026-09-25) and the PM Synthesis attribution
(FIX-019), slice G1.

Pinned here:
- the PM's rating label is normalised by an EXACT match after strip,
  whitespace collapse and case-fold; never a substring or prefix match;
- an off-enum label takes the no-usable-synthesis path with a visible soft
  degradation instead of failing the whole memo run (it raised a
  ValidationError in compose before);
- where the rating came from is recorded (`scores.rating_source_llm`,
  `scores.pm_rating_score`), with no schema change;
- end to end under a fake provider client, the PM Synthesis row is
  attributed: agent "PM Synthesis", action `pm.synthesis`, role "pm", the
  run's ticker, and it is not an unattributed call.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from app.agents import graph, llm, llm_attribution, prompts
from app.config import settings
from app.schemas import score_from_rating_label
from app.tests import llm_fakes

_PM_REPLY = {"final_pm_view": "PM view.", "one_sentence_thesis": "Thesis.", "confidence_score": 62}


@pytest.mark.parametrize("raw, want", [
    ("Bullish", "Bullish"),
    ("bullish ", "Bullish"),
    ("  very   BULLISH\n", "Very Bullish"),
    ("NEUTRAL", "Neutral"),
    ("very bearish", "Very Bearish"),
    ("Moderately Bullish", None),     # never a substring match
    ("Bullish.", None),
    ("Buy", None),
    ("Sell-side", None),
    ("Bull", None),                   # never a prefix match
    ("", None),
    (None, None),
    (70, None),
])
def test_pm_label_normalised_exact(raw, want):
    assert graph.normalize_rating_label(raw) == want


def _spy_pm(monkeypatch, label: Any) -> list[str]:
    seen: list[str] = []

    def spy(prompt, **kw):
        if prompt.startswith(prompts.PM_SYNTHESIS_PROMPT):
            seen.append(prompt)
            return {**_PM_REPLY, "rating_label": label}
        return None

    monkeypatch.setattr(llm, "chat_json", spy)
    return seen


def test_pm_synthesis_returns_the_canonical_label(monkeypatch):
    _spy_pm(monkeypatch, "bullish ")
    out = graph._pm_synthesis({"ticker": "TEST"}, {}, None)
    assert out["rating_label"] == "Bullish" and out[graph.RATING_SOURCE_KEY] == "llm"
    assert out["final_pm_view"] == "PM view."


def test_invalid_pm_label_degrades_not_crashes(monkeypatch):
    """REGRESSION: an off-enum label passed straight into `StockMemoOut`,
    whose `RatingLabel` is a strict Literal, and the ValidationError failed
    the whole memo run."""
    seen = _spy_pm(monkeypatch, "Moderately Bullish")
    memo = graph.run_stock_memo("MSFT")
    assert seen, "the PM model was never asked"
    assert memo.rating_label in ("Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish")
    assert "PM Synthesis" in memo.degraded_agents
    (event,) = [e for e in memo.degradation_events if e["agent"] == "PM Synthesis"]
    assert "invalid rating_label" in event["message"]
    # The deterministic view shipped, not the model's prose under a guessed label.
    assert memo.final_pm_view != "PM view." and "PM view." not in memo.final_pm_view
    assert memo.scores["rating_source_llm"] == 0.0


def test_rating_source_recorded(monkeypatch):
    # The keyword (deterministic) PM: demo mode, no model.
    memo = graph.run_stock_memo("MSFT")
    assert memo.scores["rating_source_llm"] == 0.0
    assert memo.scores["pm_rating_score"] in (10.0, 30.0, 50.0, 70.0, 90.0)
    assert memo.scores["pm_rating_score"] == score_from_rating_label(
        memo.quality.rating_reconciliation.pm_rating)

    # The LLM PM, with a label only normalisation makes valid.
    _spy_pm(monkeypatch, "  very bullish ")
    memo = graph.run_stock_memo("MSFT")
    assert memo.scores["rating_source_llm"] == 1.0
    # The PM's own label before risk recs, the blend and 7(b) moved it.
    assert memo.scores["pm_rating_score"] == 90.0
    assert memo.quality.rating_reconciliation.pm_rating == "Very Bullish"


def test_p6_keys_recorded_but_never_shown_to_the_legacy_critic(monkeypatch):
    """REGRESSION (G1 review): the P6 keys were written into `memo.scores` at
    compose, and `_review_memo` dumps the whole draft into the legacy Risk
    Committee prompt. With both modes off that changed the bytes the live
    critic reads and pushed memo text out of its 60k window (plan P4)."""
    monkeypatch.setattr(settings, "enable_agent_critic", True)
    assert settings.debate_mode == "off" and settings.reviewer_mode == "legacy"
    critic_prompts: list[str] = []

    def spy(prompt, **kw):
        if llm.current_call_context().get("agent_name") == "Risk Committee":
            critic_prompts.append(prompt)
        return None

    monkeypatch.setattr(llm, "chat_json", spy)
    memo = graph.run_stock_memo("MSFT")
    assert critic_prompts, "the critic model was never asked"
    for prompt in critic_prompts:
        assert "rating_source_llm" not in prompt and "pm_rating_score" not in prompt
    # Still recorded on the stored memo.
    assert memo.scores["rating_source_llm"] == 0.0
    assert memo.scores["pm_rating_score"] == score_from_rating_label(
        memo.quality.rating_reconciliation.pm_rating)


# --- FIX-019 end to end -----------------------------------------------------------------


class _PMClient(llm_fakes.FakeClient):
    """Answers the PM synthesis with a valid reply and every other call with
    an empty object (each specialist then takes its deterministic path)."""

    def _next(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        text = json.dumps(kwargs.get("messages") or kwargs.get("input") or "", default=str)
        if json.dumps(prompts.PM_SYNTHESIS_PROMPT[:200])[1:-1] in text:
            return llm_fakes.openai_response(json.dumps({**_PM_REPLY, "rating_label": "Neutral"}))
        return llm_fakes.openai_response("{}")


def test_pm_synthesis_row_attributed_under_fake_client(monkeypatch):
    """REGRESSION (FIX-019, G0 end to end): the PM Synthesis call named no
    action, and the run context named an umbrella agent, so its row could
    not be told apart from any other call of the run."""
    monkeypatch.setattr(settings, "llm_attribution_mode", "warn")
    llm_fakes.live(monkeypatch, openai=_PMClient(llm_fakes.openai_response("{}")))
    llm_attribution.reset_violations()
    run_id = f"g1-pm-{uuid.uuid4().hex[:8]}"
    memo = graph.run_stock_memo("MSFT", run_id=run_id)
    assert memo.scores["rating_source_llm"] == 1.0
    rows = llm_fakes.rows_for(run_id)
    pm = [r for r in rows if r.action == "pm.synthesis"]
    assert pm, [(r.agent_name, r.action) for r in rows]
    assert all(r.agent_name == "PM Synthesis" and r.role == "pm" and r.ticker == "MSFT" for r in pm)
    # The specialists this slice owns are attributed as analysts.
    sector = [r for r in rows if r.action == "analyst.sector"]
    assert sector and all(r.agent_name == "Sector Analyst" and r.role == "analyst" for r in sector)
    # No call from the files this slice owns is an unattributed site.
    owned = ("app/agents/graph.py", "app/agents/sector_agents.py", "app/agents/industry_analysts.py",
             "app/agents/intake.py", "app/agents/roster.py", "app/agents/news_context.py")
    assert not [site for site in llm_attribution.VIOLATIONS if site.startswith(owned)]
    # The run context is an umbrella: no row carries its old agent name.
    assert not [r for r in rows if r.agent_name == "run_stock_memo"]
