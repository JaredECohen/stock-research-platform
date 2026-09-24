"""S13 — with routing ON, nothing a routed memo produces or sends carries the
taxonomy: no "GICS", no code this company's classification or mandate carries,
no internal version key, no registry name that differs from our label.

`test_memo_industry_labels` checks the industry surfaces one by one. Routing
adds two channels those checks cannot see, so this walks everything at once:

- the WHOLE memo JSON (every view, every `long_form_report`, the thesis, the
  PM view, `agent_influence` keys, degradation messages), because a leak into
  a field nobody thought to name is still a leak;
- the PM Synthesis prompt, which now carries the routed read as a digest
  ahead of the capped findings JSON (and the PM can echo it into
  `final_pm_view`);
- the Risk Committee prompt, which carries the draft memo.

The analyst's model is stubbed to answer with deliberately leaky prose — the
brand, the group code, a sub-industry code, the registry name — so the scrub
has something to remove and the digest is really in the PM prompt (a keyless
demo read is a template and is withheld from the PM).
"""
from __future__ import annotations

import pytest

from app.agents import graph, prompts
from app.agents import industry_analysts as ia
from app.agents import llm as llm_mod
from app.config import settings
from app.services import gics_registry as reg
from app.services import industry_classification as ic
from app.tests.gating_helpers import seed_demo_universe
from app.tests.test_memo_industry_labels import _leaks, _leaky_prose

TICKERS = ("MSFT", "NVDA", "JPM")


@pytest.fixture(scope="module", autouse=True)
def _universe():
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    ic.classify_all(tickers=list(TICKERS))
    yield info


@pytest.fixture(autouse=True)
def _fresh_cache():
    ia.clear_cache()
    yield
    ia.clear_cache()


def _leaky_read(analyst: ia.IndustryAnalyst, sub_code: str) -> dict:
    leaky = _leaky_prose(analyst, sub_code)
    return {
        "headline": leaky, "summary": leaky, "key_points": [leaky, f"Unit volumes [{sub_code}]"],
        "confidence": 0.75, "mandate_type": "inflection", "placement": leaky,
        "causal_chain": [{"stage": "world_change", "text": leaky}],
        "kpis_to_watch": [{"kpi": "Unit volumes", "industry_code": sub_code, "why": leaky}],
        "falsifiers": [leaky, f"GICS {analyst.code} rerates"], "traps": [leaky],
    }


@pytest.mark.parametrize("ticker", TICKERS)
def test_routed_memo_json_and_pm_risk_prompts_carry_no_taxonomy_leak(monkeypatch, ticker):
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    row = ic.current_for([ticker])[ticker]
    analyst = ia.analyst_for_classification(row)
    assert analyst is not None
    read = _leaky_read(analyst, str(row["sub_industry_code"]))

    seen: dict[str, list[str]] = {"pm": [], "critic": [], "industry": []}
    real = llm_mod.chat_json

    def spy(prompt, **kwargs):
        if prompt.startswith(prompts.PM_SYNTHESIS_PROMPT):
            seen["pm"].append(prompt)
            return None
        if prompt.startswith(prompts.CRITIC_PROMPT):
            seen["critic"].append(prompt)
            return None
        if llm_mod.current_call_context().get("agent_name") == ia.AGENT_NAME:
            seen["industry"].append(prompt)
            return dict(read)
        return real(prompt, **kwargs)

    monkeypatch.setattr(llm_mod, "chat_json", spy)
    memo = graph.run_stock_memo(ticker)

    # The routed read really took the LLM path and really reached the PM.
    assert seen["industry"] and seen["pm"] and seen["critic"]
    finding = memo.extra_agent_views["industry_group"]
    assert finding.confidence == 0.75 and finding.long_form_report
    (pm_prompt,) = seen["pm"]
    assert f"## Industry group read — {analyst.label}" in pm_prompt

    assert _leaks(memo.model_dump(mode="json"), row, analyst) == []
    assert _leaks(pm_prompt, row, analyst) == []
    assert _leaks(seen["critic"], row, analyst) == []
    # The analyst's own prompt was sent labels only (S9); pinned here too so
    # routing ON cannot reintroduce a code through the memo path.
    assert _leaks(seen["industry"], row, analyst) == []
