"""PM intake (slice G1: integration plan P3, the news critique, FIX-020).

Pinned here:
- intake receives the memo's headlines (titles and severity only, at most
  5, sanitised) under an untrusted-data label; without news the prompt is
  byte-identical to the pre-G1 one;
- the gate is `settings.llm_enabled`, not the OpenAI key;
- with `DEBATE_MODE=on` the sector analyst cannot be skipped, even when the
  model names it;
- the call is attributed as `pm.intake`;
- the log line carries the skip list, never the model's rationale text.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.agents import intake
from app.config import settings

_PROFILE = {"ticker": "MSFT", "company_name": "Microsoft", "sector": "Technology",
            "industry": "Software", "business_description": "Cloud and software."}
_SPECIALISTS = ["sector", "earnings", "filing", "valuation", "comps", "macro", "risk", "technical"]


@pytest.fixture
def spy(monkeypatch):
    """Intake's model: records the prompt and kwargs, answers `reply`."""
    calls: list[dict] = []
    state = {"reply": {"skip": [], "rationale": ""}}

    def fake(prompt, **kw):
        calls.append({"prompt": prompt, **kw})
        return state["reply"]

    monkeypatch.setattr(intake.llm, "chat_json", fake)
    monkeypatch.setattr(type(settings), "llm_enabled", property(lambda self: True))
    # The pre-G1 gate, so the base commit reaches the model in these tests too.
    monkeypatch.setattr(settings, "openai_api_key", "stub-openai")
    monkeypatch.setattr(settings, "debate_mode", "off")

    def set_reply(reply):
        state["reply"] = reply

    fake.calls = calls  # type: ignore[attr-defined]
    fake.set_reply = set_reply  # type: ignore[attr-defined]
    return fake


def _legacy_prompt(available: list[str], payload: dict) -> str:
    """The intake prompt exactly as the base commit built it (recent_news
    was always [] there)."""
    return (
        f"You are the PM doing intake on a memo run. {len(available)} "
        "specialists are available — " + ", ".join(available) + ". "
        "Default: run them all. You may "
        "DEPRIORITIZE up to 3 specialists for this memo "
        "ONLY when running them adds little to the thesis (e.g., a "
        "regulated bank rarely needs a technical read; a name with "
        "no recent material news doesn't need a fresh filings pass). "
        "Each skip MUST cite a specific reason — generic 'low value' "
        "is not acceptable.\n\n"
        "Return strict JSON: { \"skip\": [\"<specialist>\", ...], "
        "\"rationale\": \"<one paragraph explaining the skips, or "
        "empty string if running all>\" }.\n\n"
        + json.dumps(payload, default=str)[:6000]
    )


def _payload(recent_news: list) -> dict:
    return {
        "ticker": "MSFT", "company_name": "Microsoft", "sector": "Technology",
        "industry": "Software", "business_description": "Cloud and software.",
        "drivers": [], "risks": [], "macro_regime": None, "recent_news": recent_news,
    }


def test_intake_prompt_byte_identical_without_news(spy):
    intake.run_intake(_PROFILE, specialists=_SPECIALISTS)
    intake.run_intake(_PROFILE, news_alerts=[], specialists=_SPECIALISTS)
    assert [c["prompt"] for c in spy.calls] == [_legacy_prompt(_SPECIALISTS, _payload([]))] * 2


def test_intake_receives_news_titles_untrusted(spy):
    """REGRESSION (news critique): intake was always handed `recent_news=[]`
    while its prompt skips specialists on "no recent material news"."""
    alerts = [
        {"title": "Microsoft wins \n`big` <contract>", "summary": "SUMMARY-MARKER 12.5% growth",
         "severity": "material", "url": "https://reuters.com/x"},
        *[{"title": f"Story {i}", "summary": "s", "severity": "advisory"} for i in range(6)],
        {"title": "Odd severity", "severity": "SHOUTING"},
    ]
    intake.run_intake(_PROFILE, news_alerts=alerts, specialists=_SPECIALISTS)
    prompt = spy.calls[-1]["prompt"]
    assert intake.NEWS_LABEL in prompt
    assert prompt.index(intake.NEWS_LABEL) < prompt.index('{"ticker"')
    payload = json.loads(prompt.split(intake.NEWS_LABEL, 1)[1])
    news = payload["recent_news"]
    assert len(news) == 5                                     # at most 5
    assert news[0] == {"title": "Microsoft wins big contract", "severity": "material"}
    assert all(set(n) == {"title", "severity"} for n in news)  # titles and severity only
    assert "SUMMARY-MARKER" not in prompt and "reuters.com" not in prompt


def test_intake_gate_is_llm_enabled_not_the_openai_key(monkeypatch, spy):
    """REGRESSION: the gate read the OpenAI key although the call runs on
    the active provider, so an Anthropic-only deployment never ran intake."""
    monkeypatch.setattr(settings, "openai_api_key", "")
    intake.run_intake(_PROFILE, specialists=_SPECIALISTS)
    assert len(spy.calls) == 1
    monkeypatch.setattr(type(settings), "llm_enabled", property(lambda self: False))
    monkeypatch.setattr(settings, "openai_api_key", "stub-openai")
    assert intake.run_intake(_PROFILE, specialists=_SPECIALISTS).skipped == set()
    assert len(spy.calls) == 1                                # not called in demo mode


def test_sector_not_skippable_with_debate_on(monkeypatch, spy):
    spy.set_reply({"skip": ["sector", "technical"], "rationale": "r"})
    off = intake.run_intake(_PROFILE, specialists=_SPECIALISTS)
    assert off.skipped == {"sector", "technical"}             # unchanged with the debate off
    assert intake.SECTOR_REQUIRED not in spy.calls[-1]["prompt"]

    monkeypatch.setattr(settings, "debate_mode", "on")
    on = intake.run_intake(_PROFILE, specialists=_SPECIALISTS)
    assert on.skipped == {"technical"}
    assert intake.SECTOR_REQUIRED in spy.calls[-1]["prompt"]
    # A run without the sector analyst has nothing to require.
    intake.run_intake(_PROFILE, specialists=["earnings", "technical"])
    assert intake.SECTOR_REQUIRED not in spy.calls[-1]["prompt"]


def test_intake_call_is_attributed(spy):
    intake.run_intake(_PROFILE, specialists=_SPECIALISTS)
    assert spy.calls[-1]["action"] == "pm.intake" and spy.calls[-1]["ticker"] == "MSFT"
    # Plan P3 moves intake by its action's tier; the route itself is unchanged.
    assert spy.calls[-1]["route"] == "cheap"


def test_intake_logs_the_skip_list_not_the_rationale(spy, caplog):
    """REGRESSION (FIX-020): 100 characters of the model's rationale went to
    the logs."""
    spy.set_reply({"skip": ["technical"], "rationale": "RATIONALE-SENTINEL because the bank is regulated"})
    with caplog.at_level(logging.INFO, logger="app.agents.intake"):
        decision = intake.run_intake(_PROFILE, specialists=_SPECIALISTS)
    assert decision.rationale.startswith("RATIONALE-SENTINEL")   # kept on the audit record
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "skipped=['technical']" in text
    assert "RATIONALE-SENTINEL" not in text and "rationale_chars=" in text
