"""Offline harness for the bull/bear debate engine tests (slice B8-D3).

The engine takes an injectable `call`; `ScriptedCall` answers it from a
script keyed on (side, phase), the harness the design names ("a scripted
chat_json keyed on (agent name, phase)"). Retrieval is injected the same
way, so no test reaches the vector store, a provider or the network.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from app.agents import debate
from app.agents.debate import CallResult, DebateInputs, DebateRequest, DebateRoute
from app.agents.source_ledger import SourceLedger
from app.config import settings
from app.schemas.agents import AgentFinding

ROUTE = DebateRoute(
    configured=True, provider="anthropic", model="claude-opus-5-5",
    efforts={"research": "low", "debate": "high"},
    partner_provider="openai", partner_model="gpt-6-sol",
    partner_efforts={"research": "low", "debate": "high"},
)

FIN = {
    "income": [
        {"period": "2025Q1", "revenue": 100.0, "net_income": 10.0},
        {"period": "2025Q2", "revenue": 104.0, "net_income": 11.0},
        {"period": "2025Q3", "revenue": 108.0, "net_income": 12.0},
        {"period": "2025Q4", "revenue": 112.0, "net_income": 13.0},
    ],
}

PASSAGE_TEXT = {
    "filing": "Management expects data center demand to remain strong through the fiscal year. "
              "Competition from custom accelerators is a risk to pricing.",
    "transcript": "Our backlog doubled and we see supply constraints easing in the second half.",
}


def usage_for(req: DebateRequest, *, served: str | None = None, refused: bool = False,
              error_type: str | None = None, tokens: tuple[int, int] = (1000, 500),
              cost: float = 0.01) -> dict[str, Any]:
    return {
        "provider": req.provider, "model": req.model, "served_model": served or f"{req.model}-served",
        "input_tokens": tokens[0], "output_tokens": tokens[1], "cost_usd": cost,
        "refused": refused, "error_type": error_type,
    }


def plan(corpus: str = "filings", query: str = "data center demand") -> dict[str, Any]:
    return {"queries": [{"corpus": corpus, "query": query, "why": "tests the thesis"}]}


def opening(side: str, *, n: int = 3, evidence: tuple[str, ...] = ("E01", "financials:ACME")) -> dict[str, Any]:
    word = "strength" if side == "bull" else "weakness"
    return {
        "headline": f"Demand {word} decides the next year",
        "claims": [{
            "pillar": f"Pillar {i}", "claim": f"Point {i}: demand {word} is visible in the filing",
            "category": "valuation" if i == 1 else "growth", "materiality": "high" if i <= 2 else "medium",
            "evidence": list(evidence), "quote": None, "analyst_refs": ["earnings"],
            "contests_analyst": None, "falsifier": f"Observable {i} within the horizon",
        } for i in range(1, n + 1)],
    }


def rebuttal(side: str, *, n: int = 3, stance: str = "rebut") -> dict[str, Any]:
    opp = "BEAR" if side == "bull" else "BULL"
    return {
        "responses": [{"target": f"{opp}-{i}", "stance": stance if i == 1 else "partial",
                       "argument": f"Answer {i}: the passage says otherwise", "evidence": ["E02"]}
                      for i in range(1, n + 1)],
        "revised_headline": "Revised headline", "crux": "Whether demand holds",
    }


def default_script() -> dict[tuple[str, str], list[Any]]:
    return {
        ("bull", "research"): [plan("filings", "data center demand")],
        ("bear", "research"): [plan("transcripts", "supply constraints backlog")],
        ("bull", "openings"): [opening("bull")],
        ("bear", "openings"): [opening("bear")],
        ("bull", "rebuttals"): [rebuttal("bull")],
        ("bear", "rebuttals"): [rebuttal("bear")],
    }


class ScriptedCall:
    """Answers `DebateRequest`s from a script keyed on (side, phase).

    A script entry is a dict (the parsed output), None (the call returned
    nothing), an Exception (raised), a CallResult (used as is), or a
    callable taking the request. The last entry repeats. Every request is
    recorded in order."""

    def __init__(self, script: dict[tuple[str, str], list[Any]] | None = None, *,
                 served: Callable[[DebateRequest], str] | None = None) -> None:
        self.script = default_script() if script is None else script
        self.requests: list[DebateRequest] = []
        self._served = served
        self._counts: dict[tuple[str, str], int] = {}

    def __call__(self, req: DebateRequest) -> CallResult:
        self.requests.append(req)
        key = (req.side, req.phase)
        entries = self.script.get(key) or [None]
        i = self._counts.get(key, 0)
        self._counts[key] = i + 1
        item = entries[min(i, len(entries) - 1)]
        if callable(item) and not isinstance(item, (dict, CallResult)):
            item = item(req)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, CallResult):
            return item
        served = self._served(req) if self._served else None
        return CallResult(item, usage_for(req, served=served))

    def calls(self, side: str | None = None, phase: str | None = None) -> list[DebateRequest]:
        return [r for r in self.requests if (side is None or r.side == side)
                and (phase is None or r.phase == phase)]


def findings() -> dict[str, AgentFinding]:
    sector = AgentFinding(
        agent="Sector Analyst", headline="Sector view: accelerators", summary="Cycle is mid-way.",
        key_points=["Pricing holds"], confidence=0.6,
        data={"bull_bear_analysis": {
            "bull_case": {"headline": "Demand outruns supply", "key_points": ["Backlog"]},
            "bear_case": {"headline": "Custom chips erode share", "key_points": ["Pricing"]},
            "key_disagreement": "Durability of demand", "sector_synthesis": "Balanced",
            "sector_lean": "balanced", "falsifiable_tests": [
                {"statement": "Backlog falls two quarters", "invalidates_side": "bull"}],
        }},
    )
    earnings = AgentFinding(agent="Earnings Analyst", headline="Beat and raise",
                            summary="Revenue ahead of guidance.", key_points=["Guide raised"], confidence=0.7)
    filing = AgentFinding(agent="Filing Analyst", headline="Filing Analyst unavailable",
                          summary="stand-in", data={"deterministic_fallback": True})
    return {"sector": sector, "earnings": earnings, "filing": filing}


def inputs(run_id: str = "run-debate-1", **kw: Any) -> DebateInputs:
    base: dict[str, Any] = dict(
        ticker="ACME", run_id=run_id, company_name="Acme Corp", industry_label="Chips & Chipmaking Equipment",
        price=120.0, memo_date="2026-09-20", findings=findings(), digests="",
        valuation={"dcf_pm_adjusted": {"base": {"implied_share_price": 140.0}}},
        financials=FIN, news_items=(), news_rows=[],
    )
    base.update(kw)
    return DebateInputs(**base)


def vector_search(query: str, *, ticker: str | None = None, source_types: list[str] | None = None,
                  top_k: int = 3, **_: Any) -> list[dict[str, Any]]:
    """Deterministic ticker-scoped hits: one passage per corpus, id by corpus."""
    assert ticker, "the debate must always scope vector search to a ticker"
    st = (source_types or ["filing"])[0]
    return [{"id": {"filing": 101, "transcript": 202}[st], "source_type": st, "source_id": f"{st}-src",
             "section": "mda" if st == "filing" else "qa", "text": PASSAGE_TEXT[st], "score": 0.8,
             "meta": {"accession": "0000000001-25-000001"} if st == "filing" else {}}]


def search_many(ticker: str, queries: list[str], **_: Any) -> list[list[dict[str, Any]]]:
    return [[] for _ in queries]


def zero_costs(run_id: str, agents: Any = None, **_: Any) -> dict[str, Any]:
    return {"cost_usd_total": 0.0, "n_calls": 0}


def budget(run_id: str, **kw: Any) -> debate.DebateBudget:
    kw.setdefault("cost_reader", zero_costs)
    return debate.DebateBudget(run_id, **kw)


def ledger_with_facts(ticker: str = "ACME") -> SourceLedger:
    ledger = SourceLedger()
    with ledger.activate():
        from app.agents.source_ledger import register_source
        register_source("financials", f"financials:{ticker}", FIN)
        register_source("dcf", "dcf:pm_adjusted", {"base": {"implied_share_price": 140.0}})
    return ledger


def enable(monkeypatch, *, route: DebateRoute | None = ROUTE, parallel: bool = False) -> None:
    """DEBATE_MODE=on with an LLM available, the route fixed (unless None:
    then the real resolver runs), sequential pairs (sqlite)."""
    monkeypatch.setattr(settings, "debate_mode", "on")
    monkeypatch.setattr(settings, "debate_parallel", parallel)
    monkeypatch.setattr(type(settings), "llm_enabled", property(lambda self: True))
    if route is not None:
        monkeypatch.setattr(debate, "resolve_route", lambda: route.model_copy(deep=True))


def run(call: ScriptedCall, *, run_id: str = "run-debate-1", ledger: SourceLedger | None = None,
        **kw: Any):
    ledger = ledger or ledger_with_facts()
    kw.setdefault("vector_search", vector_search)
    kw.setdefault("search_many", search_many)
    kw.setdefault("budget", budget(run_id))
    inp = kw.pop("inputs", None) or inputs(run_id)
    with ledger.activate():
        return debate.run_debate(inp, call=call, **kw)


def run_ids_by_parity() -> dict[str, str]:
    """A run id of each presentation order."""
    out: dict[str, str] = {}
    for i in itertools.count():
        rid = f"parity-run-{i}"
        out.setdefault(debate.presentation_order(rid), rid)
        if len(out) == 2:
            return out
    raise AssertionError("unreachable")
