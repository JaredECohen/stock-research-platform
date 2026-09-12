"""FEAT-003 — Industry Group analysts: one per GICS industry group, built
lazily, cached per taxonomy version, and run as a single roster entry.

Why a factory rather than a fan-out
-----------------------------------
The registry carries as many industry groups as the active taxonomy says
(never a literal here); a memo concerns exactly one company, so a memo run
constructs at most ONE analyst — the company's group — through
``get_industry_analyst``. Definitions are memoised per ``(code,
taxonomy_version_id)``: the version id is resolved from the database on
every call (``gics_registry.resolve_version``), so a worker that has been
up for a week still builds against the taxonomy the web process activated
this morning, while the immutable parts (nodes for that version id, the
checked-in knowledge base) are what the cache actually holds.

How it reaches the memo
-----------------------
``roster.AGENTS`` carries one ``AgentSpec`` for this analyst whose
``applies_to`` is ``applies_to`` below: routing on AND the company has a
routable classification. When the predicate says no, the spec does not
run — an unmapped ticker records a soft "no mapping" degradation and the
sector analyst stays primary, exactly as it is with routing off (the
default until a production A/B is read).

Stale rows still route
----------------------
``industry_classification`` flips a drifted row to ``stale`` IN PLACE: it
keeps its ``industry_group_code`` and files the state it held under
``evidence.previous_state``. Dropping those rows would make the analyst
vanish from memos — and the sector analyst lose its mandate block —
between a drift flag and the next classification run, which is a silent
capability loss on exactly the names whose classification is moving. So a
stale row routes on its previous state (``routed_state`` below) and the
provenance string SAYS it is stale, with the reason and the detection
date: a reader is told the mapping is being re-checked rather than
shown nothing.

Provenance travels with every display: the classification source label
("research map" vs "derived from provider classification", plus the
staleness when there is any), the sub-industry name, the taxonomy version
and the mapping caveat all ride on ``finding.data["industry_group"]``.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import settings
from ..memory import SectorMemory
from ..prompts import load_prompt
from ..schemas import AgentFinding, Citation
from ..services import gics_registry, industry_classification, industry_knowledge
from ..services.industry_group_knowledge import (
    GroupMandate,
    group_mandate,
    methodology_prompt_block,
    thesis_stages,
    universal_rules_prompt_block,
)
from . import llm, prompts
from .log_safety import log_safely
from .safe_runner import note_soft

if TYPE_CHECKING:
    from .memo_context import MemoInputs

log = logging.getLogger(__name__)

AGENT_NAME = "Industry Group Analyst"
NO_MAPPING_KIND = "NoMapping"
# States whose row names an industry group the analyst may route on. A
# `fallback` row knows only the sector, and a `missing` row names nothing.
ROUTABLE_STATES: tuple[str, ...] = (
    industry_classification.STATE_MAPPED,
    industry_classification.STATE_CONFLICT,
)

_SYSTEM_FALLBACK = (
    "You are an Industry Group Analyst at MarketMosaic. Reason from the world "
    "change to industry consequences before any KPI; separate observed fact, "
    "causal inference and forecast assumption; research and education only."
)

# Header field caps for `company_context_block`. The company name and the
# classification label are the only free-text fields in the template; the
# rest are codes and versions. Capping them keeps the header's length
# predictable so the mandate's budget is real.
_NAME_CHARS = 120
_LABEL_CHARS = 320
_TICKER_CHARS = 24
# Below this, a mandate block cannot carry a line of prose AND its
# attribution, so the block says the mandate was omitted instead.
_MIN_MANDATE_CHARS = 240

_CACHE: dict[tuple[str, int], IndustryAnalyst] = {}
_CACHE_LOCK = threading.Lock()
_CONSTRUCTIONS = 0


@dataclass(frozen=True)
class IndustryAnalyst:
    """The definition of one Industry Group analyst (immutable, cacheable)."""

    code: str
    name: str
    sector_code: str
    sector_name: str
    taxonomy_version_id: int
    taxonomy_version_key: str
    mandate: GroupMandate

    @property
    def display_name(self) -> str:
        return f"{AGENT_NAME} {self.code}"

    def system_prompt(self) -> str:
        """Identity prose + the loaded methodology + the ten universal rules
        + this group's mandate. Nothing about the framework is retyped."""
        identity = load_prompt("industry_analyst") or _SYSTEM_FALLBACK
        parts = [identity, methodology_prompt_block(), universal_rules_prompt_block(),
                 self.mandate.as_prompt_block()]
        return "\n\n".join(p for p in parts if p)

    def memory(self) -> SectorMemory:
        """The group's long-term memory file, alongside the sector files
        (`memory/sectors/industry_group_<code>.md`, same gitignore rule)."""
        return SectorMemory.for_sector(f"industry_group_{self.code}")

    def company_context_block(
        self, profile: dict[str, Any], classification: dict[str, Any] | None, *,
        max_chars: int = 4000, mandate_chars: int | None = None,
    ) -> str:
        """The company-context block: who the company is inside the group,
        how we know (classification source label), then the mandate.

        The mandate's budget is what is LEFT after the header is rendered,
        measured rather than guessed. A fixed reserve was wrong in both
        directions: the header grows with the company name and with a stale
        row's provenance clause, and when it overran, the tail that fell off
        the end was the mandate's attribution line — the one line
        ``as_prompt_block`` reserves budget for, because every sub-industry
        brief is original analysis and must say so wherever it is shown.
        `mandate_chars`, when given, is a CEILING on that budget, never a
        licence to overrun `max_chars`.
        """
        sub = _sub_industry_of(classification)
        cls = classification or {}
        fields = {
            "ticker": _clip(str(profile.get("ticker") or "").upper(), _TICKER_CHARS) or "n/a",
            # The only two free-text fields in the header; bounded so a
            # pathological company name cannot crowd out the mandate.
            "company_name": _clip(profile.get("company_name") or "n/a", _NAME_CHARS),
            "group_code": self.code, "group_name": self.name,
            "sector_code": self.sector_code, "sector_name": self.sector_name,
            "taxonomy_version": self.taxonomy_version_key,
            "knowledge_version": self.mandate.knowledge_version or "n/a",
            "sub_industry": (f"{sub['code']} {sub['name']}" if sub else "n/a (classified at group level only)"),
            "classification_label": _clip(classification_source_label(classification), _LABEL_CHARS),
            "state": cls.get("state") or "n/a",
            "source_as_of": cls.get("source_as_of") or "n/a",
        }
        header = prompts.INDUSTRY_GROUP_COMPANY_CONTEXT.format(mandate_block="", **fields)
        budget = max_chars - len(header)
        if mandate_chars is not None:
            budget = min(mandate_chars, budget)
        if budget < _MIN_MANDATE_CHARS:
            # No honest room for a mandate: say so instead of shipping a
            # block whose provenance has been sliced off the end. Only the
            # omission note can be cut here, and there is no mandate behind
            # it whose attribution could go missing.
            block = prompts.INDUSTRY_GROUP_COMPANY_CONTEXT.format(
                mandate_block="(mandate omitted: no prompt budget left after the company context.)",
                **fields,
            )
            return block if len(block) <= max_chars else block[: max(0, max_chars - 1)].rstrip() + "…"
        return prompts.INDUSTRY_GROUP_COMPANY_CONTEXT.format(
            mandate_block=self.mandate.as_prompt_block(max_chars=budget), **fields,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "name": self.name,
            "sector_code": self.sector_code, "sector_name": self.sector_name,
            "taxonomy_version_id": self.taxonomy_version_id,
            "taxonomy_version": self.taxonomy_version_key,
            "display_name": self.display_name,
        }


# --- provenance helpers --------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    """A header field bounded to `limit`, ellipsis included in the count."""
    text = str(text)
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _no_mapping_reason(classification: dict[str, Any] | None) -> str:
    """Why the analyst is absent, in the banner's own words. A stale row
    names the state it routed on as well as `stale`, so the reader is not
    told "stale" when the real reason is that it was only ever a
    sector-level fallback."""
    cls = classification or {}
    state = cls.get("state")
    if not state:
        return "no mapping: no classification row"
    routed = routed_state(cls)
    if routed and routed != state:
        return f"no mapping: classification state {state} (was {routed})"
    return f"no mapping: classification state {state}"


def routed_state(classification: dict[str, Any] | None) -> str:
    """The state routing decides on. A ``stale`` row was flipped in place
    and kept the state it held under ``evidence.previous_state``, so that
    is what it routes on; everything else routes on its own state."""
    cls = classification or {}
    state = str(cls.get("state") or "")
    if state != industry_classification.STATE_STALE:
        return state
    previous = (cls.get("evidence") or {}).get("previous_state")
    return str(previous or state)


def _staleness_note(classification: dict[str, Any] | None) -> str:
    """The clause appended to a stale row's provenance, so no display ever
    presents a drifted mapping as settled."""
    cls = classification or {}
    if cls.get("state") != industry_classification.STATE_STALE:
        return ""
    evidence = cls.get("evidence") or {}
    reason = evidence.get("stale_reason") or "inputs_changed"
    detected = evidence.get("stale_detected_at") or "unknown date"
    return (f"; mapping STALE since {detected} ({reason}) — routed on its previous "
            f"state pending the next classification run")


def classification_source_label(classification: dict[str, Any] | None) -> str:
    """The sentence a reader sees next to a mapping: which source placed
    the company, who authored it, and whether the mapping is stale."""
    if not classification:
        return "unclassified"
    source = classification.get("source")
    author = classification.get("author") or ""
    stale = _staleness_note(classification)
    if source == industry_classification.SOURCE_RESEARCH_MAP:
        return (f"research map ({author})" if author else "research map") + stale
    if source == industry_classification.SOURCE_PROVIDER_ALIAS:
        return ((f"derived from provider classification ({author})" if author
                 else "derived from provider classification") + stale)
    return f"unmapped ({classification.get('state') or 'unknown'})"


def _sub_industry_of(classification: dict[str, Any] | None) -> dict[str, str] | None:
    code8 = (classification or {}).get("sub_industry_code")
    if not code8:
        return None
    entry = industry_knowledge.get_sub_industry(code8)
    return {"code": str(code8), "name": entry["name"] if entry else "unknown"}


def industry_group_summary(
    classification: dict[str, Any] | None, analyst: IndustryAnalyst | None,
) -> dict[str, Any]:
    """The compact provenance block that rides on `finding.data["industry_group"]`
    (and, when routing is on, on the sector finding)."""
    cls = classification or {}
    return {
        "code": cls.get("industry_group_code"),
        "name": analyst.name if analyst else None,
        "state": cls.get("state") or "missing",
        # What routing actually decided on: differs from `state` only for a
        # stale row, which routes on the state it held before the drift.
        "routed_state": routed_state(classification) or "missing",
        "source": cls.get("source") or industry_classification.SOURCE_NONE,
        "source_label": classification_source_label(classification),
        "author": cls.get("author") or "",
        "source_as_of": cls.get("source_as_of") or "",
        "sub_industry": _sub_industry_of(classification),
        "taxonomy_version": analyst.taxonomy_version_key if analyst else None,
        # Filled by the report store once slice 3/4 publish reports; None
        # says "no edition on file" rather than pretending one exists.
        "report_version": None,
        "mapping_caveat": gics_registry.MAPPING_CAVEAT,
    }


# --- the factory ---------------------------------------------------------------


def get_industry_analyst(
    code: str, *, version: gics_registry.VersionInfo | str | int | None = None,
) -> IndustryAnalyst:
    """The analyst for a 4-digit group code under `version` (default: the
    active taxonomy — a DB read on every call, by design). Cached per
    ``(code, taxonomy_version_id)``; raises ``UnknownNode`` /
    ``UnknownIndustryGroup`` for a code the registry or the knowledge base
    does not carry."""
    global _CONSTRUCTIONS
    key = str(code or "").strip()
    info = gics_registry.resolve_version(version)
    cache_key = (key, info.id)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    node = gics_registry.group(key, version=info)
    mandate = group_mandate(key, version_key=info.version_key)
    sector = gics_registry.node(node.sector_code, version=info)
    analyst = IndustryAnalyst(
        code=node.code, name=node.name,
        sector_code=node.sector_code,
        sector_name=sector.name if sector is not None else mandate.sector_name,
        taxonomy_version_id=info.id, taxonomy_version_key=info.version_key,
        mandate=mandate,
    )
    with _CACHE_LOCK:
        existing = _CACHE.get(cache_key)
        if existing is not None:
            return existing
        _CACHE[cache_key] = analyst
        _CONSTRUCTIONS += 1
    return analyst


def construction_count() -> int:
    """How many analysts this process has built (tests assert <= 1 per memo)."""
    return _CONSTRUCTIONS


def clear_cache() -> None:
    global _CONSTRUCTIONS
    with _CACHE_LOCK:
        _CACHE.clear()
        _CONSTRUCTIONS = 0


def is_routable(classification: dict[str, Any] | None) -> bool:
    """Whether the row names a group this analyst may reason about. Stale
    rows are included on their previous state — see the module docstring."""
    return bool(
        classification
        and routed_state(classification) in ROUTABLE_STATES
        and classification.get("industry_group_code")
    )


def analyst_for_classification(classification: dict[str, Any] | None) -> IndustryAnalyst | None:
    """The analyst a classification row routes to, or None when the row
    names no group.

    The row's own ``taxonomy_version_id`` selects the REGISTRY edition —
    the group's code and name, and the sector it hangs under — so a memo
    written against version N keeps N's names even mid-switch. The mandate
    text does NOT follow it: ``group_mandate`` reads the single bundled
    knowledge base, so the prose is always the bundled edition
    (``mandate.knowledge_version``). Versioned mandates would need a
    per-version knowledge document, which does not exist; the prompt header
    names both editions rather than implying one.
    """
    if not is_routable(classification):
        return None
    assert classification is not None
    return get_industry_analyst(
        str(classification["industry_group_code"]),
        version=classification.get("taxonomy_version_id"),
    )


def lookup_classification(ticker: str) -> dict[str, Any] | None:
    """The gather stage's one read: the current row, or the on-demand
    classification of a symbol no loop has seen yet (one company read + one
    insert). None when there is no taxonomy or no `companies` row."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    row = industry_classification.current_for([symbol]).get(symbol)
    if row is not None:
        return row
    return industry_classification.classify_ticker(symbol)


def applies_to(inputs: MemoInputs) -> bool:
    """The roster predicate: routing on AND a routable mapping. Records the
    "no mapping" soft degradation on the run's log when routing is on but
    the company has no group, so the banner says why the analyst is absent."""
    if not settings.enable_industry_analyst_routing:
        return False
    if is_routable(inputs.industry_group):
        return True
    inputs.degradation.record_soft(AGENT_NAME, _no_mapping_reason(inputs.industry_group),
                                   kind=NO_MAPPING_KIND)
    return False


def sector_prompt_block(
    profile: dict[str, Any], classification: dict[str, Any] | None, *, max_chars: int = 2600,
) -> tuple[str, dict[str, Any] | None, IndustryAnalyst | None]:
    """What the sector analyst splices in as `{industry_group_block}` when
    routing is on: ``(block, summary, analyst)``. The block is '' and the
    summary carries the state when the company is not routable."""
    analyst = analyst_for_classification(classification)
    if analyst is None:
        return "", (industry_group_summary(classification, None) if classification else None), None
    block = "\n\n" + analyst.company_context_block(
        profile, classification, max_chars=max_chars, mandate_chars=max(400, max_chars - 900),
    )
    return block, industry_group_summary(classification, analyst), analyst


# --- the runner ----------------------------------------------------------------


def _profile_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    keys = ("ticker", "company_name", "sector", "industry", "sub_industry", "market_cap",
            "business_description", "drivers", "risks")
    out: dict[str, Any] = {}
    for k in keys:
        v = profile.get(k)
        if v is None:
            continue
        out[k] = v[:600] if isinstance(v, str) else v
    return out


def _ratios_snapshot(ratios: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (ratios or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = round(float(v), 4)
        if len(out) >= 24:
            break
    return out


def _unmapped_finding(ticker: str, classification: dict[str, Any] | None) -> AgentFinding:
    state = (classification or {}).get("state") or "missing"
    summary = industry_group_summary(classification, None)
    note_soft(AGENT_NAME, _no_mapping_reason(classification), kind=NO_MAPPING_KIND)
    return AgentFinding(
        agent=AGENT_NAME,
        headline="Industry group read unavailable: no mapping.",
        summary=(
            f"{ticker} has no routable industry-group classification (state: {state}); "
            "the sector analyst's view stands alone. The mapping is derived from provider "
            "classification and can be extended through the alias map."
        ),
        key_points=[f"Classification state: {state}", "Industry group: n/a (no mapping)"],
        confidence=0.0,
        sources=[],
        data={"industry_group": summary, "no_mapping": True},
    )


def _deterministic_finding(
    analyst: IndustryAnalyst, profile: dict[str, Any], classification: dict[str, Any] | None,
    *, llm_outcome: str = "returned no usable output",
) -> AgentFinding:
    """The mandate-grounded read with no LLM: observed placement plus the
    mandate's own tests. Every line names its source; the causal chain is
    honest that stage 1 is n/a rather than inventing a world change.

    `llm_outcome` names WHY this path was taken, so the memo banner
    distinguishes a model that answered with nothing from one that answered
    with the wrong shape. It is only read when an LLM was configured."""
    m = analyst.mandate
    ticker = str(profile.get("ticker") or "").upper()
    sub = _sub_industry_of(classification)
    brief = next((s for s in m.sub_industries if sub and s.code == sub["code"]), None)
    label = classification_source_label(classification)

    lines = [
        f"{ticker} is classified in {m.code} {m.name} ({label}); "
        + (f"sub-industry {sub['code']} {sub['name']}." if sub else "no sub-industry recorded."),
    ]
    if brief and brief.economics:
        lines.append(f"Sub-industry economics (original analyst brief, {brief.industry_code}): {brief.economics}")
    engines = m.items("economic_engine", 2)
    if engines:
        lines.append("Group economic engine: " + "; ".join(engines) + ".")
    if m.research_priority is not None:
        lines.append(f"Research priority {m.research_priority}/5 (set by {m.research_priority_source}).")
    lines.append(
        "Deterministic edition: no analyst narrative; the causal chain below is the mandate's "
        "test list, not an asserted thesis (stage 1 world change: n/a — no dated external change on file)."
    )

    key_points: list[str] = []
    for r in m.lists.get("core_kpis", ())[:5]:
        key_points.append(f"KPI to test: {r.text} [{','.join(r.industry_codes)}]")
    for r in m.lists.get("leading_indicators", ())[:3]:
        key_points.append(f"Leading indicator: {r.text} [{','.join(r.industry_codes)}]")
    for r in m.lists.get("accounting_data_traps", ())[:2]:
        key_points.append(f"Trap: {r.text} [{','.join(r.industry_codes)}]")
    if brief and brief.advantage_test:
        key_points.append(f"Advantage test ({brief.code}): {brief.advantage_test}")
    for code, q in m.highest_evi_questions[:2]:
        key_points.append(f"Highest-EVI question [{code}]: {q}")

    stages = thesis_stages()
    chain = []
    for s in stages:
        if s["id"] == "industry_consequences" and engines:
            chain.append({"stage": s["id"], "text": "Mandate economic engine: " + engines[0]})
        elif s["id"] == "financial_evidence_falsification" and key_points:
            chain.append({"stage": s["id"], "text": key_points[0]})
        else:
            chain.append({"stage": s["id"], "text": "n/a: deterministic edition — not asserted"})

    falsifiers = [f"Failure mode observed: {r.text} [{','.join(r.industry_codes)}]"
                  for r in m.lists.get("common_failure_modes", ())[:3]]
    data: dict[str, Any] = {
        "industry_group": industry_group_summary(classification, analyst),
        "mandate_type": "unclear",
        "placement": lines[0],
        "causal_chain": chain,
        "kpis_to_watch": [
            {"kpi": r.text, "industry_code": r.industry_codes[0], "why": "mandate core KPI"}
            for r in m.lists.get("core_kpis", ())[:5]
        ],
        "falsifiers": falsifiers,
        "traps": m.items("accounting_data_traps", 4),
        "checklists": m.as_checklists(),
        "attribution": m.attribution,
    }
    if settings.has_llm:
        # The graph promotes this into `degraded_agents` when an LLM was
        # configured; in no-key deterministic mode this path is the design.
        data["deterministic_fallback"] = (
            f"Industry Group LLM {llm_outcome}; mandate-grounded "
            "deterministic read shipped instead."
        )
    return AgentFinding(
        agent=AGENT_NAME,
        headline=f"{m.name} ({m.code}): mandate read for {ticker}" + (f" — {sub['name']}" if sub else ""),
        summary=" ".join(lines),
        key_points=key_points or ["See the industry group mandate for the KPI list."],
        confidence=0.55,
        sources=[f"industry_knowledge:{m.code}", *[f"industry:{c}" for c in m.industry_codes]],
        evidence=[Citation(kind="other", ref=f"gics:{m.code}", excerpt=m.name[:300])],
        data=data,
    )


def run_industry_group_agent(
    profile: dict[str, Any], ratios: dict[str, Any], *,
    prior_round_critique: str | None = None,
    classification: dict[str, Any] | None = None,
) -> AgentFinding:
    """The Industry Group Analyst's read on one company.

    `classification` is the row the gather stage already read; when None
    (a direct call outside a memo run) it is looked up here. An unmapped
    company gets an explicit "no mapping" finding rather than a guess.
    """
    ticker = str(profile.get("ticker") or "").upper()
    if not ticker:
        return _unmapped_finding("n/a", None)
    if classification is None:
        try:
            classification = lookup_classification(ticker)
        except Exception as exc:
            log_safely(log, f"industry classification lookup failed for {ticker}", exc)
            classification = None
    analyst = analyst_for_classification(classification)
    if analyst is None:
        return _unmapped_finding(ticker, classification)

    critique_block = ""
    if prior_round_critique:
        critique_block = (
            "\n## PM FOLLOW-UP (deep-research round)\n"
            "Address this question directly with mandate KPIs and observed figures:\n"
            f"{prior_round_critique}\n"
        )
    memory_block = ""
    if settings.enable_long_term_memory:
        try:
            memory_block = analyst.memory().as_prompt_context_for(ticker, max_chars=1500)
        except Exception:  # pragma: no cover — memory must never block a memo
            memory_block = ""
    user_prompt = prompts.INDUSTRY_GROUP_ANALYST_PROMPT.format(
        company_context=analyst.company_context_block(profile, classification, max_chars=5000),
        profile_snapshot=json.dumps(_profile_snapshot(profile), default=str)[:2500],
        ratios_snapshot=json.dumps(_ratios_snapshot(ratios), default=str)[:1500],
        critique_block=critique_block,
    )
    if memory_block:
        user_prompt += "\n\nPrior context from the group's long-term memory:\n" + memory_block

    llm_out = llm.chat_json(
        user_prompt, system=analyst.system_prompt(), route="cheap",
        model=llm.resolve_role_model("sector"),
    )
    if not llm_out:
        return _deterministic_finding(analyst, profile, classification)
    if not isinstance(llm_out, dict):
        # `chat_json` is annotated `dict | None`, but a provider in JSON mode
        # can and does answer with a top-level array — and reading `.get` off
        # a list raises AttributeError, which failed the whole memo instead of
        # degrading it. Every other agent here treats a malformed response as
        # no response; this one now does too, and the wrong SHAPE is recorded
        # as its own outcome rather than collapsed into "no usable output".
        outcome = f"returned a JSON {type(llm_out).__name__}, not an object"
        log.warning("Industry Group LLM %s for %s", outcome, ticker)
        note_soft(AGENT_NAME, f"Industry Group LLM {outcome}; deterministic read shipped")
        return _deterministic_finding(analyst, profile, classification, llm_outcome=outcome)

    chain = llm_out.get("causal_chain") or []
    data: dict[str, Any] = {
        "industry_group": industry_group_summary(classification, analyst),
        "mandate_type": llm_out.get("mandate_type") if llm_out.get("mandate_type") in
        ("compounder", "inflection", "unclear") else "unclear",
        "placement": str(llm_out.get("placement") or "")[:800],
        "causal_chain": [c for c in chain if isinstance(c, dict)][:12],
        "kpis_to_watch": [k for k in (llm_out.get("kpis_to_watch") or []) if isinstance(k, dict)][:8],
        "falsifiers": [str(f) for f in (llm_out.get("falsifiers") or [])][:5],
        "traps": [str(t) for t in (llm_out.get("traps") or [])][:6],
        "checklists": analyst.mandate.as_checklists(),
        "attribution": analyst.mandate.attribution,
    }
    try:
        confidence = float(llm_out.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    return AgentFinding(
        agent=AGENT_NAME,
        headline=str(llm_out.get("headline") or f"{analyst.name} read for {ticker}")[:240],
        summary=str(llm_out.get("summary") or ""),
        key_points=[str(p) for p in (llm_out.get("key_points") or [])][:12],
        confidence=max(0.0, min(1.0, confidence)),
        sources=[f"industry_knowledge:{analyst.code}",
                 *[f"industry:{c}" for c in analyst.mandate.industry_codes]],
        evidence=[Citation(kind="other", ref=f"gics:{analyst.code}", excerpt=analyst.name[:300])],
        data=data,
    )
