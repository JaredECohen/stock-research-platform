"""FEAT-003 — one Industry Group's research mandate, aggregated from its
industries' encyclopedia entries and their sub-industry briefs.

The knowledge base (``industry_knowledge``) is written per 6-digit
industry; an Industry Group analyst reasons one level up. Every field in
the encyclopedia is prose ("Rig count; frac spreads; utilization; …",
"5/5."), so the aggregation here is deliberately mechanical and
auditable rather than clever:

* list-like fields are split on semicolons, deduplicated
  case-insensitively and ranked by how many of the group's industries name
  them — with the naming industries kept on every item, so a prompt or a
  report can always say *which* industry a KPI came from;
* ``research_priority`` is the maximum of the parsed ``(\\d)/5`` scores,
  with the industry that set it named;
* the sub-industry briefs are listed (name, one-line economics, the
  advantage test) with per-brief provenance, bounded by a character
  budget, because a group of eight sub-industries would otherwise swamp
  the prompt.

Nothing here reads the database. A mandate is a pure function of the
checked-in JSON, memoised per ``(code, version_key)`` — the version key
is the registry's taxonomy id the caller resolved from the database, so a
taxonomy switch (which lands by regenerating the JSON) never serves a
stale mandate from a long-lived worker. A code the knowledge base does
not carry raises ``UnknownIndustryGroup``: an analyst reasoning from a
blank mandate is the silent-partial failure the loader itself refuses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from . import industry_knowledge

# Encyclopedia fields that read as "item; item; item" and are aggregated
# as ranked lists. The remaining fields (research_priority, cadence,
# preferred_archetype, highest_evi_question) are handled by name below.
LIST_FIELDS: tuple[str, ...] = (
    "economic_engine",
    "core_kpis",
    "leading_indicators",
    "typical_moats",
    "capital_cycle_supply_response",
    "valuation_lenses",
    "accounting_data_traps",
    "common_failure_modes",
    "ideal_compounder_setup",
    "ideal_inflection_setup",
)

# Prompt budgets. The whole mandate block rides inside the analyst's system
# prompt next to the methodology and the ten universal rules, and the
# sector analyst gets a shorter cut of it; both are bounded here so the
# memo's context stays inside `max_agent_context_chars` regardless of how
# many industries a group carries.
PROMPT_BLOCK_MAX_CHARS = 6000
SUB_INDUSTRY_BLOCK_MAX_CHARS = 2400
_ITEMS_PER_FIELD = 8
_PRIORITY_RE = re.compile(r"(\d)\s*/\s*5")


class UnknownIndustryGroup(LookupError):
    """A 4-digit code the knowledge base does not carry."""


@dataclass(frozen=True)
class RankedItem:
    """One deduplicated list entry with the industries that named it."""

    text: str
    count: int
    industry_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "count": self.count, "industry_codes": list(self.industry_codes)}


@dataclass(frozen=True)
class SubIndustryBrief:
    """The bounded view of one sub-industry brief the mandate carries."""

    code: str
    name: str
    industry_code: str
    economics: str
    advantage_test: str
    source_ids: tuple[str, ...]
    attribution: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "industry_code": self.industry_code,
            "economics": self.economics,
            "advantage_test": self.advantage_test,
            "source_ids": list(self.source_ids),
            "attribution": self.attribution,
        }


@dataclass(frozen=True)
class GroupMandate:
    """Everything an Industry Group analyst knows before it sees a company."""

    code: str
    name: str
    sector_code: str
    sector_name: str
    knowledge_version: str
    knowledge_source_sha256: str
    version_key: str
    # Verbatim industry entries — `{code, name, fields}` — so a consumer
    # can always drop back to the un-aggregated text with provenance.
    industries: tuple[dict[str, Any], ...]
    lists: dict[str, tuple[RankedItem, ...]]
    research_priority: int | None
    research_priority_source: str
    cadence: tuple[RankedItem, ...]
    preferred_archetypes: tuple[RankedItem, ...]
    # (industry_code, question) pairs — one per industry, never merged.
    highest_evi_questions: tuple[tuple[str, str], ...]
    sub_industries: tuple[SubIndustryBrief, ...]
    sub_industries_truncated: int = 0
    attribution: str = industry_knowledge.BRIEF_ATTRIBUTION
    extra: dict[str, Any] = field(default_factory=dict)

    # --- convenience accessors ------------------------------------------

    def items(self, field_name: str, limit: int | None = None) -> list[str]:
        ranked = self.lists.get(field_name, ())
        texts = [r.text for r in ranked]
        return texts[:limit] if limit is not None else texts

    @property
    def industry_codes(self) -> tuple[str, ...]:
        return tuple(i["code"] for i in self.industries)

    @property
    def industry_names(self) -> tuple[str, ...]:
        return tuple(i["name"] for i in self.industries)

    def as_checklists(self) -> dict[str, list[dict[str, Any]]]:
        """Compounder / inflection checklist items with the industry that
        supplied each one — the two mandates stay distinct by design."""
        return {
            "compounder": [r.as_dict() for r in self.lists.get("ideal_compounder_setup", ())],
            "inflection": [r.as_dict() for r in self.lists.get("ideal_inflection_setup", ())],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "sector_code": self.sector_code,
            "sector_name": self.sector_name,
            "knowledge_version": self.knowledge_version,
            "knowledge_source_sha256": self.knowledge_source_sha256,
            "version_key": self.version_key,
            "industries": [{"code": i["code"], "name": i["name"]} for i in self.industries],
            "lists": {k: [r.as_dict() for r in v] for k, v in self.lists.items()},
            "research_priority": self.research_priority,
            "research_priority_source": self.research_priority_source,
            "cadence": [r.as_dict() for r in self.cadence],
            "preferred_archetypes": [r.as_dict() for r in self.preferred_archetypes],
            "highest_evi_questions": [
                {"industry_code": c, "question": q} for c, q in self.highest_evi_questions
            ],
            "sub_industries": [s.as_dict() for s in self.sub_industries],
            "sub_industries_truncated": self.sub_industries_truncated,
            "attribution": self.attribution,
        }

    def as_prompt_block(self, max_chars: int = PROMPT_BLOCK_MAX_CHARS) -> str:
        """The mandate as prompt prose, bounded, provenance on every line.

        Item provenance is the industry code list in brackets; a reader
        (or the validator) can trace any KPI back to the encyclopedia entry
        that named it.
        """
        lines: list[str] = [
            f"## Industry Group mandate — {self.code} {self.name} "
            f"(sector {self.sector_code} {self.sector_name}; taxonomy {self.version_key})",
            "Industries: " + "; ".join(f"{i['code']} {i['name']}" for i in self.industries) + ".",
        ]
        if self.research_priority is not None:
            lines.append(
                f"Research priority: {self.research_priority}/5 "
                f"(set by {self.research_priority_source})."
            )
        if self.cadence:
            lines.append("Cadence: " + "; ".join(_with_codes(r) for r in self.cadence) + ".")
        if self.preferred_archetypes:
            lines.append(
                "Preferred archetypes: " + "; ".join(_with_codes(r) for r in self.preferred_archetypes) + "."
            )
        labels = {
            "economic_engine": "Economic engine",
            "core_kpis": "Core KPIs",
            "leading_indicators": "Leading indicators",
            "typical_moats": "Typical moats",
            "capital_cycle_supply_response": "Capital cycle / supply response",
            "valuation_lenses": "Valuation lenses",
            "accounting_data_traps": "Accounting and data traps",
            "common_failure_modes": "Common failure modes",
            "ideal_compounder_setup": "Ideal compounder setup",
            "ideal_inflection_setup": "Ideal inflection setup",
        }
        for key in LIST_FIELDS:
            ranked = self.lists.get(key, ())
            if not ranked:
                continue
            lines.append(
                f"{labels[key]}: " + "; ".join(_with_codes(r) for r in ranked[:_ITEMS_PER_FIELD]) + "."
            )
        if self.highest_evi_questions:
            lines.append("Highest-EVI questions:")
            lines.extend(f"- [{code}] {q}" for code, q in self.highest_evi_questions)
        sub_block = self.sub_industry_block()
        if sub_block:
            lines.append(sub_block)
        lines.append(f"Attribution: {self.attribution}")
        return _bounded("\n".join(lines), max_chars)

    def sub_industry_block(self, max_chars: int = SUB_INDUSTRY_BLOCK_MAX_CHARS) -> str:
        """Sub-industries as `code name [industry] — economics | advantage
        test (sources)`. Empty when the group carries none."""
        if not self.sub_industries:
            return ""
        lines = ["Sub-industries (original analyst briefs; provenance per line):"]
        for sub in self.sub_industries:
            src = ",".join(sub.source_ids) if sub.source_ids else "n/a"
            lines.append(
                f"- {sub.code} {sub.name} [industry {sub.industry_code}; sources {src}] — "
                f"{sub.economics} | Advantage test: {sub.advantage_test}"
            )
        if self.sub_industries_truncated:
            lines.append(
                f"- … {self.sub_industries_truncated} more sub-industr"
                f"{'y' if self.sub_industries_truncated == 1 else 'ies'} omitted for length."
            )
        return _bounded("\n".join(lines), max_chars)


# --- parsing helpers ----------------------------------------------------------


def split_items(text: Any) -> list[str]:
    """`"A; b; C."` → `["A", "b", "C"]`. Trailing periods are dropped, blanks
    skipped, and the order of first appearance is preserved."""
    if not isinstance(text, str):
        return []
    out: list[str] = []
    for raw in text.split(";"):
        item = raw.strip().strip(".").strip()
        if item:
            out.append(item)
    return out


def parse_priority(text: Any) -> int | None:
    """`"5/5."` → 5; anything without an `n/5` reads as unknown."""
    if not isinstance(text, str):
        return None
    m = _PRIORITY_RE.search(text)
    return int(m.group(1)) if m else None


def aggregate_field(industries: list[dict[str, Any]], field_name: str) -> tuple[RankedItem, ...]:
    """Dedupe (case-insensitive) and rank by industry count, keeping the
    naming industries per item.

    Ties break on an item's position in its own industry's list, then on
    industry order — so a group of two industries lists both industries'
    first KPIs before either's fifth, instead of exhausting the first
    industry's list and cutting the second one off at the prompt budget.
    """
    seen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for idx, industry in enumerate(industries):
        code = str(industry.get("code", ""))
        fields = industry.get("fields") or {}
        for pos, item in enumerate(split_items(fields.get(field_name))):
            key = " ".join(item.lower().split())
            entry = seen.get(key)
            if entry is None:
                seen[key] = {"text": item, "codes": [code], "pos": pos, "idx": idx}
                order.append(key)
            elif code not in entry["codes"]:
                entry["codes"].append(code)
                entry["pos"] = min(entry["pos"], pos)
    ranked = sorted(order, key=lambda k: (-len(seen[k]["codes"]), seen[k]["pos"], seen[k]["idx"]))
    return tuple(
        RankedItem(text=seen[k]["text"], count=len(seen[k]["codes"]), industry_codes=tuple(seen[k]["codes"]))
        for k in ranked
    )


def first_sentence(text: Any, max_chars: int = 220) -> str:
    """The opening sentence of a brief, bounded — the one-line economics."""
    if not isinstance(text, str):
        return ""
    s = " ".join(text.split())
    if not s:
        return ""
    m = re.search(r"[.!?](\s|$)", s)
    head = s[: m.end()].strip() if m else s
    return _bounded(head, max_chars)


def _with_codes(item: RankedItem) -> str:
    return f"{item.text} [{','.join(item.industry_codes)}]"


def _bounded(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


# --- the mandate ---------------------------------------------------------------


def _sub_industry_briefs(
    industries: list[dict[str, Any]], budget: int,
) -> tuple[tuple[SubIndustryBrief, ...], int]:
    """Briefs in taxonomy order until `budget` characters are spent; the
    rest are counted, not dropped silently."""
    briefs: list[SubIndustryBrief] = []
    spent = 0
    truncated = 0
    for industry in industries:
        for sub in industry.get("sub_industries") or []:
            fields = sub.get("fields") or {}
            brief = SubIndustryBrief(
                code=str(sub.get("code", "")),
                name=str(sub.get("name", "")),
                industry_code=str(industry.get("code", "")),
                economics=first_sentence(fields.get("economics")),
                advantage_test=first_sentence(fields.get("advantage_test")),
                source_ids=tuple(str(s) for s in (sub.get("source_ids") or [])),
                attribution=industry_knowledge.BRIEF_ATTRIBUTION,
            )
            size = len(brief.economics) + len(brief.advantage_test) + len(brief.name) + 40
            if spent + size > budget and briefs:
                truncated += 1
                continue
            briefs.append(brief)
            spent += size
    return tuple(briefs), truncated


def build_mandate(group: dict[str, Any], *, version_key: str, payload: dict[str, Any]) -> GroupMandate:
    """Aggregate one group record (the shape ``get_industry_group`` returns)."""
    industries = list(group.get("industries") or [])
    lists = {name: aggregate_field(industries, name) for name in LIST_FIELDS}

    priority: int | None = None
    priority_source = ""
    for industry in industries:
        score = parse_priority((industry.get("fields") or {}).get("research_priority"))
        if score is not None and (priority is None or score > priority):
            priority, priority_source = score, f"{industry['code']} {industry['name']}"

    evi = tuple(
        (str(i["code"]), " ".join(str((i.get("fields") or {}).get("highest_evi_question", "")).split()))
        for i in industries
        if (i.get("fields") or {}).get("highest_evi_question")
    )
    subs, truncated = _sub_industry_briefs(industries, SUB_INDUSTRY_BLOCK_MAX_CHARS)
    return GroupMandate(
        code=str(group["code"]),
        name=str(group["name"]),
        sector_code=str(group.get("sector_code", "")),
        sector_name=str(group.get("sector_name", "")),
        knowledge_version=str(payload.get("taxonomy_version", "")),
        knowledge_source_sha256=str(payload.get("source_sha256", "")),
        version_key=version_key,
        industries=tuple({"code": i["code"], "name": i["name"], "fields": dict(i.get("fields") or {})}
                         for i in industries),
        lists=lists,
        research_priority=priority,
        research_priority_source=priority_source,
        cadence=aggregate_field(industries, "cadence"),
        preferred_archetypes=aggregate_field(industries, "preferred_archetype"),
        highest_evi_questions=evi,
        sub_industries=subs,
        sub_industries_truncated=truncated,
        extra={"map_as_of": str(payload.get("map_as_of", ""))},
    )


@lru_cache(maxsize=64)
def _cached_mandate(code: str, version_key: str) -> GroupMandate:
    group = industry_knowledge.get_industry_group(code)
    if group is None:
        raise UnknownIndustryGroup(f"industry group {code!r} is not in the knowledge base")
    return build_mandate(group, version_key=version_key, payload=industry_knowledge.load_industry_knowledge())


def group_mandate(code: str, *, version_key: str | None = None) -> GroupMandate:
    """The mandate for a 4-digit group code, memoised per ``(code, version_key)``.

    ``version_key`` defaults to the knowledge base's own taxonomy version;
    callers holding a registry version pass its key so two taxonomy
    editions never share a cache slot.
    """
    key = str(code or "").strip()
    vk = version_key or str(industry_knowledge.load_industry_knowledge().get("taxonomy_version", ""))
    return _cached_mandate(key, vk)


def clear_cache() -> None:
    _cached_mandate.cache_clear()


# --- universal framework, loaded not retyped ---------------------------------


def universal_research_rules() -> list[str]:
    """The encyclopedia's universal rules, in order, as written."""
    return [str(r) for r in industry_knowledge.load_industry_knowledge().get("universal_research_rules", [])]


def primary_sources() -> list[dict[str, Any]]:
    """The encyclopedia's source library (`domain, name, url, description`)."""
    return [dict(s) for s in industry_knowledge.load_industry_knowledge().get("primary_sources", [])]


def thesis_stages() -> list[dict[str, Any]]:
    """The eight-stage causal order from ``governing_methodology`` —
    ``[{order, id, question}]`` sorted by ``order``. Empty only if the
    knowledge base carries no methodology, which the writer treats as a
    degradation rather than inventing an order."""
    stages = industry_knowledge.governing_methodology().get("thesis_construction_order") or []
    return sorted(
        ({"order": int(s.get("order", 0)), "id": str(s.get("id", "")), "question": str(s.get("question", ""))}
         for s in stages if isinstance(s, dict)),
        key=lambda s: s["order"],
    )


def methodology_prompt_block(max_chars: int = 3200) -> str:
    """The governing methodology rendered for a system prompt: the ordered
    stages, the per-link requirements and the operating rules. Every line
    is read from the JSON so the prompt cannot drift from the handbook."""
    gm = industry_knowledge.governing_methodology()
    if not gm:
        return ""
    lines = [f"## Governing methodology — {gm.get('name', '')} ({gm.get('version', '')})"]
    if gm.get("objective"):
        lines.append(str(gm["objective"]))
    stages = thesis_stages()
    if stages:
        lines.append("Thesis construction order (reason in this order; never start from a KPI forecast):")
        lines.extend(f"{s['order']}. {s['id']}: {s['question']}" for s in stages)
    req = gm.get("required_for_each_link") or []
    if req:
        lines.append("Required for each causal link: " + ", ".join(str(r) for r in req) + ".")
    rules = gm.get("operating_rules") or {}
    if isinstance(rules, dict):
        lines.append("Operating rules:")
        for key, value in rules.items():
            if isinstance(value, list):
                lines.append(f"- {key}: " + ", ".join(str(v) for v in value))
            else:
                lines.append(f"- {key}: {value}")
    for anti in gm.get("anti_patterns") or []:
        if isinstance(anti, dict):
            seq = " → ".join(str(x) for x in anti.get("sequence") or [])
            lines.append(f"Anti-pattern {anti.get('id', '')}: {seq}. {anti.get('warning', '')}")
    return _bounded("\n".join(lines), max_chars)


def universal_rules_prompt_block(max_chars: int = 1200) -> str:
    rules = universal_research_rules()
    if not rules:
        return ""
    lines = ["## Universal research rules"]
    lines.extend(f"{i}. {r}" for i, r in enumerate(rules, start=1))
    return _bounded("\n".join(lines), max_chars)
