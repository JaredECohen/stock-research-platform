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

from . import industry_knowledge, industry_labels

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
# When the mandate prose alone would fill a prompt block, list fields are
# trimmed by whole ranked items down to this floor before anything is cut
# mid-sentence; every trim is counted in an omission line.
_MIN_ITEMS_PER_FIELD = 3
# The sub-industry layer is guaranteed this share of a prompt block (up to
# its own cap) so a long mandate can never push it — or the attribution
# line after it — off the end of the prompt.
_SUB_BLOCK_SHARE = 3
SUB_INDUSTRY_HEADER = "Sub-industries (original analyst briefs; provenance per line):"
# The public edition of the sub-industry layer (owner decision 2026-09-24:
# no taxonomy codes or registry names on anything a reader can see, and a
# model that never sees them cannot echo them into memo prose).
PUBLIC_SUB_INDUSTRY_HEADER = "Sub-industry briefs (original analyst research; economics | advantage test):"
PROVENANCE_MODES: tuple[str, ...] = ("codes", "public")
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

    def render(self) -> str:
        """The one prompt line for this brief — the unit every char budget
        is measured against, so the budget counts what is actually shown."""
        src = ",".join(self.source_ids) if self.source_ids else "n/a"
        return (
            f"- {self.code} {self.name} [industry {self.industry_code}; sources {src}] — "
            f"{self.economics} | Advantage test: {self.advantage_test}"
        )

    def render_public(self) -> str:
        """The brief as an unnamed ``economics | advantage test`` line: no
        8-digit code, no registry name, no industry code, no source ids.
        The prose goes through the one scrubber in case it names a code."""
        return "- " + industry_labels.scrub_text(
            f"{self.economics} | Advantage test: {self.advantage_test}"
        )

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

    def as_prompt_block(self, max_chars: int = PROMPT_BLOCK_MAX_CHARS, *, provenance: str = "codes") -> str:
        """The mandate as prompt prose, bounded, provenance on every line.

        ``provenance="codes"`` (the default, unchanged): item provenance is
        the industry code list in brackets; a reader (or the validator) can
        trace any KPI back to the encyclopedia entry that named it.

        ``provenance="public"``: what every model prompt whose output can
        reach a reader uses (the analyst system prompt, which the report
        writer also sends, and the memo's company context). The header is
        our label, there is no "Industries:" line, no ``[code]`` brackets
        anywhere (list items, highest-EVI questions, the priority source),
        the sub-industry briefs are unnamed ``economics | advantage test``
        lines, and the attribution is the public one. Raises
        ``industry_labels.UnknownLabel`` for a group with no public label.

        The budget is split so nothing is lost silently: the attribution
        line is reserved first (every brief is original analysis and must
        say so wherever it is displayed), the sub-industry layer is
        guaranteed a share, and the mandate prose is trimmed by whole
        ranked items — each trim counted — before the ellipsis fallback.
        """
        if provenance not in PROVENANCE_MODES:
            raise ValueError(f"provenance must be one of {PROVENANCE_MODES}, not {provenance!r}")
        public = provenance == "public"
        attribution = industry_labels.PUBLIC_BRIEF_ATTRIBUTION if public else self.attribution
        attribution_line = f"Attribution: {attribution}"
        has_subs = bool(self.sub_industries or self.sub_industries_truncated)
        sub_reserve = min(SUB_INDUSTRY_BLOCK_MAX_CHARS, max_chars // _SUB_BLOCK_SHARE) if has_subs else 0
        head_budget = max_chars - (len(attribution_line) + 1) - (sub_reserve + 1 if sub_reserve else 0)
        head = self._fit_head(head_budget, public=public)
        parts = [head]
        if has_subs:
            remaining = max_chars - len(head) - 1 - (len(attribution_line) + 1)
            sub_block = self.sub_industry_block(min(SUB_INDUSTRY_BLOCK_MAX_CHARS, remaining), public=public)
            if sub_block:
                parts.append(sub_block)
        parts.append(attribution_line)
        return _bounded("\n".join(parts), max_chars)

    def _head_lines(self, items_per_field: int, *, public: bool = False) -> list[str]:
        item = _public_item if public else _with_codes
        if public:
            # The identity line is our label and nothing else: no code, no
            # registry name, no internal taxonomy key, no industry list.
            lines: list[str] = [
                f"## Industry Group mandate — {industry_labels.label(self.code)} "
                f"({industry_labels.label(self.sector_code)})",
            ]
        else:
            lines = [
                f"## Industry Group mandate — {self.code} {self.name} "
                f"(sector {self.sector_code} {self.sector_name}; taxonomy {self.version_key})",
                "Industries: " + "; ".join(f"{i['code']} {i['name']}" for i in self.industries) + ".",
            ]
        if self.research_priority is not None:
            # The priority's source is "<industry code> <registry name>":
            # internal provenance, so the public edition gives the score only.
            lines.append(
                f"Research priority: {self.research_priority}/5."
                if public else
                f"Research priority: {self.research_priority}/5 "
                f"(set by {self.research_priority_source})."
            )
        if self.cadence:
            lines.append("Cadence: " + "; ".join(item(r) for r in self.cadence) + ".")
        if self.preferred_archetypes:
            lines.append(
                "Preferred archetypes: " + "; ".join(item(r) for r in self.preferred_archetypes) + "."
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
        omitted = 0
        for key in LIST_FIELDS:
            ranked = self.lists.get(key, ())
            if not ranked:
                continue
            omitted += max(0, len(ranked) - items_per_field)
            lines.append(
                f"{labels[key]}: " + "; ".join(item(r) for r in ranked[:items_per_field]) + "."
            )
        if omitted:
            lines.append(
                f"({omitted} lower-ranked list item{'s' if omitted != 1 else ''} omitted for length; "
                "the full ranked lists are in the mandate record.)"
            )
        if self.highest_evi_questions:
            lines.append("Highest-EVI questions:")
            if public:
                lines.extend(f"- {industry_labels.scrub_text(q)}" for _code, q in self.highest_evi_questions)
            else:
                lines.extend(f"- [{code}] {q}" for code, q in self.highest_evi_questions)
        return lines

    def _fit_head(self, budget: int, *, public: bool = False) -> str:
        """Mandate prose within `budget`, cut by whole units and counted at
        every step: fewer items per list field first, then whole trailing
        lines (the highest-EVI questions go before the ranked lists), and
        the ellipsis only when not even the title and industry list fit."""
        lines: list[str] = []
        for n in range(_ITEMS_PER_FIELD, _MIN_ITEMS_PER_FIELD - 1, -1):
            lines = self._head_lines(n, public=public)
            text = "\n".join(lines)
            if len(text) <= budget:
                return text
        # The title and the industry list are the identity of the mandate
        # (the public edition has the title only); everything after them is
        # droppable, one whole line at a time.
        identity = 1 if public else 2
        keep = list(lines)
        while len(keep) > identity:
            dropped = len(lines) - len(keep) + 1
            note = (
                f"({dropped} further mandate line{'s' if dropped != 1 else ''} omitted for length; "
                "the full mandate is in the record.)"
            )
            candidate = "\n".join(keep[:-1] + [note])
            if len(candidate) <= budget:
                return candidate
            keep.pop()
        return _bounded("\n".join(lines[:identity]), max(0, budget))

    def sub_industry_block(self, max_chars: int | None = None, *, public: bool = False) -> str:
        """Sub-industries as `code name [industry] — economics | advantage
        test (sources)` (public: unnamed `economics | advantage test`
        lines), refitted to `max_chars` with every brief that does not fit
        counted in the omission line. Empty when the group carries none."""
        if not (self.sub_industries or self.sub_industries_truncated):
            return ""
        budget = SUB_INDUSTRY_BLOCK_MAX_CHARS if max_chars is None else max_chars
        _, _, text = _fit_briefs(self.sub_industries, budget, self.sub_industries_truncated, public=public)
        return text


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


def _public_item(item: RankedItem) -> str:
    """A ranked item without its provenance brackets, prose scrubbed: the
    public prompt carries no taxonomy code at all."""
    return industry_labels.scrub_text(item.text)


def _bounded(text: str, max_chars: int) -> str:
    """Never returns more than `max_chars` — the ellipsis costs a character
    too, and a non-positive budget buys nothing at all."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    return text[: max_chars - 1].rstrip() + "…"


# --- the mandate ---------------------------------------------------------------


def _omission_line(count: int) -> str:
    return f"- … {count} more sub-industr{'y' if count == 1 else 'ies'} omitted for length."


def _fit_briefs(
    briefs: tuple[SubIndustryBrief, ...] | list[SubIndustryBrief], max_chars: int, already_omitted: int = 0,
    *, public: bool = False,
) -> tuple[tuple[SubIndustryBrief, ...], int, str]:
    """Keep briefs in taxonomy order while the RENDERED block — header, one
    line per brief and the omission line whenever anything is left out —
    fits `max_chars`. Returns `(kept, omitted, text)`; `omitted` counts
    `already_omitted` (briefs an earlier, larger budget already dropped)
    so the reader always sees the true shortfall."""
    header = PUBLIC_SUB_INDUSTRY_HEADER if public else SUB_INDUSTRY_HEADER
    rendered = [b.render_public() if public else b.render() for b in briefs]
    lines = [header, *rendered]
    if already_omitted:
        lines.append(_omission_line(already_omitted))
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return tuple(briefs), already_omitted, text
    total = len(briefs) + already_omitted
    # Reserve the omission line at its widest so adding it never overruns.
    reserve = len(_omission_line(total)) + 1
    used = len(header)
    if used + reserve > max_chars:
        # Not even the header fits: say only what was left out.
        return (), total, _bounded(_omission_line(total), max_chars)
    kept: list[SubIndustryBrief] = []
    kept_lines: list[str] = []
    for brief, line in zip(briefs, rendered):
        if used + 1 + len(line) + reserve > max_chars:
            continue
        kept.append(brief)
        kept_lines.append(line)
        used += 1 + len(line)
    omitted = total - len(kept)
    text = "\n".join([header, *kept_lines, _omission_line(omitted)])
    return tuple(kept), omitted, text


def _sub_industry_briefs(
    industries: list[dict[str, Any]], budget: int,
) -> tuple[tuple[SubIndustryBrief, ...], int]:
    """Briefs in taxonomy order until `budget` characters of the rendered
    block are spent; the rest are counted, not dropped silently."""
    briefs: list[SubIndustryBrief] = []
    for industry in industries:
        for sub in industry.get("sub_industries") or []:
            fields = sub.get("fields") or {}
            briefs.append(SubIndustryBrief(
                code=str(sub.get("code", "")),
                name=str(sub.get("name", "")),
                industry_code=str(industry.get("code", "")),
                economics=first_sentence(fields.get("economics")),
                advantage_test=first_sentence(fields.get("advantage_test")),
                source_ids=tuple(str(s) for s in (sub.get("source_ids") or [])),
                attribution=industry_knowledge.BRIEF_ATTRIBUTION,
            ))
    if not briefs:
        return (), 0
    kept, omitted, _ = _fit_briefs(briefs, budget)
    return kept, omitted


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
