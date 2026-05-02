"""Read/write helpers for long-term agent memory files.

File format:

    ---
    subject: AMZN
    kind: company
    last_updated: 2026-04-30
    entry_count: 47
    ---

    # AMZN — Long-term memory

    ## Historical context (condensed)
    <free-form summary updated on eviction>

    ## Recent entries

    ### 2026-04-30 — earnings
    **Trigger:** Q1 2026 10-Q (accession 0001018724-26-000123)
    **Observation:** AWS growth re-accelerated to 19% YoY; ads up 24% — ...
    **Update to thesis:** ...

    ### 2026-03-15 — material_news
    ...

Entries are parsed into `MemoryEntry` dataclasses; the file is rewritten
atomically on every mutation (write to temp + rename) so a crash never
leaves a half-written state.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import json as _json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _root() -> Path:
    p = Path(settings.memory_dir).expanduser()
    if not p.is_absolute():
        # Anchor relative paths to the backend directory so callers in any
        # CWD see the same files. Fallback: process CWD.
        backend_root = Path(__file__).resolve().parent.parent.parent
        p = (backend_root / p).resolve()
    return p


def company_memory_path(ticker: str) -> Path:
    return _root() / "companies" / f"{ticker.upper()}.md"


def sector_slug(sector: str) -> str:
    """Stable filesystem-safe slug for a sector name."""
    s = (sector or "unknown").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def sector_memory_path(sector: str) -> Path:
    return _root() / "sectors" / f"{sector_slug(sector)}.md"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """One structured entry in the 'Recent entries' section.

    Wave 3D: when a filing/transcript trigger fires, the reflection step
    can attach a `structured_facts` dict (segments, guidance changes,
    capex commentary, M&A, leadership changes). It's persisted as a JSON
    fenced code block following the markdown body so the parser can round-
    trip it without breaking older entries (the parser ignores the block
    if it's missing).
    """
    date: str  # ISO date (YYYY-MM-DD)
    trigger: str  # e.g. "earnings", "filing:10-K", "material_news"
    body: str  # Markdown — multiple paragraphs allowed
    structured_facts: Optional[Dict[str, Any]] = None

    def render(self) -> str:
        out = f"### {self.date} — {self.trigger}\n\n{self.body.strip()}\n"
        if self.structured_facts:
            try:
                blob = _json.dumps(self.structured_facts, indent=2, default=str, sort_keys=True)
                out += f"\n```structured-facts\n{blob}\n```\n"
            except Exception:
                pass
        return out


@dataclass
class CrossCompanyPattern:
    """A transferable lesson learned on one company that applies to peers.

    Lives in sector memory only. The sector agent reads patterns whose
    `applies_to` contains the current target ticker at run start, so a
    lesson learned on AMZN's AWS gets surfaced when running GOOGL or MSFT.
    """
    date: str
    source_company: str
    applies_to: List[str]
    lesson: str  # plain markdown, multi-paragraph allowed

    def render(self) -> str:
        applies = ", ".join(self.applies_to) if self.applies_to else "(unspecified)"
        return (
            f"### {self.date} — {self.source_company} → applies to: {applies}\n\n"
            f"{self.lesson.strip()}\n"
        )


# Regex for parsing entry headers from the file
_ENTRY_HEADER_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2}) — (.+)$", re.MULTILINE)
_PATTERN_HEADER_RE = re.compile(
    r"^### (\d{4}-\d{2}-\d{2}) — (\S+) → applies to: (.+)$", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Base memory class
# ---------------------------------------------------------------------------

@dataclass
class _MemoryFile:
    """In-memory representation of a memory markdown file."""
    path: Path
    subject: str
    kind: str  # "company" or "sector"
    last_updated: str = ""
    entry_count: int = 0
    static_section: str = ""  # rarely-changing seed content
    historical_context: str = ""  # condensed summary block
    entries: List[MemoryEntry] = field(default_factory=list)
    # Cross-company patterns are populated only on sector memory files. The
    # field lives on the base class so parsing/rendering stays uniform; for
    # company files it just stays empty.
    cross_company_patterns: List[CrossCompanyPattern] = field(default_factory=list)

    # ----- I/O -----

    @classmethod
    def load(cls, path: Path, subject: str, kind: str) -> "_MemoryFile":
        if not path.exists():
            return cls(path=path, subject=subject, kind=kind)
        text = path.read_text(encoding="utf-8")
        return cls._parse(path, subject, kind, text)

    def save(self) -> None:
        """Atomic write: temp file in the same dir + rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = self._render()
        # NamedTemporaryFile in same dir so rename stays on the same fs.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ----- Mutation -----

    def append_entry(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        self.entry_count = len(self.entries)
        self.last_updated = date.today().isoformat()

    def needs_condensation(self) -> bool:
        return len(self.entries) > settings.memory_max_entries

    def condense_oldest(self, summarizer=None) -> int:
        """Move the oldest `memory_condense_batch` entries into the historical
        context block. Returns how many entries were folded.

        `summarizer(entries: List[MemoryEntry], existing: str) -> str` is
        optional; when provided (e.g. an LLM-backed summarizer) it produces
        the new condensed block. When omitted, a deterministic template is
        used so this works in demo mode.
        """
        n = settings.memory_condense_batch
        if len(self.entries) <= settings.memory_max_entries:
            return 0
        # Take the oldest n entries; the rest stay as recent entries.
        oldest = self.entries[:n]
        remaining = self.entries[n:]
        if summarizer is None:
            new_block = _deterministic_summary(oldest, self.historical_context)
        else:
            try:
                new_block = summarizer(oldest, self.historical_context)
            except Exception as exc:  # pragma: no cover
                log.warning("memory summarizer failed: %s", exc)
                new_block = _deterministic_summary(oldest, self.historical_context)
        self.historical_context = new_block.strip()
        self.entries = remaining
        self.entry_count = len(self.entries)
        return len(oldest)

    # ----- Rendering for prompts -----

    def as_prompt_context(self, max_chars: int = 4000) -> str:
        """The chunk an analyst prepends to its system prompt at run-start."""
        if (not self.path.exists()
                and not self.entries
                and not self.historical_context
                and not self.cross_company_patterns):
            return ""
        parts: List[str] = [f"### Long-term memory: {self.subject} ({self.kind})\n"]
        if self.static_section.strip():
            parts.append(self.static_section.strip() + "\n")
        if self.historical_context.strip():
            parts.append("Historical context:\n" + self.historical_context.strip() + "\n")
        if self.cross_company_patterns:
            parts.append("Cross-company patterns (lessons from peer names):\n")
            # Most recent first
            for p in list(reversed(self.cross_company_patterns))[:8]:
                parts.append(p.render())
        if self.entries:
            parts.append("Recent entries:\n")
            # Newest first so the most recent context is closest to the model
            for entry in reversed(self.entries[-12:]):
                parts.append(entry.render())
        out = "\n".join(parts)
        if len(out) > max_chars:
            out = out[: max_chars - 60] + "\n…[truncated for prompt budget]"
        return out

    # ----- Internal: parse / render -----

    @classmethod
    def _parse(cls, path: Path, subject: str, kind: str, text: str) -> "_MemoryFile":
        front = {}
        body = text
        if text.startswith("---"):
            # Pull front matter
            end = text.find("\n---", 3)
            if end != -1:
                fm_block = text[3:end].strip()
                body = text[end + 4 :].lstrip("\n")
                for line in fm_block.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        front[k.strip()] = v.strip()
        # Split body into sections by H2 headings
        sections: dict[str, str] = {}
        current = "_intro"
        buf: List[str] = []
        for line in body.splitlines():
            if line.startswith("## "):
                sections[current] = "\n".join(buf).strip()
                current = line[3:].strip().lower()
                buf = []
            else:
                buf.append(line)
        sections[current] = "\n".join(buf).strip()
        static_section = sections.get("static profile", "")
        historical = sections.get("historical context (condensed)", "") or sections.get(
            "historical context", ""
        )
        recent_block = sections.get("recent entries", "")
        entries = _parse_entries(recent_block)
        patterns_block = sections.get("cross-company patterns", "")
        patterns = _parse_patterns(patterns_block)
        return cls(
            path=path,
            subject=front.get("subject") or subject,
            kind=front.get("kind") or kind,
            last_updated=front.get("last_updated", ""),
            entry_count=int(front.get("entry_count", "0") or 0),
            static_section=static_section,
            historical_context=historical,
            entries=entries,
            cross_company_patterns=patterns,
        )

    def _render(self) -> str:
        lines: List[str] = []
        lines.append("---")
        lines.append(f"subject: {self.subject}")
        lines.append(f"kind: {self.kind}")
        lines.append(f"last_updated: {self.last_updated or date.today().isoformat()}")
        lines.append(f"entry_count: {len(self.entries)}")
        lines.append("---\n")
        title = (
            f"# {self.subject} — Long-term memory"
            if self.kind == "company"
            else f"# {self.subject} sector — Long-term memory"
        )
        lines.append(title + "\n")
        if self.static_section.strip():
            lines.append("## Static profile\n")
            lines.append(self.static_section.strip() + "\n")
        lines.append("## Historical context (condensed)\n")
        lines.append((self.historical_context.strip() or "_No condensed history yet._") + "\n")
        # Sector files use the cross-company-patterns section; for company
        # files this section is empty and we omit it entirely so the file
        # stays clean.
        if self.kind == "sector" or self.cross_company_patterns:
            lines.append("## Cross-company patterns\n")
            if self.cross_company_patterns:
                for p in self.cross_company_patterns:
                    lines.append(p.render())
            else:
                lines.append("_No transferable lessons recorded yet._\n")
        lines.append("## Recent entries\n")
        if self.entries:
            for entry in self.entries:
                lines.append(entry.render())
        else:
            lines.append("_No entries yet._\n")
        return "\n".join(lines)


_STRUCTURED_FACTS_RE = re.compile(
    r"```structured-facts\s*\n(.*?)\n```", re.DOTALL,
)


def _split_structured_facts(body: str) -> "tuple[str, Optional[Dict[str, Any]]]":
    """Pull a trailing ```structured-facts``` JSON block out of an entry body.

    Wave 3D round-trips structured-fact dicts by appending them as a
    fenced code block. Older entries lack the block; parsing returns
    `(body_unchanged, None)` for those.
    """
    m = _STRUCTURED_FACTS_RE.search(body)
    if not m:
        return body, None
    try:
        facts = _json.loads(m.group(1))
    except (ValueError, TypeError):
        return body, None
    cleaned = (body[: m.start()] + body[m.end():]).rstrip()
    if not isinstance(facts, dict):
        return cleaned, None
    return cleaned, facts


def _parse_entries(block: str) -> List[MemoryEntry]:
    if not block.strip():
        return []
    # Skip pattern-shaped headers — they go through `_parse_patterns`.
    matches = [
        m for m in _ENTRY_HEADER_RE.finditer(block)
        if " → applies to:" not in m.group(2)
    ]
    out: List[MemoryEntry] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        body = block[start:end].strip()
        body, facts = _split_structured_facts(body)
        out.append(MemoryEntry(
            date=m.group(1), trigger=m.group(2).strip(), body=body,
            structured_facts=facts,
        ))
    return out


def _parse_patterns(block: str) -> List[CrossCompanyPattern]:
    if not block.strip():
        return []
    matches = list(_PATTERN_HEADER_RE.finditer(block))
    out: List[CrossCompanyPattern] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        lesson = block[start:end].strip()
        applies = [t.strip().upper() for t in m.group(3).split(",") if t.strip()]
        out.append(CrossCompanyPattern(
            date=m.group(1),
            source_company=m.group(2).strip().upper(),
            applies_to=applies,
            lesson=lesson,
        ))
    return out


def _deterministic_summary(
    entries: List[MemoryEntry], prior: str
) -> str:
    """Demo-mode condenser. Preserves a one-line takeaway per entry plus the
    date range and trigger histogram so future-you can still see what shaped
    the agent's view."""
    if not entries:
        return prior
    dates = [e.date for e in entries]
    triggers: dict[str, int] = {}
    for e in entries:
        triggers[e.trigger] = triggers.get(e.trigger, 0) + 1
    lines: List[str] = []
    if prior.strip() and not prior.strip().startswith("_"):
        lines.append(prior.strip())
        lines.append("")
    lines.append(
        f"**Condensed {dates[0]} → {dates[-1]}** "
        f"({len(entries)} entries; "
        + ", ".join(f"{k}={v}" for k, v in sorted(triggers.items()))
        + ")"
    )
    for e in entries:
        # First non-empty paragraph as the takeaway
        first_para = next(
            (p for p in e.body.split("\n\n") if p.strip() and not p.strip().startswith("**")),
            e.body[:160],
        )
        takeaway = first_para.strip().replace("\n", " ")[:200]
        lines.append(f"- {e.date} ({e.trigger}): {takeaway}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public per-subject classes
# ---------------------------------------------------------------------------

class CompanyMemory(_MemoryFile):
    """Per-company memory file (`memory/companies/<TICKER>.md`)."""

    @classmethod
    def for_ticker(cls, ticker: str) -> "CompanyMemory":
        path = company_memory_path(ticker)
        loaded = cls.load(path, ticker.upper(), "company")
        # mypy/pylance: cast to CompanyMemory since load returns _MemoryFile
        return loaded  # type: ignore[return-value]


class SectorMemory(_MemoryFile):
    """Per-sector memory file (`memory/sectors/<slug>.md`).

    Acts as a secondary system prompt: heuristics that worked, regimes that
    fooled the agent, KPI weights to lean on for this sector. Also stores
    `CrossCompanyPattern` entries — transferable lessons learned on one
    company that the sector agent applies to peers (e.g., AWS lessons →
    GCP / Azure). Read at the start of every sector run, written by the
    reflection step after a run."""

    @classmethod
    def for_sector(cls, sector: str) -> "SectorMemory":
        path = sector_memory_path(sector)
        loaded = cls.load(path, sector, "sector")
        return loaded  # type: ignore[return-value]

    def add_pattern(self, p: CrossCompanyPattern) -> None:
        """Append a cross-company pattern, deduping by (date, source, applies_to)."""
        key = (p.date, p.source_company, tuple(sorted(p.applies_to)))
        for existing in self.cross_company_patterns:
            ek = (existing.date, existing.source_company,
                  tuple(sorted(existing.applies_to)))
            if ek == key:
                return  # dedupe
        self.cross_company_patterns.append(p)
        self.last_updated = date.today().isoformat()

    def patterns_for(self, ticker: str) -> List[CrossCompanyPattern]:
        """Return patterns applicable to `ticker`. Source = the company we
        learned the lesson on; we exclude self-patterns (a pattern with
        source==ticker is the agent's own past entry, already covered by the
        company memory file)."""
        t = (ticker or "").upper()
        return [
            p for p in self.cross_company_patterns
            if t in p.applies_to and p.source_company != t
        ]

    def as_prompt_context_for(self, ticker: str, max_chars: int = 4000) -> str:
        """Like `as_prompt_context` but filters cross-company patterns to those
        relevant to `ticker`. Use this when running the sector agent so the
        prompt only carries patterns the current ticker can actually use."""
        relevant = self.patterns_for(ticker)
        # Temporarily swap so the standard renderer picks up only the relevant ones
        full = list(self.cross_company_patterns)
        self.cross_company_patterns = relevant
        try:
            return self.as_prompt_context(max_chars=max_chars)
        finally:
            self.cross_company_patterns = full
