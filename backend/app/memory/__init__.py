"""Long-term agent memory (filesystem markdown).

Each ticker has a `companies/<TICKER>.md` and each sector has a
`sectors/<slug>.md`. Files are append-only narratives the analysts read at
the start of every run and update at the end *only when a delta event
fires* — a new earnings release, a new 10-K/10-Q/8-K accession, or a
material/breaking news alert.

Entries are bounded: when the recent-entries section exceeds
`settings.memory_max_entries`, the oldest `memory_condense_batch` entries are
summarized (LLM if available, deterministic template otherwise) and folded
into a single "Historical context" block. We never drop information — the
condensed block keeps the high-level lessons and a rough timeline so the
agent can still reason over its own past mistakes.
"""
from .longterm import (
    CompanyMemory,
    MacroMemory,
    MemoryEntry,
    PMMemory,
    SectorMemory,
    company_memory_path,
    macro_memory_path,
    pm_memory_path,
    sector_memory_path,
    sector_slug,
)

__all__ = [
    "CompanyMemory",
    "MacroMemory",
    "MemoryEntry",
    "PMMemory",
    "SectorMemory",
    "company_memory_path",
    "macro_memory_path",
    "pm_memory_path",
    "sector_memory_path",
    "sector_slug",
]
