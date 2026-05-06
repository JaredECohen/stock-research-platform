"""Curatable markdown prompts loaded at runtime.

Files in this package (`pm_identity.md`, `macro_primer.md`, etc.) are
read by the agent layer at startup via `load_prompt(name)`. Keeping
them as markdown rather than f-strings makes them easier to iterate
on without touching Python — a senior buy-side voice belongs in prose,
not in escaped triple-quotes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> Optional[str]:
    """Read `<name>.md` from this package.

    Returns None when the file is missing — caller falls back to its
    embedded default. Cached for the process lifetime; restart the
    backend to pick up edits (no hot-reload is intentional).
    """
    if not name.endswith(".md"):
        name = f"{name}.md"
    path = PROMPTS_DIR / name
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()
