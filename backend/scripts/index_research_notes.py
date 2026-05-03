"""Wave 7A / 8E — research-notes indexer.

Walks `research_notes/`, parses every `.md` file's YAML frontmatter,
fills missing fields with sensible defaults, optionally regenerates
summaries via the LLM (when keys are present + `--llm-summaries` set),
and writes back the file (preserving body) plus an aggregate
`_index.json` for tooling.

Idempotent — running twice on the same corpus is a no-op when nothing
has changed; otherwise updates only the affected files.

Run from `backend/`:

    python -m scripts.index_research_notes              # deterministic only
    python -m scripts.index_research_notes --llm        # use LLM for summaries
    python -m scripts.index_research_notes --check      # dry-run; nonzero
                                                          exit if any file
                                                          would be updated
                                                          (good for CI)

The deterministic path runs without keys: missing summary → first-paragraph
extract; missing weight → 0.5; missing status → "active"; etc.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Allow `python -m scripts.index_research_notes` from `backend/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.research_notes import (  # noqa: E402
    NoteFrontmatter,
    _deterministic_summary,
    _root_default,
    list_notes,
)


log = logging.getLogger("index_research_notes")


def _llm_summarize(body: str) -> str:
    """LLM enrichment fallback. Returns deterministic summary on failure."""
    try:
        from app.agents import llm
        from app.config import settings
        if not settings.has_llm:
            return _deterministic_summary(body)
        prompt = (
            "You are summarizing a discretionary research note in one or two "
            "sentences (max 220 chars). Extract the takeaway, not the topic. "
            "No quotes, no preamble.\n\n"
            "Note body:\n" + (body[:4000] if body else "")
        )
        text = llm.chat_text(
            prompt,
            system="You are a careful equity-research editor.",
            route="cheap", model=settings.openai_tool_model,
        )
        if text and text.strip():
            return text.strip()[:240]
    except Exception as exc:
        log.warning("LLM summary failed; falling back: %s", exc)
    return _deterministic_summary(body)


def _serialize_frontmatter(fm: NoteFrontmatter) -> str:
    """Render frontmatter as a compact YAML block. Preserves empty lists
    as `[]` rather than YAML's verbose multi-line list form."""
    import yaml
    data: Dict[str, Any] = fm.model_dump()
    # yaml.safe_dump emits some keys we don't want bare-empty in the file —
    # drop None values so they get re-defaulted on next read.
    data = {k: v for k, v in data.items() if v is not None}
    return yaml.safe_dump(data, default_flow_style=False, sort_keys=False).strip()


def _split_body(text: str) -> tuple[str, str]:
    """Pull the existing frontmatter block off `text`, return `(fm_yaml, body)`."""
    import re
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return "", text
    return m.group(1), m.group(2)


def _rewrite_note(path: Path, *, use_llm: bool, dry_run: bool) -> bool:
    """Re-emit the note with normalized frontmatter. Returns True if a
    change WOULD be (or was) written."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        log.warning("read failed for %s: %s", path, exc)
        return False

    _, body = _split_body(text)
    # Re-parse via the canonical loader so defaults + date coercion apply.
    from app.services.research_notes import parse_note
    note = parse_note(path)
    if note is None:
        return False
    fm = note.frontmatter

    # Backfill `chars` and `summary`.
    new_chars = len(note.body)
    if fm.chars != new_chars:
        fm.chars = new_chars
    if not fm.summary or fm.summary == _deterministic_summary(note.body):
        if use_llm:
            fm.summary = _llm_summarize(note.body)
        else:
            fm.summary = _deterministic_summary(note.body)

    rewritten = "---\n" + _serialize_frontmatter(fm) + "\n---\n\n" + note.body.lstrip("\n")
    if rewritten.strip() == text.strip():
        return False
    if not dry_run:
        path.write_text(rewritten, encoding="utf-8")
    return True


def _build_index(notes) -> Dict[str, Any]:
    """Aggregate index of every parseable note. Used by `_index.json`."""
    return {
        "schema_version": 1,
        "count": len(notes),
        "notes": [
            {
                "path": str(Path(n.path).relative_to(_root_default()))
                if Path(n.path).is_relative_to(_root_default())
                else n.path,
                "title": n.frontmatter.title,
                "source": n.frontmatter.source,
                "applies_to_agents": n.frontmatter.applies_to_agents,
                "applies_to_sectors": n.frontmatter.applies_to_sectors,
                "applies_to_tickers": n.frontmatter.applies_to_tickers,
                "weight": n.frontmatter.weight,
                "status": n.frontmatter.status,
                "summary": n.frontmatter.summary,
                "chars": n.frontmatter.chars,
            }
            for n in notes
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Override the corpus root.")
    parser.add_argument("--llm", action="store_true",
                        help="Regenerate missing summaries via LLM (cheap route).")
    parser.add_argument("--check", action="store_true",
                        help="Dry-run; exit nonzero if any file would change.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[indexer] %(message)s",
    )
    root = Path(args.root) if args.root else _root_default()
    if not root.exists():
        log.warning("corpus root does not exist: %s", root)
        return 0

    paths = sorted(root.rglob("*.md"))
    log.info("found %d note(s) under %s", len(paths), root)

    changed = 0
    for p in paths:
        if _rewrite_note(p, use_llm=args.llm, dry_run=args.check):
            changed += 1
            log.info("%s %s",
                     "would update" if args.check else "updated",
                     p.relative_to(root) if p.is_relative_to(root) else p)

    notes = list_notes(corpus_root=root)
    index = _build_index(notes)
    index_path = root / "_index.json"
    new_blob = json.dumps(index, indent=2, default=str)
    if not index_path.exists() or index_path.read_text() != new_blob:
        if not args.check:
            index_path.write_text(new_blob, encoding="utf-8")
            log.info("wrote %s", index_path.relative_to(root))
        changed += 1

    log.info("done — %d file(s) %s",
             changed, "would change" if args.check else "changed")
    if args.check and changed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
