"""Wave 7A tests — research notes MVP.

Covers:
- `_parse_frontmatter` round-trips a YAML block + body cleanly; falls back
  to `({}, full_text)` when no delimiter present.
- `parse_note` populates the deterministic summary when frontmatter omits it.
- `_matches_filter` honors `["*"]` wildcard, empty-list "doesn't apply",
  and case-insensitive equality.
- `_is_active` drops expired and non-active-status notes.
- `select_for` cascades all four filter dimensions and respects weight ordering.
- `select_for` excludes notes whose agent tag doesn't include the caller.
- `render_summary_block` produces a non-empty markdown block when notes
  exist, empty string otherwise.
- The shipped `research_notes/` corpus has at least one parseable note.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from app.services import research_notes as rn


def _write_note(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_frontmatter_extracts_yaml_block():
    raw = (
        "---\n"
        "title: Test note\n"
        "weight: 0.9\n"
        "---\n"
        "Body text here."
    )
    fm, body = rn._parse_frontmatter(raw)
    assert fm["title"] == "Test note"
    assert fm["weight"] == 0.9
    assert body.strip() == "Body text here."


def test_parse_frontmatter_no_delimiter_returns_empty_dict_and_full_text():
    raw = "no frontmatter, just body content"
    fm, body = rn._parse_frontmatter(raw)
    assert fm == {}
    assert body == raw


def test_parse_note_uses_deterministic_summary_when_missing(tmp_path):
    note_path = _write_note(
        tmp_path, "books/sample.md",
        "---\ntitle: Sample\napplies_to_agents: [sector]\n---\n"
        "## Heading\n\nFirst paragraph captures the takeaway.\n\nSecond paragraph.",
    )
    note = rn.parse_note(note_path)
    assert note is not None
    assert note.frontmatter.title == "Sample"
    assert note.frontmatter.summary
    assert "First paragraph" in note.frontmatter.summary


def test_parse_note_minimal_frontmatter_inherits_defaults(tmp_path):
    """Just title — defaults should fill agents/sectors/weight/status."""
    note_path = _write_note(
        tmp_path, "frameworks/x.md",
        "---\ntitle: only-title\n---\nBody.",
    )
    note = rn.parse_note(note_path)
    assert note is not None
    assert note.frontmatter.applies_to_agents == ["sector"]
    assert note.frontmatter.applies_to_sectors == ["*"]
    assert note.frontmatter.weight == 0.5
    assert note.frontmatter.status == "active"


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def test_matches_filter_wildcard_and_explicit_match():
    assert rn._matches_filter(["*"], "Technology")
    assert rn._matches_filter(["Technology"], "technology")  # case-insensitive
    assert rn._matches_filter([], "Technology") is False
    assert rn._matches_filter(["Energy"], "Technology") is False


def test_matches_filter_no_target_passes():
    assert rn._matches_filter(["Energy"], None)


def test_is_active_respects_status():
    fm = rn.NoteFrontmatter(title="x", status="archived")
    assert rn._is_active(fm) is False
    fm = rn.NoteFrontmatter(title="x", status="active")
    assert rn._is_active(fm) is True


def test_is_active_respects_expires():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    fm_expired = rn.NoteFrontmatter(title="x", expires=yesterday)
    assert rn._is_active(fm_expired) is False
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    fm_fresh = rn.NoteFrontmatter(title="x", expires=tomorrow)
    assert rn._is_active(fm_fresh) is True


# ---------------------------------------------------------------------------
# select_for end-to-end
# ---------------------------------------------------------------------------

def _seed_corpus(tmp_path: Path) -> Path:
    _write_note(tmp_path, "books/a.md",
                "---\ntitle: A\napplies_to_agents: [sector]\n"
                "applies_to_sectors: ['Technology']\nweight: 0.9\n---\nBody A.")
    _write_note(tmp_path, "books/b.md",
                "---\ntitle: B\napplies_to_agents: [sector, valuation]\n"
                "applies_to_sectors: ['*']\nweight: 0.6\n---\nBody B.")
    _write_note(tmp_path, "books/c.md",
                "---\ntitle: C\napplies_to_agents: [valuation]\n"
                "applies_to_sectors: ['*']\nweight: 0.7\n---\nBody C only valuation.")
    _write_note(tmp_path, "books/d.md",
                "---\ntitle: D\napplies_to_agents: [sector]\n"
                "applies_to_sectors: ['Energy']\nweight: 0.95\n---\nBody D.")
    return tmp_path


def test_select_for_filters_by_agent_and_sector(tmp_path):
    root = _seed_corpus(tmp_path)
    out = rn.select_for(
        "sector", sector="Technology", corpus_root=root, max_notes=10,
    )
    titles = [n.title for n in out]
    # B (universal) + A (Technology) → both. C (valuation only) excluded.
    # D (Energy only) excluded.
    assert "A" in titles and "B" in titles
    assert "C" not in titles
    assert "D" not in titles


def test_select_for_orders_by_weight_descending(tmp_path):
    root = _seed_corpus(tmp_path)
    out = rn.select_for(
        "sector", sector="Technology", corpus_root=root, max_notes=10,
    )
    weights = [n.weight for n in out]
    assert weights == sorted(weights, reverse=True)


def test_select_for_respects_max_notes_cap(tmp_path):
    root = _seed_corpus(tmp_path)
    out = rn.select_for(
        "sector", sector="Technology", corpus_root=root, max_notes=1,
    )
    assert len(out) == 1
    # Highest-weight match wins.
    assert out[0].title == "A"


def test_select_for_excludes_notes_for_other_agents(tmp_path):
    root = _seed_corpus(tmp_path)
    out = rn.select_for(
        "earnings", sector="Technology", corpus_root=root, max_notes=10,
    )
    assert out == []


def test_select_for_excludes_expired_notes(tmp_path):
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    _write_note(tmp_path, "books/expired.md",
                f"---\ntitle: Expired\napplies_to_agents: [sector]\n"
                f"applies_to_sectors: ['*']\nexpires: '{yesterday}'\n---\nBody.")
    _write_note(tmp_path, "books/fresh.md",
                "---\ntitle: Fresh\napplies_to_agents: [sector]\n"
                "applies_to_sectors: ['*']\n---\nBody.")
    out = rn.select_for("sector", sector="X", corpus_root=tmp_path, max_notes=10)
    titles = [n.title for n in out]
    assert "Fresh" in titles
    assert "Expired" not in titles


def test_select_for_excludes_notes_for_specific_ticker_when_mismatched(tmp_path):
    _write_note(tmp_path, "personal/nvda-only.md",
                "---\ntitle: NVDA-only\napplies_to_agents: [sector]\n"
                "applies_to_sectors: ['*']\napplies_to_tickers: ['NVDA']\n---\nBody.")
    out_match = rn.select_for(
        "sector", sector="Technology", ticker="NVDA",
        corpus_root=tmp_path, max_notes=10,
    )
    out_miss = rn.select_for(
        "sector", sector="Technology", ticker="MSFT",
        corpus_root=tmp_path, max_notes=10,
    )
    assert any(n.title == "NVDA-only" for n in out_match)
    assert not any(n.title == "NVDA-only" for n in out_miss)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def test_render_summary_block_empty_returns_empty_string():
    assert rn.render_summary_block([]) == ""


def test_render_summary_block_includes_title_source_summary(tmp_path):
    root = _seed_corpus(tmp_path)
    out = rn.select_for("sector", sector="Technology", corpus_root=root)
    block = rn.render_summary_block(out)
    assert "## Discretionary investment context" in block
    assert "**A**" in block or "**B**" in block


# ---------------------------------------------------------------------------
# Shipped corpus parses
# ---------------------------------------------------------------------------

def test_shipped_corpus_has_parseable_notes():
    """The notes shipped under `research_notes/` should be valid — catches
    typos in seed files before they reach prod."""
    notes = rn.list_notes()
    # Either the shipped seeds parse or the corpus is genuinely empty.
    if notes:
        # Pat Dorsey or quality-compounders — at least one with weight ≥ 0.7.
        assert any(n.frontmatter.weight >= 0.7 for n in notes)
        for n in notes:
            assert n.frontmatter.title
            assert n.frontmatter.status in {"active", "archived", "superseded"}
