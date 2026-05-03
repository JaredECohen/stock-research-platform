"""Wave 7B tests — BM25 body retrieval for research notes.

Covers:
- `_tokenize_body` strips non-word characters and lowercases.
- `select_bodies` returns no excerpts when no notes match the routing
  filter (no false positives from the BM25 over an empty subset).
- BM25 ranks the most-relevant note first; tie-breaking is deterministic
  on (score desc, weight desc, title asc).
- `top_k` caps the result count.
- `char_cap` caps combined body bytes; oversize bodies are truncated.
- `render_body_block` emits a non-empty markdown block when excerpts exist.
"""
from __future__ import annotations

from pathlib import Path

from app.services import research_notes as rn


def _write(tmp: Path, rel: str, content: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_tokenize_body_lowercases_and_strips_punct():
    out = rn._tokenize_body("AI capex flowed THROUGH; Hyperscalers!!")
    assert out == ["ai", "capex", "flowed", "through", "hyperscalers"]


def test_select_bodies_returns_empty_when_no_notes_match_filter(tmp_path):
    _write(tmp_path, "books/x.md",
           "---\ntitle: X\napplies_to_agents: [valuation]\n"
           "applies_to_sectors: ['*']\n---\nBody about valuation.")
    out = rn.select_bodies(
        "sector", "anything", corpus_root=tmp_path, top_k=2,
    )
    assert out == []


def test_select_bodies_ranks_most_relevant_first(tmp_path):
    _write(tmp_path, "books/moats.md",
           "---\ntitle: Moats\napplies_to_agents: [sector]\n"
           "applies_to_sectors: ['*']\nweight: 0.5\n---\n"
           "Moats are durability tests for cohort placement.")
    _write(tmp_path, "books/macro.md",
           "---\ntitle: Macro\napplies_to_agents: [sector]\n"
           "applies_to_sectors: ['*']\nweight: 0.5\n---\n"
           "Macro regime affects sector-fit. Hyperscaler capex.")
    out = rn.select_bodies(
        "sector", "moats durability", corpus_root=tmp_path, top_k=2,
    )
    assert len(out) >= 1
    # Moats note should rank first for a "moats durability" query.
    assert out[0].title == "Moats"


def test_select_bodies_tie_break_uses_weight_then_title(tmp_path):
    """Two notes with equal BM25 score — higher weight wins."""
    _write(tmp_path, "books/a.md",
           "---\ntitle: AAA\napplies_to_agents: [sector]\n"
           "applies_to_sectors: ['*']\nweight: 0.5\n---\n"
           "Cohort placement matters.")
    _write(tmp_path, "books/b.md",
           "---\ntitle: BBB\napplies_to_agents: [sector]\n"
           "applies_to_sectors: ['*']\nweight: 0.9\n---\n"
           "Cohort placement matters.")
    out = rn.select_bodies(
        "sector", "cohort placement", corpus_root=tmp_path, top_k=2,
    )
    # Both should have the same BM25 score; weight 0.9 wins → BBB first.
    assert out[0].title == "BBB"


def test_select_bodies_respects_top_k_cap(tmp_path):
    for i in range(5):
        _write(tmp_path, f"books/n{i}.md",
               f"---\ntitle: N{i}\napplies_to_agents: [sector]\n"
               f"applies_to_sectors: ['*']\nweight: 0.5\n---\n"
               f"Cohort placement is real. {i}")
    out = rn.select_bodies(
        "sector", "cohort", corpus_root=tmp_path, top_k=2,
    )
    assert len(out) <= 2


def test_select_bodies_respects_char_cap(tmp_path):
    big = "lorem ipsum cohort " * 500  # ~10KB
    _write(tmp_path, "books/big.md",
           f"---\ntitle: Big\napplies_to_agents: [sector]\n"
           f"applies_to_sectors: ['*']\n---\n{big}")
    out = rn.select_bodies(
        "sector", "cohort", corpus_root=tmp_path, top_k=5, char_cap=2_000,
    )
    assert len(out) == 1
    assert len(out[0].body) <= 2_000 + 1  # cap honored (truncated suffix +1 char)


def test_select_bodies_skips_notes_with_zero_bm25_score(tmp_path):
    """A note that doesn't match any query token should not be returned."""
    _write(tmp_path, "books/unrelated.md",
           "---\ntitle: Unrelated\napplies_to_agents: [sector]\n"
           "applies_to_sectors: ['*']\n---\n"
           "Cooking recipes and travel guides.")
    out = rn.select_bodies(
        "sector", "valuation discount premium", corpus_root=tmp_path, top_k=5,
    )
    assert out == []


def test_render_body_block_empty_returns_empty():
    assert rn.render_body_block([]) == ""


def test_render_body_block_includes_titles_and_bodies(tmp_path):
    _write(tmp_path, "books/x.md",
           "---\ntitle: Sample\nsource: Test source\n"
           "applies_to_agents: [sector]\napplies_to_sectors: ['*']\n---\n"
           "Sample body content here.")
    out = rn.select_bodies(
        "sector", "sample body", corpus_root=tmp_path, top_k=1,
    )
    block = rn.render_body_block(out)
    assert "Discretionary investment context (relevant excerpts)" in block
    assert "Sample" in block
    assert "Test source" in block
    assert "Sample body content here." in block


def test_select_bodies_ticker_filter(tmp_path):
    _write(tmp_path, "personal/nvda.md",
           "---\ntitle: NVDA personal\napplies_to_agents: [sector]\n"
           "applies_to_sectors: ['*']\napplies_to_tickers: ['NVDA']\n---\n"
           "Cohort placement notes for NVDA specifically.")
    match = rn.select_bodies(
        "sector", "cohort placement", ticker="NVDA",
        corpus_root=tmp_path, top_k=2,
    )
    miss = rn.select_bodies(
        "sector", "cohort placement", ticker="MSFT",
        corpus_root=tmp_path, top_k=2,
    )
    assert any(e.title == "NVDA personal" for e in match)
    assert not any(e.title == "NVDA personal" for e in miss)
