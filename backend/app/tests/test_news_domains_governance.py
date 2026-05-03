"""Wave 6C tests — news allow-list governance file.

Covers:
- The shipped `app/data/news_domains.json` parses cleanly.
- `_load_domain_lists` falls back to empty sets when the file is
  missing or malformed.
- `_filter_grounded_sources` honors the JSON-driven allow + block lists.
- `reload_domain_lists` picks up edits without a process restart.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.agents import news_agent


def test_shipped_domains_file_parses():
    """The committed governance file should always be valid JSON with
    `allowed` and `blocked` keys. Catches typos before they ship."""
    path = Path(__file__).resolve().parent.parent / "data" / "news_domains.json"
    data = json.loads(path.read_text())
    assert isinstance(data.get("allowed"), list)
    assert isinstance(data.get("blocked"), list)
    # Reuters should always be allowed; if someone removed it, that's
    # a likely typo / broken merge.
    assert "reuters.com" in data["allowed"]


def test_allowed_and_blocked_domains_are_loaded_at_runtime():
    news_agent.reload_domain_lists()
    allowed = news_agent.allowed_domains()
    blocked = news_agent.blocked_domains()
    assert "reuters.com" in allowed
    assert "msn.com" in blocked


def test_filter_grounded_sources_honors_allow_and_block():
    news_agent.reload_domain_lists()
    items = [
        {"title": "Reuters story", "url": "https://reuters.com/x"},
        {"title": "MSN aggregator", "url": "https://msn.com/x"},
        {"title": "Random blog", "url": "https://random-blog-xyz.com/x"},
        {"title": "Reuters subdomain", "url": "https://nordics.reuters.com/x"},
        {"title": "no url field", "url": ""},
    ]
    out = news_agent._filter_grounded_sources(items)
    titles = [i["title"] for i in out]
    assert "Reuters story" in titles
    assert "Reuters subdomain" in titles  # subdomain leniency preserved
    assert "MSN aggregator" not in titles  # blocked
    assert "Random blog" not in titles     # not in allow-list
    assert "no url field" in titles        # no domain → pass through


def test_reload_picks_up_edits(tmp_path, monkeypatch):
    """When the JSON file is rewritten, `reload_domain_lists` should
    surface the new lists on the next call."""
    custom = {"allowed": ["custom.example"], "blocked": ["bad.example"]}
    custom_path = tmp_path / "news_domains.json"
    custom_path.write_text(json.dumps(custom))
    monkeypatch.setattr(news_agent, "_NEWS_DOMAINS_PATH", custom_path)
    news_agent.reload_domain_lists()
    assert news_agent.allowed_domains() == {"custom.example"}
    assert news_agent.blocked_domains() == {"bad.example"}


def test_load_returns_empty_on_missing_or_bad_file(tmp_path, monkeypatch):
    """A missing or malformed file should default to empty sets, not crash."""
    bad_path = tmp_path / "missing.json"
    monkeypatch.setattr(news_agent, "_NEWS_DOMAINS_PATH", bad_path)
    news_agent.reload_domain_lists()
    assert news_agent.allowed_domains() == set()
    assert news_agent.blocked_domains() == set()

    # Now drop a malformed file in place and verify the same.
    bad_path.write_text("{ not json }")
    news_agent.reload_domain_lists()
    assert news_agent.allowed_domains() == set()
    assert news_agent.blocked_domains() == set()
