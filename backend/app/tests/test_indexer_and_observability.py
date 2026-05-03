"""Wave 8E tests — research-notes indexer + history-backfill observability.

Covers:
- The indexer normalizes a minimally-tagged note (title only, body)
  into a fully-defaulted file with deterministic summary.
- `--check` mode is a dry-run that exits nonzero when changes would land.
- `_index.json` is written with one entry per parseable note.
- `history_backfill.run_once` classifies rate-limit + auth failures
  separately from generic errors, surfaces them in the loop note.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.monitoring import history_backfill
from app.services import history_service


def _seed_minimal_note(tmp: Path) -> Path:
    p = tmp / "books" / "minimal.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "---\n"
        "title: Minimal note\n"
        "applies_to_agents: [sector]\n"
        "applies_to_sectors: ['*']\n"
        "---\n"
        "Body paragraph one.\n\nBody paragraph two.\n",
        encoding="utf-8",
    )
    return p


def _run_indexer(tmp: Path, *args: str) -> subprocess.CompletedProcess:
    backend_root = Path(__file__).resolve().parent.parent.parent
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.index_research_notes",
            "--root", str(tmp), *args,
        ],
        cwd=str(backend_root),
        capture_output=True, text=True, timeout=60,
    )


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

def test_indexer_writes_index_json_for_corpus(tmp_path):
    _seed_minimal_note(tmp_path)
    out = _run_indexer(tmp_path)
    assert out.returncode == 0, f"indexer failed: {out.stderr}"
    index_path = tmp_path / "_index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert data["count"] == 1
    assert data["notes"][0]["title"] == "Minimal note"
    assert data["notes"][0]["summary"]
    assert data["schema_version"] == 1


def test_indexer_idempotent_on_second_run(tmp_path):
    _seed_minimal_note(tmp_path)
    _run_indexer(tmp_path)
    # Second invocation should now exit 0 with no further file changes
    # detected by --check.
    out = _run_indexer(tmp_path, "--check")
    assert out.returncode == 0, f"--check should pass after first run: {out.stdout} {out.stderr}"


def test_indexer_check_mode_exits_nonzero_when_changes_pending(tmp_path):
    """A fresh corpus has no _index.json — --check should exit 1."""
    _seed_minimal_note(tmp_path)
    out = _run_indexer(tmp_path, "--check")
    assert out.returncode == 1
    # And the file is not actually written when in check mode.
    assert not (tmp_path / "_index.json").exists()


def test_indexer_handles_empty_corpus(tmp_path):
    out = _run_indexer(tmp_path)
    # Empty corpus → exit 0, no index written (count would be 0 → file
    # WOULD be written for traceability; we just check no crash).
    assert out.returncode == 0


# ---------------------------------------------------------------------------
# Backfill observability
# ---------------------------------------------------------------------------

def test_history_backfill_classifies_rate_limit_failures(monkeypatch):
    """Simulated 429s should bump `rate_limited` separately from `errors`."""
    monkeypatch.setattr(
        history_backfill, "_tier1_tickers",
        lambda: ["FAKE_TKR_RATE"],
    )

    def boom(*args, **kwargs):
        raise RuntimeError("FMP responded 429: rate limit exceeded")

    # The loop binds `backfill_ticker` at module import; patch the
    # locally-bound name, not the underlying service module.
    monkeypatch.setattr(history_backfill, "backfill_ticker", boom)
    res = history_backfill.run_once()
    assert res["errors"] == 1
    assert res["rate_limited"] == 1
    assert res["auth_errors"] == 0
    # Loop status flagged as failed.
    from app.monitoring import status_snapshot
    snap = status_snapshot()
    assert snap.get("history_backfill", {}).get("success") is False
    assert "rate_limited=1" in snap.get("history_backfill", {}).get("note", "")


def test_history_backfill_classifies_auth_failures(monkeypatch):
    monkeypatch.setattr(
        history_backfill, "_tier1_tickers",
        lambda: ["FAKE_TKR_AUTH"],
    )

    def boom(*args, **kwargs):
        raise RuntimeError("403 Forbidden")

    # The loop binds `backfill_ticker` at module import; patch the
    # locally-bound name, not the underlying service module.
    monkeypatch.setattr(history_backfill, "backfill_ticker", boom)
    res = history_backfill.run_once()
    assert res["errors"] == 1
    assert res["auth_errors"] == 1


def test_history_backfill_clean_run_marks_success(monkeypatch):
    monkeypatch.setattr(
        history_backfill, "_tier1_tickers",
        lambda: [],  # nothing to do — vacuously clean
    )
    res = history_backfill.run_once()
    assert res["errors"] == 0
    from app.monitoring import status_snapshot
    snap = status_snapshot()
    assert snap.get("history_backfill", {}).get("success") is True
