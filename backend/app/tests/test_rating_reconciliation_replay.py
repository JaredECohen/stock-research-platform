"""The S14 merge gate: `scripts/rating_reconciliation_replay.py`.

The replay decides whether 7(b) is enforced or the owner is asked first
(stop if more than 25% of recent Bullish full runs would be downgraded, or
if `overvalued` would be the majority verdict). It runs on archived bodies
that live outside the repo, so these tests pin its arithmetic on trimmed
SYNTHETIC bodies in both archive shapes, and that its receipt carries no
memo prose.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

replay = importlib.import_module("scripts.rating_reconciliation_replay")

_THESIS = "PROSE-MUST-NOT-LEAK: the thesis text of a stored memo."


def _body(ticker: str, version: int, *, rating: str, trigger: str = "full_reanalysis",
          generated: str = "2026-09-15T10:00:00", mode: str = "live", family: float | None = None,
          prem: float | None = None, dcf: float | None = None, wrapped: bool = True,
          thesis: str = _THESIS, dcf_initial: float | None = None,
          sector: str | None = None) -> dict[str, Any]:
    memo: dict[str, Any] = {
        "ticker": ticker, "rating_label": rating, "generation_mode": mode,
        "generated_at": generated, "one_sentence_thesis": thesis,
        "valuation_verdict": {"verdict": "fairly_priced", "comps_ev_ebitda_premium": prem},
        "dcf_summary": {"base_upside": dcf, "tv_clamped": False},
        # Empty when no PM adjustment fired (the initial model IS the final one).
        "dcf_initial_summary": ({"base_upside": dcf_initial, "tv_clamped": False}
                                if dcf_initial is not None else {}),
        "sector": sector,
        "risk_committee_challenge": {"overall_assessment": "x", "review_mode": "rule_based"},
        "scorecard": ({"coverage": 1.0, "categories": {"valuation": {"percentile": family, "coverage": 1.0}}}
                      if family is not None else None),
    }
    if not wrapped:
        return {**memo, "version": version, "trigger": trigger}
    return {"status": 200, "headers": {"x-memo-version": str(version), "x-memo-trigger": trigger},
            "body": memo}


def _write(tmp_path: Path, bodies: list[dict[str, Any]]) -> Path:
    d = tmp_path / "bodies"
    d.mkdir()
    for i, b in enumerate(bodies):
        m = b.get("body", b)
        (d / f"{m['ticker']}_v{i + 1}.json").write_text(json.dumps(b))
    return d


def _sample(n_divergent_bull: int = 1, *, overvalued_neutrals: int = 0) -> list[dict[str, Any]]:
    rich = dict(prem=0.40, dcf=-0.60)       # comps rich + DCF agrees -> overvalued
    fair = dict(prem=0.02, dcf=-0.05)
    bodies = []
    for i in range(4):   # four Bullish full runs in the window
        kw = rich if i < n_divergent_bull else fair
        bodies.append(_body(f"B{i}", 1, rating="Bullish", **kw))
    bodies += [
        _body("N0", 1, rating="Neutral", family=50.0, **fair),
        _body("E0", 1, rating="Bearish", prem=-0.3, family=90.0, wrapped=False),  # Bearish on undervalued
        # Outside the full-run set: a patch, an out-of-window run, a demo run.
        _body("B0", 2, rating="Bullish", trigger="incremental_patch", **rich),
        _body("OLD", 1, rating="Bullish", generated="2026-07-01T00:00:00", **rich),
        _body("DEM", 1, rating="Bullish", mode="demo", **rich),
    ]
    for i in range(overvalued_neutrals):
        bodies.append(_body(f"OV{i}", 1, rating="Neutral", **rich))
    return bodies


def test_replay_on_captured_bodies(tmp_path):
    report = replay.replay(replay.load_bodies(_write(tmp_path, _sample(1))))
    fs = report["full_runs"]
    assert fs["n"] == 6
    assert fs["verdicts"] == {"undervalued": 1, "fairly_priced": 4, "overvalued": 1, "mixed": 0}
    assert (fs["bullish"], fs["bullish_downgrade_upper_bound"]) == (4, 1)
    assert fs["bullish_downgrade_rate"] == pytest.approx(0.25)     # at, not over, the threshold
    assert (fs["bearish"], fs["bearish_downgrade_upper_bound"]) == (1, 1)
    assert (fs["divergent"], fs["accepted"], fs["accept_rate"]) == (2, 0, 0.0)
    assert report["stop"] is False
    # Stored live = the latest live version per ticker: B0 is its v2 patch;
    # DEM is demo and dropped.
    stored = {(r["ticker"], r["version"]) for r in report["stored_live_rows"]}
    assert ("B0", 2) in stored and ("B0", 1) not in stored
    assert "DEM" not in {t for t, _ in stored} and ("OLD", 1) in stored
    rows = {r["ticker"]: r for r in report["full_run_rows"]}
    assert rows["B0"]["outcome"] == "downgraded" and rows["B0"]["final_rating"] == "Neutral"
    assert rows["B1"]["outcome"] == "consistent"


def test_stop_when_downgrade_rate_exceeds_threshold(tmp_path, capsys):
    d = _write(tmp_path, _sample(2))
    report = replay.replay(replay.load_bodies(d))
    assert report["full_runs"]["bullish_downgrade_rate"] == pytest.approx(0.5)
    assert report["stop"] and "exceeds 25%" in report["stop_reasons"][0]
    assert replay.main(["--bodies", str(d)]) == 3
    assert "STOP:" in capsys.readouterr().out


def test_stop_when_overvalued_is_the_majority(tmp_path):
    report = replay.replay(replay.load_bodies(_write(tmp_path, _sample(1, overvalued_neutrals=5))))
    fs = report["full_runs"]
    assert fs["verdicts"]["overvalued"] == 6 and fs["n"] == 11
    assert fs["bullish_downgrade_rate"] == pytest.approx(0.25)
    assert report["stop"] and any("majority" in r for r in report["stop_reasons"])


def test_receipt_has_counts_and_no_memo_prose(tmp_path):
    d = _write(tmp_path, _sample(1))
    md_out, json_out = tmp_path / "receipt.md", tmp_path / "report.json"
    assert replay.main(["--bodies", str(d), "--md-out", str(md_out), "--json-out", str(json_out)]) == 0
    md = md_out.read_text()
    assert "below threshold — enforce" in md and "| Live full runs | 6 |" in md
    assert "| B0 | 1 | full_reanalysis |" in md
    for text in (md, json_out.read_text()):
        assert "PROSE-MUST-NOT-LEAK" not in text


def test_thesis_rewrite_counts_under_the_new_guard(tmp_path):
    bodies = [
        # Neutral on overvalued evidence, thesis says undervalued: both disagree.
        _body("R1", 1, rating="Neutral", prem=0.4, dcf=-0.6, thesis="R1 is undervalued — x."),
        # Agrees with the rating: kept.
        _body("R2", 1, rating="Neutral", prem=0.4, dcf=-0.6, thesis="R2 is fairly priced — x."),
    ]
    report = replay.replay(replay.load_bodies(_write(tmp_path, bodies)))
    rows = {r["ticker"]: r["thesis_rewrite"] for r in report["full_run_rows"]}
    assert rows == {"R1": True, "R2": False}
    assert report["full_runs"]["thesis_rewrites"] == 1


def test_unusable_input_is_an_error_not_a_smaller_sample(tmp_path):
    d = _write(tmp_path, _sample(1))
    (d / "junk.json").write_text(json.dumps({"hello": "world"}))
    assert replay.main(["--bodies", str(d)]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert replay.main(["--bodies", str(empty)]) == 2


def test_replay_votes_on_the_initial_dcf_not_the_pm_adjusted_one(tmp_path):
    """The PM who rates the name also moved the adjusted model, so only the
    INITIAL model may vote, exactly as in the pipeline."""
    bodies = [
        # Initial -21% (inside the band), PM-adjusted -45%: comps alone -> fairly priced.
        _body("I1", 1, rating="Bullish", prem=0.54, dcf_initial=-0.21, dcf=-0.45),
        # Initial -45%, PM-adjusted -21%: the initial agrees with comps -> overvalued.
        _body("I2", 1, rating="Bullish", prem=0.54, dcf_initial=-0.45, dcf=-0.21),
    ]
    rows = {r["ticker"]: r for r in replay.replay(replay.load_bodies(_write(tmp_path, bodies)))["full_run_rows"]}
    assert rows["I1"]["verdict"] == "fairly_priced" and rows["I1"]["dcf_initial"] == -0.21
    assert rows["I2"]["verdict"] == "overvalued" and rows["I2"]["outcome"] == "downgraded"


def test_replay_withholds_the_comps_vote_for_financials(tmp_path):
    bodies = [_body("BK", 1, rating="Bullish", prem=0.40, dcf=-0.60, sector="Financial Services"),
              _body("TK", 1, rating="Bullish", prem=0.40, dcf=-0.60, sector="Technology")]
    rows = {r["ticker"]: r for r in replay.replay(replay.load_bodies(_write(tmp_path, bodies)))["full_run_rows"]}
    assert rows["BK"]["verdict"] == "fairly_priced" and "comps_ev_ebitda" not in rows["BK"]["votes"]
    assert rows["TK"]["verdict"] == "overvalued"


def test_stop_when_only_the_stored_memos_are_overvalued_majority(tmp_path):
    """Full runs below both thresholds; the latest stored versions (patches
    outside the window) are overvalued-majority: stop for that reason alone."""
    rich = dict(prem=0.40, dcf=-0.60)
    fair = dict(prem=0.02, dcf=-0.05)
    bodies = [_body(f"S{i}", 1, rating="Neutral", **fair) for i in range(3)]
    bodies += [_body(f"S{i}", 2, rating="Neutral", trigger="incremental_patch", **rich) for i in range(3)]
    report = replay.replay(replay.load_bodies(_write(tmp_path, bodies)))
    assert report["full_runs"]["verdicts"]["overvalued"] == 0
    assert report["stored_live"]["verdicts"]["overvalued"] == 3
    assert report["stop"] is True
    assert report["stop_reasons"] == ["overvalued is the majority verdict over the stored live memos"]
