"""Structural tests for `services/calibration_service.py`.

The aggregators are read-only over `memo_outcomes` + `memo_postmortems`
keyed by horizon. Every test here uses horizon 7, which no production
cadence (30/90/180/365) or other test writes, so the "empty DB" case is
genuinely empty regardless of collection order and the seeded case
sees only its own rows.
"""
from __future__ import annotations

import socket
from datetime import datetime, timedelta

import pytest

from app.database import SessionLocal
from app.models import MemoOutcome, MemoPostmortem, MemoSnapshot
from app.services import calibration_service as cal

H = 7                                   # horizon reserved for this file
PREFIX = "TSTCAL"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _refuse(*_a, **_k):
        raise RuntimeError("network access attempted during an offline structural test")
    monkeypatch.setattr(socket.socket, "connect", _refuse)


def _purge() -> None:
    with SessionLocal() as db:
        for model in (MemoPostmortem, MemoOutcome, MemoSnapshot):
            db.query(model).filter(model.ticker.like(f"{PREFIX}%")).delete(synchronize_session=False)
        db.query(MemoPostmortem).filter(MemoPostmortem.horizon_days == H).delete()
        db.query(MemoOutcome).filter(MemoOutcome.horizon_days == H).delete()
        db.commit()


@pytest.fixture(autouse=True)
def _clean():
    _purge()
    yield
    _purge()


def _seed(
    ticker: str, rating: str, *, alpha: float | None,
    verdict: str | None = None, attribution=None, regime: str | None = None,
    realized: float | None = None, bench: float | None = None,
) -> None:
    with SessionLocal() as db:
        snap = MemoSnapshot(
            ticker=ticker, version=1, trigger="first_run",
            memo_json={"ticker": ticker, "rating_label": rating}, revision_log=[],
            generated_at=datetime.utcnow() - timedelta(days=30),
        )
        db.add(snap)
        db.flush()
        db.add(MemoOutcome(
            memo_snapshot_id=snap.id, ticker=ticker, rating_at_memo=rating,
            horizon_days=H, forward_return=realized, benchmark_return=bench, alpha=alpha,
        ))
        if verdict is not None:
            db.add(MemoPostmortem(
                memo_snapshot_id=snap.id, ticker=ticker, horizon_days=H,
                verdict=verdict, lesson="seed", agent_attribution=attribution,
                realized_return=realized, benchmark_return=bench, regime_at_memo=regime,
            ))
        db.commit()


EMPTY_BUCKET = {"n": 0, "mean_alpha": None, "median": None, "p25": None, "p75": None, "win_rate": None}
EMPTY_AGENT = {"n": 0, "mean_attribution": None, "contrib_to_right": 0, "contrib_to_right_pct": None}


# ---------------------------------------------------------------------------
# Empty DB
# ---------------------------------------------------------------------------

def test_calibration_by_rating_on_empty_horizon():
    out = cal.calibration_by_rating(horizon_days=H)
    assert set(out) == {"horizon_days", "rating_order", "buckets"}
    assert out["horizon_days"] == H
    assert out["rating_order"] == cal.RATING_ORDER
    assert list(out["buckets"]) == cal.RATING_ORDER      # stable display order
    assert all(b == EMPTY_BUCKET for b in out["buckets"].values())


def test_per_agent_attribution_on_empty_horizon():
    out = cal.per_agent_attribution(horizon_days=H)
    assert set(out) == {"horizon_days", "total_postmortems", "total_right", "agents"}
    assert out["total_postmortems"] == 0 and out["total_right"] == 0
    assert list(out["agents"]) == cal.ALL_AGENTS
    assert all(a == EMPTY_AGENT for a in out["agents"].values())


def test_regime_conditional_accuracy_on_empty_horizon():
    assert cal.regime_conditional_accuracy(horizon_days=H) == {"horizon_days": H, "regimes": {}}


def test_summary_composes_the_three_views():
    out = cal.summary(horizon_days=H)
    assert set(out) == {"calibration", "per_agent", "regime_conditional"}
    assert out["calibration"]["horizon_days"] == H
    assert out["per_agent"]["total_postmortems"] == 0
    assert out["regime_conditional"]["regimes"] == {}


# ---------------------------------------------------------------------------
# Seeded
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded():
    _seed(f"{PREFIX}VB", "Very Bullish", alpha=0.10, verdict="right",
          attribution={"sector": 0.8, "valuation": 0.5, "bogus_agent": 1.0, "macro": "n/a"},
          regime=" Soft_Landing ", realized=0.16, bench=0.06)
    _seed(f"{PREFIX}B1", "Bullish", alpha=0.05, verdict="right",
          attribution={"sector": 0.1}, regime=None, realized=0.07, bench=0.02)
    _seed(f"{PREFIX}B2", "Bullish", alpha=-0.01, verdict="wrong",
          attribution="junk-string", regime="", realized=0.01, bench=0.02)
    _seed(f"{PREFIX}BR", "Bearish", alpha=0.08, verdict="wrong",
          attribution={"sector": -0.6}, regime="soft_landing", realized=0.10, bench=0.02)
    _seed(f"{PREFIX}NA", "Neutral", alpha=None)                 # pending: no alpha, no postmortem
    _seed(f"{PREFIX}XX", "Strong Buy", alpha=0.3)               # label outside RATING_ORDER


def test_calibration_buckets(seeded):
    buckets = cal.calibration_by_rating(horizon_days=H)["buckets"]
    assert buckets["Very Bullish"] == {
        "n": 1, "mean_alpha": 0.10, "median": 0.10, "p25": 0.10, "p75": 0.10, "win_rate": 1.0,
    }
    b = buckets["Bullish"]
    assert b["n"] == 2
    assert abs(b["mean_alpha"] - 0.02) < 1e-12
    assert abs(b["median"] - 0.02) < 1e-12
    assert abs(b["p25"] - 0.005) < 1e-12 and abs(b["p75"] - 0.035) < 1e-12
    assert b["win_rate"] == 0.5
    assert buckets["Bearish"]["n"] == 1 and buckets["Bearish"]["win_rate"] == 1.0
    assert buckets["Neutral"] == EMPTY_BUCKET                    # alpha None excluded
    assert buckets["Very Bearish"] == EMPTY_BUCKET
    assert sum(b["n"] for b in buckets.values()) == 4          # unknown label dropped


def test_per_agent_attribution(seeded):
    out = cal.per_agent_attribution(horizon_days=H)
    assert out["total_postmortems"] == 4
    assert out["total_right"] == 2
    agents = out["agents"]
    assert set(agents) == set(cal.ALL_AGENTS)                  # bogus_agent never appears
    sector = agents["sector"]
    assert sector["n"] == 3
    assert abs(sector["mean_attribution"] - (0.8 + 0.1 - 0.6) / 3) < 1e-12
    assert sector["contrib_to_right"] == 1                     # 0.8 counts, 0.1 is under 0.2
    assert sector["contrib_to_right_pct"] == 0.5
    assert agents["valuation"] == {
        "n": 1, "mean_attribution": 0.5, "contrib_to_right": 1, "contrib_to_right_pct": 0.5,
    }
    # Non-numeric score ignored; pct is 0.0 (not None) once any memo was right.
    unseen = {"n": 0, "mean_attribution": None, "contrib_to_right": 0, "contrib_to_right_pct": 0.0}
    assert agents["macro"] == unseen
    assert agents["technical"] == unseen


def test_regime_conditional_accuracy(seeded):
    out = cal.regime_conditional_accuracy(horizon_days=H)
    regimes = out["regimes"]
    assert set(regimes) == {"soft_landing", "unknown"}         # trimmed + lower-cased
    soft = regimes["soft_landing"]
    assert set(soft) == {"n", "right", "wrong", "mixed", "accuracy", "mean_alpha"}
    assert (soft["n"], soft["right"], soft["wrong"], soft["mixed"]) == (2, 1, 1, 0)
    assert soft["accuracy"] == 0.5
    assert abs(soft["mean_alpha"] - ((0.16 - 0.06) + (0.10 - 0.02)) / 2) < 1e-12
    unknown = regimes["unknown"]
    assert (unknown["n"], unknown["right"], unknown["wrong"]) == (2, 1, 1)


def test_percentile_helper():
    assert cal._percentile([], 0.5) is None
    assert cal._percentile([3.0], 0.25) == 3.0
    assert cal._percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert cal._percentile([4.0, 1.0, 3.0, 2.0], 0.0) == 1.0
    assert cal._percentile([4.0, 1.0, 3.0, 2.0], 1.0) == 4.0
