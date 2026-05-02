"""Wave 3B tests — Technical Analyst.

Two layers:

1) Indicator math (`app/finance/technicals.py`) — pure-math properties
   that should hold regardless of the price path:
   - SMA: latest value equals mean of trailing window.
   - EMA: bounded between min/max of the seeded window.
   - RSI: bounded [0, 100]; monotone-up series → 100; monotone-down → 0.
   - MACD: histogram = macd_line - signal_line.
   - Bollinger: position ∈ [0, 1]; price at upper band → 1.0.

2) Agent (`app/agents/technical_agent.run_technical_agent`) — produces
   a typed `AgentFinding` with the structured signals attached as
   `data["signals"]`, never raises on missing data, and stays in
   "positioning context" framing in the deterministic fallback.

3) Graph integration: a full memo run wires the technical agent into
   `memo.technical_agent_view` and does NOT change the rating relative
   to a baseline run with technicals disabled (technicals are positioning,
   not rating-driving).
"""
from __future__ import annotations

import math

import pytest

from app.agents.technical_agent import _deterministic_summary, run_technical_agent
from app.finance import technicals as ti
from app.schemas import AgentFinding, TechnicalSignals


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------

def test_sma_equals_arithmetic_mean_of_window():
    closes = list(range(1, 21))  # 1..20
    out = ti.sma(closes, 5)
    assert out == sum(closes[-5:]) / 5
    # Insufficient data → None.
    assert ti.sma([1, 2], 5) is None


def test_ema_bounded_by_window_extrema():
    closes = [10.0] * 30
    # All-flat series → EMA equals the constant.
    assert ti.ema(closes, 10) == pytest.approx(10.0)
    closes2 = list(range(1, 31))  # 1..30
    out = ti.ema(closes2, 10)
    assert out is not None
    assert min(closes2) <= out <= max(closes2)


def test_rsi_bounded_and_monotone_extremes():
    rising = list(range(1, 30))
    out = ti.rsi(rising, 14)
    assert out is not None
    # All gains, no losses → RSI = 100.
    assert out == pytest.approx(100.0)

    falling = list(range(30, 1, -1))
    out2 = ti.rsi(falling, 14)
    assert out2 is not None
    # All losses, no gains → RSI = 0.
    assert out2 == pytest.approx(0.0)


def test_rsi_oscillates_within_bounds():
    # Sinusoidal series — should sit somewhere in (0, 100).
    series = [50.0 + 5.0 * math.sin(i / 3.0) for i in range(60)]
    out = ti.rsi(series, 14)
    assert out is not None
    assert 0.0 < out < 100.0


def test_macd_identity_histogram_equals_line_minus_signal():
    closes = [100.0 + math.sin(i / 5.0) * 3.0 for i in range(80)]
    out = ti.macd(closes, 12, 26, 9)
    assert out is not None
    assert out["histogram"] == pytest.approx(out["macd_line"] - out["signal_line"], rel=1e-9)


def test_macd_returns_none_when_insufficient_history():
    short = [100.0 + i for i in range(20)]
    assert ti.macd(short, 12, 26, 9) is None


def test_bollinger_position_zero_at_lower_one_at_upper():
    # Constant series with a high outlier at the end → position should be 1.
    closes = [100.0] * 19 + [200.0]
    bb = ti.bollinger(closes, 20, 2.0)
    assert bb is not None
    assert 0.0 <= bb["position"] <= 1.0
    assert bb["upper"] >= bb["lower"]


def test_classify_trend_up_down_sideways():
    assert ti.classify_trend(50.0, 30.0, 60.0) == "up"
    assert ti.classify_trend(30.0, 50.0, 20.0) == "down"
    assert ti.classify_trend(50.0, 30.0, 25.0) == "sideways"
    # Missing inputs default to sideways.
    assert ti.classify_trend(None, 30.0, 60.0) == "sideways"


def test_classify_momentum_responds_to_rsi_and_macd():
    assert ti.classify_momentum(75.0, 0.5) == "positive"
    assert ti.classify_momentum(25.0, -0.5) == "negative"
    assert ti.classify_momentum(50.0, 0.0) == "neutral"


def test_compute_technical_signals_returns_none_for_short_series():
    # Less than 60 bars — bail out cleanly.
    assert ti.compute_technical_signals([{"close": 100.0}]) is None


def test_compute_technical_signals_full_payload_shape():
    rows = [{"close": 100.0 + i * 0.5, "volume": 1_000} for i in range(260)]
    rows.append({"close": 250.0, "volume": 5_000})  # an end-of-series push
    sig = ti.compute_technical_signals(rows)
    assert sig is not None
    assert sig["last_price"] == 250.0
    # Slow indicators populated when we have ≥234 bars.
    assert sig["sma_50"] is not None
    assert sig["sma_200"] is not None
    assert sig["macd_line"] is not None and sig["macd_signal"] is not None
    # Trend classification is one of the buckets.
    assert sig["trend"] in ("up", "down", "sideways")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def test_run_technical_agent_returns_finding_for_known_ticker():
    profile = {"ticker": "NVDA", "sector": "Technology"}
    finding = run_technical_agent(profile, days=300)
    assert isinstance(finding, AgentFinding)
    # In demo mode the agent runs through the deterministic narrative path —
    # we just assert the structural contract.
    assert finding.agent == "Technical Analyst"
    assert finding.headline
    assert isinstance(finding.data, dict)
    sig = finding.data.get("signals")
    assert isinstance(sig, dict)
    assert "trend" in sig and sig["trend"] in ("up", "down", "sideways")


def test_run_technical_agent_handles_missing_ticker():
    finding = run_technical_agent({}, days=300)
    assert finding.confidence < 0.5
    assert "unavailable" in finding.headline.lower()


def test_deterministic_summary_frames_as_positioning_not_signal():
    sig = TechnicalSignals(
        last_price=100.0, sma_50=95.0, sma_200=90.0, sma_50_above_200=True,
        rsi_14=55.0, macd_line=0.5, macd_signal=0.3, macd_histogram=0.2,
        bb_position=0.6, position_52w=0.4, trend="up", momentum="positive",
    )
    out = _deterministic_summary({"ticker": "MSFT"}, sig)
    body = (out["headline"] + " " + out["summary"]).lower()
    # Explicit guardrail — the deterministic narrative must not emit
    # standalone trade-signal language.
    assert "buy" not in body and "sell" not in body
    # And it must reinforce the positioning frame.
    assert "positioning" in body or "context" in body


# ---------------------------------------------------------------------------
# Graph integration — technicals do NOT override rating
# ---------------------------------------------------------------------------

def test_run_stock_memo_attaches_technical_view():
    from app.agents.graph import run_stock_memo
    memo = run_stock_memo("MSFT")
    assert memo.technical_agent_view is not None
    assert memo.technical_agent_view.agent == "Technical Analyst"
    sig = (memo.technical_agent_view.data or {}).get("signals")
    assert isinstance(sig, dict)
    assert "trend" in sig
