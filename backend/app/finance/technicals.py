"""Pure-math technical indicators for the Technical Analyst (Wave 3B).

All functions are deterministic, dependency-free (stdlib only), and operate
on simple price series. Kept separate from `agents/technical_agent.py` so
they're easy to unit-test in isolation against indicator math properties
(RSI bounded 0-100, MACD line - signal = histogram, SMA recency, etc.).

Why no NumPy/Pandas:
- The platform stays portable (no NumPy compile in lightweight runtimes).
- Indicator math is small (252 daily bars × ~10 indicators is trivial).
- Vectorization wouldn't speed up at this scale; pure Python is simpler.

All functions accept either a list of dicts (with `close` / `high` / `low`)
or a list of floats. Inputs are assumed chronologically ordered, oldest
first; the helper `_extract_closes` handles both shapes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Series helpers
# ---------------------------------------------------------------------------

def _extract(rows_or_floats: Sequence[Any], key: str) -> List[float]:
    """Pull a price column out of either a list of dicts or a list of floats.

    Accepts the shape returned by `market_data_service.get_price_series`
    (dicts with `close`/`high`/`low`/`open`/`volume`) and also works on a
    raw list of floats (for tests). Returns floats, dropping rows where
    the value is missing or non-numeric.
    """
    out: List[float] = []
    for r in rows_or_floats:
        if isinstance(r, dict):
            v = r.get(key)
        else:
            v = r
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _last_date(rows: Sequence[Any]) -> Optional[str]:
    if not rows:
        return None
    last = rows[-1]
    if isinstance(last, dict):
        return str(last.get("date") or "")
    return None


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(values: Sequence[float], window: int) -> Optional[float]:
    """Simple moving average over the LAST `window` observations."""
    if window <= 0 or len(values) < window:
        return None
    chunk = values[-window:]
    return sum(chunk) / len(chunk)


def ema(values: Sequence[float], window: int) -> Optional[float]:
    """Exponential moving average. Seeded with the SMA of the first `window`
    points so the first emitted EMA equals SMA — matches conventional charting
    libraries (TradingView, TA-Lib) within rounding."""
    if window <= 0 or len(values) < window:
        return None
    alpha = 2.0 / (window + 1.0)
    seed = sum(values[:window]) / window
    out = seed
    for v in values[window:]:
        out = alpha * v + (1.0 - alpha) * out
    return out


def vwma(rows: Sequence[Dict[str, Any]], window: int) -> Optional[float]:
    """Volume-weighted moving average. Requires dict rows with `close` + `volume`."""
    if window <= 0 or len(rows) < window:
        return None
    chunk = rows[-window:]
    num = 0.0
    den = 0.0
    for r in chunk:
        try:
            c = float(r.get("close")) if isinstance(r, dict) else float(r)
        except (TypeError, ValueError):
            continue
        try:
            v = float(r.get("volume")) if isinstance(r, dict) else 0.0
        except (TypeError, ValueError):
            v = 0.0
        num += c * v
        den += v
    if den == 0:
        # Fallback to plain SMA if volume is missing/zero across the window.
        return sma(_extract(chunk, "close"), len(chunk))
    return num / den


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

def rsi(values: Sequence[float], window: int = 14) -> Optional[float]:
    """Wilder's RSI. Bounded [0, 100]. Requires len(values) >= window + 1."""
    if window <= 0 or len(values) < window + 1:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    # Initial averages over the first `window` deltas.
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window
    # Wilder smoothing for the remainder.
    for i in range(window, len(gains)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(values: Sequence[float], fast: int = 12, slow: int = 26,
         signal: int = 9) -> Optional[Dict[str, float]]:
    """Standard MACD: 12/26 EMAs, 9-period signal line.

    Returns `{macd_line, signal_line, histogram}` or None if insufficient data.
    """
    if len(values) < slow + signal:
        return None
    # Compute EMAs at every step so we can build the signal line.
    fast_series = _ema_series(values, fast)
    slow_series = _ema_series(values, slow)
    if not fast_series or not slow_series:
        return None
    # Align tails — fast_series is longer (starts earlier).
    n = min(len(fast_series), len(slow_series))
    macd_series = [fast_series[-n + i] - slow_series[-n + i] for i in range(n)]
    if len(macd_series) < signal:
        return None
    signal_series = _ema_series(macd_series, signal)
    if not signal_series:
        return None
    macd_line = macd_series[-1]
    signal_line = signal_series[-1]
    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": macd_line - signal_line,
    }


def _ema_series(values: Sequence[float], window: int) -> List[float]:
    """All EMA values from index `window-1` onward — used for MACD lines."""
    if window <= 0 or len(values) < window:
        return []
    alpha = 2.0 / (window + 1.0)
    seed = sum(values[:window]) / window
    out: List[float] = [seed]
    for v in values[window:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger(values: Sequence[float], window: int = 20,
              num_stdev: float = 2.0) -> Optional[Dict[str, float]]:
    """Bollinger Bands. Returns `{upper, lower, middle, bandwidth, position}`.

    `position` is the latest price's relative location within the band:
    0.0 = at lower, 1.0 = at upper, 0.5 = at midline.
    """
    if window <= 0 or len(values) < window:
        return None
    chunk = values[-window:]
    mean = sum(chunk) / window
    var = sum((x - mean) ** 2 for x in chunk) / window
    std = var ** 0.5
    upper = mean + num_stdev * std
    lower = mean - num_stdev * std
    band_width = upper - lower
    last = values[-1]
    if band_width == 0:
        position = 0.5
    else:
        position = (last - lower) / band_width
    return {
        "upper": upper,
        "lower": lower,
        "middle": mean,
        "bandwidth": band_width,
        "position": max(0.0, min(1.0, position)),
    }


# ---------------------------------------------------------------------------
# 52-week + trend / momentum classification
# ---------------------------------------------------------------------------

def fifty_two_week(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    window = values[-252:] if len(values) > 252 else list(values)
    high = max(window)
    low = min(window)
    last = values[-1]
    rng = high - low
    pos = 0.5 if rng == 0 else (last - low) / rng
    return {
        "high_52w": high, "low_52w": low,
        "position_52w": max(0.0, min(1.0, pos)),
    }


def classify_trend(sma_50: Optional[float], sma_200: Optional[float],
                   last: Optional[float]) -> str:
    """Crude trend label using the golden-cross / death-cross convention."""
    if sma_50 is None or sma_200 is None or last is None:
        return "sideways"
    if sma_50 > sma_200 and last > sma_50:
        return "up"
    if sma_50 < sma_200 and last < sma_50:
        return "down"
    return "sideways"


def classify_momentum(rsi_value: Optional[float],
                      macd_hist: Optional[float]) -> str:
    """Momentum bucket from RSI extremes + MACD histogram sign."""
    if rsi_value is None and macd_hist is None:
        return "neutral"
    pos_signals = 0
    neg_signals = 0
    if rsi_value is not None:
        if rsi_value > 60:
            pos_signals += 1
        elif rsi_value < 40:
            neg_signals += 1
    if macd_hist is not None:
        if macd_hist > 0:
            pos_signals += 1
        elif macd_hist < 0:
            neg_signals += 1
    if pos_signals > neg_signals:
        return "positive"
    if neg_signals > pos_signals:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Full bundle
# ---------------------------------------------------------------------------

def compute_technical_signals(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One-shot computation of every indicator the Technical Analyst uses.

    Returns None if the price series is too short for the slowest indicator
    (we want SMA200 + MACD signal — call it 234 bars of headroom).
    """
    closes = _extract(rows, "close")
    if len(closes) < 60:
        return None  # need enough bars for the cheaper indicators at minimum

    sma_50 = sma(closes, 50)
    sma_200 = sma(closes, 200)
    ema_10 = ema(closes, 10)
    ema_20 = ema(closes, 20)
    rsi_14 = rsi(closes, 14)
    macd_d = macd(closes, 12, 26, 9)
    bb = bollinger(closes, 20, 2.0)
    vwma_20 = vwma(rows, 20)
    fw = fifty_two_week(closes)

    notes: List[str] = []
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200:
            notes.append("Golden-cross alignment (SMA50 above SMA200).")
        else:
            notes.append("Death-cross alignment (SMA50 below SMA200).")
    if rsi_14 is not None:
        if rsi_14 > 70:
            notes.append(f"RSI {rsi_14:.0f} — overbought.")
        elif rsi_14 < 30:
            notes.append(f"RSI {rsi_14:.0f} — oversold.")
    if bb and bb.get("position") is not None:
        p = bb["position"]
        if p > 0.95:
            notes.append("Price hugging upper Bollinger band.")
        elif p < 0.05:
            notes.append("Price hugging lower Bollinger band.")

    last_close = closes[-1]
    last_date = _last_date(rows)
    return {
        "last_price": last_close,
        "last_date": last_date,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "sma_50_above_200": (
            None if sma_50 is None or sma_200 is None else sma_50 > sma_200
        ),
        "ema_10": ema_10,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "macd_line": (macd_d or {}).get("macd_line"),
        "macd_signal": (macd_d or {}).get("signal_line"),
        "macd_histogram": (macd_d or {}).get("histogram"),
        "bb_upper": (bb or {}).get("upper"),
        "bb_lower": (bb or {}).get("lower"),
        "bb_middle": (bb or {}).get("middle"),
        "bb_position": (bb or {}).get("position"),
        "vwma_20": vwma_20,
        "high_52w": (fw or {}).get("high_52w"),
        "low_52w": (fw or {}).get("low_52w"),
        "position_52w": (fw or {}).get("position_52w"),
        "trend": classify_trend(sma_50, sma_200, last_close),
        "momentum": classify_momentum(rsi_14, (macd_d or {}).get("histogram")),
        "notes": notes,
    }
