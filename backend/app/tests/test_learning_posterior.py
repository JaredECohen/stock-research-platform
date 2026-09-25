"""W7: the posterior arithmetic that turns evidence into a prior's credibility.

Owner decision 9: memory must "update priors and learn as evidence comes in"
without over-indexing. These pin the numbers that make that true: Beta(1,1)
updated per verdict, a 365-day half-life, and a stance table under which a
lesson needs three weighted later outcomes to be "supported" and retires
after four failures. Pure functions, injected `now`, no wall clock.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.learning.ledger import (
    evidence_weight,
    independence_key,
    parse_hypothesis,
    posterior,
    should_retire,
    trailing_sentences,
    verdict_from_alpha,
)

NOW = datetime(2026, 12, 1)


def _p(*verdicts: str):
    return posterior([(v, NOW) for v in verdicts], now=NOW)


def test_beta_updates_per_verdict():
    assert (_p().alpha, _p().beta) == (1.0, 1.0)
    assert (_p("held").alpha, _p("held").beta) == (2.0, 1.0)
    assert (_p("failed").alpha, _p("failed").beta) == (1.0, 2.0)
    assert (_p("mixed").alpha, _p("mixed").beta) == (1.5, 1.5)
    irrelevant = _p("irrelevant", "irrelevant")
    assert (irrelevant.alpha, irrelevant.beta, irrelevant.judged) == (1.0, 1.0, 0)
    assert _p("held", "failed", "irrelevant").judged == 2
    with pytest.raises(ValueError):
        _p("maybe")


def test_half_life_halves_weight_at_365d():
    assert evidence_weight(NOW, NOW) == 1.0
    assert evidence_weight(NOW - timedelta(days=365), NOW) == pytest.approx(0.5)
    assert evidence_weight(NOW - timedelta(days=730), NOW) == pytest.approx(0.25)
    # Future-dated evidence never weighs more than fresh evidence.
    assert evidence_weight(NOW + timedelta(days=30), NOW) == 1.0
    old = posterior([("held", NOW - timedelta(days=365))], now=NOW)
    assert old.alpha == pytest.approx(1.5) and old.n_eff == pytest.approx(0.5)
    assert old.stance == "untested"


def test_stance_thresholds_table():
    assert _p("held", "held").stance == "contested"            # n_eff 2 < 3
    five = _p(*["held"] * 5)
    assert five.stance == "supported" and five.lo80 == pytest.approx(0.70, abs=0.01)
    two_failed = _p("failed", "failed")
    assert two_failed.stance == "weakened" and two_failed.hi80 == pytest.approx(0.498, abs=0.001)
    four_failed = _p(*["failed"] * 4)
    assert should_retire(four_failed) and four_failed.hi80 == pytest.approx(0.348, abs=0.001)
    assert not should_retire(_p(*["failed"] * 3))
    assert _p("failed", "failed", "failed", "held").stance == "contested"
    assert _p().stance == "untested"


def test_interval_clipped_to_unit():
    # Beta(10, 1): mean 0.909 + 1.2816 * 0.083 = 1.015 before clipping.
    many = _p(*["held"] * 9)
    assert many.hi80 == 1.0 and 0.0 <= many.lo80 < many.hi80
    few = _p(*["failed"] * 9)
    assert few.lo80 == 0.0 and few.lo80 < few.hi80 <= 1.0


def test_verdict_follows_alpha_not_model():
    """held / failed come from realized alpha with the postmortem thresholds
    (strict > 2% / < -5% for an outperform call); nothing a model says can
    set them."""
    assert verdict_from_alpha("outperform", 0.021) == "held"
    assert verdict_from_alpha("outperform", 0.02) == "mixed"
    assert verdict_from_alpha("outperform", -0.051) == "failed"
    assert verdict_from_alpha("underperform", -0.021) == "held"
    assert verdict_from_alpha("underperform", 0.051) == "failed"
    assert verdict_from_alpha("underperform", 0.0) == "mixed"
    assert verdict_from_alpha("outperform", None) is None
    with pytest.raises(ValueError):
        verdict_from_alpha("up", 0.1)


def test_independence_key_is_one_window_per_scope():
    a = datetime(2026, 1, 5)
    same_window = a + timedelta(days=1)
    assert independence_key("MSFT", 90, a) == independence_key("MSFT", 90, same_window)
    assert independence_key("MSFT", 90, a) != independence_key("MSFT", 90, a + timedelta(days=91))
    assert independence_key("4510", 90, a).startswith("4510:90:")
    with pytest.raises(ValueError):
        independence_key("", 90, a)


@pytest.mark.parametrize("raw, reason", [
    (None, "empty"), ("", "empty"), ({}, "empty"), ({"condition": " "}, "empty"),
    ("When x, expect y", "not_an_object"),
    ({"condition": "x" * 161, "observable": "outperform"}, "condition_too_long"),
    ({"condition": "the stock outperforms peers", "observable": "outperform"}, "condition_names_outcome"),
    ({"condition": "gross margin guidance is raised", "observable": "up"}, "observable_invalid"),
    ({"condition": 5, "observable": "outperform"}, "condition_not_text"),
])
def test_grammar_rejects_nonconforming_hypotheses(raw, reason):
    assert parse_hypothesis(raw) == (None, reason)


def test_grammar_accepts_and_normalizes():
    h, why = parse_hypothesis({"condition": "  When gross margin guidance is raised twice. ",
                               "observable": "Outperform"})
    assert why is None and h is not None
    assert h.condition == "gross margin guidance is raised twice" and h.observable == "outperform"


def test_trailing_sentences_are_whole():
    text = "We said buy. The stock fell 12%. Margins compressed. **Next time, weigh the capex cycle.**"
    assert trailing_sentences(text, 40) == "Next time, weigh the capex cycle."
    assert trailing_sentences(text, 70) == "Margins compressed. Next time, weigh the capex cycle."
    assert trailing_sentences("One very long sentence " * 20, 30) == ""
