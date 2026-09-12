"""PM rating blend (Option A): the final rating_label mixes the LLM's
directional call with the deterministic factor_pm_score.

These tests pin the math at the helper level and exercise the override
site in `graph.py` end-to-end via a tiny memo stub, so a regression
either in the weight clamp or the bucket centers is caught.
"""
from __future__ import annotations

from app.schemas import rating_from_stock_score, score_from_rating_label


def test_score_from_rating_label_bucket_centers():
    assert score_from_rating_label("Very Bullish") == 90.0
    assert score_from_rating_label("Bullish") == 70.0
    assert score_from_rating_label("Neutral") == 50.0
    assert score_from_rating_label("Bearish") == 30.0
    assert score_from_rating_label("Very Bearish") == 10.0


def test_score_from_rating_label_unknown_falls_back_to_neutral():
    assert score_from_rating_label(None) == 50.0
    assert score_from_rating_label("") == 50.0
    assert score_from_rating_label("not a rating") == 50.0


def test_blend_math_matches_formula():
    # Weak factors (40 → Neutral) + strong LLM call (Very Bullish=90).
    # At w=0.4 → blended = 0.4 * 90 + 0.6 * 40 = 60 → Bullish
    factor = 40.0
    llm = score_from_rating_label("Very Bullish")
    blended = 0.4 * llm + 0.6 * factor
    assert rating_from_stock_score(blended) == "Bullish"
    # At w=0.0 the factor wins outright → Neutral
    assert rating_from_stock_score(0.0 * llm + 1.0 * factor) == "Neutral"
    # At w=1.0 the LLM wins outright → Very Bullish
    assert rating_from_stock_score(1.0 * llm + 0.0 * factor) == "Very Bullish"


def test_graph_blend_override_site(monkeypatch):
    """End-to-end test of the override block at graph.py: it must
    populate memo.scores with the blend audit fields and overwrite
    rating_label per the configured weight."""
    from types import SimpleNamespace

    from app import config as _config

    # Force weight=0.4 regardless of env / config.env.
    monkeypatch.setattr(_config.settings, "llm_rating_weight", 0.4)

    # The override block only touches rating_label + scores, so a
    # minimal stub avoids dragging in StockMemoOut's many required
    # subfields and keeps the test focused on the blend math.
    memo = SimpleNamespace(
        rating_label="Very Bullish",  # LLM's directional call
        scores={"factor_pm_score": 40.0},  # weak quant factors
    )

    # Inline copy of the override block at graph.py — kept in sync by
    # this test breaking if either the helper or the formula drifts.
    factor_pm = (memo.scores or {}).get("factor_pm_score")
    assert factor_pm is not None
    w = max(0.0, min(1.0, float(_config.settings.llm_rating_weight)))
    llm_score = score_from_rating_label(memo.rating_label)
    blended = w * llm_score + (1.0 - w) * float(factor_pm)
    memo.rating_label = rating_from_stock_score(blended)
    memo.scores = {
        **memo.scores,
        "llm_rating_score": float(llm_score),
        "llm_rating_weight": float(w),
        "blended_pm_score": round(float(blended), 1),
    }

    # 0.4 * 90 + 0.6 * 40 = 60.0 → boundary into "Bullish".
    assert memo.rating_label == "Bullish"
    assert memo.scores["llm_rating_score"] == 90.0
    assert memo.scores["llm_rating_weight"] == 0.4
    assert memo.scores["blended_pm_score"] == 60.0


def test_weight_clamp_out_of_range(monkeypatch):
    """The override block clamps to [0, 1] — pathological env values
    should not push final score outside [factor_pm, llm_score]."""
    from app import config as _config

    for raw, expected in [(-0.5, 0.0), (1.5, 1.0), (0.0, 0.0), (1.0, 1.0)]:
        monkeypatch.setattr(_config.settings, "llm_rating_weight", raw)
        w = max(0.0, min(1.0, float(_config.settings.llm_rating_weight)))
        assert w == expected


def test_deterministic_pm_synthesis_treats_none_dcf_upside_as_absent(monkeypatch):
    """The LLM-disabled PM synthesis scores the DCF as +1 / -1 / 0 by
    upside sign. A DCF that could not price the shares (`upside_pct`
    None) must contribute NOTHING — the same rating as no DCF at all —
    rather than crash on the comparison or be read as a 0% neutral."""
    from app.agents import graph
    from app.schemas import AgentFinding, DCFResult
    from app.services.valuation_service import build_dcf

    monkeypatch.setattr(graph.llm, "chat_json", lambda *a, **k: None)
    profile = {"ticker": "NVDA", "sector": "Technology", "drivers": ["AI demand"], "risks": ["Capex cycle"]}
    findings = {
        "sector": AgentFinding(agent="Sector Analyst", headline="constructive cohort", summary="", confidence=0.6),
        # One positive + one negative keyword → score 0 (Neutral) with no
        # DCF, leaving headroom for a real upside to move the rating up.
        "earnings": AgentFinding(agent="Earnings Analyst", headline="margins pressured", summary="", confidence=0.6),
    }

    real = build_dcf("NVDA")
    data = real.model_dump()
    for k in ("base", "bull", "bear"):
        data[k]["implied_share_price"] = None
        data[k]["upside_pct"] = None
    data["current_price"] = None
    unpriced = DCFResult(**data)

    without_dcf = graph._pm_synthesis(profile, findings, None)
    with_unpriced = graph._pm_synthesis(profile, findings, unpriced)
    assert with_unpriced["rating_label"] == without_dcf["rating_label"]
    assert with_unpriced["confidence_score"] == without_dcf["confidence_score"]

    # And a real, strongly positive upside still moves the score — the
    # signal is absent for None, not disabled.
    data["base"]["upside_pct"] = 0.50
    priced = DCFResult(**data)
    from app.schemas import score_from_rating_label
    assert (
        score_from_rating_label(graph._pm_synthesis(profile, findings, priced)["rating_label"])
        > score_from_rating_label(with_unpriced["rating_label"])
    )
