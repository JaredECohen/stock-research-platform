"""Scorecard methodology spec (fs-v1): structure, masks, sector aliases and
the spec fingerprint.

The pinned hash at the bottom is deliberate: ``fs-v1`` rows persisted in
production were computed with these exact definitions, so any edit to a
formula, weight, sign, mask or normalization parameter must ship as a new
``VERSION_KEY`` rather than silently redefining v1. If this test fails
because you changed the spec on purpose, bump the version and re-pin.
"""
from __future__ import annotations

import json
import random

import pytest

from app.finance import scorecard_spec as S

# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_version_key_and_family_count():
    assert S.VERSION_KEY == "fs-v1"
    assert len(S.FAMILIES) == 8
    assert S.FAMILY_NAMES == (
        "valuation", "quality", "growth", "profitability",
        "efficiency", "leverage", "capital_allocation", "earnings_quality",
    )


def test_family_weights_are_equal_and_sum_to_one():
    assert set(S.FAMILY_WEIGHTS) == set(S.FAMILY_NAMES)
    assert all(w == pytest.approx(1 / 8) for w in S.FAMILY_WEIGHTS.values())
    assert sum(S.FAMILY_WEIGHTS.values()) == pytest.approx(1.0)


def test_thirty_one_features_unique_and_every_family_populated():
    names = [f.name for f in S.FEATURE_SPEC]
    assert len(names) == 31
    assert len(set(names)) == 31
    by_family = S.features_by_family()
    assert {fam: len(v) for fam, v in by_family.items()} == {
        "valuation": 4, "quality": 4, "growth": 5, "profitability": 4,
        "efficiency": 3, "leverage": 4, "capital_allocation": 4, "earnings_quality": 3,
    }


def test_feature_signs_and_weights():
    for f in S.FEATURE_SPEC:
        assert f.sign in (1, -1), f.name
        assert f.weight > 0, f.name
        assert f.inputs, f.name
        assert f.formula and f.description, f.name
    lower_is_better = {f.name for f in S.FEATURE_SPEC if f.sign == -1}
    assert lower_is_better == {
        "opex_ratio", "capex_intensity", "net_debt_to_ebitda", "debt_to_equity",
        "net_share_change_1y", "sbc_to_revenue", "goodwill_to_assets", "accruals_ratio",
    }


def test_specs_are_frozen():
    f = S.get_feature("roic")
    with pytest.raises(AttributeError):
        f.weight = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Applicability masks
# ---------------------------------------------------------------------------


def test_financials_get_no_ev_or_leverage_features():
    fin = S.applicable_features(S.SECTOR_FINANCIALS)
    excluded = set(S.FEATURE_NAMES) - fin
    assert excluded == {
        "ebitda_ev_yield", "sales_ev_yield", "gross_margin", "asset_turnover",
        "net_debt_to_ebitda", "debt_to_equity", "interest_coverage", "current_ratio",
    }
    # Financials still get the market-cap yields: price against earnings/FCF
    # is meaningful for a bank even when EV is not.
    assert {"earnings_yield", "fcf_yield"} <= fin


def test_real_estate_and_utilities_masks():
    re_ = S.applicable_features(S.SECTOR_REAL_ESTATE)
    assert "ebitda_ev_yield" not in re_
    assert "sales_ev_yield" in re_
    util = S.applicable_features(S.SECTOR_UTILITIES)
    assert "current_ratio" not in util
    assert "net_debt_to_ebitda" in util


def test_unknown_or_missing_sector_gets_every_feature():
    assert S.applicable_features(None) == frozenset(S.FEATURE_NAMES)
    assert S.applicable_features(S.SECTOR_INFORMATION_TECHNOLOGY) == frozenset(S.FEATURE_NAMES)


def test_masks_only_name_canonical_sectors():
    for f in S.FEATURE_SPEC:
        for s in f.exclude_sectors:
            assert s in S.CANONICAL_SECTORS, (f.name, s)


# ---------------------------------------------------------------------------
# Sector alias table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("Financial Services", S.SECTOR_FINANCIALS),        # FMP profile vocabulary
    ("Financials", S.SECTOR_FINANCIALS),                # sector_configs.json / GICS
    ("Technology", S.SECTOR_INFORMATION_TECHNOLOGY),
    ("Information Technology", S.SECTOR_INFORMATION_TECHNOLOGY),
    ("tech", S.SECTOR_INFORMATION_TECHNOLOGY),
    ("Healthcare", S.SECTOR_HEALTH_CARE),
    ("Health Care", S.SECTOR_HEALTH_CARE),
    ("Consumer Cyclical", S.SECTOR_CONSUMER_DISCRETIONARY),
    ("Consumer Defensive", S.SECTOR_CONSUMER_STAPLES),
    ("Basic Materials", S.SECTOR_MATERIALS),
    ("Communication Services", S.SECTOR_COMMUNICATION_SERVICES),
    ("Real Estate", S.SECTOR_REAL_ESTATE),
    ("  real-estate ", S.SECTOR_REAL_ESTATE),           # punctuation / spacing insensitive
    ("UTILITIES", S.SECTOR_UTILITIES),                  # case insensitive
    ("Energy", S.SECTOR_ENERGY),
    ("Industrials", S.SECTOR_INDUSTRIALS),
])
def test_normalize_sector_aliases(raw, expected):
    assert S.normalize_sector(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "No Such Sector", "X", "Consumer"])
def test_normalize_sector_unknown_is_none_not_a_guess(raw):
    assert S.normalize_sector(raw) is None


def test_every_canonical_sector_round_trips():
    for s in S.CANONICAL_SECTORS:
        assert S.normalize_sector(s) == s


# ---------------------------------------------------------------------------
# Serialisation and hash
# ---------------------------------------------------------------------------


def test_spec_as_dict_is_json_serialisable_and_ordered():
    d = S.spec_as_dict()
    json.dumps(d)  # must not raise
    assert d["version_key"] == "fs-v1"
    assert [fam["name"] for fam in d["families"]] == list(S.FAMILY_NAMES)
    assert sum(len(fam["features"]) for fam in d["families"]) == 31
    assert d["normalization"] == {
        "winsor_pct": 0.025, "clip_z": 3.0, "sector_neutral": True,
        "min_sector_n": 5, "min_feature_coverage": 0.5, "min_categories": 5,
    }
    feat = d["families"][0]["features"][2]
    assert feat["name"] == "ebitda_ev_yield"
    assert feat["applicability"] == {"exclude_sectors": ["Financials", "Real Estate"]}


def _shuffle_keys(obj, rng: random.Random):
    """Rebuild dicts with a random key order (lists keep their order —
    order inside a list is meaningful)."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        rng.shuffle(keys)
        return {k: _shuffle_keys(obj[k], rng) for k in keys}
    if isinstance(obj, list):
        return [_shuffle_keys(x, rng) for x in obj]
    return obj


def test_spec_hash_is_stable_across_dict_order_permutations():
    base = S.spec_as_dict()
    rng = random.Random(0)
    for _ in range(10):
        permuted = _shuffle_keys(base, rng)
        assert S.hash_spec_dict(permuted) == S.spec_hash()


def test_spec_hash_changes_when_a_weight_changes():
    d = S.spec_as_dict()
    d["families"][1]["features"][0]["weight"] = 2.0
    assert S.hash_spec_dict(d) != S.spec_hash()
    d2 = S.spec_as_dict()
    d2["families"][1]["weight"] = 0.2
    assert S.hash_spec_dict(d2) != S.spec_hash()
    d3 = S.spec_as_dict()
    d3["normalization"]["min_sector_n"] = 8
    assert S.hash_spec_dict(d3) != S.spec_hash()


def test_spec_hash_is_sha256_of_canonical_json():
    import hashlib
    expected = hashlib.sha256(S.canonical_json(S.spec_as_dict()).encode("utf-8")).hexdigest()
    assert S.spec_hash() == expected
    assert len(expected) == 64


def test_fs_v1_hash_is_pinned():
    # Changing fs-v1 in place invalidates every persisted fs-v1 row. Bump
    # VERSION_KEY instead and re-pin here.
    assert S.spec_hash() == "f88cd6d0d5a12b8c413e8855a063d1f55e3e59c3679795daa79e8d6482dc6056"


def test_no_numpy_at_import():
    # These modules are imported by the web process; numpy must stay off
    # that path (numpy laziness is a repo constraint).
    from pathlib import Path

    for mod in ("scorecard_spec", "scorecard_normalize", "scorecard_features"):
        src = (Path(S.__file__).parent / f"{mod}.py").read_text()
        assert "import numpy" not in src and "from numpy" not in src, mod
