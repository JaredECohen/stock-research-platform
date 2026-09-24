"""A dual-class ticker the provider spells differently is not a missing company.

`/api/admin/universe-review` reported `diff_vs_db: {"missing_in_db": ["BRK.B"]}`.
Berkshire is in `app/data/sp500.json` twice over — in `tickers`, and in the
`_top_10_by_market_cap_2026_05` list that drives `Company.auto_update_memo` —
and it had no `companies` row at all. So one of the ten automatic-memo pins
named a ticker that did not exist and could never fire anything: a tenth of
that budget pointed at nothing, silently, and the only surface that said so
was a diff nobody reads.

The cause is spelling, not data. There is no agreed form for a dual-class
symbol, and this repo already carries three: the universe file says `BRK.B`,
the GICS map keys it `BRK-B`, and `sec_edgar_provider` rewrites the dot to a
hyphen on every lookup with the comment "SEC uses BRK-B format". Ask the
provider chain for `BRK.B`, get nothing back because that feed spells it
`BRK-B`, and `seed_universe` counts a `missing_profile` and moves on — the
same code path as a genuinely delisted symbol.

The fix is not a rename and not an alias table for one ticker. `BRK.B` stays
canonical for the database, the cache key and the universe file; the provider
chain is simply asked for each plausible spelling before a miss is believed.
The next dual-class name — `BF.B`, `LEN.B`, whatever the index adds — then
works with no further change, which is the property these tests pin.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings as app_settings
from app.database import SessionLocal
from app.models import Company
from app.services.data_service import get_data_service
from app.services.ticker_symbols import is_multi_class, symbol_variants

SP500 = Path(__file__).resolve().parent.parent / "data" / "sp500.json"
BERKSHIRE = "BRK.B"
PROVIDER_SPELLING = "BRK-B"


# ---------------------------------------------------------------------------
# The spelling rule
# ---------------------------------------------------------------------------

def test_an_ordinary_ticker_has_exactly_one_spelling():
    """The general case must cost nothing — one pass over the chain."""
    for ticker in ("AAPL", "NVDA", "MSFT", "GOOGL"):
        assert symbol_variants(ticker) == [ticker]
        assert is_multi_class(ticker) is False


def test_a_share_class_ticker_offers_the_separators_providers_actually_use():
    variants = symbol_variants(BERKSHIRE)

    assert variants[0] == BERKSHIRE, "our own spelling stays canonical"
    assert PROVIDER_SPELLING in variants
    assert "BRK/B" in variants
    assert is_multi_class(BERKSHIRE) is True


def test_the_separator_free_form_is_never_guessed():
    """`BRKB` is a different security, not a spelling of `BRK-B`.

    Guessing it would seed the wrong company under Berkshire's name, which
    is worse than the missing row this fix is for.
    """
    assert "BRKB" not in symbol_variants(BERKSHIRE)
    assert "BFB" not in symbol_variants("BF.B")


def test_the_rule_is_general_rather_than_a_berkshire_alias():
    for ticker, expected in (("BF.B", "BF-B"), ("LEN.B", "LEN-B"), ("HEI.A", "HEI-A")):
        assert expected in symbol_variants(ticker)


def test_the_spelling_rule_has_one_definition():
    """The GICS map's symbol lookup and the provider chain must agree.

    They disagreed by construction before — one had a variant helper, the
    other had none — and the disagreement is the bug.
    """
    from app.services import industry_knowledge as ik
    assert ik._symbol_variants(BERKSHIRE)[:2] == symbol_variants(BERKSHIRE)[:2]


# ---------------------------------------------------------------------------
# The universe file
# ---------------------------------------------------------------------------

def test_berkshire_is_pinned_for_automatic_memos_in_the_universe_file():
    """Guards the premise: the pin is what made the missing row expensive."""
    cfg = json.loads(SP500.read_text())

    assert BERKSHIRE in cfg["tickers"]
    assert BERKSHIRE in cfg["_top_10_by_market_cap_2026_05"]


def test_every_universe_ticker_that_needs_a_variant_is_a_share_class():
    """A file symbol with a separator is a share class, not a typo."""
    cfg = json.loads(SP500.read_text())
    odd = [t for t in cfg["tickers"] if is_multi_class(t)]

    assert odd == [BERKSHIRE], (
        f"a new separator-bearing symbol appeared in the universe file: {odd}"
    )


# ---------------------------------------------------------------------------
# End to end through the provider chain and the seeder
# ---------------------------------------------------------------------------

class OneSpellingProvider:
    """A feed that knows Berkshire only as `BRK-B`, like FMP and EDGAR.

    Records every symbol it was asked for, so a test can show the chain
    tried our spelling first and only then the provider's.
    """

    name = "one-spelling"

    def __init__(self, known: str = PROVIDER_SPELLING) -> None:
        self.known = known
        self.asked: list[str] = []

    def get_company_profile(self, ticker: str):
        self.asked.append(ticker)
        if ticker != self.known:
            return None
        return {
            "ticker": self.known,
            "company_name": "Berkshire Hathaway Inc.",
            "exchange": "NYSE",
            "sector": "Financial Services",
            "industry": "Insurance—Diversified",
            "country": "US",
            "currency": "USD",
            "market_cap": 1_050_000_000_000.0,
            "cik": "0001067983",
            "business_description": "Conglomerate holding company.",
            "beta": 0.86,
            "shares_outstanding": 1_300_000_000.0,
            "last_price": 486.12,
        }


@pytest.fixture()
def one_spelling():
    ds = get_data_service()
    previous = ds._test_provider
    provider = OneSpellingProvider()
    ds.register_test_provider(provider)
    try:
        yield provider
    finally:
        ds.register_test_provider(previous)


@pytest.fixture()
def no_berkshire_row():
    with SessionLocal() as db:
        db.query(Company).filter(Company.ticker == BERKSHIRE).delete()
        db.commit()
    yield
    with SessionLocal() as db:
        db.query(Company).filter(Company.ticker == BERKSHIRE).delete()
        db.commit()


def test_the_profile_chain_resolves_our_spelling_through_the_providers(one_spelling):
    """The defect itself: this returned None, and the seeder believed it."""
    profile = get_data_service().get_company_profile(BERKSHIRE)

    assert profile is not None, (
        "the provider chain still gives up when the feed spells a share "
        "class differently; that miss is what left Berkshire unseeded"
    )
    assert profile["company_name"] == "Berkshire Hathaway Inc."
    assert one_spelling.asked[0] == BERKSHIRE, (
        "our own spelling must be tried first — the alternates are a "
        "fallback, not a rewrite"
    )
    assert PROVIDER_SPELLING in one_spelling.asked


def test_an_ordinary_ticker_costs_exactly_one_provider_call(one_spelling):
    get_data_service().get_company_profile("AAPL")
    assert one_spelling.asked == ["AAPL"], (
        "a ticker with no share-class separator must not pay for retries"
    )


def test_a_genuinely_unknown_symbol_is_still_a_miss(one_spelling):
    assert get_data_service().get_company_profile("ZZNOPE") is None


def test_primary_provider_tries_every_spelling_before_fallback(monkeypatch):
    """FIX-006: provider-major order. The primary (FMP spells it `BRK-B`)
    must be asked under every spelling before a fallback that happens to
    accept our `BRK.B` spelling gets to answer."""
    from app.services import data_service

    primary = OneSpellingProvider()
    primary.name = "fmp"
    fallback = OneSpellingProvider(known=BERKSHIRE)
    fallback.name = "alpha_vantage"
    ds = data_service.DataService()
    monkeypatch.setattr(ds, "_live_chain", lambda capability: [primary, fallback])
    profile = ds._try_chain_symbol("profile", "get_company_profile", BERKSHIRE)
    assert profile is not None and profile["ticker"] == PROVIDER_SPELLING, "the fallback answered first"
    assert primary.asked[:2] == [BERKSHIRE, PROVIDER_SPELLING]
    assert fallback.asked == []
    primary.asked.clear()
    ds._try_chain_symbol("profile", "get_company_profile", "AAPL")
    assert primary.asked == ["AAPL"] and fallback.asked == ["AAPL"]


def test_seed_universe_gives_berkshire_a_row_under_the_canonical_ticker(
    one_spelling, no_berkshire_row, monkeypatch,
):
    """The whole chain: universe file → provider chain → `companies` row.

    The universe passed to the seeder is the one already in the database
    plus Berkshire, so this exercises the insert without demoting anything
    the rest of the suite relies on.
    """
    import app.seed_universe as su

    with SessionLocal() as db:
        current = sorted(
            t for (t,) in db.query(Company.ticker)
            .filter(Company.universe_tier == "auto_analysis").all()
        )
        pinned = sorted(
            t for (t,) in db.query(Company.ticker)
            .filter(Company.auto_update_memo.is_(True)).all()
        )
    universe = sorted(set(current) | {BERKSHIRE})
    pins = sorted(set(pinned) | {BERKSHIRE})
    monkeypatch.setattr(su, "_load_universe", lambda: (universe, pins))
    monkeypatch.setattr(app_settings, "enable_long_term_memory", False)

    result = su.seed_universe(refresh=False)

    assert result["inserted"] >= 1
    with SessionLocal() as db:
        row = db.get(Company, BERKSHIRE)
    assert row is not None, (
        "seed_universe still skips Berkshire; the auto_update_memo pin "
        "continues to name a company that does not exist"
    )
    assert row.ticker == BERKSHIRE, (
        "the row must be keyed by our canonical spelling — the universe "
        "file, the pin list and universe-review all compare against it"
    )
    assert row.universe_tier == "auto_analysis"
    assert row.auto_update_memo is True, (
        "the pin is the reason this mattered; a row that is not pinned "
        "leaves the automatic-memo budget one short"
    )
    assert row.company_name == "Berkshire Hathaway Inc."
    assert row.cik == "0001067983"


def test_universe_review_no_longer_reports_berkshire_missing(
    one_spelling, no_berkshire_row, monkeypatch,
):
    """The surface that reported the gap must now agree."""
    import app.seed_universe as su
    from app.services import universe_review

    before = universe_review.review_universe()
    assert BERKSHIRE in before["diff_vs_db"]["missing_in_db"]

    with SessionLocal() as db:
        current = sorted(
            t for (t,) in db.query(Company.ticker)
            .filter(Company.universe_tier == "auto_analysis").all()
        )
        pinned = sorted(
            t for (t,) in db.query(Company.ticker)
            .filter(Company.auto_update_memo.is_(True)).all()
        )
    monkeypatch.setattr(
        su, "_load_universe",
        lambda: (sorted(set(current) | {BERKSHIRE}), sorted(set(pinned) | {BERKSHIRE})),
    )
    monkeypatch.setattr(app_settings, "enable_long_term_memory", False)
    su.seed_universe(refresh=False)

    after = universe_review.review_universe()
    assert BERKSHIRE not in after["diff_vs_db"]["missing_in_db"]
