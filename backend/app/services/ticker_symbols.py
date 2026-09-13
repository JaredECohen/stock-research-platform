"""How a share-class ticker is spelled, and who spells it which way.

There is no agreed spelling for a dual-class symbol. The same security is
`BRK.B` on some feeds, `BRK-B` on others, `BRK/B` or `BRK B` on a few. Our
own corpus already carries three of them: `app/data/sp500.json` lists
`BRK.B`, the GICS map in `app/data/industry_knowledge/` keys it `BRK-B`,
and `providers/sec_edgar_provider.py` rewrites the dot to a hyphen on every
lookup because EDGAR insists on it.

That disagreement cost us a company. `seed_universe` asked the provider
chain for a profile for `BRK.B`, the chain answered on a different spelling,
the seeder counted a `missing_profile` and moved on — so Berkshire had no
`companies` row at all. It was invisible twice over: `/api/admin/
universe-review` reported it as `missing_in_db`, and one of the ten
`auto_update_memo` pins in `_top_10_by_market_cap_2026_05` named a ticker
that did not exist, so a tenth of the automatic-memo budget pointed at
nothing and could never fire.

The fix is not to rename Berkshire. It is to stop assuming that the
spelling we hold is the spelling the provider uses: ask for each plausible
spelling, in a defined order, and keep our own as the canonical key for the
database, the cache and the universe file. The next dual-class name to
arrive — `BF.B`, `LEN.B`, a new listing — then works with no further
change, which a hardcoded alias for one ticker would not give.

Deliberately NOT included: the separator-free form. `BRKB` is a different
ticker symbol, not a spelling of `BRK-B`, and guessing it would silently
seed the wrong security.
"""
from __future__ import annotations

from datetime import date

# Separators a provider might use in place of ours, in the order we try
# them. Hyphen first because it is what both EDGAR and FMP use for the
# share classes in our universe today.
_SEPARATORS = (".", "-", "/", " ")


def symbol_variants(symbol: str) -> list[str]:
    """Plausible spellings of `symbol`, canonical form first, no duplicates.

    A symbol with no separator — the overwhelming majority — returns a
    single-element list, so callers pay nothing for the general case.
    """
    sym = str(symbol or "").strip().upper()
    if not sym:
        return []
    out = [sym]
    for sep in _SEPARATORS:
        if sep not in sym:
            continue
        for other in _SEPARATORS:
            if other == sep:
                continue
            candidate = sym.replace(sep, other)
            if candidate not in out:
                out.append(candidate)
        break
    return out


def is_multi_class(symbol: str) -> bool:
    """True when `symbol` carries a share-class separator."""
    return len(symbol_variants(symbol)) > 1


def market_data_symbols(symbol: str, *, on: date | None = None) -> list[str]:
    """Resolve verified same-security renames for market data only.

    Keep canonical company/memo keys intact. Acquisitions with an exchange
    ratio are not aliases. BNY confirms unchanged CUSIP/capital structure:
    https://www.bny.com/corporate/global/en/about-us/newsroom/press-release/bny-announces-planned-change-of-stock-ticker-symbol-to-bny-130465.html
    """
    variants = symbol_variants(symbol)
    if variants and variants[0] == "BK" and (on or date.today()) >= date(2026, 5, 21):
        return ["BNY", *variants]
    return variants
