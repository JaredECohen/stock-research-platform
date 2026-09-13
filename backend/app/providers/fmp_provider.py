"""Financial Modeling Prep provider — `/stable/` namespace (Wave 9b).

FMP retired the `/api/v3/` and `/api/v4/` URL families on 2025-08-31; the
old endpoints now reply with HTTP 403 and `"Legacy Endpoint"` for every
key, regardless of plan tier. This module hits the current `/stable/`
endpoints exclusively. Field names follow the new shape (see the
docstring on each method).

Every method catches network/HTTP errors and returns None so the data
service can fall through to the next provider in the chain.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from ..config import settings
from .base import ProviderStatus, log_safely
from .price_history import history_start, normalize_history

log = logging.getLogger(__name__)
BASE_URL = "https://financialmodelingprep.com/stable"
TIMEOUT = 10.0


def _to_float(v: Any) -> float | None:
    if v in (None, "None", "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class FMPProvider:
    name: str = "fmp"
    price_history_provenance = {
        "provider": "fmp", "endpoint": "/stable/historical-price-eod/full",
        "close_basis": "provider_eod_close", "adjusted_close_basis": "same_as_close_not_total_return",
        "adjustment_note": "Existing full endpoint close mapping retained; no dividend-adjusted endpoint requested.",
    }

    def __init__(self) -> None:
        self.api_key = settings.fmp_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set FMP_API_KEY to enable.",
            capabilities=[
                "profile", "prices", "quote", "financials", "ratios",
                "key_metrics", "earnings", "estimates", "news",
            ],
        )

    def _get(self, path: str, **params: Any) -> Any | None:
        if not self.api_key:
            return None
        try:
            params["apikey"] = self.api_key
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(f"{BASE_URL}{path}", params=params)
                if r.status_code != 200:
                    log.warning("FMP %s -> %s", path, r.status_code)
                    return None
                return r.json()
        except Exception as exc:  # pragma: no cover — network paths
            # httpx errors quote the URL, which carries `?apikey=`.
            log_safely(log, f"FMP request failed for {path}", exc)
            return None

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_sp500_constituents(self) -> list[str] | None:
        """`/stable/sp500-constituent` — current S&P 500 ticker list.

        Used by the universe seeder + `scripts/refresh_universe_lists.py`
        to keep `data/sp500.json` in sync. Returns None on auth failure
        so callers can fall back to whatever's already on disk.
        """
        data = self._get("/sp500-constituent")
        if not isinstance(data, list):
            return None
        tickers: list[str] = []
        for row in data:
            if isinstance(row, dict):
                sym = row.get("symbol") or row.get("ticker")
            else:
                sym = row
            if isinstance(sym, str) and sym:
                tickers.append(sym.upper())
        return tickers or None

    def get_company_profile(self, ticker: str) -> dict[str, Any] | None:
        """`/stable/profile?symbol=…`. Includes sector / industry / CIK /
        market cap / beta / price / description. Shares outstanding lives
        on a separate endpoint; pulled inline so consumers get one payload."""
        data = self._get("/profile", symbol=ticker.upper())
        if not data:
            return None
        item = data[0] if isinstance(data, list) and data else None
        if not item:
            return None
        # Shares outstanding moved out of /profile in /stable/. Best-effort
        # follow-up call; failure leaves the field None.
        shares: float | None = None
        sf = self._get("/shares-float", symbol=ticker.upper())
        if isinstance(sf, list) and sf:
            shares = _to_float(sf[0].get("outstandingShares"))
        return dict(
            ticker=item.get("symbol"),
            company_name=item.get("companyName"),
            exchange=item.get("exchange") or "",
            sector=item.get("sector") or "",
            industry=item.get("industry") or "",
            sub_industry=item.get("industry"),
            country=item.get("country") or "US",
            currency=item.get("currency") or "USD",
            market_cap=_to_float(item.get("marketCap")),
            cik=item.get("cik"),
            business_description=item.get("description") or "",
            fiscal_year_end=None,
            is_active=item.get("isActivelyTrading", True),
            is_etf=item.get("isEtf", False),
            beta=_to_float(item.get("beta")),
            shares_outstanding=shares,
            last_price=_to_float(item.get("price")),
        )

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_quote(self, ticker: str) -> dict[str, Any] | None:
        """`/stable/quote?symbol=…` — near-real-time intraday price.

        FMP returns ~15-min-delayed prices on Starter, real-time on
        Premium. Either way, far fresher than reading `last_price` off
        the 7-day-cached `/profile` blob, which is what the DCF
        comparison drifted on for fast movers like NVDA.
        """
        data = self._get("/quote", symbol=ticker.upper())
        if not isinstance(data, list) or not data:
            return None
        item = data[0]
        return dict(
            ticker=item.get("symbol"),
            price=_to_float(item.get("price")),
            previous_close=_to_float(item.get("previousClose")),
            change=_to_float(item.get("change")),
            change_pct=_to_float(item.get("changesPercentage")),
            day_low=_to_float(item.get("dayLow")),
            day_high=_to_float(item.get("dayHigh")),
            volume=_to_float(item.get("volume")),
            timestamp=item.get("timestamp"),
        )

    def get_price_history(self, ticker: str, days: int = 252) -> list[dict[str, Any]] | None:
        """Explicit inclusive dates; retain all returned daily bars.

        A `limit` alone is not a historical date request on this endpoint.
        Long histories are split into bounded five-calendar-year requests.
        Any failed window rejects the result rather than claiming completion.
        """
        end = date.today()
        start = history_start(end, days)
        if start is None or not self.api_key:
            return None
        rows = []
        window_start = start
        while window_start <= end:
            window_end = min(window_start + timedelta(days=5 * 365), end)
            data = self._get(
                "/historical-price-eod/full", symbol=ticker.upper(),
                **{"from": window_start.isoformat(), "to": window_end.isoformat()},
            )
            if not isinstance(data, list):
                log.warning("FMP price history window unavailable ticker=%s from=%s to=%s", ticker, window_start, window_end)
                return None
            rows.extend(
                dict(
                    date=r.get("date"), open=r.get("open"), high=r.get("high"),
                    low=r.get("low"), close=r.get("close"),
                    adjusted_close=r.get("close"), volume=r.get("volume"),
                ) if isinstance(r, dict) else r
                for r in data
            )
            window_start = window_end + timedelta(days=1)
        return normalize_history(rows, provider=self.name, ticker=ticker, start=start, end=end, log=log)

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------

    @staticmethod
    def _period_label(date_str: str, period: str | None) -> str:
        """Map (`fiscalDateEnding`, `period`) to a `2024Q4` / `FY2024` label."""
        if not date_str:
            return ""
        period_upper = (period or "").upper()
        if period_upper == "FY":
            return f"FY{date_str[:4]}"
        if period_upper.startswith("Q") and len(period_upper) == 2:
            return f"{date_str[:4]}{period_upper}"
        # Fall back to deriving quarter from the month.
        try:
            month = int(date_str[5:7])
            q = (month - 1) // 3 + 1
            return f"{date_str[:4]}Q{q}"
        except (ValueError, IndexError):
            return date_str[:4]

    @staticmethod
    def _filing_dates(r: dict[str, Any]) -> dict[str, Any]:
        """Phase 6 (scorecard): pass the statement's own filing dates through
        when FMP supplies them (`fillingDate` — FMP's spelling — and
        `acceptedDate`). Additive keys only, and only when present, so
        the normalised row contract is unchanged for rows without them.
        `history_service` records `filing_date` (else `accepted_date`) as
        the row's point-in-time `available_at`; without either the lag
        rule applies."""
        out: dict[str, Any] = {}
        filing = r.get("fillingDate") or r.get("filingDate")
        if filing:
            out["filing_date"] = str(filing)
        accepted = r.get("acceptedDate")
        if accepted:
            out["accepted_date"] = str(accepted)
        return out

    @classmethod
    def _income_row(cls, r: dict[str, Any]) -> dict[str, Any]:
        date_str = r.get("date") or ""
        return dict(
            period=cls._period_label(date_str, r.get("period")),
            period_end=date_str,
            currency=r.get("reportedCurrency") or "USD",
            revenue=_to_float(r.get("revenue")),
            cost_of_revenue=_to_float(r.get("costOfRevenue")),
            gross_profit=_to_float(r.get("grossProfit")),
            r_and_d=_to_float(r.get("researchAndDevelopmentExpenses")),
            sga=_to_float(r.get("sellingGeneralAndAdministrativeExpenses")),
            operating_income=_to_float(r.get("operatingIncome")),
            ebit=_to_float(r.get("ebit")),
            ebitda=_to_float(r.get("ebitda")),
            net_income=_to_float(r.get("netIncome")),
            eps_diluted=_to_float(r.get("epsDiluted")),
            weighted_avg_shares_diluted=_to_float(r.get("weightedAverageShsOutDil")),
            interest_expense=_to_float(r.get("interestExpense")),
            pretax_income=_to_float(r.get("incomeBeforeTax")),
            tax_expense=_to_float(r.get("incomeTaxExpense")),
            **cls._filing_dates(r),
        )

    @classmethod
    def _balance_row(cls, r: dict[str, Any]) -> dict[str, Any]:
        date_str = r.get("date") or ""
        st_debt = _to_float(r.get("shortTermDebt")) or 0
        lt_debt = _to_float(r.get("longTermDebt")) or 0
        total_debt = (st_debt + lt_debt) or None
        return dict(
            period=cls._period_label(date_str, r.get("period")),
            period_end=date_str,
            currency=r.get("reportedCurrency") or "USD",
            total_assets=_to_float(r.get("totalAssets")),
            total_liabilities=_to_float(r.get("totalLiabilities")),
            shareholders_equity=_to_float(r.get("totalStockholdersEquity")),
            cash_and_equivalents=_to_float(r.get("cashAndCashEquivalents")),
            short_term_investments=_to_float(r.get("shortTermInvestments")),
            short_term_debt=st_debt or None,
            long_term_debt=lt_debt or None,
            total_debt=total_debt,
            goodwill=_to_float(r.get("goodwill")),
            current_assets=_to_float(r.get("totalCurrentAssets")),
            current_liabilities=_to_float(r.get("totalCurrentLiabilities")),
            **cls._filing_dates(r),
        )

    @classmethod
    def _cash_row(cls, r: dict[str, Any]) -> dict[str, Any]:
        date_str = r.get("date") or ""
        return dict(
            period=cls._period_label(date_str, r.get("period")),
            period_end=date_str,
            currency=r.get("reportedCurrency") or "USD",
            cash_from_operations=_to_float(r.get("operatingCashFlow")),
            capex=_to_float(r.get("capitalExpenditure")),
            free_cash_flow=_to_float(r.get("freeCashFlow")),
            depreciation_and_amortization=_to_float(r.get("depreciationAndAmortization")),
            dividends_paid=_to_float(r.get("commonDividendsPaid")) or _to_float(r.get("netDividendsPaid")),
            share_repurchases=_to_float(r.get("commonStockRepurchased")),
            stock_based_compensation=_to_float(r.get("stockBasedCompensation")),
            **cls._filing_dates(r),
        )

    def get_financial_statements(self, ticker: str) -> dict[str, Any] | None:
        """`/stable/income-statement` + `/balance-sheet-statement` +
        `/cash-flow-statement`, all keyed by `?symbol=`. Period defaults
        to `annual` on /stable/; pass `period=quarter` for Q-by-Q."""
        # FMP answers some failures with HTTP 200 and a JSON *object*
        # ({"Error Message": ...}); iterating that as rows raised out of the
        # provider chain. Anything that is not a list of rows is "no data".
        def _rows(path: str) -> list[dict[str, Any]]:
            payload = self._get(path, symbol=ticker.upper(), limit=8)
            if not isinstance(payload, list):
                return []
            return [r for r in payload if isinstance(r, dict)]

        income = _rows("/income-statement")
        balance = _rows("/balance-sheet-statement")
        cash = _rows("/cash-flow-statement")
        if not income:
            return None
        return dict(
            income=[self._income_row(r) for r in income],
            balance=[self._balance_row(r) for r in balance],
            cash=[self._cash_row(r) for r in cash],
        )

    def get_financial_history(self, ticker: str, start_date) -> dict[str, Any]:
        """Fetch annual AND quarterly statements to the requested history boundary.

        This dedicated backfill path does not change the normal eight-annual-row
        read. Requests include one earlier period for boundary coverage; provider
        truncation/entitlements remain visible in the caller's coverage report.
        """
        from datetime import date

        start = date.fromisoformat(str(start_date)[:10])
        years = max(1, date.today().year - start.year + 1)
        result: dict[str, Any] = {"income": [], "balance": [], "cash": [], "_history_issues": []}
        for statement, path, mapper in (
            ("income", "/income-statement", self._income_row),
            ("balance", "/balance-sheet-statement", self._balance_row),
            ("cash", "/cash-flow-statement", self._cash_row),
        ):
            for cadence, api_period, limit in (("annual", "annual", years + 2), ("quarterly", "quarter", years * 4 + 4)):
                raw = self._get(path, symbol=ticker.upper(), period=api_period, limit=limit)
                if not isinstance(raw, list) or not raw:
                    result["_history_issues"].append({"kind": "provider_no_data", "statement": statement, "cadence": cadence})
                    continue
                for row in raw:
                    if not isinstance(row, dict):
                        result["_history_issues"].append({"kind": "invalid_provider_row", "statement": statement, "cadence": cadence})
                        continue
                    normalized = mapper(row)
                    # Stable API fiscalYear is authoritative for non-calendar FYs.
                    year = str(row.get("fiscalYear") or str(row.get("date") or "")[:4])
                    quarter = str(row.get("period") or "").upper()
                    if cadence == "quarterly" and quarter not in {"Q1", "Q2", "Q3", "Q4"}:
                        result["_history_issues"].append({"kind": "invalid_fiscal_quarter", "statement": statement, "cadence": cadence, "period_end": row.get("date")})
                        continue
                    normalized["period"] = f"FY{year}" if cadence == "annual" else f"{year}{quarter}"
                    normalized["currency"] = row.get("reportedCurrency") or ""
                    normalized["source"] = self.name
                    normalized["cadence"] = cadence
                    result[statement].append(normalized)
        return result

    # ------------------------------------------------------------------
    # Ratios + key metrics (price-derived; recomputed daily by FMP)
    # ------------------------------------------------------------------

    def get_ratios(self, ticker: str) -> dict[str, Any] | None:
        """`/stable/ratios?symbol=…`. Field names changed in /stable/ —
        `priceEarningsRatio` → `priceToEarningsRatio`, etc. We expose the
        same vocabulary the demo / downstream callers used (`PE`,
        `EV_EBITDA`, `PFCF`, `FCF_yield`, `ROIC`, margins)."""
        data = self._get("/ratios", symbol=ticker.upper(), limit=1)
        if not isinstance(data, list) or not data:
            return None
        r = data[0]
        # /stable/key-metrics carries EV-based multiples.
        km_data = self._get("/key-metrics", symbol=ticker.upper(), limit=1)
        km = km_data[0] if isinstance(km_data, list) and km_data else {}
        return dict(
            PE=_to_float(r.get("priceToEarningsRatio")),
            EV_Revenue=_to_float(km.get("evToSales")),
            EV_EBITDA=_to_float(km.get("evToEBITDA")),
            PFCF=_to_float(r.get("priceToFreeCashFlowRatio")),
            FCF_yield=_to_float(km.get("freeCashFlowYield")),
            ROIC=_to_float(km.get("returnOnInvestedCapital")),
            ROE=_to_float(km.get("returnOnEquity")),
            gross_margin=_to_float(r.get("grossProfitMargin")),
            operating_margin=_to_float(r.get("operatingProfitMargin")),
            ebitda_margin=_to_float(r.get("ebitdaMargin")),
            fcf_margin=None,  # derive elsewhere
            net_margin=_to_float(r.get("netProfitMargin")),
            debt_to_ebitda=_to_float(km.get("netDebtToEBITDA")),
            dividend_yield=_to_float(r.get("dividendYield")),
        )

    def get_key_metrics(self, ticker: str) -> dict[str, Any] | None:
        data = self._get("/key-metrics", symbol=ticker.upper(), limit=1)
        if not isinstance(data, list) or not data:
            return None
        return data[0]

    # ------------------------------------------------------------------
    # Earnings + estimates
    # ------------------------------------------------------------------

    def get_earnings(self, ticker: str) -> dict[str, Any] | None:
        """`/stable/earnings?symbol=…` returns past + future quarters in one
        list. Past rows have `epsActual` set; future rows leave it null."""
        data = self._get("/earnings", symbol=ticker.upper(), limit=12)
        if not isinstance(data, list) or not data:
            return None
        quarters: list[dict[str, Any]] = []
        for q in data:
            actual = _to_float(q.get("epsActual"))
            estimate = _to_float(q.get("epsEstimated"))
            if actual is None and estimate is None:
                continue
            surprise_pct = None
            if actual is not None and estimate not in (None, 0):
                surprise_pct = (actual - estimate) / abs(estimate)
            quarters.append(dict(
                period=q.get("date") or "",
                report_date=q.get("date") or "",
                eps_actual=actual,
                eps_estimate=estimate,
                surprise_pct=surprise_pct,
                revenue_actual=_to_float(q.get("revenueActual")),
                revenue_estimate=_to_float(q.get("revenueEstimated")),
            ))
        if not quarters:
            return None
        return dict(quarters=quarters)

    def get_estimates(self, ticker: str) -> dict[str, Any] | None:
        """`/stable/analyst-estimates?symbol=…&period=annual` — sell-side
        consensus for the next few fiscal years (revenue + EPS Avg/Low/High
        + analyst counts). Plus `/price-target-consensus` for target-price
        averages.

        Output includes both the rich `annual` rows and the legacy
        `revenue` / `revenue_growth` keys consumed by
        `finance.dcf._consensus_growth_path`, so DCF defaults pick up
        consensus growth without per-call bridging.
        """
        data = self._get(
            "/analyst-estimates",
            symbol=ticker.upper(), period="annual", limit=6,
        )
        if not isinstance(data, list):
            data = []
        # FMP returns most-recent-first; flip to chronological so YoY
        # deltas land in order downstream.
        ascending = list(reversed(data))
        years: list[dict[str, Any]] = []
        revenue_rows: list[dict[str, Any]] = []
        revenue_growth: list[float] = []
        prev_rev: float | None = None
        for r in ascending:
            period = r.get("date") or ""
            rev_avg = _to_float(r.get("revenueAvg"))
            years.append(dict(
                period=period,
                revenue_avg=rev_avg,
                revenue_low=_to_float(r.get("revenueLow")),
                revenue_high=_to_float(r.get("revenueHigh")),
                eps_avg=_to_float(r.get("epsAvg")),
                eps_low=_to_float(r.get("epsLow")),
                eps_high=_to_float(r.get("epsHigh")),
                num_analysts_revenue=r.get("numAnalystsRevenue"),
                num_analysts_eps=r.get("numAnalystsEps"),
            ))
            if rev_avg is not None:
                revenue_rows.append(dict(period=period, value=rev_avg))
                if prev_rev is not None and prev_rev > 0:
                    revenue_growth.append((rev_avg - prev_rev) / prev_rev)
                prev_rev = rev_avg
        # Consensus target price.
        ptc_data = self._get("/price-target-consensus", symbol=ticker.upper())
        target = None
        if isinstance(ptc_data, list) and ptc_data:
            target = dict(
                target_high=_to_float(ptc_data[0].get("targetHigh")),
                target_low=_to_float(ptc_data[0].get("targetLow")),
                target_consensus=_to_float(ptc_data[0].get("targetConsensus")),
                target_median=_to_float(ptc_data[0].get("targetMedian")),
            )
        if not years and not target:
            return None
        return dict(
            annual=years,
            revenue=revenue_rows,             # legacy shape: dcf._consensus_growth_path reads this
            revenue_growth=revenue_growth,    # already-derived YoY deltas
            price_target=target,
        )

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_news(self, ticker: str) -> list[dict[str, Any]] | None:
        """`/stable/news/stock?symbols=…`."""
        data = self._get("/news/stock", symbols=ticker.upper(), limit=20)
        if not isinstance(data, list) or not data:
            return None
        return [
            dict(
                title=n.get("title"),
                source=n.get("publisher"),
                published_at=n.get("publishedDate"),
                url=n.get("url") or n.get("link"),
                summary=n.get("text"),
                tickers=[ticker.upper()],
                topics=[],
                sentiment=None,
                relevance_score=0.7,
            )
            for n in data
        ]

    # ------------------------------------------------------------------
    # BaseProvider stubs we don't implement on FMP
    # ------------------------------------------------------------------

    def get_earnings_transcripts(self, ticker: str) -> list[dict[str, Any]] | None:
        return None

    def get_filings(self, ticker: str, *, cik: str | None = None) -> list[dict[str, Any]] | None:
        return None

    def get_macro_series(self, series_id: str) -> dict[str, Any] | None:
        return None

    def list_tickers(self) -> list[str]:
        return []
