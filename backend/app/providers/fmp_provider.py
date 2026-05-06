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
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
BASE_URL = "https://financialmodelingprep.com/stable"
TIMEOUT = 10.0


def _to_float(v: Any) -> Optional[float]:
    if v in (None, "None", "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class FMPProvider:
    name: str = "fmp"

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

    def _get(self, path: str, **params: Any) -> Optional[Any]:
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
            log.warning("FMP request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
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
        shares: Optional[float] = None
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

    def get_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
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

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        """`/stable/historical-price-eod/full?symbol=…`. Returns OHLCV bars
        most-recent first; reversed to oldest-first to match the demo /
        downstream expectation."""
        data = self._get(
            "/historical-price-eod/full",
            symbol=ticker.upper(), limit=days,
        )
        if not isinstance(data, list) or not data:
            return None
        rows = list(reversed(data))
        return [
            dict(
                date=r.get("date"),
                open=_to_float(r.get("open")),
                high=_to_float(r.get("high")),
                low=_to_float(r.get("low")),
                close=_to_float(r.get("close")),
                adjusted_close=_to_float(r.get("close")),  # /stable/ doesn't split adj
                volume=_to_float(r.get("volume")),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------

    @staticmethod
    def _period_label(date_str: str, period: Optional[str]) -> str:
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

    @classmethod
    def _income_row(cls, r: Dict[str, Any]) -> Dict[str, Any]:
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
        )

    @classmethod
    def _balance_row(cls, r: Dict[str, Any]) -> Dict[str, Any]:
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
        )

    @classmethod
    def _cash_row(cls, r: Dict[str, Any]) -> Dict[str, Any]:
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
        )

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]:
        """`/stable/income-statement` + `/balance-sheet-statement` +
        `/cash-flow-statement`, all keyed by `?symbol=`. Period defaults
        to `annual` on /stable/; pass `period=quarter` for Q-by-Q."""
        income = self._get("/income-statement", symbol=ticker.upper(), limit=8) or []
        balance = self._get("/balance-sheet-statement", symbol=ticker.upper(), limit=8) or []
        cash = self._get("/cash-flow-statement", symbol=ticker.upper(), limit=8) or []
        if not income:
            return None
        return dict(
            income=[self._income_row(r) for r in income],
            balance=[self._balance_row(r) for r in balance],
            cash=[self._cash_row(r) for r in cash],
        )

    # ------------------------------------------------------------------
    # Ratios + key metrics (price-derived; recomputed daily by FMP)
    # ------------------------------------------------------------------

    def get_ratios(self, ticker: str) -> Optional[Dict[str, Any]]:
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

    def get_key_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = self._get("/key-metrics", symbol=ticker.upper(), limit=1)
        if not isinstance(data, list) or not data:
            return None
        return data[0]

    # ------------------------------------------------------------------
    # Earnings + estimates
    # ------------------------------------------------------------------

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]:
        """`/stable/earnings?symbol=…` returns past + future quarters in one
        list. Past rows have `epsActual` set; future rows leave it null."""
        data = self._get("/earnings", symbol=ticker.upper(), limit=12)
        if not isinstance(data, list) or not data:
            return None
        quarters: List[Dict[str, Any]] = []
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

    def get_estimates(self, ticker: str) -> Optional[Dict[str, Any]]:
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
        years: List[Dict[str, Any]] = []
        revenue_rows: List[Dict[str, Any]] = []
        revenue_growth: List[float] = []
        prev_rev: Optional[float] = None
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

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
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

    def get_earnings_transcripts(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_filings(self, ticker: str, *, cik: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        return None

    def list_tickers(self) -> List[str]:
        return []
