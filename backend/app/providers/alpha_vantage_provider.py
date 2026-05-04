"""Alpha Vantage provider — earnings transcripts and news/sentiment."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings
from .base import ProviderStatus

log = logging.getLogger(__name__)
BASE_URL = "https://www.alphavantage.co/query"
TIMEOUT = 10.0


class AlphaVantageProvider:
    name: str = "alpha_vantage"

    def __init__(self) -> None:
        self.api_key = settings.alpha_vantage_api_key

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            configured=bool(self.api_key),
            healthy=bool(self.api_key),
            notes="" if self.api_key else "Set ALPHA_VANTAGE_API_KEY to enable.",
            capabilities=["transcripts", "news", "fundamentals_fallback"],
        )

    def _get(self, **params: Any) -> Optional[Any]:
        if not self.api_key:
            return None
        try:
            params["apikey"] = self.api_key
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(BASE_URL, params=params)
                if r.status_code != 200:
                    log.warning("AlphaVantage %s -> %s", params, r.status_code)
                    return None
                return r.json()
        except Exception as exc:  # pragma: no cover
            log.warning("AlphaVantage request failed: %s", exc)
            return None

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = self._get(function="OVERVIEW", symbol=ticker)
        if not data or "Symbol" not in data:
            return None
        return dict(
            ticker=data.get("Symbol"),
            company_name=data.get("Name"),
            exchange=data.get("Exchange") or "",
            sector=data.get("Sector") or "",
            industry=data.get("Industry") or "",
            country=data.get("Country") or "US",
            currency=data.get("Currency") or "USD",
            market_cap=float(data.get("MarketCapitalization") or 0) or None,
            business_description=data.get("Description") or "",
            beta=float(data.get("Beta") or 0) or None,
            shares_outstanding=float(data.get("SharesOutstanding") or 0) or None,
            last_price=None,
        )

    def get_price_history(self, ticker: str, days: int = 252) -> Optional[List[Dict[str, Any]]]:
        return None

    # ------------------------------------------------------------------
    # Wave 9b — fundamentals fallback
    # ------------------------------------------------------------------
    # FMP's free / Starter tier returns 403 on financial statements for
    # most tickers. Alpha Vantage covers the gap on paid plans (75 rpm
    # on Premium). Field mappings hew to the same vocabulary
    # `history_service` expects (`revenue`, `gross_profit`, …).

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        if v in (None, "None", "", "-"):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _income_row(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        date_str = r.get("fiscalDateEnding") or ""
        period_label = date_str
        if date_str:
            # Map fiscal end to a 4-quarter label so downstream period
            # parsing matches what FMP / demo emit (`2024Q4`, etc.).
            try:
                year, month = date_str[:4], int(date_str[5:7])
                quarter = (month - 1) // 3 + 1
                period_label = f"{year}Q{quarter}"
            except (ValueError, IndexError):
                pass
        revenue = cls._to_float(r.get("totalRevenue"))
        cogs = cls._to_float(r.get("costOfRevenue"))
        gross = cls._to_float(r.get("grossProfit"))
        if gross is None and revenue is not None and cogs is not None:
            gross = revenue - cogs
        return dict(
            period=period_label,
            period_end=date_str,
            currency=r.get("reportedCurrency") or "USD",
            revenue=revenue,
            cost_of_revenue=cogs,
            gross_profit=gross,
            r_and_d=cls._to_float(r.get("researchAndDevelopment")),
            sga=cls._to_float(r.get("sellingGeneralAndAdministrative")),
            operating_income=cls._to_float(r.get("operatingIncome")),
            ebit=cls._to_float(r.get("ebit")),
            ebitda=cls._to_float(r.get("ebitda")),
            net_income=cls._to_float(r.get("netIncome")),
            eps_diluted=None,
            weighted_avg_shares_diluted=None,
            interest_expense=cls._to_float(r.get("interestExpense")),
            pretax_income=cls._to_float(r.get("incomeBeforeTax")),
            tax_expense=cls._to_float(r.get("incomeTaxExpense")),
        )

    @classmethod
    def _balance_row(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            period=r.get("fiscalDateEnding") or "",
            period_end=r.get("fiscalDateEnding") or "",
            currency=r.get("reportedCurrency") or "USD",
            total_assets=cls._to_float(r.get("totalAssets")),
            total_liabilities=cls._to_float(r.get("totalLiabilities")),
            shareholders_equity=cls._to_float(r.get("totalShareholderEquity")),
            cash_and_equivalents=cls._to_float(r.get("cashAndCashEquivalentsAtCarryingValue")),
            short_term_investments=cls._to_float(r.get("shortTermInvestments")),
            short_term_debt=cls._to_float(r.get("shortTermDebt")),
            long_term_debt=cls._to_float(r.get("longTermDebt")),
            total_debt=cls._to_float(r.get("shortLongTermDebtTotal")),
            goodwill=cls._to_float(r.get("goodwill")),
            current_assets=cls._to_float(r.get("totalCurrentAssets")),
            current_liabilities=cls._to_float(r.get("totalCurrentLiabilities")),
        )

    @classmethod
    def _cash_row(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        ops = cls._to_float(r.get("operatingCashflow"))
        capex = cls._to_float(r.get("capitalExpenditures"))
        fcf = None
        if ops is not None and capex is not None:
            fcf = ops - abs(capex)
        return dict(
            period=r.get("fiscalDateEnding") or "",
            period_end=r.get("fiscalDateEnding") or "",
            currency=r.get("reportedCurrency") or "USD",
            cash_from_operations=ops,
            capex=capex,
            free_cash_flow=fcf,
            depreciation_and_amortization=cls._to_float(r.get("depreciationDepletionAndAmortization")),
            dividends_paid=cls._to_float(r.get("dividendPayoutCommonStock")),
            share_repurchases=cls._to_float(r.get("paymentsForRepurchaseOfCommonStock")),
            stock_based_compensation=cls._to_float(r.get("stockBasedCompensation")),
        )

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]:
        income_raw = self._get(function="INCOME_STATEMENT", symbol=ticker) or {}
        balance_raw = self._get(function="BALANCE_SHEET", symbol=ticker) or {}
        cash_raw = self._get(function="CASH_FLOW", symbol=ticker) or {}
        income = [self._income_row(r) for r in (income_raw.get("annualReports") or [])][:8]
        balance = [self._balance_row(r) for r in (balance_raw.get("annualReports") or [])][:8]
        cash = [self._cash_row(r) for r in (cash_raw.get("annualReports") or [])][:8]
        if not income:
            return None
        return dict(income=income, balance=balance, cash=cash)

    def get_ratios(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_key_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = self._get(function="EARNINGS", symbol=ticker)
        if not data:
            return None
        quarterly = data.get("quarterlyEarnings") or []
        if not quarterly:
            return None
        rows: List[Dict[str, Any]] = []
        for q in quarterly[:8]:
            rows.append(dict(
                period=q.get("fiscalDateEnding") or "",
                report_date=q.get("reportedDate") or "",
                eps_actual=self._to_float(q.get("reportedEPS")),
                eps_estimate=self._to_float(q.get("estimatedEPS")),
                surprise_pct=self._to_float(q.get("surprisePercentage")),
            ))
        return dict(quarters=rows)

    @staticmethod
    def _recent_quarters(n: int = 4) -> List[str]:
        """Return the last `n` fiscal quarters as `YYYYQM` strings,
        most-recent first. AV's `EARNINGS_CALL_TRANSCRIPT` requires a
        specific quarter; without one it 200s with no transcript."""
        from datetime import date
        today = date.today()
        # Most-recent *completed* quarter — current calendar quarter
        # almost never has a transcript yet.
        cur_q = (today.month - 1) // 3 + 1
        year, q = today.year, cur_q
        # Step back one full quarter to start at the latest reported one.
        q -= 1
        if q == 0:
            q, year = 4, year - 1
        out: List[str] = []
        for _ in range(n):
            out.append(f"{year}Q{q}")
            q -= 1
            if q == 0:
                q, year = 4, year - 1
        return out

    def get_earnings_transcripts(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """Pull up to 4 recent quarterly transcripts. AV requires the
        `quarter=YYYYQN` param; without it the endpoint returns 200 with
        an empty body, which is why our backfill was getting nothing.
        Iterate the last 4 fiscal quarters and stitch together what
        comes back."""
        out: List[Dict[str, Any]] = []
        for quarter in self._recent_quarters(4):
            data = self._get(
                function="EARNINGS_CALL_TRANSCRIPT",
                symbol=ticker, quarter=quarter,
            )
            if not data:
                continue
            transcript = data.get("transcript")
            # AV returns either a list of {speaker, content, ...} dicts
            # or an empty/missing field for quarters without a transcript.
            if not isinstance(transcript, list) or not transcript:
                continue
            prepared = " ".join(
                str(item.get("content", "")) for item in transcript
                if item.get("type") == "presentation" or item.get("section") == "Prepared Remarks"
            ).strip()
            qa = " ".join(
                str(item.get("content", "")) for item in transcript
                if item.get("type") == "qa" or item.get("section") in ("Q&A", "Q & A", "Question and Answer")
            ).strip()
            # Some AV plans return one big list with no section tag — fall
            # back to using the whole thing as prepared remarks so the
            # filing/earnings analysts still get content.
            if not prepared and not qa:
                prepared = " ".join(
                    str(item.get("content", "")) for item in transcript
                ).strip()
            speakers = [item.get("speaker") for item in transcript if item.get("speaker")]
            out.append(dict(
                ticker=ticker,
                period=data.get("quarter") or quarter,
                date=data.get("date"),  # may be None depending on tier
                speakers=speakers,
                prepared_remarks=prepared,
                qa=qa,
            ))
        return out or None

    def get_filings(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        data = self._get(function="NEWS_SENTIMENT", tickers=ticker, limit=20)
        if not data or "feed" not in data:
            return None
        return [
            dict(
                title=n.get("title"),
                source=n.get("source"),
                published_at=n.get("time_published"),
                url=n.get("url"),
                summary=n.get("summary"),
                tickers=[ticker],
                topics=[t.get("topic") for t in n.get("topics", [])],
                sentiment=n.get("overall_sentiment_label"),
                relevance_score=float(n.get("relevance_score") or 0.5),
            )
            for n in data.get("feed", [])
        ]

    def get_estimates(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def get_macro_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        return None

    def list_tickers(self) -> List[str]:
        return []
