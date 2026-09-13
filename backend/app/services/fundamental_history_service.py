"""Durable annual/quarterly fundamentals; no filings, transcripts, or LLM calls.

FinancialPeriod stores latest reported values, not a revision ledger. Availability
is retained on updates, so this is explicitly NOT an as-originally-reported PIT
archive. Provider/coverage failures are returned with their full identities.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import FinancialPeriod
from . import history_service as history
from . import scorecard_pit
from .data_service import get_data_service
from .ticker_symbols import symbol_variants

log = logging.getLogger(__name__)
LINES = {"income": history._INCOME_LINES, "balance": history._BALANCE_LINES, "cash": history._CASH_LINES}
PRIMARY = {"income": "revenue", "balance": "total_assets", "cash": "cash_from_operations"}
LEGACY_SOURCES = {"", "live", "demo", "unknown"}


def _today() -> date:
    return date.today()


def read_stored_financials(ticker: str, *, start_date: date | None = None, cadence: str | None = None, db: Session | None = None) -> dict[str, list[dict]]:
    """Reconstitute provider-shaped statements from durable rows, newest first.

    No network or seeding. Optional start is inclusive; omit it for all history.
    A row's line_sources makes a mixed-provider statement explicit.
    """
    if cadence not in {None, "annual", "quarterly"}:
        raise ValueError("cadence must be annual or quarterly")
    own = db is None
    db = db or SessionLocal()
    try:
        if not inspect(db.get_bind()).has_table(FinancialPeriod.__tablename__):
            return {s: [] for s in LINES}
        query = select(FinancialPeriod).where(FinancialPeriod.ticker == ticker.strip().upper())
        if start_date:
            query = query.where(FinancialPeriod.period_end >= start_date)
        rows = db.execute(query.order_by(FinancialPeriod.period_end.desc(), FinancialPeriod.period.desc())).scalars()
        out: dict[str, list[dict]] = {s: [] for s in LINES}
        grouped: dict[tuple[str, str], dict] = {}
        for row in rows:
            if row.source == "demo" or row.statement not in LINES or row.value is None or not math.isfinite(row.value):
                continue
            if cadence and (row.fiscal_quarter is not None) != (cadence == "quarterly"):
                continue
            key = (row.statement, row.period)
            if key not in grouped:
                item = {"period": row.period, "period_end": row.period_end.isoformat() if row.period_end else None,
                        "currency": row.currency, "source": row.source, "line_sources": {},
                        "available_at": row.available_at.isoformat() if row.available_at else None,
                        "available_at_source": row.available_at_source,
                        "fetched_at": row.fetched_at.isoformat() if row.fetched_at else None}
                grouped[key] = item
                out[row.statement].append(item)
            item = grouped[key]
            item[row.line_item] = row.value
            item["line_sources"][row.line_item] = row.source
            if item["source"] != row.source:
                item["source"] = "mixed"
            if item["currency"] != row.currency:
                item["currency"] = "mixed"
        return out
    finally:
        if own:
            db.close()


def _coverage(statements: dict, start: date, end: date) -> dict:
    result = {}
    for statement in LINES:
        result[statement] = {}
        for cadence in ("annual", "quarterly"):
            points = []
            for row in statements.get(statement, []):
                fy, fq = history._parse_period(row.get("period"))
                d = history._coerce_date(row.get("period_end"))
                value = row.get(PRIMARY[statement])
                provenance = row.get("line_sources", {}).get(PRIMARY[statement], row.get("source"))
                if provenance in LEGACY_SOURCES or not row.get("currency") or row.get("currency") == "mixed":
                    continue
                if not d or d > end or fy is None or (fq is not None) != (cadence == "quarterly"):
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                points.append((d, fy * 4 + fq - 1 if fq else fy, str(row["period"])))
            points = sorted(set(points))
            oldest = points[0][0] if points else None
            newest = points[-1][0] if points else None
            relevant = [p for p in points if p[0] >= start]
            anchors = [p for p in points if p[0] < start]
            if anchors:
                relevant.insert(0, anchors[-1])
            keys = {p[1] for p in relevant}
            missing = []
            if keys:
                for key in range(min(keys), max(keys) + 1):
                    if key not in keys:
                        missing.append(f"{key // 4}Q{key % 4 + 1}" if cadence == "quarterly" else f"FY{key}")
            stale_days = 180 if cadence == "quarterly" else 460
            stale = newest is None or (end - newest).days > stale_days
            covers = oldest is not None and oldest <= start
            result[statement][cadence] = {
                "oldest": oldest.isoformat() if oldest else None, "newest": newest.isoformat() if newest else None,
                "period_count": len(points), "covers_start": covers, "stale": stale,
                "missing_periods": missing, "primary_line_item": PRIMARY[statement],
                "complete": bool(covers and not stale and not missing),
            }
    return result


def fundamental_coverage(ticker: str, start_date: date, *, db: Session | None = None) -> dict:
    """Read durable coverage only; never call providers or seed documents."""
    start = history._coerce_date(start_date)
    if start is None or start > _today():
        raise ValueError("start_date on or before today required")
    coverage = _coverage(read_stored_financials(ticker, db=db), start, _today())
    issues = [{"kind": "coverage_gap", "statement": statement, "cadence": cadence, **bucket}
              for statement, buckets in coverage.items() for cadence, bucket in buckets.items()
              if not bucket["complete"]]
    return {"ticker": ticker.strip().upper(), "requested_start": start.isoformat(), "requested_end": _today().isoformat(),
            "coverage": coverage, "success": _complete(coverage), "issues": issues}


def _complete(coverage: dict) -> bool:
    return all(bucket["complete"] for s in coverage.values() for bucket in s.values())


def _clean_payload(raw: dict, provider: str, symbol: str, issues: list[dict]) -> dict:
    clean = {s: [] for s in LINES}
    for issue in raw.get("_history_issues", []):
        issues.append({**issue, "provider": provider, "symbol": symbol})
    for statement, whitelist in LINES.items():
        rows = raw.get(statement) or []
        if not isinstance(rows, list):
            issues.append({"kind": "invalid_statement_payload", "provider": provider, "statement": statement})
            continue
        for row in rows:
            identity = {"provider": provider, "symbol": symbol, "statement": statement}
            if not isinstance(row, dict):
                issues.append({"kind": "invalid_provider_row", **identity})
                continue
            period = str(row.get("period") or "")
            identity["period"] = period
            fy, fq = history._parse_period(period)
            period_end = history._coerce_date(row.get("period_end") or row.get("date"))
            if fy is None or fq not in (None, 1, 2, 3, 4) or not period_end or period_end > _today():
                issues.append({"kind": "invalid_period", **identity})
                continue
            currency = str(row.get("currency") or "").strip().upper()
            if not currency or len(currency) > 8:
                issues.append({"kind": "missing_or_invalid_currency", **identity})
                continue
            item = {"period": f"{fy}Q{fq}" if fq else f"FY{fy}", "period_end": period_end.isoformat(),
                    "currency": currency, "source": provider,
                    "filing_date": row.get("filing_date"), "accepted_date": row.get("accepted_date")}
            for line in whitelist:
                value = row.get(line)
                if value is None:
                    continue
                try:
                    number = float(value)
                    if isinstance(value, bool) or not math.isfinite(number):
                        raise ValueError("nonfinite")
                except (ValueError, TypeError, OverflowError):
                    issues.append({"kind": "invalid_value", **identity, "line_item": line})
                    continue
                item[line] = number
            if PRIMARY[statement] not in item:
                issues.append({"kind": "missing_primary_value", **identity, "line_item": PRIMARY[statement]})
            if any(line in item for line in whitelist):
                clean[statement].append(item)
    return clean


def _fetch_financial_history(ticker: str, start: date) -> tuple[list[dict], list[dict], list[dict]]:
    """Retain partial providers; try later providers until combined coverage suffices."""
    payloads, issues, attempts = [], [], []
    combined = {s: [] for s in LINES}
    for provider in get_data_service()._live_chain("financials"):
        name = str(getattr(provider, "name", type(provider).__name__))
        if name in LEGACY_SOURCES:
            issues.append({"kind": "provider_identity_unusable", "provider": name})
            continue
        for symbol in symbol_variants(ticker):
            attempt = {"provider": name, "symbol": symbol}
            try:
                method = getattr(provider, "get_financial_history", None)
                raw = method(symbol, start) if method else provider.get_financial_statements(symbol)
            except Exception as exc:
                issues.append({"kind": "provider_error", **attempt, "error_type": type(exc).__name__})
                attempts.append({**attempt, "received": False})
                continue
            if not isinstance(raw, dict) or not any(raw.get(s) for s in LINES):
                issues.append({"kind": "provider_no_data", **attempt})
                if isinstance(raw, dict):
                    issues.extend({**i, **attempt} for i in raw.get("_history_issues", []))
                attempts.append({**attempt, "received": False})
                continue
            clean = _clean_payload(raw, name, symbol, issues)
            if not any(clean.values()):
                attempts.append({**attempt, "received": False})
                continue
            attempts.append({**attempt, "received": True, "periods_received": {s: len(clean[s]) for s in LINES}})
            payloads.append(clean)
            for s in LINES:
                combined[s].extend(clean[s])
            coverage = _coverage(combined, start, _today())
            if _complete(coverage):
                return payloads, issues, attempts
            issues.append({"kind": "provider_partial_coverage", **attempt, "coverage": _coverage(clean, start, _today())})
            break  # A valid alias resolved this security; next provider may add missing cadence.
    return payloads, issues, attempts


def backfill_fundamentals(ticker: str, start_date: date, force_refresh: bool = False, *, db: Session | None = None) -> dict:
    """Populate durable fundamentals only, with explicit complete/partial coverage.

    With an external session, writes are flushed but its caller owns commit and
    the report says committed=False. Missing/nonfinite values never erase data;
    conflicting values from a different provider are retained and reported.
    """
    ticker = ticker.strip().upper()
    start = history._coerce_date(start_date)
    if not ticker or len(ticker) > 16 or start is None or start > _today():
        raise ValueError("valid ticker and start_date on or before today required")
    own = db is None
    db = db or SessionLocal()
    report: dict[str, Any] = {"ticker": ticker, "requested_start": start.isoformat(), "requested_end": _today().isoformat(),
                              "rows_written": 0, "periods_received": {s: 0 for s in LINES}, "provider": [], "attempts": [], "issues": [],
                              "point_in_time": "latest_restated_values_at_original_availability", "committed": False}
    try:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        stored = read_stored_financials(ticker, db=db)
        before = _coverage(stored, start, _today())
        if not force_refresh and _complete(before):
            report.update(coverage=before, success=True, source="database", committed=own)
            return report
        # End the owned read transaction before spending time on provider IO.
        if own:
            db.rollback()
        payloads, issues, attempts = _fetch_financial_history(ticker, start)
        report["issues"].extend(issues)
        report["attempts"] = attempts
        existing = {(r.period, r.statement, r.line_item): r for r in db.execute(
            select(FinancialPeriod).where(FinancialPeriod.ticker == ticker)).scalars()}
        currencies: dict[tuple[str, str], set[str]] = {}
        for (period, statement, _), row in existing.items():
            if row.currency and row.source != "demo":
                currencies.setdefault((period, statement), set()).add(row.currency)
        now = datetime.utcnow()
        for payload in payloads:
            for statement, whitelist in LINES.items():
                report["periods_received"][statement] += len(payload[statement])
                for row in payload[statement]:
                    source = row["source"]
                    if source not in report["provider"]:
                        report["provider"].append(source)
                    period = row["period"]
                    end = history._coerce_date(row["period_end"])
                    fy, fq = history._parse_period(period)
                    available, rule = scorecard_pit.derive_available_at(ticker=ticker, period_end=end, fiscal_year=fy,
                        fiscal_quarter=fq, provider_date=row.get("filing_date") or row.get("accepted_date"), fetched_at=now, filings=[])
                    for line in whitelist:
                        if line not in row:
                            continue
                        key = (period, statement, line)
                        prior = existing.get(key)
                        other_currencies = currencies.get((period, statement), set())
                        if prior is None and other_currencies and row["currency"] not in other_currencies:
                            report["issues"].append({"kind": "stored_value_conflict", "statement": statement,
                                "period": period, "line_item": line, "provider": source,
                                "stored_currencies": sorted(other_currencies), "incoming_currency": row["currency"]})
                            continue
                        if prior and prior.value is not None and math.isfinite(prior.value):
                            if (prior.currency and prior.currency != row["currency"]) or (
                                prior.source != source and prior.source != "demo" and prior.value != row[line]
                            ):
                                report["issues"].append({"kind": "stored_value_conflict", "statement": statement,
                                    "period": period, "line_item": line, "provider": source, "stored_source": prior.source,
                                    "stored_currency": prior.currency, "incoming_currency": row["currency"]})
                                continue
                        if history._upsert_financial_period(db, ticker=ticker, period=period, statement=statement,
                            line_item=line, value=row[line], period_end=end, fiscal_year=fy, fiscal_quarter=fq,
                            source=source, currency=row["currency"], fetched_at=now, available_at=available, available_at_source=rule, existing_rows=existing):
                            report["rows_written"] += 1
                            currencies.setdefault((period, statement), set()).add(row["currency"])
        db.flush()
        report["coverage"] = _coverage(read_stored_financials(ticker, db=db), start, _today())
        for statement, buckets in report["coverage"].items():
            for cadence, bucket in buckets.items():
                if not bucket["complete"]:
                    report["issues"].append({"kind": "coverage_gap", "statement": statement, "cadence": cadence, **bucket})
        report["success"] = _complete(report["coverage"]) and not any(i["kind"] in {"stored_value_conflict", "invalid_value"} for i in report["issues"])
        if own:
            db.commit()
            report["committed"] = True
        if report["issues"]:
            log.warning("fundamentals history %s: %d issues: %s", ticker, len(report["issues"]), report["issues"])
        return report
    except Exception as exc:
        if own:
            db.rollback()
        log.warning("fundamentals history %s failed: %s", ticker, type(exc).__name__)
        report.update(success=False, rows_written=0)
        report["issues"].append({"kind": "persistence_or_read_error", "error_type": type(exc).__name__})
        return report
    finally:
        if own:
            db.close()
