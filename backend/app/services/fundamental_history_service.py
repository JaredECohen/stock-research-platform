"""Durable annual/quarterly fundamentals; no filings, transcripts, or LLM calls.

FinancialPeriod stores latest reported values, not a revision ledger. Availability
is retained on updates, so this is explicitly NOT an as-originally-reported PIT
archive. Provider/coverage failures are returned with their full identities.
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import FinancialPeriod
from . import history_service as history
from . import scorecard_pit
from .data_service import get_data_service
from .ticker_symbols import market_data_symbols

log = logging.getLogger(__name__)
LINES = {"income": history._INCOME_LINES, "balance": history._BALANCE_LINES, "cash": history._CASH_LINES}
PRIMARY = {"income": "revenue", "balance": "total_assets", "cash": "cash_from_operations"}
LEGACY_SOURCES = {"", "live", "demo", "unknown"}
REFRESH_TTL_DAYS = 7
BLOCKING_ISSUES = {
    "stored_value_conflict", "invalid_value", "stored_period_end_conflict", "conflicting_provider_period",
    "invalid_period", "invalid_stored_period", "conflicting_stored_statement", "invalid_stored_value_or_currency",
    "missing_stored_primary_value", "invalid_coverage_period", "stored_period_alias_conflict", "refresh_incomplete",
    "coverage_gap", "stored_fetch_stale",
}


def _has_blockers(issues: list[dict]) -> bool:
    """Optional NULL warnings remain observable without denying usable primary history."""
    return any(not issue.get("resolved") and issue.get("kind") in BLOCKING_ISSUES for issue in issues)


def _today() -> date:
    return date.today()


def _valid_period(period: Any, period_end: Any, *, end: date) -> tuple[int | None, int | None, date | None]:
    """Validate before sorting, grouping, or enumerating fiscal gaps."""
    fy, fq = history._parse_period(period)
    d = history._coerce_date(period_end)
    if fy is None or d is None or d > end or abs(fy - d.year) > 1:
        return None, None, d
    return fy, fq, d


def read_stored_financials(ticker: str, *, start_date: date | None = None, cadence: str | None = None, db: Session | None = None) -> dict[str, list[dict]]:
    """Read durable rows without collapsing conflicting dates or currencies.

    Excluded stored rows/groups are named in additive `_history_issues` metadata.
    This read never repairs or deletes preexisting data and never calls providers.
    """
    if cadence not in {None, "annual", "quarterly"}:
        raise ValueError("cadence must be annual or quarterly")
    own = db is None
    db = db or SessionLocal()
    out: dict[str, list[dict]] = {s: [] for s in LINES}
    issues: list[dict] = []
    try:
        if not inspect(db.get_bind()).has_table(FinancialPeriod.__tablename__):
            return out
        query = select(FinancialPeriod).where(FinancialPeriod.ticker == ticker.strip().upper())
        # Inspect each entire group before applying a requested date/cadence so
        # filtering cannot hide the other half of a contradictory statement.
        rows = db.execute(query.order_by(FinancialPeriod.period_end.desc(), FinancialPeriod.period.desc())).scalars()
        groups: dict[tuple[str, str], list] = {}
        for row in rows:
            if row.source == "demo" or row.statement not in LINES:
                continue
            identity = {"id": row.id, "ticker": row.ticker, "statement": row.statement,
                        "period": row.period, "period_end": str(row.period_end), "line_item": row.line_item,
                        "fiscal_year": row.fiscal_year, "fiscal_quarter": row.fiscal_quarter, "source": row.source,
                        "currency": row.currency}
            fy, fq, d = _valid_period(row.period, row.period_end, end=_today())
            if fy is None or (row.fiscal_year is not None and row.fiscal_year != fy) or row.fiscal_quarter != fq:
                issues.append({"kind": "invalid_stored_period", **identity})
                continue
            if row.value is None:
                kind = "missing_stored_primary_value" if row.line_item == PRIMARY[row.statement] else "missing_stored_optional_value"
                issues.append({"kind": kind, **identity})
                continue
            if not row.currency or not math.isfinite(row.value):
                issues.append({"kind": "invalid_stored_value_or_currency", **identity})
                continue
            canonical = f"{fy:04d}Q{fq}" if fq else f"FY{fy:04d}"
            groups.setdefault((row.statement, canonical), []).append(row)
        for (statement, period), group in groups.items():
            ends = {r.period_end for r in group}
            currencies = {r.currency for r in group}
            values: dict[str, set[float]] = {}
            for r in group:
                values.setdefault(r.line_item, set()).add(r.value)
            if len(ends) > 1 or len(currencies) > 1 or any(len(v) > 1 for v in values.values()):
                issues.append({"kind": "conflicting_stored_statement", "ticker": ticker.strip().upper(),
                    "statement": statement, "period": period,
                    "rows": [{"id": r.id, "period": r.period, "line_item": r.line_item, "period_end": str(r.period_end),
                              "currency": r.currency, "value": r.value, "source": r.source} for r in group]})
                continue
            row = group[0]
            if start_date and row.period_end < start_date:
                continue
            if cadence and (row.fiscal_quarter is not None) != (cadence == "quarterly"):
                continue
            item = {"period": period, "period_end": row.period_end.isoformat(), "currency": row.currency,
                    "source": row.source, "line_sources": {}, "line_fetched_at": {}, "available_at": row.available_at.isoformat() if row.available_at else None,
                    "available_at_source": row.available_at_source, "fetched_at": row.fetched_at.isoformat() if row.fetched_at else None}
            for r in sorted(group, key=lambda r: r.source in LEGACY_SOURCES):
                if r.line_item not in item["line_sources"]:
                    item[r.line_item] = r.value
                    item["line_sources"][r.line_item] = r.source
                    item["line_fetched_at"][r.line_item] = r.fetched_at.isoformat() if r.fetched_at else None
                if item["source"] != r.source:
                    item["source"] = "mixed"
            out[statement].append(item)
        if issues:
            out["_history_issues"] = issues
            log.warning("fundamentals stored read %s: %d issues: %s", ticker, len(issues), issues)
        return out
    finally:
        if own:
            db.close()


def _coverage(statements: dict, start: date, end: date, *, issues: list[dict] | None = None) -> dict:
    result = {}
    for statement in LINES:
        result[statement] = {}
        for cadence in ("annual", "quarterly"):
            points = []
            fetched_by_point = {}
            for row in statements.get(statement, []):
                fy, fq, d = _valid_period(row.get("period"), row.get("period_end"), end=end)
                if fy is None:
                    if issues is not None and cadence == "annual":
                        issues.append({"kind": "invalid_coverage_period", "statement": statement,
                                       "period": row.get("period"), "period_end": str(row.get("period_end")), "provider": row.get("source")})
                    continue
                value = row.get(PRIMARY[statement])
                provenance = row.get("line_sources", {}).get(PRIMARY[statement], row.get("source"))
                if provenance in LEGACY_SOURCES or not row.get("currency") or row.get("currency") == "mixed":
                    continue
                if not d or d > end or fy is None or (fq is not None) != (cadence == "quarterly"):
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                point = (d, fy * 4 + fq - 1 if fq else fy, str(row["period"]))
                points.append(point)
                fetched_by_point[point] = row.get("line_fetched_at", {}).get(PRIMARY[statement], row.get("fetched_at"))
            points = sorted(set(points))
            oldest = points[0][0] if points else None
            newest = points[-1][0] if points else None
            fetched_at = fetched_by_point.get(points[-1]) if points else None
            try:
                fetched_dt = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
                if fetched_dt.tzinfo is None:
                    fetched_dt = fetched_dt.replace(tzinfo=UTC)
                fetch_age = (datetime.now(UTC) - fetched_dt).total_seconds()
                fresh = 0 <= fetch_age <= REFRESH_TTL_DAYS * 86400
            except (TypeError, ValueError, OverflowError):
                fresh = False
            # Operational freshness guard, not a regulatory filing calendar.
            # Q3 may wait longer for the annual report that supplies Q4.
            stale_days = 460 if cadence == "annual" else (185 if points and points[-1][1] % 4 == 2 else 140)
            relevant = [p for p in points if p[0] >= start]
            anchors = [p for p in points if p[0] < start]
            if anchors and (start - anchors[-1][0]).days <= stale_days:
                relevant.insert(0, anchors[-1])
            keys = {p[1] for p in relevant}
            missing = []
            if keys:
                for key in range(min(keys), max(keys) + 1):
                    if key not in keys:
                        missing.append(f"{key // 4}Q{key % 4 + 1}" if cadence == "quarterly" else f"FY{key}")
            stale = newest is None or (end - newest).days > stale_days
            covers = bool(relevant and relevant[0][0] <= start)
            result[statement][cadence] = {
                "oldest": oldest.isoformat() if oldest else None, "newest": newest.isoformat() if newest else None,
                "period_count": len(points), "covers_start": covers, "stale": stale,
                "stale_threshold_days": stale_days,
                "latest_primary_fetched_at": fetched_at, "fresh": fresh, "refresh_ttl_days": REFRESH_TTL_DAYS,
                "missing_periods": missing, "primary_line_item": PRIMARY[statement],
                "complete": bool(covers and not stale and not missing),
            }
    return result


def fundamental_coverage(ticker: str, start_date: date, *, db: Session | None = None) -> dict:
    """Read durable coverage only; never call providers or seed documents."""
    start = history._coerce_date(start_date)
    if start is None or start > _today():
        raise ValueError("start_date on or before today required")
    stored = read_stored_financials(ticker, db=db)
    issues = list(stored.get("_history_issues", []))
    coverage = _coverage(stored, start, _today(), issues=issues)
    issues.extend([{"kind": "coverage_gap", "statement": statement, "cadence": cadence, **bucket}
              for statement, buckets in coverage.items() for cadence, bucket in buckets.items()
              if not bucket["complete"]])
    issues.extend([{"kind": "stored_fetch_stale", "statement": statement, "cadence": cadence,
                    "latest_primary_fetched_at": bucket["latest_primary_fetched_at"], "refresh_ttl_days": REFRESH_TTL_DAYS}
                   for statement, buckets in coverage.items() for cadence, bucket in buckets.items()
                   if bucket["complete"] and not bucket["fresh"]])
    return {"ticker": ticker.strip().upper(), "requested_start": start.isoformat(), "requested_end": _today().isoformat(),
            "coverage": coverage, "success": _complete(coverage) and _fresh(coverage) and not _has_blockers(issues), "issues": issues}


def _complete(coverage: dict) -> bool:
    return all(bucket["complete"] for s in coverage.values() for bucket in s.values())


def _fresh(coverage: dict) -> bool:
    return all(bucket["fresh"] for s in coverage.values() for bucket in s.values())


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
            fy, fq, period_end = _valid_period(period, row.get("period_end") or row.get("date"), end=_today())
            if fy is None or (row.get("cadence") == "annual" and fq is not None) or (row.get("cadence") == "quarterly" and fq is None):
                issues.append({"kind": "invalid_period", **identity, "period_end": str(row.get("period_end") or row.get("date"))})
                continue
            currency = str(row.get("currency") or "").strip().upper()
            if not currency or len(currency) > 8:
                issues.append({"kind": "missing_or_invalid_currency", **identity})
                continue
            item = {"period": f"{fy:04d}Q{fq}" if fq else f"FY{fy:04d}", "period_end": period_end.isoformat(),
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
                if row.get("period_basis"):
                    issues.append({"kind": "fiscal_period_derivation", **identity,
                                   "period_end": period_end.isoformat(), "period_basis": row["period_basis"],
                                   "anchor_date": row.get("period_anchor_date")})
                clean[statement].append(item)
    for statement, rows in clean.items():
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["period"], []).append(row)
        clean[statement] = []
        for period, group in grouped.items():
            combined = dict(group[0])
            conflicts = set()
            for other in group[1:]:
                for key, value in other.items():
                    if key in combined and combined[key] is not None and value is not None and combined[key] != value:
                        conflicts.add(key)
                    elif value is not None:
                        combined[key] = value
            if conflicts:
                issues.append({"kind": "conflicting_provider_period", "provider": provider, "symbol": symbol,
                               "statement": statement, "period": period, "fields": sorted(conflicts),
                               "occurrences": [{"row_index": index, "period_end": r["period_end"], "currency": r["currency"],
                                                "values": {key: r.get(key) for key in sorted(conflicts)}} for index, r in enumerate(group)]})
            else:
                clean[statement].append(combined)
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
        for symbol in market_data_symbols(ticker):
            attempt = {"provider": name, "symbol": symbol}
            try:
                method = getattr(provider, "get_financial_history", None)
                if not method:
                    issues.append({"kind": "history_adapter_unavailable", **attempt})
                    attempts.append({**attempt, "received": False})
                    break
                raw = method(symbol, start)
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
                              "source_upgrades": [], "refresh_complete": False, "rows_refreshed": 0,
                              "point_in_time": "latest_restated_values_at_original_availability", "committed": False}
    try:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        stored = read_stored_financials(ticker, db=db)
        report["issues"].extend(stored.get("_history_issues", []))
        before = _coverage(stored, start, _today(), issues=report["issues"])
        if not force_refresh and _complete(before) and _fresh(before) and not _has_blockers(report["issues"]):
            report.update(coverage=before, success=True, source="database", committed=own)
            return report
        # End the owned read transaction before spending time on provider IO.
        if own:
            db.rollback()
        payloads, issues, attempts = _fetch_financial_history(ticker, start)
        report["issues"].extend(issues)
        report["attempts"] = attempts
        incoming = {s: [row for payload in payloads for row in payload[s]] for s in LINES}
        report["refresh_complete"] = bool(payloads) and _complete(_coverage(incoming, start, _today()))
        if force_refresh and not report["refresh_complete"]:
            report["issues"].append({"kind": "refresh_incomplete", "ticker": ticker,
                                     "coverage": _coverage(incoming, start, _today())})
        existing = {}
        aliases: dict[tuple[str, str, str], list] = {}
        for r in db.execute(select(FinancialPeriod).where(FinancialPeriod.ticker == ticker)).scalars():
            fy, fq = history._parse_period(r.period)
            canonical = (f"{fy:04d}Q{fq}" if fq else f"FY{fy:04d}") if fy else r.period
            aliases.setdefault((canonical, r.statement, r.line_item), []).append(r)
        blocked_keys = set()
        for key, group in aliases.items():
            if len(group) > 1:
                blocked_keys.add(key)
                report["issues"].append({"kind": "stored_period_alias_conflict", "ticker": ticker,
                    "period": key[0], "statement": key[1], "line_item": key[2],
                    "rows": [{"id": r.id, "period": r.period, "period_end": str(r.period_end),
                              "currency": r.currency, "source": r.source} for r in group]})
            existing[key] = group[0]
        currencies: dict[tuple[str, str], set[str]] = {}
        for (period, statement, _), row in existing.items():
            if row.currency and row.source not in LEGACY_SOURCES:
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
                        if key in blocked_keys:
                            continue  # Never choose an arbitrary alias row to restate.
                        prior = existing.get(key)
                        other_currencies = currencies.get((period, statement), set())
                        if prior is None and other_currencies and row["currency"] not in other_currencies:
                            report["issues"].append({"kind": "stored_value_conflict", "statement": statement,
                                "period": period, "line_item": line, "provider": source,
                                "stored_currencies": sorted(other_currencies), "incoming_currency": row["currency"]})
                            continue
                        if prior and prior.period_end is not None and prior.period_end != end:
                            report["issues"].append({"kind": "stored_period_end_conflict", "ticker": ticker, "id": prior.id,
                                "statement": statement, "period": period, "line_item": line, "provider": source,
                                "stored_source": prior.source, "stored_period_end": str(prior.period_end), "incoming_period_end": str(end)})
                            continue
                        if prior and prior.source not in LEGACY_SOURCES and prior.currency and prior.currency != row["currency"]:
                            report["issues"].append({"kind": "stored_value_conflict", "ticker": ticker, "id": prior.id,
                                "statement": statement, "period": period, "line_item": line, "provider": source,
                                "stored_source": prior.source, "stored_currency": prior.currency, "incoming_currency": row["currency"]})
                            continue
                        if prior and prior.value is not None and math.isfinite(prior.value):
                            if prior.source != source and prior.source not in LEGACY_SOURCES and prior.value == row[line] and prior.currency == row["currency"]:
                                continue  # Corroboration does not transfer ownership of a stored fact.
                            if prior.source != source and prior.source not in LEGACY_SOURCES and prior.value != row[line]:
                                report["issues"].append({"kind": "stored_value_conflict", "statement": statement,
                                    "period": period, "line_item": line, "provider": source, "stored_source": prior.source,
                                    "stored_currency": prior.currency, "incoming_currency": row["currency"]})
                                continue
                        upgrade = None
                        if prior and prior.source in LEGACY_SOURCES and source not in LEGACY_SOURCES:
                            upgrade = {"ticker": ticker, "id": prior.id, "statement": statement, "period": period,
                                       "period_end": str(end), "line_item": line, "old_source": prior.source,
                                       "new_source": source, "old_value": prior.value if prior.value is not None and math.isfinite(prior.value) else None,
                                       "new_value": row[line], "old_currency": prior.currency, "new_currency": row["currency"]}
                        if history._upsert_financial_period(db, ticker=ticker, period=period, statement=statement,
                            line_item=line, value=row[line], period_end=end, fiscal_year=fy, fiscal_quarter=fq,
                            source=source, currency=row["currency"], fetched_at=now, available_at=available, available_at_source=rule, existing_rows=existing):
                            report["rows_written"] += 1
                            if upgrade:
                                report["source_upgrades"].append(upgrade)
                            currencies.setdefault((period, statement), set()).add(row["currency"])
                        elif prior is not None and prior.source == source:
                            # A same-provider verification refreshes its timestamp
                            # even when facts are identical. Count metadata writes
                            # explicitly; a different provider cannot date-stamp
                            # an older source's observation by corroboration.
                            prior.fetched_at = now
                            report["rows_written"] += 1
                            report["rows_refreshed"] += 1
        db.flush()
        after = read_stored_financials(ticker, db=db)
        for issue in report["issues"]:
            if issue in stored.get("_history_issues", []) and issue not in after.get("_history_issues", []):
                issue["resolved"] = True
        for issue in after.get("_history_issues", []):
            if issue not in report["issues"]:
                report["issues"].append(issue)
        report["coverage"] = _coverage(after, start, _today(), issues=report["issues"])
        for statement, buckets in report["coverage"].items():
            for cadence, bucket in buckets.items():
                if not bucket["complete"]:
                    report["issues"].append({"kind": "coverage_gap", "statement": statement, "cadence": cadence, **bucket})
                elif not bucket["fresh"]:
                    report["issues"].append({"kind": "stored_fetch_stale", "statement": statement, "cadence": cadence,
                                            "latest_primary_fetched_at": bucket["latest_primary_fetched_at"], "refresh_ttl_days": REFRESH_TTL_DAYS})
        report["success"] = _complete(report["coverage"]) and _fresh(report["coverage"]) and not _has_blockers(report["issues"])
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
        report.update(success=False, rows_written=0, rows_refreshed=0)
        report["issues"].append({"kind": "persistence_or_read_error", "error_type": type(exc).__name__})
        return report
    finally:
        if own:
            db.close()
