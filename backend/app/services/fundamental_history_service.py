"""Durable annual/quarterly fundamentals; no filings, transcripts, or LLM calls.

FinancialPeriod stores latest reported values, not a revision ledger. Availability
is retained on updates, so this is explicitly NOT an as-originally-reported PIT
archive. Provider/coverage failures are returned with their full identities.

Ownership (owner decision 2026-09-24, FIX-006): FMP owns every observation it
reports. An equal FMP verification adopts a secondary or legacy observation
(source -> fmp, fetch time -> now, availability kept). An FMP disagreement
quarantines the other row (`fundamental_quarantine`, never a delete) and stores
FMP's observation, carrying the quarantined row's availability when it is
valid. Secondary providers still cannot date-stamp or adopt an FMP observation,
and never fill a period FMP reports. Between non-primary providers the earlier
rules are unchanged. Every quarantine, adoption and restatement is audited in
`financial_data_repairs` with before-images.
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, date, datetime
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import FinancialDataRepair, FinancialPeriod
from . import history_service as history
from . import scorecard_pit
from .data_service import get_data_service
from .ticker_symbols import symbol_variants

log = logging.getLogger(__name__)
LINES = {"income": history._INCOME_LINES, "balance": history._BALANCE_LINES, "cash": history._CASH_LINES}
PRIMARY = {"income": "revenue", "balance": "total_assets", "cash": "cash_from_operations"}
LEGACY_SOURCES = {"", "live", "demo", "unknown"}
# Owner decision 2026-09-24: FMP is primary everywhere; other providers only
# fill periods FMP lacks.
PRIMARY_PROVIDER = "fmp"
# 52/53-week fiscal years move a period end by a few days between providers.
PERIOD_END_TOLERANCE_DAYS = 7
# Unattended (scheduled) refreshes may quarantine only small, unambiguous
# sets. Anything larger, or any label/alias/period-end shift, becomes a
# planned repair a person reviews through the bk-repair GET/apply routes.
QUARANTINE_AUTO_MAX_ROWS = 20
QUARANTINE_AUTO_MAX_SHARE = 0.05
# Integration critique names: exact_key_value_conflict,
# exact_key_currency_conflict, availability_before_period_end.
SAFE_AUTO_REASONS = frozenset({"value_conflicts_with_primary", "currency_conflicts_with_primary",
                               "invalid_availability_precedes_period_end"})
TakeoverMode = Literal["supervised", "unattended"]
REFRESH_TTL_DAYS = 7
BLOCKING_ISSUES = {
    "stored_value_conflict", "invalid_value", "stored_period_end_conflict", "conflicting_provider_period",
    "invalid_period", "invalid_stored_period", "conflicting_stored_statement", "invalid_stored_value_or_currency",
    "missing_stored_primary_value", "invalid_coverage_period", "stored_period_alias_conflict", "refresh_incomplete",
    "coverage_gap", "stored_fetch_stale",
    "legacy_period_relabel_conflict",
    "ambiguous_provider_period_end",
    "duplicate_stored_period_end",
    # An unattended takeover that needs review has not happened yet.
    "primary_takeover_planned",
    "repull_plan_mismatch",
}
# Stored exactly as FMP reports them (owner decision 2026-09-24); flagged on
# read, never normalized. The flag retires itself once the stored value
# moves more than 0.5% away from the evidenced provider value (an FMP fix).
KNOWN_DEFINITION_BREAKS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("BK", "income", "revenue"): {
        "period": "2026Q1", "provider_value": 9.863e9, "issuer_value": 5.409e9,
        "note": "FMP 2026Q1 revenue is gross of interest expense (issuer net revenue = 9.863bn - 4.454bn); "
                "2026Q2 is net. Do not compare across the break.",
        "evidence": "docs/reviews/2026-09-13-bk-provider-comparability.json"},
    ("BK", "income", "eps_diluted"): {
        "period": "2026Q2", "provider_value": 2.43, "issuer_value": 2.45,
        "note": "FMP 2026Q2 diluted EPS differs from the issuer's reported 2.45.",
        "evidence": "docs/reviews/2026-09-13-bk-provider-comparability.json"},
    ("BK", "income", "weighted_avg_shares_diluted"): {
        "period": "2026Q2", "provider_value": 698.164e6, "issuer_value": 692.223e6,
        "note": "FMP 2026Q2 diluted share count differs from the issuer's reported 692.223m.",
        "evidence": "docs/reviews/2026-09-13-bk-provider-comparability.json"},
}


class PlanMismatch(RuntimeError):
    """An executed re-pull would quarantine a different set than its reviewed dry run."""

    def __init__(self, expected: dict[int, str], actual: dict[int, str]) -> None:
        super().__init__("fundamentals repull plan mismatch")
        self.expected, self.actual = expected, actual


def _has_blockers(issues: list[dict]) -> bool:
    """Optional NULL warnings remain observable without denying usable primary history."""
    return any(not issue.get("resolved") and issue.get("kind") in BLOCKING_ISSUES for issue in issues)


def _today() -> date:
    return date.today()


def _valid_currency(value: Any) -> bool:
    code = str(value or "").strip().upper()
    return len(code) == 3 and code.isascii() and code.isalpha() and code not in {"NAN", "XXX", "XTS"}


def _valid_period(period: Any, period_end: Any, *, end: date) -> tuple[int | None, int | None, date | None]:
    """Validate before sorting, grouping, or enumerating fiscal gaps."""
    fy, fq = history._parse_period(period)
    d = history._coerce_date(period_end)
    if fy is None or d is None or d > end or abs(fy - d.year) > 1:
        return None, None, d
    return fy, fq, d


def _stored_period_valid(row: FinancialPeriod) -> bool:
    fy, fq, _ = _valid_period(row.period, row.period_end, end=_today())
    return fy is not None and (row.fiscal_year is None or row.fiscal_year == fy) and row.fiscal_quarter == fq


def _unusable_legacy_alias(row: FinancialPeriod, canonical: str) -> bool:
    return row.source in LEGACY_SOURCES and row.period != canonical and not _stored_period_valid(row)


def _canonical(period: Any) -> str:
    fy, fq = history._parse_period(period)
    return (f"{fy:04d}Q{fq}" if fq else f"FY{fy:04d}") if fy else str(period)


def _near(a: date | None, b: date | None) -> bool:
    return a is not None and b is not None and abs((a - b).days) <= PERIOD_END_TOLERANCE_DAYS


def _json_value(value: float | None) -> float | str | None:
    return value if value is None or math.isfinite(value) else str(value)


def _row_identity(row: FinancialPeriod) -> dict:
    return {"id": row.id, "ticker": row.ticker, "statement": row.statement,
            "period": row.period, "period_end": str(row.period_end), "line_item": row.line_item,
            "fiscal_year": row.fiscal_year, "fiscal_quarter": row.fiscal_quarter, "source": row.source,
            "currency": row.currency}


def _stored_observation_identity(row: FinancialPeriod) -> tuple:
    """An observed statement line, independent of a possibly stale FY label."""
    return (row.period_end, row.statement, row.line_item, row.fiscal_quarter is not None)


def _stored_observation_evidence(row: FinancialPeriod) -> dict:
    return {"id": row.id, "ticker": row.ticker, "period": row.period,
            "fiscal_year": row.fiscal_year, "fiscal_quarter": row.fiscal_quarter,
            "period_end": row.period_end.isoformat(), "statement": row.statement,
            "line_item": row.line_item, "value": row.value, "source": row.source,
            "currency": row.currency,
            "available_at": row.available_at.isoformat() if row.available_at else None,
            "available_at_source": row.available_at_source,
            "fetched_at": row.fetched_at.isoformat() if row.fetched_at else None}


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
        # The session's own connection: `backfill_fundamentals` calls this read
        # mid-transaction, and checking out a second pooled connection there
        # can reset the shared one (sqlite StaticPool) and drop its writes.
        if not inspect(db.connection()).has_table(FinancialPeriod.__tablename__):
            return out
        query = select(FinancialPeriod).where(FinancialPeriod.ticker == ticker.strip().upper())
        # Inspect each entire group before applying a requested date/cadence so
        # filtering cannot hide the other half of a contradictory statement.
        rows = list(db.execute(query.order_by(FinancialPeriod.period_end.desc(), FinancialPeriod.period.desc())).scalars())
        confirmed = {(r.period, r.statement, r.line_item): r for r in rows
                     if r.source not in LEGACY_SOURCES and _stored_period_valid(r)
                     and r.value is not None and math.isfinite(r.value) and _valid_currency(r.currency)}
        confirmed_observations: dict[tuple, list] = {}
        for row in confirmed.values():
            confirmed_observations.setdefault(_stored_observation_identity(row), []).append(row)
        groups: dict[tuple[str, str], list] = {}
        for row in rows:
            if row.source == "demo" or row.statement not in LINES:
                continue
            identity = _row_identity(row)
            fy, fq, d = _valid_period(row.period, row.period_end, end=_today())
            if fy is None or (row.fiscal_year is not None and row.fiscal_year != fy) or row.fiscal_quarter != fq:
                parsed_fy, parsed_fq = history._parse_period(row.period)
                canonical = (f"{parsed_fy:04d}Q{parsed_fq}" if parsed_fq else f"FY{parsed_fy:04d}") if parsed_fy else row.period
                replacement = confirmed.get((canonical, row.statement, row.line_item))
                if _unusable_legacy_alias(row, canonical) and replacement:
                    issues.append({"kind": "unusable_legacy_observation", **identity,
                        "value": row.value if row.value is None or math.isfinite(row.value) else str(row.value),
                        "reason": "invalid_legacy_alias_excluded", "canonical_period": canonical,
                        "usable_canonical_id": replacement.id, "usable_canonical_source": replacement.source,
                        "usable_canonical_period_end": replacement.period_end.isoformat()})
                else:
                    issues.append({"kind": "invalid_stored_period", **identity})
                continue
            if row.value is None:
                kind = "missing_stored_primary_value" if row.line_item == PRIMARY[row.statement] else "missing_stored_optional_value"
                issues.append({"kind": kind, **identity})
                continue
            if not _valid_currency(row.currency) or not math.isfinite(row.value):
                issues.append({"kind": "invalid_stored_value_or_currency", **identity})
                continue
            canonical = f"{fy:04d}Q{fq}" if fq else f"FY{fy:04d}"
            # A legacy FY label may duplicate a separately stored, dated
            # provider fact. Exclude only this legacy line from usable reads;
            # every stored field remains untouched, even when values differ.
            replacements = confirmed_observations.get(_stored_observation_identity(row), [])
            primary_replacements = [r for r in replacements if r.source == PRIMARY_PROVIDER]
            if len(replacements) > 1 and len(primary_replacements) == 1:
                replacements = primary_replacements  # R3: the primary candidate decides.
            if row.source in LEGACY_SOURCES and len(replacements) == 1:
                replacement = replacements[0]
                replacement_fy, replacement_fq = history._parse_period(replacement.period)
                replacement_period = (f"{replacement_fy:04d}Q{replacement_fq}" if replacement_fq
                                      else f"FY{replacement_fy:04d}")
                if replacement_period != canonical:
                    issues.append({"kind": "legacy_duplicate_observation_excluded",
                        **_stored_observation_evidence(row),
                        "reason": "unique_named_provider_observation_under_different_fiscal_label",
                        "replacement": _stored_observation_evidence(replacement)})
                    continue
            groups.setdefault((row.statement, canonical), []).append(row)
        periods_by_end: dict[tuple, set] = {}
        for (statement, period), group in groups.items():
            for row in group:
                periods_by_end.setdefault((statement, history._parse_period(period)[1] is not None, row.period_end), set()).add(period)
        duplicate_keys = set()
        superseded_labels = set()
        for (statement, _, end), periods in periods_by_end.items():
            if len(periods) > 1:
                # R2: one period end under several labels. When exactly one
                # label group holds primary rows it is the observation; the
                # others stay stored and are named, not silently dropped.
                primary_periods = [p for p in periods if any(r.source == PRIMARY_PROVIDER for r in groups[(statement, p)])]
                if len(primary_periods) == 1:
                    for period in sorted(periods - {primary_periods[0]}):
                        superseded_labels.add((statement, period))
                        issues.extend({"kind": "label_superseded_by_primary", **_row_identity(row),
                                       "value": _json_value(row.value), "primary_period": primary_periods[0]}
                                      for row in groups[(statement, period)])
                    continue
                duplicate_keys.update((statement, period) for period in periods)
                issues.append({"kind": "duplicate_stored_period_end", "ticker": ticker.strip().upper(),
                    "statement": statement, "period_end": end.isoformat(), "periods": sorted(periods),
                    "rows": [{"id": row.id, "period": row.period, "line_item": row.line_item,
                              "period_end": str(row.period_end), "source": row.source}
                             for period in sorted(periods) for row in groups[(statement, period)]]})
        for (statement, period), group in groups.items():
            if (statement, period) in duplicate_keys or (statement, period) in superseded_labels:
                continue
            primary_rows = [r for r in group if r.source == PRIMARY_PROVIDER]
            if primary_rows:
                # R1: inside a primary period, named secondaries never
                # contribute, and a legacy line is kept only when the primary
                # does not report it and agrees on date and currency.
                primary_values = {r.line_item: r.value for r in primary_rows}
                primary_ends = {r.period_end for r in primary_rows}
                primary_currencies = {r.currency for r in primary_rows}
                kept = []
                for r in group:
                    if (r.source == PRIMARY_PROVIDER or (r.source in LEGACY_SOURCES and r.line_item not in primary_values
                            and r.period_end in primary_ends and r.currency in primary_currencies)):
                        kept.append(r)
                        continue
                    issues.append({"kind": "superseded_by_primary", **_row_identity(r), "value": _json_value(r.value),
                                   "primary_value": _json_value(primary_values.get(r.line_item))})
                group = kept
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
            group = sorted(group, key=lambda r: (r.source != PRIMARY_PROVIDER, r.source in LEGACY_SOURCES))
            row = group[0]
            if start_date and row.period_end < start_date:
                continue
            if cadence and (row.fiscal_quarter is not None) != (cadence == "quarterly"):
                continue
            item = {"period": period, "period_end": row.period_end.isoformat(), "currency": row.currency,
                    "source": row.source, "line_sources": {}, "line_fetched_at": {}, "available_at": row.available_at.isoformat() if row.available_at else None,
                    "available_at_source": row.available_at_source, "fetched_at": row.fetched_at.isoformat() if row.fetched_at else None}
            for r in sorted(group, key=lambda r: (r.source != PRIMARY_PROVIDER, r.source in LEGACY_SOURCES)):
                if r.line_item not in item["line_sources"]:
                    item[r.line_item] = r.value
                    item["line_sources"][r.line_item] = r.source
                    item["line_fetched_at"][r.line_item] = r.fetched_at.isoformat() if r.fetched_at else None
                if item["source"] != r.source:
                    item["source"] = "mixed"
            out[statement].append(item)
        issues.extend(_definition_break_flags(ticker.strip().upper(), out))
        if issues:
            out["_history_issues"] = issues
            log.warning("fundamentals stored read %s: %d issues: %s", ticker, len(issues), issues)
        return out
    finally:
        if own:
            db.close()


def _definition_break_flags(ticker: str, statements: dict) -> list[dict]:
    """Nonblocking flags for evidenced provider definition breaks (BK)."""
    flags = []
    for (flag_ticker, statement, line), spec in KNOWN_DEFINITION_BREAKS.items():
        if flag_ticker != ticker:
            continue
        for item in statements.get(statement, []):
            value = item.get(line)
            if (item.get("period") == spec["period"] and isinstance(value, (int, float))
                    and abs(value - spec["provider_value"]) <= 0.005 * abs(spec["provider_value"])):
                flags.append({"kind": "provider_definition_break", "ticker": ticker, "statement": statement,
                              "period": spec["period"], "line_item": line, "stored_value": value,
                              "source": item.get("line_sources", {}).get(line), **spec})
    return flags


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
                if provenance in LEGACY_SOURCES or not _valid_currency(row.get("currency")):
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
            if not _valid_currency(currency):
                issues.append({"kind": "missing_or_invalid_currency", **identity, "currency": currency,
                               "period_end": period_end.isoformat()})
                continue
            item = {"period": f"{fy:04d}Q{fq}" if fq else f"FY{fy:04d}", "period_end": period_end.isoformat(),
                    "currency": currency, "source": provider,
                    "fiscal_label_source": row.get("fiscal_label_source"),
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
        by_end: dict[tuple, list] = {}
        for row in clean[statement]:
            key = (row["period_end"], history._parse_period(row["period"])[1] is not None)
            by_end.setdefault(key, []).append(row)
        ambiguous = [group for group in by_end.values() if len({row["period"] for row in group}) > 1]
        rejected_periods = {row["period"] for group in ambiguous for row in group}
        for group in ambiguous:
            issues.append({"kind": "ambiguous_provider_period_end", "provider": provider, "symbol": symbol,
                "statement": statement, "period_end": group[0]["period_end"],
                "periods": [row["period"] for row in group],
                "line_items": sorted({line for row in group for line in LINES[statement] if line in row})})
        clean[statement] = [row for row in clean[statement] if row["period"] not in rejected_periods]
    return clean


def _fetch_financial_history(ticker: str, start: date, *, required_start: date | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Fetch older label evidence while judging fallback against required coverage."""
    coverage_start = required_start or start
    payloads: list[dict] = []
    issues: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    combined = {s: [] for s in LINES}
    for provider in get_data_service()._live_chain("financials"):
        name = str(getattr(provider, "name", type(provider).__name__))
        if name in LEGACY_SOURCES:
            issues.append({"kind": "provider_identity_unusable", "provider": name})
            continue
        # A verified price-series rename does not establish statement identity.
        # BNY historical financial endpoints mix fiscal labels after BK's rename.
        for symbol in symbol_variants(ticker):
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
            # A 401/402/403 under one spelling that the same provider answers
            # under another is a symbol-coverage refusal (BRK.B -> BRK-B), not
            # a plan loss. Only unresolved denials mean an entitlement gap.
            for issue in issues:
                if (issue.get("kind") == "provider_entitlement_denied" and issue.get("provider") == name
                        and issue.get("symbol") != symbol and not issue.get("resolved")):
                    issue.update(resolved=True, resolved_by_symbol=symbol)
            attempts.append({**attempt, "received": True, "periods_received": {s: len(clean[s]) for s in LINES}})
            payloads.append(clean)
            for s in LINES:
                combined[s].extend(clean[s])
            coverage = _coverage(combined, coverage_start, _today())
            if _complete(coverage):
                return payloads, issues, attempts
            issues.append({"kind": "provider_partial_coverage", **attempt, "coverage": _coverage(clean, coverage_start, _today())})
            break  # A valid alias resolved this security; next provider may add missing cadence.
    return payloads, issues, attempts


def _relabel_legacy_periods(db: Session, ticker: str, rows: list, payloads: list[dict], report: dict) -> None:
    """Move only provider-confirmed legacy identities, preserving every row.

    The whole destination graph is validated before temporary keys are used.
    Cycles/chains move atomically; a blocked destination blocks its dependents.
    Inferred provider labels never authorize a migration of existing facts.
    """
    labels: dict[tuple, set[tuple[str, str]]] = {}
    for payload in payloads:
        for statement, whitelist in LINES.items():
            for incoming in payload[statement]:
                if incoming.get("fiscal_label_source") != "provider":
                    continue
                end = history._coerce_date(incoming["period_end"])
                for line in whitelist:
                    if line in incoming:
                        labels.setdefault((statement, line, end), set()).add((incoming["period"], incoming["source"]))
    by_key: dict[tuple, list] = {}
    moves = {}
    for row in rows:
        fy, fq = history._parse_period(row.period)
        canonical = (f"{fy:04d}Q{fq}" if fq else f"FY{fy:04d}") if fy else row.period
        by_key.setdefault((canonical, row.statement, row.line_item), []).append(row)
        if row.source not in LEGACY_SOURCES or row.period_end is None:
            continue
        old_quarter = fq if fy else row.fiscal_quarter
        candidates = {(period, provider) for period, provider in labels.get((row.statement, row.line_item, row.period_end), set())
                      if (history._parse_period(period)[1] is not None) == (old_quarter is not None)}
        destinations = {period for period, _ in candidates}
        if len(destinations) > 1:
            report["issues"].append({"kind": "legacy_period_relabel_conflict", "ticker": ticker, "id": row.id,
                "statement": row.statement, "line_item": row.line_item, "period": row.period,
                "period_end": row.period_end.isoformat(), "reason": "ambiguous_provider_labels",
                "candidates": [{"period": p, "provider": s} for p, s in sorted(candidates)]})
        elif destinations and (new_period := next(iter(destinations))) != canonical:
            moves[row.id] = {"row": row, "destination": (new_period, row.statement, row.line_item),
                             "providers": sorted({provider for _, provider in candidates})}
    blocked = set()
    destination_ids: dict[tuple, list[int]] = {}
    for row_id, move in moves.items():
        destination_ids.setdefault(move["destination"], []).append(row_id)
    for row_ids in destination_ids.values():
        if len(row_ids) > 1:
            optional_nulls = [row_id for row_id in row_ids if moves[row_id]["row"].value is None
                              and moves[row_id]["row"].line_item != PRIMARY.get(moves[row_id]["row"].statement)]
            blocked.update(optional_nulls)
            populated = [row_id for row_id in row_ids if row_id not in optional_nulls]
            if len(populated) > 1:
                blocked.update(populated)
    changed = True
    while changed:
        changed = False
        for row_id, move in moves.items():
            if row_id in blocked:
                continue
            occupants = [r for r in by_key.get(move["destination"], []) if r.id != row_id]
            if any(r.id not in moves or r.id in blocked for r in occupants):
                blocked.add(row_id)
                changed = True
    for row_id in blocked:
        move = moves[row_id]
        row = move["row"]
        optional_null = row.value is None and row.line_item != PRIMARY.get(row.statement)
        report["issues"].append({"kind": "legacy_period_relabel_optional_null" if optional_null else "legacy_period_relabel_conflict",
            "ticker": ticker, "id": row.id, "statement": row.statement, "line_item": row.line_item,
            "period": row.period, "period_end": row.period_end.isoformat(), "destination_period": move["destination"][0],
            "reason": "destination_collision", "source": row.source,
            "destination_rows": [{"id": r.id, "period": r.period, "period_end": str(r.period_end), "source": r.source}
                                 for r in by_key.get(move["destination"], [])]})
    approved = [move for row_id, move in moves.items() if row_id not in blocked]
    if not approved:
        return
    audits = []
    for move in approved:
        row = move["row"]
        fy, fq = history._parse_period(move["destination"][0])
        audits.append({"ticker": ticker, "id": row.id, "statement": row.statement, "line_item": row.line_item,
            "old_period": row.period, "new_period": move["destination"][0],
            "old_fiscal_year": row.fiscal_year, "new_fiscal_year": fy,
            "old_fiscal_quarter": row.fiscal_quarter, "new_fiscal_quarter": fq,
            "old_source": row.source, "new_source": row.source,
            "old_value": row.value if row.value is None or math.isfinite(row.value) else str(row.value),
            "new_value": row.value if row.value is None or math.isfinite(row.value) else str(row.value),
            "old_period_end": row.period_end.isoformat(), "new_period_end": row.period_end.isoformat(),
            "currency": row.currency, "available_at_preserved": row.available_at.isoformat() if row.available_at else None,
            "label_providers": move["providers"]})
    # SQLite defers its physical BEGIN until a write; a SAVEPOINT after only
    # SELECTs could otherwise become a standalone transaction and commit on
    # release, escaping the outer backfill rollback.
    connection = db.connection()
    if connection.dialect.name == "sqlite" and not connection.connection.dbapi_connection.in_transaction:
        connection.exec_driver_sql("BEGIN")
    with db.begin_nested():
        for move in approved:
            move["row"].period = "~" + uuid4().hex[:15]
        db.flush()
        for move in approved:
            row = move["row"]
            row.period = move["destination"][0]
            row.fiscal_year, row.fiscal_quarter = history._parse_period(row.period)
        db.flush()
    report["period_relabels"].extend(audits)
    report["rows_relabelled"] += len(approved)


def _primary_facts(payloads: list[dict]) -> dict[tuple[str, str, str], dict]:
    facts: dict[tuple[str, str, str], dict] = {}
    for payload in payloads:
        for statement, whitelist in LINES.items():
            for row in payload[statement]:
                if row.get("source") != PRIMARY_PROVIDER:
                    continue
                end = history._coerce_date(row["period_end"])
                quarterly = history._parse_period(row["period"])[1] is not None
                for line in whitelist:
                    if line in row:
                        facts[(row["period"], statement, line)] = {
                            "period": row["period"], "period_end": end, "value": row[line],
                            "currency": row["currency"], "quarterly": quarterly}
    return facts


def _fact_evidence(fact: dict | None) -> dict | None:
    if fact is None:
        return None
    return {**fact, "period_end": fact["period_end"].isoformat() if fact["period_end"] else None}


def _takeover_reason(row: FinancialPeriod, facts: dict, observations: dict, period_ends: dict) -> tuple[str | None, dict | None]:
    """First matching reason to move a non-primary row aside, with the primary fact it contradicts."""
    legacy = row.source in LEGACY_SOURCES
    key = (row.period, row.statement, row.line_item)
    canonical = _canonical(row.period)
    fy, fq = history._parse_period(row.period)
    quarterly = (fq is not None) if fy else row.fiscal_quarter is not None
    fact = facts.get(key)
    if fact is None and canonical != row.period and (canonical, row.statement, row.line_item) in facts:
        return "alias_superseded_by_primary", facts[(canonical, row.statement, row.line_item)]
    if fact is not None:
        if row.period_end is not None and row.period_end != fact["period_end"]:
            return "period_end_conflicts_with_primary", fact
        if row.value is not None and math.isfinite(row.value) and row.value != fact["value"]:
            return "value_conflicts_with_primary", fact
        if not legacy and row.currency != fact["currency"]:
            return "currency_conflicts_with_primary", fact
        if row.available_at is not None and fact["period_end"] and row.available_at < fact["period_end"]:
            return "invalid_availability_precedes_period_end", fact
        return None, None  # Equal: adopted (named) or upgraded in place (legacy) by the write loop.
    if row.period_end is None:
        return None, None
    near = [k for end, k in observations.get((row.statement, row.line_item, quarterly), ()) if _near(end, row.period_end)]
    if near:
        return "label_conflicts_with_primary_observation", facts[sorted(near)[0]]
    if not legacy and any(_near(end, row.period_end) for end in period_ends.get((row.statement, quarterly), ())):
        return "secondary_row_in_primary_period", None
    # Legacy lines FMP does not report came from the same FMP-first chain;
    # missing never erases, so they stay.
    return None, None


def _primary_takeover(db: Session, ticker: str, rows: list, payloads: list[dict], report: dict, *,
                      mode: TakeoverMode, pre_images: dict[int, dict], expected: dict[int, str] | None,
                      audit_key: str | None) -> tuple[dict[tuple, tuple], set[tuple]]:
    """Quarantine (or plan) every stored row that contradicts an FMP observation.

    Returns `(available_override, blocked_keys)`. `rows` is updated in place so
    moved rows can no longer act as existing occupants. A primary row whose
    stored availability precedes its period end (LULU 35607/35609) is flagged,
    not moved or re-inserted: the PIT exclusion documented in
    docs/ops/scorecard-pit-eligibility.md must keep holding.
    """
    from . import fundamental_quarantine as quarantine

    facts = _primary_facts(payloads)
    if not facts:
        if expected:
            raise PlanMismatch(expected, {})
        return {}, set()
    observations: dict[tuple, list] = {}
    period_ends: dict[tuple, set] = {}
    for key, observed in facts.items():
        observations.setdefault((key[1], key[2], observed["quarterly"]), []).append((observed["period_end"], key))
        period_ends.setdefault((key[1], observed["quarterly"]), set()).add(observed["period_end"])
    actions = []
    fact: dict | None
    for row in rows:
        if row.source == PRIMARY_PROVIDER:
            fact = facts.get((row.period, row.statement, row.line_item))
            if fact and row.available_at is not None and fact["period_end"] and row.available_at < fact["period_end"]:
                report["issues"].append({"kind": "primary_availability_precedes_period_end", **_row_identity(row),
                    "available_at": row.available_at.isoformat(), "value": _json_value(row.value),
                    "action": "flagged_not_moved", "reason": "pit_exclusion_preserved"})
            continue
        reason, fact = _takeover_reason(row, facts, observations, period_ends)
        if reason:
            before = pre_images.get(row.id)
            actions.append({"row": row, "reason": reason, "primary_fact": _fact_evidence(fact),
                            "pre_run": before if before is not None and before != quarantine._snapshot(row) else None,
                            "fact": fact})
    actual = {a["row"].id: a["reason"] for a in actions}
    if expected is not None and actual != expected:
        raise PlanMismatch(expected, actual)
    if not actions:
        return {}, set()
    for action in actions:
        row, fact = action["row"], action["fact"]
        report["quarantined"].append({"id": row.id, "reason": action["reason"], "statement": row.statement,
            "period": row.period, "line_item": row.line_item,
            "period_end": row.period_end.isoformat() if row.period_end else None, "source": row.source,
            "value": _json_value(row.value),
            "primary_period": fact["period"] if fact else None,
            "primary_period_end": fact["period_end"].isoformat() if fact and fact["period_end"] else None,
            "primary_value": _json_value(fact["value"]) if fact else None})
    unsafe = [a["reason"] for a in actions if a["reason"] not in SAFE_AUTO_REASONS]
    too_many = len(actions) > QUARANTINE_AUTO_MAX_ROWS or len(actions) > QUARANTINE_AUTO_MAX_SHARE * max(len(rows), 1)
    repair_id = quarantine.repair_id_for(audit_key, quarantine.QUARANTINE_KIND, ticker)
    if mode == "unattended" and (unsafe or too_many):
        planned = quarantine.quarantine_rows(db, ticker=ticker, actions=actions, status="planned", mode=mode,
                                             repair_id=repair_id)
        blocked = set()
        for action in actions:
            row, fact = action["row"], action["fact"]
            blocked.add((row.period, row.statement, row.line_item))
            if fact:
                blocked.add((fact["period"], row.statement, row.line_item))
        report["planned_repair_id"] = planned["repair_id"]
        report["issues"].append({"kind": "primary_takeover_planned", "ticker": ticker,
            "repair_id": planned["repair_id"], "rows": len(actions), "counts": planned["counts"],
            "reason": "unsafe_reasons" if unsafe else "exceeds_unattended_limits",
            "limits": {"max_rows": QUARANTINE_AUTO_MAX_ROWS, "max_share": QUARANTINE_AUTO_MAX_SHARE,
                       "safe_reasons": sorted(SAFE_AUTO_REASONS)}})
        return {}, blocked
    override = {}
    for action in actions:
        row, fact = action["row"], action["fact"]
        # INSERT-only availability: the replacement at the same key keeps the
        # observation's original availability when that date is possible
        # (not before the period end). Nothing stored is re-dated.
        if (fact and row.period == fact["period"] and row.available_at is not None and fact["period_end"]
                and row.available_at >= fact["period_end"]):
            override[(row.period, row.statement, row.line_item)] = (row.available_at, row.available_at_source)
    moved = {id(a["row"]) for a in actions}  # object identity: moved rows are expired
    applied = quarantine.quarantine_rows(db, ticker=ticker, actions=actions, mode=mode, repair_id=repair_id)
    rows[:] = [r for r in rows if id(r) not in moved]
    report["rows_quarantined"] = len(actions)
    report["quarantine_repair_id"] = applied["repair_id"]
    report["quarantine_namespace"] = applied["namespace"]
    return override, set()


def backfill_fundamentals(ticker: str, start_date: date, force_refresh: bool = False, *, db: Session | None = None,
                          dry_run: bool = False, mode: TakeoverMode = "supervised",
                          expected_quarantine: dict[int, str] | None = None, audit_key: str | None = None) -> dict:
    """Populate durable fundamentals only, with explicit complete/partial coverage.

    With an external session, writes are flushed but its caller owns commit and
    the report says committed=False. Missing/nonfinite values never erase data;
    conflicting values from a different non-primary provider are retained and
    reported. FMP-primary takeover rules are in the module docstring.

    `dry_run` performs the whole run inside a transaction and rolls it back, so
    the report is the exact plan (quarantines, adoptions, restatements) and
    nothing persists; it needs an owned session. `mode="unattended"` (scheduled
    refreshes) quarantines only small safe sets and otherwise records a planned
    repair. `expected_quarantine` ({row id: reason} from a reviewed dry run)
    aborts the whole ticker transaction when the executed set differs.
    """
    from . import fundamental_quarantine as quarantine

    ticker = ticker.strip().upper()
    start = history._coerce_date(start_date)
    if not ticker or len(ticker) > 16 or start is None or start > _today():
        raise ValueError("valid ticker and start_date on or before today required")
    if mode not in ("supervised", "unattended"):
        raise ValueError("mode must be supervised or unattended")
    if dry_run and db is not None:
        raise ValueError("dry_run needs an owned session so it can roll back")
    own = db is None
    db = db or SessionLocal()
    report: dict[str, Any] = {"ticker": ticker, "requested_start": start.isoformat(), "requested_end": _today().isoformat(),
                              "rows_written": 0, "periods_received": {s: 0 for s in LINES}, "provider": [], "attempts": [], "issues": [],
                              "source_upgrades": [], "refresh_complete": False, "rows_refreshed": 0,
                              "period_relabels": [], "rows_relabelled": 0,
                              "point_in_time": "latest_restated_values_at_original_availability", "committed": False,
                              "dry_run": dry_run, "takeover_mode": mode, "rows_quarantined": 0,
                              "quarantine_repair_id": None, "quarantined": [], "adoptions": [], "restatements": [],
                              "adoption_repair_id": None, "restatement_repair_id": None, "planned_repair_id": None,
                              "secondary_rows_skipped": 0, "rows_inserted": 0,
                              "new_period_ends": {s: {"annual": [], "quarterly": []} for s in LINES}}
    try:
        FinancialPeriod.__table__.create(bind=db.get_bind(), checkfirst=True)
        # DDL before any write: SQLite would otherwise lock against the open
        # write transaction when the first audit row is inserted.
        FinancialDataRepair.__table__.create(bind=db.get_bind(), checkfirst=True)
        stored = read_stored_financials(ticker, db=db)
        report["issues"].extend(stored.get("_history_issues", []))
        before = _coverage(stored, start, _today(), issues=report["issues"])
        if not force_refresh and _complete(before) and _fresh(before) and not _has_blockers(report["issues"]):
            report.update(coverage=before, success=True, source="database", committed=own and not dry_run)
            return report
        # The provider request reaches back to the oldest stored observation
        # that FMP might contradict (legacy and named secondaries alike), so
        # the takeover sees FMP's version of every such period.
        existing_dates = list(db.execute(select(FinancialPeriod.period_end).where(
            FinancialPeriod.ticker == ticker, FinancialPeriod.source != PRIMARY_PROVIDER,
            FinancialPeriod.period_end.is_not(None), FinancialPeriod.period_end <= _today(),
        )).scalars())
        fetch_start = min([start, *existing_dates])
        report["provider_requested_start"] = fetch_start.isoformat()
        # End the owned read transaction before spending time on provider IO.
        if own:
            db.rollback()
        payloads, issues, attempts = _fetch_financial_history(ticker, fetch_start, required_start=start)
        report["issues"].extend(issues)
        report["attempts"] = attempts
        incoming = {s: [row for payload in payloads for row in payload[s]] for s in LINES}
        report["refresh_complete"] = bool(payloads) and _complete(_coverage(incoming, start, _today()))
        if force_refresh and not report["refresh_complete"]:
            report["issues"].append({"kind": "refresh_incomplete", "ticker": ticker,
                                     "coverage": _coverage(incoming, start, _today())})
        existing = {}
        aliases: dict[tuple[str, str, str], list] = {}
        existing_rows = list(db.execute(select(FinancialPeriod).where(FinancialPeriod.ticker == ticker)).scalars())
        pre_images = {r.id: quarantine._snapshot(r) for r in existing_rows}
        _relabel_legacy_periods(db, ticker, existing_rows, payloads, report)
        available_override, blocked_keys = _primary_takeover(
            db, ticker, existing_rows, payloads, report, mode=mode, pre_images=pre_images,
            expected=expected_quarantine, audit_key=audit_key)
        quarantined_ids = {q["id"] for q in report["quarantined"]} if report["rows_quarantined"] else set()
        for r in existing_rows:
            aliases.setdefault((_canonical(r.period), r.statement, r.line_item), []).append(r)
        incoming_keys = {(r["period"], s, line) for p in payloads for s in LINES
                         for r in p[s] for line in LINES[s] if line in r}
        for key, group in aliases.items():
            if key in incoming_keys:
                # Different raw keys remain untouched. A canonical provider
                # fact can be stored independently of unusable legacy aliases;
                # read-time warnings become nonblocking only after it exists.
                usable = [r for r in group if not _unusable_legacy_alias(r, key[0])]
                if len(group) > 1 and any(r.period == key[0] and not _stored_period_valid(r) for r in usable):
                    usable = group  # An exact-key occupant is not an excluded alias.
                group = usable
            if len(group) > 1:
                blocked_keys.add(key)
                report["issues"].append({"kind": "stored_period_alias_conflict", "ticker": ticker,
                    "period": key[0], "statement": key[1], "line_item": key[2],
                    "rows": [{"id": r.id, "period": r.period, "period_end": str(r.period_end),
                              "currency": r.currency, "source": r.source} for r in group]})
            if group:
                existing[key] = group[0]
        currencies: dict[tuple[str, str], set[str]] = {}
        for (period, statement, _), row in existing.items():
            if row.currency and row.source not in LEGACY_SOURCES:
                currencies.setdefault((period, statement), set()).add(row.currency)
        # Periods FMP reports (this payload or stored FMP rows) belong to FMP:
        # a secondary row for one of them is skipped whole, never merged in.
        primary_cover: dict[tuple[str, bool], list[tuple[date | None, dict]]] = {}
        for payload in payloads:
            for statement in LINES:
                for row in payload[statement]:
                    if row["source"] == PRIMARY_PROVIDER:
                        primary_cover.setdefault((statement, history._parse_period(row["period"])[1] is not None), []).append(
                            (history._coerce_date(row["period_end"]), {k: v for k, v in row.items() if k in LINES[statement]}))
        stored_primary: dict[tuple, dict] = {}
        for r in existing_rows:
            if r.source == PRIMARY_PROVIDER and r.period_end is not None:
                stored_primary.setdefault((r.statement, _canonical(r.period), r.period_end), {})[r.line_item] = r.value
        for (statement, period, end), lines in stored_primary.items():
            primary_cover.setdefault((statement, history._parse_period(period)[1] is not None), []).append((end, lines))
        adoption_audit: list[tuple[FinancialPeriod, dict]] = []
        restatement_audit: list[tuple[FinancialPeriod, dict]] = []
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
                    if source != PRIMARY_PROVIDER:
                        cover = next((lines for cover_end, lines in primary_cover.get((statement, fq is not None), ())
                                      if _near(cover_end, end)), None)
                        if cover is not None:
                            report["secondary_rows_skipped"] += 1
                            report["issues"].extend({"kind": "secondary_disagrees_with_primary", "ticker": ticker,
                                "statement": statement, "period": period, "period_end": row["period_end"],
                                "line_item": line, "provider": source, "value": row[line],
                                "primary_value": _json_value(cover[line])}
                                for line in whitelist if line in row and cover.get(line) is not None and cover[line] != row[line])
                            continue
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
                                if source == PRIMARY_PROVIDER:
                                    # Equal FMP verification adopts the observation;
                                    # its id and availability are kept.
                                    before_image = quarantine._snapshot(prior)
                                    report["adoptions"].append({"id": prior.id, "statement": statement, "period": period,
                                        "line_item": line, "old_source": prior.source, "new_source": source})
                                    report["source_upgrades"].append({"ticker": ticker, "id": prior.id, "statement": statement,
                                        "period": period, "period_end": str(end), "line_item": line, "old_source": prior.source,
                                        "new_source": source, "old_value": prior.value, "new_value": row[line],
                                        "old_currency": prior.currency, "new_currency": row["currency"], "reason": "primary_adoption"})
                                    prior.source = source
                                    prior.fetched_at = now
                                    adoption_audit.append((prior, before_image))
                                    report["rows_written"] += 1
                                continue  # Corroboration does not transfer ownership between secondaries.
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
                        prior_image = quarantine._snapshot(prior) if prior is not None else None
                        old_value = prior.value if prior is not None else None
                        write_available = available_override.get(key, (available, rule)) if prior is None else (available, rule)
                        if history._upsert_financial_period(db, ticker=ticker, period=period, statement=statement,
                            line_item=line, value=row[line], period_end=end, fiscal_year=fy, fiscal_quarter=fq,
                            source=source, currency=row["currency"], fetched_at=now, available_at=write_available[0],
                            available_at_source=write_available[1], existing_rows=existing, refusals=report["issues"]):
                            report["rows_written"] += 1
                            if upgrade:
                                report["source_upgrades"].append(upgrade)
                                if source == PRIMARY_PROVIDER and prior is not None and prior_image is not None:
                                    # A legacy upgrade is an adoption: equal value,
                                    # a filled NULL, or a currency-only correction.
                                    adoption_audit.append((prior, prior_image))
                                    report["adoptions"].append({"id": prior.id, "statement": statement, "period": period,
                                        "line_item": line, "old_source": upgrade["old_source"], "new_source": source})
                            elif prior is not None and prior.source == source and old_value != row[line] and prior_image is not None:
                                restatement_audit.append((prior, prior_image))
                                report["restatements"].append({"id": prior.id, "statement": statement, "period": period,
                                    "line_item": line, "old_value": _json_value(old_value), "new_value": row[line], "source": source})
                            if prior is None:
                                report["rows_inserted"] += 1
                                if line == PRIMARY[statement]:
                                    report["new_period_ends"][statement]["quarterly" if fq else "annual"].append(
                                        end.isoformat() if end else None)
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
        audit_extra = {"dry_run": True} if dry_run else None
        adoption = quarantine.record_in_place(db, ticker=ticker, kind=quarantine.ADOPTION_KIND, entries=adoption_audit,
            repair_id=quarantine.repair_id_for(audit_key, quarantine.ADOPTION_KIND, ticker), extra=audit_extra)
        restatement = quarantine.record_in_place(db, ticker=ticker, kind=quarantine.RESTATEMENT_KIND, entries=restatement_audit,
            repair_id=quarantine.repair_id_for(audit_key, quarantine.RESTATEMENT_KIND, ticker), extra=audit_extra)
        report["adoption_repair_id"] = adoption["repair_id"] if adoption else None
        report["restatement_repair_id"] = restatement["repair_id"] if restatement else None
        if report["quarantine_repair_id"]:
            audit = db.get(FinancialDataRepair, report["quarantine_repair_id"])
            if audit is not None:
                audit.result = {"rows_quarantined": report["rows_quarantined"], "replacement_ids": {
                    str(q["id"]): getattr(existing.get((q["primary_period"], q["statement"], q["line_item"])), "id", None)
                    for q in report["quarantined"]}}
                db.flush()
        after = read_stored_financials(ticker, db=db)
        preserved_duplicates = {i["id"]: i for i in after.get("_history_issues", [])
                                if i["kind"] == "legacy_duplicate_observation_excluded"}
        for issue in report["issues"]:
            if (issue.get("id") in quarantined_ids and not issue.get("resolved")
                    and issue.get("kind") != "secondary_disagrees_with_primary"):
                # A relabel blocked by, or a stored read issue about, a row
                # that is now quarantined no longer describes stored data.
                issue.update(resolved=True, resolution="quarantined_superseded_by_primary")
                continue
            # A blocked relabel need not make usable history fail when the
            # exact preserved row has a unique authoritative replacement.
            # Ambiguous labels and all other collisions stay blocking.
            if (issue["kind"] == "legacy_period_relabel_conflict"
                    and issue.get("reason") == "destination_collision"
                    and issue.get("id") in preserved_duplicates):
                evidence = preserved_duplicates[issue["id"]]
                issue.update(kind="legacy_period_relabel_preserved_duplicate",
                             original_kind="legacy_period_relabel_conflict",
                             preserved_observation={k: v for k, v in evidence.items()
                                                    if k not in {"kind", "reason", "replacement"}},
                             replacement=evidence["replacement"])
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
        if dry_run:
            # The report is the plan; nothing it describes may persist.
            db.rollback()
        elif own:
            db.commit()
            report["committed"] = True
        if report["issues"]:
            log.warning("fundamentals history %s: %d issues: %s", ticker, len(report["issues"]), report["issues"])
        return report
    except Exception as exc:
        if own:
            db.rollback()
        log.warning("fundamentals history %s failed: %s", ticker, type(exc).__name__)
        report.update(success=False, rows_written=0, rows_refreshed=0, rows_relabelled=0, rows_quarantined=0,
                      rows_inserted=0, quarantine_repair_id=None, adoption_repair_id=None, restatement_repair_id=None,
                      planned_repair_id=None)
        if isinstance(exc, PlanMismatch):
            report["issues"].append({"kind": "repull_plan_mismatch", "ticker": ticker,
                "expected": {str(k): v for k, v in sorted(exc.expected.items())},
                "actual": {str(k): v for k, v in sorted(exc.actual.items())}})
        else:
            report["issues"].append({"kind": "persistence_or_read_error", "error_type": type(exc).__name__})
        return report
    finally:
        if own:
            db.close()
