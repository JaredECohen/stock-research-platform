"""W6 / FIX-007 — which memo snapshots count toward the track record.

Owner decision (2026-09-24): the Track Record stays visible but provisional,
and outcomes on demo-mode memos copied from the development laptop on
2026-05-04 are EXCLUDED, never deleted. FIX-003 adds the migrated test-fixture
snapshots from the same copy. This module is the one place that decides
eligibility, and `eligible_only` is the one predicate every consumer uses
(contract C5: track record, calibration, attribution, regime accuracy,
specialist reliability, postmortem selection, and W7's learning readers).

Design (`.claude/memory/proposals/design-w6-track-record.md` plus the accepted
critique in the integration plan, slice S3):

* **Persisted, not derived.** Deriving at read time means reading
  `memo_json->>'generation_mode'`, which de-TOASTs a ~95 KB body per snapshot
  on every page load of a browser-called endpoint on the web process. A ledger
  row is also visible to both Render processes (web serves the page, the
  worker runs the loops), which a module-level cache would not be.
* **Fail closed.** A snapshot with no ledger row is "unclassified" and NOT
  eligible. Every reason is named and counted; nothing is silently dropped.
* **Exclusion only.** Classification writes `memo_outcome_eligibility` and
  nothing else. `memo_snapshots` stay byte-identical (they are documented as
  immutable), and no outcome or postmortem row is modified or deleted.
* **Versioned.** `RULE_VERSION` makes a rule change a reclassification sweep,
  not a migration.
* **No bodies in Python.** The sweep projects a handful of JSON paths as
  strings (`->>` on Postgres, `JSON_EXTRACT` on sqlite). It never loads
  `memo_json`.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NamedTuple, TypeVar

from sqlalchemy import ColumnElement, Select, and_, exists, or_, select
from sqlalchemy.orm import Session, aliased

from ..models import MemoOutcomeEligibility, MemoSnapshot
from . import outcome_eligibility_evidence as evidence

log = logging.getLogger(__name__)

RULE_VERSION = 1

# --- Evidence boundaries (FIX-003 provenance write-up, W6 design §2) --------
#
# The 2026-05-04 sqlite→postgres copy kept primary keys. No copied snapshot id
# exceeds 582, and the first production-native snapshot (id 17, a reused gap)
# was written 2026-05-12T23:53Z.
DEV_COPY_MAX_SNAPSHOT_ID = 582
DEV_COPY_BEFORE = datetime(2026, 5, 12)
# 7961289 (PR #40) reached main at 2026-06-15T00:59:06Z. Before it, production
# labelled live-data memos "demo" (Theme 3; graph.py's generation_mode comment),
# so a demo label from before this instant is not evidence of demo data.
LABEL_TRUSTWORTHY_FROM = datetime(2026, 6, 15, 1, 0, 0)

# FIX-003: symbols the test suite wrote into the laptop database that the
# copy carried to production. Exact names, never a pattern: a pattern would
# one day match a real listing. Only rows the copy could have carried count
# (id <= 582, written before 2026-05-05); a later same-ticker snapshot is
# classified by the ordinary rules. Expected ids: fix003-provenance-and-row-plan
# §2 (223, 224, 225, 239, 411-413, 516-523, 525, 554-556, 562, 563, 565-571,
# 577-582). A fixture-ticker row with as_of_date set (ASOFT1 v2, and the §4.0
# candidates 524/564) is classified by the earlier backtest rule instead.
FIXTURE_TICKERS = frozenset({
    "TSTONE", "TSTONE2", "TSTAA", "TSTBB", "TSTNEU", "TSTRFL", "TSTADM",
    "TSTSHORT", "TSTHALF", "TSTPATCH", "TSTNOMA", "TSTCAP", "RTRIP", "CHAIN",
    "AUDA", "AUDB", "AUDC", "AUDN0", "AUDN1", "AUDN2", "AUDN3", "AUDN4", "ASOFT1",
})
FIXTURE_MAX_SNAPSHOT_ID = 582
FIXTURE_BEFORE = datetime(2026, 5, 5)

# --- Reasons (String(48)) ----------------------------------------------------
REASON_LIVE = "live_generation"                          # eligible
# Eligible, but NAMED apart from production live memos: the owner's default
# for the ~20 live-mode memos the 2026-05-04 copy carried from the laptop is
# "eligible, and disclosed" (integration plan §owner questions 3c; design-w6
# §8 Q3). About 42% of the post-exclusion record, all keyword-PM rated, so a
# lump `live_generation` count would hide where the record comes from.
REASON_LIVE_DEV_COPY = "live_dev_copy_2026_05_04"        # eligible, listed and disclosed
REASON_LABEL_PREDATES_FIX = "demo_label_predates_fix"    # eligible, listed in receipts
REASON_TEST_FIXTURE = "test_fixture_migrated_2026_05_04"  # excluded (FIX-003)
REASON_BACKTEST = "backtest"                             # excluded (as_of_date set)
REASON_PATCH_PARENT_MISSING = "patch_parent_missing"     # excluded (lineage unknown)
REASON_UNRECORDED = "generation_mode_unrecorded"         # excluded (fail closed)
REASON_UNRECOGNIZED = "generation_mode_unrecognized"     # excluded (fail closed)
REASON_DEV_COPY = "demo_dev_copy_2026_05_04"             # excluded (owner decision)
REASON_NO_LLM_OR_DEMO = "no_llm_or_demo_generation"      # excluded (demo data or no LLM)

ELIGIBLE_REASONS = frozenset({REASON_LIVE, REASON_LIVE_DEV_COPY, REASON_LABEL_PREDATES_FIX})
ALL_REASONS = frozenset({
    REASON_LIVE, REASON_LIVE_DEV_COPY, REASON_LABEL_PREDATES_FIX, REASON_TEST_FIXTURE, REASON_BACKTEST,
    REASON_PATCH_PARENT_MISSING, REASON_UNRECORDED, REASON_UNRECOGNIZED,
    REASON_DEV_COPY, REASON_NO_LLM_OR_DEMO,
})

# --- Rating source (String(16)) ------------------------------------------------
# Who produced the rating label. The W3 skew diagnosis found the eligible set
# is mostly keyword-PM and patch calls, so the page must not present it as a
# measure of the LLM committee.
SOURCE_LLM_PM = "llm_pm"
SOURCE_KEYWORD_PM = "keyword_pm"
SOURCE_FALLBACK_PM = "fallback_pm"
SOURCE_PATCH = "patch"
SOURCE_UNKNOWN = "unknown"
RATING_SOURCES = (SOURCE_LLM_PM, SOURCE_KEYWORD_PM, SOURCE_FALLBACK_PM, SOURCE_PATCH, SOURCE_UNKNOWN)

# The deterministic keyword PM's `final_pm_view` literal (graph.py
# `_pm_synthesis`, stable since bc34765 on 2026-04-28), and the `safe_call`
# fallback literal written when `_pm_synthesis` raises (graph.py
# `synth_fallback`). Substrings, because later steps append to the view.
KEYWORD_PM_MARKER = "valuation-relative read is the main swing factor"
FALLBACK_PM_MARKER = "PM synthesis unavailable; relying on specialist findings only."
PM_DEGRADED_AGENT = "PM Synthesis"

PATCH_TRIGGER = "incremental_patch"


class Classification(NamedTuple):
    eligible: bool
    reason: str
    inherited_from_snapshot_id: int | None = None


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------

def parse_timestamp(value: Any) -> datetime | None:
    """Naive UTC datetime from a stored value, or None when it is unusable.

    `memo_json["generated_at"]` is pydantic's ISO string; offsets are
    converted, not stripped, so a "+02:00" stamp is not read two hours late.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def classify_snapshot(
    *,
    snapshot_id: int,
    ticker: str,
    snapshot_generated_at: datetime | None,
    analysis_generated_at: datetime | None,
    generation_mode: str | None,
    trigger: str | None,
    as_of_date: datetime | None,
    parent_classification: Classification | None = None,
    parent_snapshot_id: int | None = None,
) -> Classification:
    """Rule v1. First match wins; no I/O.

    ``parent_classification`` is only read for an ``incremental_patch``; None
    there means the parent snapshot does not exist. A patch copies its
    parent's body (including ``generation_mode`` and ``generated_at``) and
    edits a few fields, so its own label says nothing new about its data.
    """
    # Backtests first, as the integration plan orders the rule. Excluded
    # either way, but the order decides the NAME, and `_check_expected` only
    # accepts the 34 enumerated fixture ids under the fixture name: the copy
    # also carried fixture-ticker backtests that never reached any log (ASOFT1
    # v2 at 526 per test_as_of_date.py; FIX-003 §4.0 candidates 524, 564), so
    # naming those as fixtures would abort the first production sweep.
    if as_of_date is not None:
        return Classification(False, REASON_BACKTEST)
    # FIX-003 next, ahead of the patch and demo rules: a migrated test fixture
    # is not research in any mode, and naming it is more useful to the
    # receipt than "unrecorded" or an inherited reason.
    if (
        (ticker or "").upper() in FIXTURE_TICKERS
        and snapshot_id <= FIXTURE_MAX_SNAPSHOT_ID
        and snapshot_generated_at is not None
        and snapshot_generated_at < FIXTURE_BEFORE
    ):
        return Classification(False, REASON_TEST_FIXTURE)
    if (trigger or "") == PATCH_TRIGGER:
        if parent_classification is None:
            return Classification(False, REASON_PATCH_PARENT_MISSING)
        return Classification(
            parent_classification.eligible, parent_classification.reason, parent_snapshot_id,
        )
    mode = (generation_mode or "").strip().lower()
    if not mode:
        return Classification(False, REASON_UNRECORDED)
    if mode == "live":
        # Same copy boundary as the demo dev-copy rule below; only the name
        # differs, eligibility does not.
        if (
            analysis_generated_at is not None
            and snapshot_id <= DEV_COPY_MAX_SNAPSHOT_ID
            and analysis_generated_at < DEV_COPY_BEFORE
        ):
            return Classification(True, REASON_LIVE_DEV_COPY)
        return Classification(True, REASON_LIVE)
    if mode != "demo":
        return Classification(False, REASON_UNRECOGNIZED)
    if analysis_generated_at is None:
        # Cannot place it relative to either boundary; fail closed.
        return Classification(False, REASON_NO_LLM_OR_DEMO)
    if snapshot_id <= DEV_COPY_MAX_SNAPSHOT_ID and analysis_generated_at < DEV_COPY_BEFORE:
        return Classification(False, REASON_DEV_COPY)
    if analysis_generated_at < LABEL_TRUSTWORTHY_FROM:
        return Classification(True, REASON_LABEL_PREDATES_FIX)
    # After the label fix, "demo" means demo data or no LLM credentials (the
    # worker-without-keys memos: AAPL v7, AMZN v2, AVGO v1, GOOGL v7).
    return Classification(False, REASON_NO_LLM_OR_DEMO)


def rating_source(
    *, trigger: str | None, generation_mode: str | None,
    final_pm_view: str | None, degraded_agents: str | None,
) -> str:
    """Who produced the rating, from strings the sweep projected.

    ``degraded_agents`` is the JSON text of the list (both dialects return an
    array as its JSON text), checked by membership after decoding.
    """
    if (trigger or "") == PATCH_TRIGGER:
        return SOURCE_PATCH
    view = final_pm_view or ""
    if FALLBACK_PM_MARKER in view:
        return SOURCE_FALLBACK_PM
    degraded: list[Any] = []
    if degraded_agents:
        try:
            decoded = json.loads(degraded_agents)
        except ValueError:
            decoded = []
        if isinstance(decoded, list):
            degraded = decoded
    mode = (generation_mode or "").strip().lower()
    if KEYWORD_PM_MARKER in view or PM_DEGRADED_AGENT in degraded or mode == "demo":
        return SOURCE_KEYWORD_PM
    if view.strip() and mode == "live":
        return SOURCE_LLM_PM
    return SOURCE_UNKNOWN


# ---------------------------------------------------------------------------
# Shared predicates (contract C5)
# ---------------------------------------------------------------------------

def identity_matches(ledger: Any = MemoOutcomeEligibility, snapshot: Any = MemoSnapshot) -> ColumnElement[bool]:
    """The ledger row belongs to THIS snapshot, not to a reused id."""
    return and_(
        ledger.memo_snapshot_id == snapshot.id,
        ledger.ticker == snapshot.ticker,
        ledger.snapshot_generated_at == snapshot.generated_at,
    )


_S = TypeVar("_S", bound=Select[Any])


def eligible_exists(snapshot_id_col: Any) -> ColumnElement[bool]:
    """EXISTS an identity-valid, eligible ledger row for ``snapshot_id_col``.

    A correlated EXISTS rather than a join, so it neither multiplies rows nor
    collides with a ``MemoSnapshot`` the caller already joined: the snapshot
    it checks identity against is a private alias. Negate it to count the
    excluded-or-unclassified complement.
    """
    snap = aliased(MemoSnapshot)
    led = MemoOutcomeEligibility
    return exists(
        select(led.memo_snapshot_id)
        .join(snap, identity_matches(led, snap))
        .where(led.memo_snapshot_id == snapshot_id_col, led.eligible.is_(True))
        .correlate_except(led, snap)
    )


def eligible_only(stmt: _S, snapshot_id_col: Any) -> _S:
    """The ONE predicate every track-record and learning consumer uses (C5).

    ``snapshot_id_col`` is the consumer's ``memo_snapshot_id`` (outcomes,
    postmortems) or ``MemoSnapshot.id``. Unclassified counts as ineligible.
    """
    return stmt.where(eligible_exists(snapshot_id_col))


def ensure_table(db: Session) -> None:
    MemoSnapshot.__table__.create(bind=db.get_bind(), checkfirst=True)  # type: ignore[attr-defined]
    MemoOutcomeEligibility.__table__.create(bind=db.get_bind(), checkfirst=True)  # type: ignore[attr-defined]


def lookup(db: Session, snapshot_id: int) -> Classification | None:
    """The current, identity-valid classification of one snapshot."""
    row = db.execute(
        select(MemoOutcomeEligibility.eligible, MemoOutcomeEligibility.reason,
               MemoOutcomeEligibility.inherited_from_snapshot_id)
        .join(MemoSnapshot, identity_matches())
        .where(MemoOutcomeEligibility.memo_snapshot_id == snapshot_id)
    ).first()
    return Classification(bool(row[0]), row[1], row[2]) if row is not None else None


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Facts:
    id: int
    ticker: str
    version: int
    parent_version: int | None
    trigger: str | None
    generated_at: datetime | None
    as_of_date: datetime | None
    generation_mode: str | None
    analysis_generated_at: datetime | None
    sector: str | None
    final_pm_view: str | None
    degraded_agents: str | None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _project(db: Session, where: ColumnElement[bool]) -> list[_Facts]:
    """Small columns plus JSON-path strings. Never the body itself."""
    mj = MemoSnapshot.memo_json
    rows = db.execute(
        select(
            MemoSnapshot.id, MemoSnapshot.ticker, MemoSnapshot.version,
            MemoSnapshot.parent_version, MemoSnapshot.trigger,
            MemoSnapshot.generated_at, MemoSnapshot.as_of_date,
            mj["generation_mode"].as_string(),
            mj["generated_at"].as_string(),
            mj["sector"].as_string(),
            mj["final_pm_view"].as_string(),
            mj["degraded_agents"].as_string(),
        ).where(where).order_by(MemoSnapshot.id)
    ).all()
    out = []
    for (sid, ticker, version, parent_version, trigger, generated_at, as_of,
         mode, analysis_at, sector, pm_view, degraded) in rows:
        analysis = parse_timestamp(analysis_at) or generated_at
        out.append(_Facts(
            id=sid, ticker=ticker, version=version, parent_version=parent_version,
            trigger=trigger, generated_at=generated_at, as_of_date=as_of,
            generation_mode=_as_text(mode), analysis_generated_at=analysis,
            sector=_as_text(sector), final_pm_view=_as_text(pm_view),
            degraded_agents=_as_text(degraded),
        ))
    return out


class _Resolver:
    """Classifies facts, resolving patch lineage through the parent chain.

    Parents are taken from this sweep's results first, then from a current,
    identity-valid ledger row, then classified on the spot. Versions strictly
    decrease along a valid chain, so the walk terminates; a patch naming a
    parent at or above its own version is treated as having no parent.
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.done: dict[int, Classification] = {}

    def _parent_id(self, facts: _Facts) -> int | None:
        if facts.parent_version is None or facts.parent_version >= facts.version:
            return None
        return self.db.execute(
            select(MemoSnapshot.id).where(
                MemoSnapshot.ticker == facts.ticker,
                MemoSnapshot.version == facts.parent_version,
            )
        ).scalar_one_or_none()

    def _stored(self, snapshot_id: int) -> Classification | None:
        row = self.db.execute(
            select(MemoOutcomeEligibility.eligible, MemoOutcomeEligibility.reason,
                   MemoOutcomeEligibility.inherited_from_snapshot_id)
            .join(MemoSnapshot, identity_matches())
            .where(
                MemoOutcomeEligibility.memo_snapshot_id == snapshot_id,
                MemoOutcomeEligibility.rule_version >= RULE_VERSION,
            )
        ).first()
        return Classification(bool(row[0]), row[1], row[2]) if row is not None else None

    def classify(self, facts: _Facts) -> Classification:
        if facts.id in self.done:
            return self.done[facts.id]
        # Walk up the patch chain iteratively (NVDA has hundreds of versions).
        chain: list[tuple[_Facts, int | None]] = []
        current = facts
        base: Classification | None = None
        while True:
            if chain:
                # `current` is an ancestor: reuse an answer we already have.
                known = self.done.get(current.id) or self._stored(current.id)
                if known is not None:
                    base = known
                    break
            if (current.trigger or "") != PATCH_TRIGGER:
                base = self._rule(current, None, None)
                self.done[current.id] = base
                break
            parent_id = self._parent_id(current)
            parents = _project(self.db, MemoSnapshot.id == parent_id) if parent_id is not None else []
            if not parents:
                chain.append((current, None))
                base = None
                break
            chain.append((current, parent_id))
            current = parents[0]
        # Unwind: each patch inherits from the classification above it.
        for patch, parent_id in reversed(chain):
            result = self._rule(patch, base if parent_id is not None else None, parent_id)
            self.done[patch.id] = result
            base = result
        return self.done[facts.id]

    @staticmethod
    def _rule(facts: _Facts, parent: Classification | None, parent_id: int | None) -> Classification:
        return classify_snapshot(
            snapshot_id=facts.id, ticker=facts.ticker,
            snapshot_generated_at=facts.generated_at,
            analysis_generated_at=facts.analysis_generated_at,
            generation_mode=facts.generation_mode, trigger=facts.trigger,
            as_of_date=facts.as_of_date,
            parent_classification=parent, parent_snapshot_id=parent_id,
        )


def _insert_for(db: Session) -> Any:
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        return pg_insert
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        return sqlite_insert
    raise NotImplementedError(f"outcome eligibility upsert has no implementation for dialect {dialect!r}")


def _upsert(db: Session, facts: _Facts, result: Classification) -> None:
    insert = _insert_for(db)
    led = MemoOutcomeEligibility
    # Identity columns are copied by SQL from the snapshot row, not
    # round-tripped through Python, so the readers' equality join compares a
    # stored value with itself on every dialect.
    ticker_sql = select(MemoSnapshot.ticker).where(MemoSnapshot.id == facts.id).scalar_subquery()
    generated_sql = select(MemoSnapshot.generated_at).where(MemoSnapshot.id == facts.id).scalar_subquery()
    values = {
        "memo_snapshot_id": facts.id,
        "ticker": ticker_sql,
        "snapshot_generated_at": generated_sql,
        "analysis_generated_at": facts.analysis_generated_at,
        "trigger": (facts.trigger or None) and facts.trigger[:48],
        "inherited_from_snapshot_id": result.inherited_from_snapshot_id,
        "eligible": result.eligible,
        "reason": result.reason,
        "generation_mode": (facts.generation_mode or None) and facts.generation_mode[:16],
        "rating_source": rating_source(
            trigger=facts.trigger, generation_mode=facts.generation_mode,
            final_pm_view=facts.final_pm_view, degraded_agents=facts.degraded_agents,
        ),
        "sector": (facts.sector or None) and facts.sector[:64],
        "rule_version": RULE_VERSION,
        "classified_at": datetime.utcnow(),
    }
    stmt = insert(led).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[led.memo_snapshot_id],
        set_={k: getattr(stmt.excluded, k) for k in values if k != "memo_snapshot_id"},
        # A row already at the current version for the same snapshot is left
        # alone, so a concurrent sweep (web "Score now" and the worker) is
        # harmless: both would compute the same answer.
        where=or_(
            led.rule_version < RULE_VERSION,
            led.ticker != stmt.excluded.ticker,
            led.snapshot_generated_at.is_distinct_from(stmt.excluded.snapshot_generated_at),
        ),
    )
    db.execute(stmt)


def pending_condition() -> ColumnElement[bool]:
    """Snapshot has no ledger row, an older rule's row, or a reused id's row."""
    led = MemoOutcomeEligibility
    return or_(
        led.memo_snapshot_id.is_(None),
        led.rule_version < RULE_VERSION,
        led.ticker != MemoSnapshot.ticker,
        led.snapshot_generated_at.is_distinct_from(MemoSnapshot.generated_at),
    )


class ExclusionSetMismatch(RuntimeError):
    """The sweep disagreed with the enumerated historical exclusion sets."""


_EXPECTED_DEV_COPY: dict[int, tuple[str, datetime]] = {
    sid: (ticker, datetime.fromisoformat(generated))
    for sid, ticker, generated in evidence.DEV_COPY_SNAPSHOTS
}
_EXPECTED_FIXTURES: frozenset[tuple[int, str]] = frozenset(evidence.FIXTURE_SNAPSHOTS)


def _check_expected(facts: _Facts, result: Classification) -> str | None:
    """Why this classification contradicts the enumerated evidence, or None.

    The two historical exclusions are decided by rule, but the rows they may
    touch are known exactly. Over-exclusion (a snapshot outside the evidence
    classified as the dev copy or a fixture) and under-exclusion (an exact
    evidence row, matched on id + ticker + generated_at, classified as
    anything but the dev copy) both abort the sweep, so the production
    ledger can only ever hold the reviewed set. The under-exclusion check
    fingerprints on the microsecond timestamp, so it cannot fire on an
    unrelated row that merely reuses an id in a test or dev database.
    """
    # A patch inheriting either reason is judged by its parent, which was
    # itself checked; `inherited_from_snapshot_id` keeps it separable in the
    # ledger and the outcome audit.
    direct = result.inherited_from_snapshot_id is None
    if direct and result.reason == REASON_DEV_COPY:
        expected = _EXPECTED_DEV_COPY.get(facts.id)
        if expected is None or expected[0] != facts.ticker:
            return f"dev-copy classification outside the enumerated set: {facts.ticker}#{facts.id}"
    if direct and result.reason == REASON_TEST_FIXTURE and (facts.id, facts.ticker) not in _EXPECTED_FIXTURES:
        return f"fixture classification outside the enumerated set: {facts.ticker}#{facts.id}"
    expected = _EXPECTED_DEV_COPY.get(facts.id)
    if (
        expected is not None and expected == (facts.ticker, facts.generated_at)
        and result.reason != REASON_DEV_COPY
    ):
        return f"enumerated dev-copy snapshot {facts.ticker}#{facts.id} classified as {result.reason}"
    return None


# Reasons whose snapshots the summary names by id, for owner visibility (the
# W6 receipt lists them): eligible despite a demo label, eligible laptop-live
# memos from the copy, and post-fix demo.
_LISTED_REASONS = (REASON_LABEL_PREDATES_FIX, REASON_LIVE_DEV_COPY, REASON_NO_LLM_OR_DEMO)
_LISTED_CAP = 200


def classify_pending(*, db: Session, batch_size: int = 200) -> dict[str, Any]:
    """Classify every snapshot with no valid current ledger row.

    Idempotent, resumable and safe to run concurrently. DB only: no provider
    and no LLM. Writes ONLY `memo_outcome_eligibility`, in ONE transaction:
    a disagreement with the enumerated evidence (`_check_expected`) or any
    error rolls the whole sweep back and raises, so the ledger never holds a
    partial or unreviewed exclusion. Batches bound the rows held in Python,
    not the transaction (about 1k small rows on the first production sweep).

    Returns ``{"classified": n, "by_reason": {reason: n}, "rule_version": v,
    "listed": {reason: [ids]}}``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _insert_for(db)  # refuse an unsupported dialect before touching anything
    ensure_table(db)
    led = MemoOutcomeEligibility
    by_reason: Counter[str] = Counter()
    listed: dict[str, list[int]] = {}
    processed: set[int] = set()
    classified = 0
    resolver = _Resolver(db)
    try:
        while True:
            ids = list(db.execute(
                select(MemoSnapshot.id)
                .outerjoin(led, led.memo_snapshot_id == MemoSnapshot.id)
                .where(pending_condition())
                .order_by(MemoSnapshot.id)
                .limit(batch_size)
            ).scalars())
            if not ids:
                break
            if all(i in processed for i in ids):
                # Never spin: a write that does not stick is a defect to surface.
                raise RuntimeError(f"eligibility classification made no progress: ids={ids[:20]}")
            for facts in _project(db, MemoSnapshot.id.in_(ids)):
                result = resolver.classify(facts)
                problem = _check_expected(facts, result)
                if problem is not None:
                    raise ExclusionSetMismatch(problem)
                _upsert(db, facts, result)
                processed.add(facts.id)
                by_reason[result.reason] += 1
                if result.reason in _LISTED_REASONS and len(listed.get(result.reason, [])) < _LISTED_CAP:
                    listed.setdefault(result.reason, []).append(facts.id)
                classified += 1
            processed.update(ids)
        db.commit()
    except Exception as exc:
        db.rollback()
        log.error("outcome eligibility sweep aborted, nothing written: %s: %s", type(exc).__name__, exc)
        raise
    result_summary: dict[str, Any] = {
        "classified": classified,
        "by_reason": dict(sorted(by_reason.items())),
        "rule_version": RULE_VERSION,
        "listed": listed,
    }
    log.info("outcome eligibility: %s", result_summary)
    return result_summary
