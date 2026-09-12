"""Pick the tickers the monitoring loops should spend their budget on.

`news_loop` and `social_loop` used to choose their work like this::

    tickers = list(ds.list_tickers())[:10]   # demo universe sample

`DataService.list_tickers()` is `db.query(Company.ticker).all()` with no
ORDER BY, so that slice is whatever ten rows the seeder happened to insert
first — in practice the alphabetical head of the S&P 500 (A, AAPL, ABBV,
ABNB, ...). The result was that the hourly news loop burned its entire
Gemini budget on ABNB and ACN while NVDA (230 memo versions), MSFT (108)
and every name a user had just asked us to research got no news coverage
at all. Of the 172 companies in the universe only 17 have ever had a memo
generated; the alphabetical head overlapped that set by two names.

So: rank by evidence that a ticker is actually being researched, and spend
the budget from the top. A ticker with no research signal at all is simply
not selected — there is no fallback to an arbitrary slice, because an
arbitrary slice is the bug.

Bands, highest priority first:

  0 `pinned`          `Company.auto_update_memo` — the explicitly curated
                      auto-regen set (top 10 by market cap, from
                      sp500.json). Always makes the cut.
  1 `regen_requested` a `regen_jobs` row enqueued inside the window. This
                      is the truest "being researched right now" signal:
                      one row per regen request, including the manual,
                      user-triggered ones, so it catches on-demand names
                      (MELI, SHOP) that are outside the curated universe.
  2 `memo_fresh`      newest memo snapshot inside the window — the same
                      signal `update_orchestrator.should_auto_regen` uses
                      to decide a ticker is worth re-analysing.
  3 `memo_stale`      has a memo, but older than the window.

A ticker lands in exactly one band (its highest). Within a band, most
recent first, ties broken by ticker ascending, so the order is a total
function of the inputs.

Which memo table: the brief for this change named `StockMemo`
(`stock_memos`), but nothing has written to that table since `memo_store`
replaced it — it is the legacy pre-versioning store and it is empty in
production. The live memos, and the version counts the ranking is meant to
reflect, are `MemoSnapshot` (`memo_snapshots`), which is also what
`should_auto_regen` reads through `memo_store.latest_memo`. Ranking on the
empty table would have selected nobody. Backtest snapshots (`as_of_date`
set) are excluded, matching `latest_memo`'s default: a reproduced-as-of
memo is not someone watching the ticker today.

Starvation: bands 0-2 are the guaranteed head. Whatever budget is left
over is filled from band 3 through a window that rotates one step per
wall-clock hour, so consecutive runs walk the whole tail instead of
re-running the same head forever. The offset is derived from the injected
`now`, which makes the rotation a pure function and lets a test assert the
coverage property directly rather than sampling it.

Degraded path: every DB read is wrapped. On failure we fall back to the
static pin list in `app/data/sp500.json` and say so in `note()`. Monitoring
the ten most important names while the database is unhappy beats monitoring
an arbitrary alphabetical slice, and beats monitoring nothing.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..services.update_orchestrator import AUTO_REGEN_RECENCY_DAYS

log = logging.getLogger(__name__)

__all__ = [
    "AUTO_REGEN_RECENCY_DAYS",
    "BAND_DEGRADED",
    "BAND_MEMO_FRESH",
    "BAND_MEMO_STALE",
    "BAND_PINNED",
    "BAND_REGEN",
    "FocusSelection",
    "select_focus",
]

# Reason labels. Also the band names `note()` counts by, so they are part
# of what shows up in `/api/admin/cron-health`.
BAND_PINNED = "pinned"
BAND_REGEN = "regen_requested"
BAND_MEMO_FRESH = "memo_fresh"
BAND_MEMO_STALE = "memo_stale"
BAND_DEGRADED = "degraded_static_pin"

_BAND_ORDER: tuple[str, ...] = (
    BAND_PINNED, BAND_REGEN, BAND_MEMO_FRESH, BAND_MEMO_STALE, BAND_DEGRADED,
)

_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class FocusSelection:
    """What the loop should run, why, and what it could not afford.

    `reasons` maps every selected ticker to its band label. `dropped` is
    every ticker that qualified but did not fit the budget, still in rank
    order, so the note names the most important omission first.
    """

    tickers: tuple[str, ...] = ()
    reasons: dict[str, str] = field(default_factory=dict)
    dropped: tuple[str, ...] = ()
    degraded: bool = False
    # Where in the band-3 tail this run's rotation window started, and how
    # long that tail was. Diagnostics only — but they are the two numbers
    # you need to explain why a given stale ticker did or did not run.
    rotation_offset: int = 0
    tail_size: int = 0

    def note(self) -> str:
        """One-line summary for `record_run`, hence for cron-health.

        Every dropped ticker is named, deliberately and without a cap: a
        cap is exactly the silent behaviour this change exists to remove.
        The list is bounded by the size of the universe (~172).
        """
        counts = Counter(self.reasons.values())
        breakdown = ", ".join(
            f"{band}={counts[band]}" for band in _BAND_ORDER if counts.get(band)
        )
        out = f"focus {len(self.tickers)}"
        if breakdown:
            out += f" ({breakdown})"
        if self.degraded:
            out += " [degraded: static pins]"
        if self.dropped:
            out += f"; over budget, dropped {len(self.dropped)}: " + ", ".join(self.dropped)
        return out


def _hours_since_epoch(now: datetime) -> int:
    """Whole hours since the Unix epoch, for the tail rotation.

    Naive datetimes are read as UTC, which is what the whole codebase
    stores (`datetime.utcnow()`). Computed as a timedelta rather than via
    `.timestamp()` because `.timestamp()` on a naive value interprets it
    as *local* time, which is not monotonic across a DST transition and
    would make the rotation jump backwards twice a year.
    """
    if now.tzinfo is not None:
        now = now.astimezone(UTC).replace(tzinfo=None)
    return int((now - _EPOCH).total_seconds() // 3600)


def _by_recency_then_ticker(row: tuple[str, datetime]) -> tuple[timedelta, str]:
    """Sort key: most recent first, ties broken by ticker ascending."""
    return (_EPOCH - row[1], row[0])


def _gather(db: Session, *, cutoff: datetime) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return the four bands, each already in final rank order."""
    from ..models import Company, MemoSnapshot, RegenJob

    known: set[str] = set()
    pinned: list[str] = []
    for ticker, auto in db.execute(
        select(Company.ticker, Company.auto_update_memo)
    ).all():
        if not ticker:
            continue
        ticker = ticker.upper()
        known.add(ticker)
        if auto:
            pinned.append(ticker)

    # A `regen_jobs` row can outlive the company row it names (on-demand
    # tickers get pruned), and a loop cannot research a company we have no
    # profile for — so everything below is filtered against `known`.
    regen_rows = [
        (t.upper(), ts or _EPOCH)
        for t, ts in db.execute(
            select(RegenJob.ticker, func.max(RegenJob.enqueued_at))
            .where(RegenJob.enqueued_at >= cutoff)
            .group_by(RegenJob.ticker)
        ).all()
        if t and t.upper() in known
    ]

    memo_rows = [
        (t.upper(), ts or _EPOCH)
        for t, ts in db.execute(
            select(MemoSnapshot.ticker, func.max(MemoSnapshot.generated_at))
            .where(MemoSnapshot.as_of_date.is_(None))
            .group_by(MemoSnapshot.ticker)
        ).all()
        if t and t.upper() in known
    ]

    regen_rows.sort(key=_by_recency_then_ticker)
    memo_rows.sort(key=_by_recency_then_ticker)

    claimed = set(pinned)
    band_regen: list[str] = []
    for ticker, _ts in regen_rows:
        if ticker not in claimed:
            claimed.add(ticker)
            band_regen.append(ticker)

    band_fresh: list[str] = []
    band_stale: list[str] = []
    for ticker, ts in memo_rows:
        if ticker in claimed:
            continue
        claimed.add(ticker)
        (band_fresh if ts >= cutoff else band_stale).append(ticker)

    return sorted(pinned), band_regen, band_fresh, band_stale


def _select(db: Session, *, budget: int, window_days: int, now: datetime) -> FocusSelection:
    cutoff = now - timedelta(days=window_days)
    pinned, regen, fresh, stale = _gather(db, cutoff=cutoff)

    head = pinned + regen + fresh
    bands = {t: BAND_PINNED for t in pinned}
    bands.update({t: BAND_REGEN for t in regen})
    bands.update({t: BAND_MEMO_FRESH for t in fresh})
    bands.update({t: BAND_MEMO_STALE for t in stale})

    offset = 0
    if len(head) >= budget:
        # The head is already over budget. Truncate in rank order and drop
        # the rest — never rotate the head, because band 0 is the curated
        # pin list and skipping a pin for a turn is exactly what
        # `auto_update_memo` promises will not happen.
        selected = head[:budget]
        dropped = head[budget:] + stale
    else:
        room = budget - len(head)
        if stale:
            offset = _hours_since_epoch(now) % len(stale)
            rotated = stale[offset:] + stale[:offset]
        else:
            rotated = []
        picked = rotated[:room]
        taken = set(picked)
        selected = head + picked
        dropped = [t for t in stale if t not in taken]

    return FocusSelection(
        tickers=tuple(selected),
        reasons={t: bands[t] for t in selected},
        dropped=tuple(dropped),
        degraded=False,
        rotation_offset=offset,
        tail_size=len(stale),
    )


def _static_pins(budget: int) -> FocusSelection:
    """Fallback when the DB work raised: the curated top-10 from sp500.json.

    Kept in file order rather than sorted, because that file lists the pins
    by market cap — if the budget is smaller than the list, the biggest
    names are the ones worth keeping.
    """
    try:
        from ..seed_universe import load_universe_file
        pins = list(dict.fromkeys(t.upper() for t in load_universe_file().auto_update if t))
    except Exception:
        log.warning("static pin list unavailable; monitoring loop has nothing to run", exc_info=True)
        return FocusSelection(degraded=True)

    selected = pins[:budget]
    return FocusSelection(
        tickers=tuple(selected),
        reasons={t: BAND_DEGRADED for t in selected},
        dropped=tuple(pins[budget:]),
        degraded=True,
    )


def select_focus(
    *,
    budget: int,
    window_days: int = AUTO_REGEN_RECENCY_DAYS,
    now: datetime | None = None,
    db: Session | None = None,
) -> FocusSelection:
    """Choose up to `budget` tickers worth spending a monitoring run on.

    `now` and `db` are injectable so callers (and tests) can pin the clock
    and the database. Never raises and never touches the network: on any
    DB failure it degrades to the static pin list and says so.
    """
    now = now or datetime.utcnow()
    budget = max(0, int(budget))

    try:
        if db is not None:
            selection = _select(db, budget=budget, window_days=window_days, now=now)
        else:
            from ..database import SessionLocal
            with SessionLocal() as session:
                selection = _select(session, budget=budget, window_days=window_days, now=now)
    except Exception as exc:
        log.warning(
            "research focus selection failed (%s); falling back to the static pin list",
            exc, exc_info=True,
        )
        selection = _static_pins(budget)

    if selection.dropped:
        # Constraint: a cap is never silent. The note carries this to
        # cron-health; this line carries it to the worker's logs.
        log.info(
            "research focus: budget %d, dropped %d qualifying ticker(s): %s",
            budget, len(selection.dropped), ", ".join(selection.dropped),
        )
    return selection
