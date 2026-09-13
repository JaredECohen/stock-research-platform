"""Per-agent failure isolation.

Goal: a thrown exception in any single specialist (sector, earnings, filing,
valuation, comps, macro, risk, critic) must not kill the whole memo. The
memo continues with a typed "agent unavailable" stand-in for the failed
specialist plus a `degraded_agents` list the PM and the UI can surface.

Why this is its own module:
    `graph.py` already imports the specialist runners directly. Wrapping
    them inline would clutter the orchestration. Putting the safe-call
    helpers here keeps the call sites in `graph.py` readable and makes
    failure handling testable in isolation.

Failure semantics:
    - Returns a typed fallback (AgentFinding, CriticReview, or None) so
      downstream code never has to special-case missing data.
    - Records the (agent_name, exception class, message) tuple on a
      `DegradationLog` accumulator so the memo can surface a banner of
      degraded agents.
    - Logs the failure at WARNING with the exception *type* only, and the
      traceback at DEBUG. Provider exceptions quote the request that
      failed — including auth headers — so the body must not reach the
      retained production log stream (see `log_safety`).
"""
from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..schemas import AgentFinding, CriticReview
from .log_safety import redact

log = logging.getLogger(__name__)


def _log_failure(msg: str, exc: BaseException) -> None:
    log.warning("%s: %s", msg, type(exc).__name__)
    log.debug("%s — traceback follows", msg, exc_info=True)

T = TypeVar("T")

# The DegradationLog of the memo run active in this context, if any.
#
# A ContextVar rather than a module-level global on purpose: the regen
# worker runs memos back to back in one long-lived thread and the web
# process serves the synchronous `/memo` path, so a global would bleed one
# run's failures into the next. A ContextVar is per-thread (each thread
# starts from an empty context) and `DegradationLog.activate()` resets it
# in `finally`, so a run can never inherit a stale log. Nothing crosses the
# web/worker process boundary here — a memo run is single-process end to
# end — so this does not belong in the database (see CLAUDE.md).
_ACTIVE_LOG: contextvars.ContextVar[DegradationLog | None] = contextvars.ContextVar(
    "degradation_log", default=None,
)


@dataclass
class DegradationLog:
    """Accumulator passed through `run_stock_memo` so failed agents surface."""
    failures: list[dict] = field(default_factory=list)

    @contextmanager
    def activate(self) -> Iterator[DegradationLog]:
        """Make this log the target of `note_soft` for the enclosed block.

        `run_stock_memo` wraps the whole memo run in it so service code that
        has no handle on the log (valuation service, thesis builder, PM DCF
        adjuster) can still report a soft degradation. The token is reset in
        `finally`, so a run that raises leaves nothing behind for the next
        run in the same thread.
        """
        token = _ACTIVE_LOG.set(self)
        try:
            yield self
        finally:
            _ACTIVE_LOG.reset(token)

    def record(self, agent: str, exc: BaseException) -> None:
        # The message rides on the memo's `degraded_agents` banner and is
        # persisted with the memo — redact it like a log line.
        self.failures.append({
            "agent": agent,
            "error_type": type(exc).__name__,
            "message": redact(exc),
        })

    def record_soft(self, agent: str, reason: str,
                    kind: str = "DeterministicFallback") -> None:
        """Record a degradation that wasn't an exception.

        An LLM call that succeeds structurally but returns nothing usable
        (so the agent shipped its deterministic fallback) degrades the memo
        just as much as a crash — the section renders as boilerplate while
        presenting as a real analyst view. Same record shape as `record`
        so the memo banner treats both alike.
        """
        if any(f["agent"] == agent for f in self.failures):
            return
        self.failures.append({
            "agent": agent,
            "error_type": kind,
            "message": reason[:300],
        })

    def degraded_agents(self) -> list[str]:
        return [f["agent"] for f in self.failures]

    def events(self) -> list[dict]:
        """Copy of the failure records for `StockMemoOut.degradation_events`.

        Same `{agent, error_type, message}` shape as `failures`, copied so
        the memo does not alias the accumulator (a later `record` must not
        mutate an already-built memo behind its back).
        """
        return [dict(f) for f in self.failures]


def active_log() -> DegradationLog | None:
    """The DegradationLog of the memo run active in this context, or None."""
    return _ACTIVE_LOG.get()


def note_soft(agent: str, reason: str, kind: str = "DeterministicFallback") -> bool:
    """Record a soft degradation on the active memo run's log.

    The one-liner for class (b) sites — code that can change what the user
    reads in a memo but has no `DegradationLog` in scope. Returns True when
    it recorded (or deduped) on an active log and False when no memo run is
    active: chat, screener and monitoring paths call the same services, and
    for them a fallback is not a memo degradation, so the call is a no-op
    rather than an error. Callers may therefore invoke it unconditionally.
    """
    active = _ACTIVE_LOG.get()
    if active is None:
        return False
    active.record_soft(agent, reason, kind=kind)
    return True


def _fallback_finding(agent: str, error: str) -> AgentFinding:
    return AgentFinding(
        agent=agent,
        headline=f"{agent} unavailable",
        summary=(
            f"The {agent.lower()} agent failed during this run; the memo was "
            "produced without its contribution. Other specialist views remain valid."
        ),
        key_points=[f"Error: {error[:140]}"],
        confidence=0.0,
        sources=[],
        data={"degraded": True, "error": error[:300]},
    )


def safe_finding(
    agent: str,
    fn: Callable[..., AgentFinding],
    *args: Any,
    log_to: DegradationLog | None = None,
    **kwargs: Any,
) -> AgentFinding:
    """Call an agent runner and convert any exception into a fallback finding.

    The caller passes the human-readable agent name (for the memo + the log).
    """
    try:
        result = fn(*args, **kwargs)
        if result is None:
            raise RuntimeError(f"{agent} returned None")
        return result
    except Exception as exc:
        _log_failure(f"Agent {agent} failed", exc)
        if log_to is not None:
            log_to.record(agent, exc)
        return _fallback_finding(agent, redact(exc))


def safe_call(
    fn: Callable[..., T],
    *args: Any,
    fallback: T,
    name: str = "",
    log_to: DegradationLog | None = None,
    **kwargs: Any,
) -> T:
    """Generic safe wrapper for non-AgentFinding helpers (DCF, comps, etc.).

    Returns `fallback` on any exception. The same `DegradationLog` is used so
    the memo's degraded_agents list captures everything.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        _log_failure(f"Safe call {name or fn.__name__} failed", exc)
        if log_to is not None and name:
            log_to.record(name, exc)
        return fallback


def safe_critic(
    fn: Callable[..., CriticReview | None],
    *args: Any,
    log_to: DegradationLog | None = None,
    **kwargs: Any,
) -> CriticReview | None:
    """Critic-specific safe wrapper.

    The critic is allowed to legitimately return None (when disabled by
    `ENABLE_AGENT_CRITIC=false`). Only converts *exceptions* to a fallback
    review so we don't paper over an intentional opt-out.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        _log_failure("Critic failed", exc)
        if log_to is not None:
            log_to.record("Risk Committee", exc)
        return CriticReview(
            overall_assessment="Critic agent unavailable for this run.",
            review_mode="unavailable",
            challenges=[],
            underweighted_risks=[],
            suggested_revisions=[
                "Re-run the memo when the critic is available; do not act on this draft alone.",
            ],
            advice_compliance_check=(
                "Output framed as research/education only; critic stage was skipped."
            ),
        )
