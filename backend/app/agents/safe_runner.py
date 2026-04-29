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
    - Logs the exception via `logging.exception` so prod telemetry / Sentry
      sees the full traceback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, TypeVar

from ..schemas import AgentFinding, CriticReview

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class DegradationLog:
    """Accumulator passed through `run_stock_memo` so failed agents surface."""
    failures: List[dict] = field(default_factory=list)

    def record(self, agent: str, exc: BaseException) -> None:
        self.failures.append({
            "agent": agent,
            "error_type": type(exc).__name__,
            "message": str(exc)[:300],
        })

    def degraded_agents(self) -> List[str]:
        return [f["agent"] for f in self.failures]


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
    log_to: Optional[DegradationLog] = None,
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
        log.exception("Agent %s failed", agent)
        if log_to is not None:
            log_to.record(agent, exc)
        return _fallback_finding(agent, str(exc))


def safe_call(
    fn: Callable[..., T],
    *args: Any,
    fallback: T,
    name: str = "",
    log_to: Optional[DegradationLog] = None,
    **kwargs: Any,
) -> T:
    """Generic safe wrapper for non-AgentFinding helpers (DCF, comps, etc.).

    Returns `fallback` on any exception. The same `DegradationLog` is used so
    the memo's degraded_agents list captures everything.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.exception("Safe call %s failed", name or fn.__name__)
        if log_to is not None and name:
            log_to.record(name, exc)
        return fallback


def safe_critic(
    fn: Callable[..., Optional[CriticReview]],
    *args: Any,
    log_to: Optional[DegradationLog] = None,
    **kwargs: Any,
) -> Optional[CriticReview]:
    """Critic-specific safe wrapper.

    The critic is allowed to legitimately return None (when disabled by
    `ENABLE_AGENT_CRITIC=false`). Only converts *exceptions* to a fallback
    review so we don't paper over an intentional opt-out.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.exception("Critic failed")
        if log_to is not None:
            log_to.record("Risk Committee", exc)
        return CriticReview(
            overall_assessment="Critic agent unavailable for this run.",
            challenges=[],
            underweighted_risks=[],
            suggested_revisions=[
                "Re-run the memo when the critic is available; do not act on this draft alone.",
            ],
            advice_compliance_check=(
                "Output framed as research/education only; critic stage was skipped."
            ),
        )
