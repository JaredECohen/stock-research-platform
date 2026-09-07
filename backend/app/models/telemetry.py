"""Telemetry — LLM call log, SDK traces, UI activity log."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class LLMCallLog(Base):
    """Append-only audit log of every provider LLM call (Wave 1A).

    One row per real LLM call with attribution to the agent that made it
    and the memo run it belongs to. Used for cost-per-run / cost-per-agent
    breakdowns + slow-call audit. Distinct from `CacheCostLog` (which
    tracks snapshot-write savings) — this one tracks actual provider
    spend regardless of caching.

    GC: rows older than 90 days are deleted by a daily monitoring job
    when ENABLE_MONITORING is on; otherwise they accumulate harmlessly
    (rows are small).
    """
    __tablename__ = "llm_call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    ticker: Mapped[Optional[str]] = mapped_column(String(16), index=True, nullable=True)
    agent_name: Mapped[str] = mapped_column(String(64), index=True, default="unknown")
    provider: Mapped[str] = mapped_column(String(16), default="openai")
    model: Mapped[str] = mapped_column(String(64), default="")
    route: Mapped[str] = mapped_column(String(16), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str] = mapped_column(Text, default="")
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )


Index("ix_llm_call_run", LLMCallLog.run_id, LLMCallLog.generated_at)


class SDKTrace(Base):
    """Wave 10 — persisted OpenAI Agents SDK exchange trace.

    When `USE_AGENTS_SDK=true` and the official `openai-agents` package +
    `OPENAI_API_KEY` are present, every memo run kicks off a parallel
    SDK exchange (real LLM-driven handoffs / tool calls). The trace is
    informational — the canonical `StockMemoOut` still comes from the
    legacy graph — but it's load-bearing for evals + debugging memos
    that look wrong + tool-gap discovery.

    One row per memo run (or per chat turn that routes through the SDK).
    `new_items` carries the raw SDK item stream (handoffs, tool calls,
    reasoning steps); the admin trace viewer joins this against
    `LLMCallLog` rows by `run_id` so reviewers can see SDK reasoning + the
    deterministic graph's calls in the same timeline.

    GC: rows older than 90 days are deleted by the same daily monitoring
    job that cleans `LLMCallLog`. Rows are bigger (`new_items` JSON), so
    don't keep them forever.
    """
    __tablename__ = "sdk_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ticker: Mapped[Optional[str]] = mapped_column(String(16), index=True, nullable=True)
    # Source surface: "memo" (run_stock_memo path) or "chat" (freeform Q&A).
    surface: Mapped[str] = mapped_column(String(16), default="memo", index=True)
    final_output: Mapped[str] = mapped_column(Text, default="")
    # Raw SDK item stream serialized to JSON. Format is provider-defined; we
    # don't constrain it here so model upgrades that add fields don't require
    # a migration.
    new_items: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )


Index("ix_sdk_trace_run", SDKTrace.run_id, SDKTrace.generated_at)


class UILog(Base):
    """Append-only trace of UI activity + backend HTTP requests.

    Frontend posts route changes, API calls (with duration + status),
    button clicks, and uncaught errors here via `POST /api/admin/ui-log`.
    Backend middleware also writes a row per request so server-side
    traces and client-side actions sit in one queryable timeline.

    Schema is intentionally loose — `payload` carries arbitrary JSON
    so new event kinds can ship without migrations.
    """
    __tablename__ = "ui_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source: Mapped[str] = mapped_column(String(16), default="frontend")  # frontend | backend
    kind: Mapped[str] = mapped_column(String(32), index=True)
    # Common dimensions surfaced as columns for cheap filtering; everything
    # else (request body, error stack, etc.) rides in `payload`.
    path: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    method: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


Index("ix_ui_log_ts_kind", UILog.ts, UILog.kind)
