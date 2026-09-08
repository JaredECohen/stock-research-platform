"""FEAT-002 Phase 1 — per-action unit-cost measurement.

Runs N representative actions per billable feature against the *real*
LLM + provider chain and reports, per action: estimated LLM spend (from
`LLMCallLog` via `llm_metrics.cost_per_run`), provider-cache misses (rows
written to `provider_cache` / snapshot writes in `cache_cost_logs` during
the action) and wall-clock / worker seconds (`RegenJob.started_at →
finished_at` for research runs). The JSON summary feeds
`docs/economics/unit-costs-2026-09.md`, which holds the method, the
Free/Pro worst-case formulas and the go/no-go thresholds.

**This script spends real money.** It refuses to run unless BOTH
`RUN_LIVE_TESTS=1` is set AND an LLM key is configured, and exits 0 with a
message otherwise, so a stray invocation in CI or on a laptop with a bare
`.env` costs nothing. Run it in the nightly-live environment
(`.github/workflows/nightly-live.yml` sets the same env):

    RUN_LIVE_TESTS=1 ENABLE_LIVE_DATA=true USE_DEMO_DATA=false \\
        python -m scripts.audit_unit_costs --tickers NVDA,COST,JPM --n 3

Options:
    --tickers   comma list (default `SAMPLE_TICKERS` env or NVDA,COST,JPM)
    --n         actions per feature (default 3)
    --research-n  actions for `research_run` (default 1 — each is a full
                  5-9 minute memo run; raise for a tighter estimate)
    --features  comma subset of memo_view,research_run,pm_chat,dcf,comps,
                chart_commentary (default all)
    --out       JSON path (default docs/economics/unit-costs-measured-<date>.json)

What it deliberately does NOT do: print prompts, answers, memo bodies or
provider payloads (counts and dollars only), start the regen worker thread
(jobs are executed synchronously in this process so the timing is
attributable), or generate memos inside `pm_chat` — under `AUTH_ENABLED`
chat never runs a memo in-request, so the measurement emulates that by
answering from the stored memo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# The gate runs BEFORE any `app.*` import: importing `app.config` loads
# `.env`, and a developer `.env` carries live keys. Nothing below this
# line executes without the explicit opt-in.
RUN_LIVE = os.environ.get("RUN_LIVE_TESTS", "") == "1"

FEATURES = ("memo_view", "research_run", "pm_chat", "dcf", "comps", "chart_commentary")

# DEVPLAN FEAT-002 launch allowances — the multipliers in the worst-case
# formulas. Mirrors docs/economics/unit-costs-2026-09.md §3; change both.
ALLOWANCES = {
    "free": {"memo_view": 3, "research_run": 1, "pm_chat": 10, "chart_commentary": 5,
             # DCF/comps follow the memo allowance: 3 tickers' worth.
             "dcf": 3, "comps": 3},
    "pro": {"research_run": 20, "pm_chat": 300, "chart_commentary": 100},
}
PRO_PRICE_USD = 29.99
THRESHOLDS = {"pro_max_usd": round(PRO_PRICE_USD * 0.5, 3), "free_max_usd": 1.50}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
REPO_ROOT = Path(__file__).resolve().parents[2]


def _refuse(reason: str) -> int:
    print(f"audit_unit_costs: not running — {reason}.")
    print("This script spends real LLM/provider money; it runs only with "
          "RUN_LIVE_TESTS=1 and an LLM key configured (nightly-live env).")
    return 0


def _utcnow() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Provider / snapshot accounting over a time window
# ---------------------------------------------------------------------------

def _provider_activity(since: datetime) -> dict[str, int]:
    """Cache misses proxied by rows written since `since`.

    `provider_cache.put` writes/refreshes a `provider_cache` row on every
    miss (`fetched_at` moves), and the snapshot layer appends a
    `cache_cost_logs` row per computation (`kind` without `:hit`) and per
    hit (`:hit`). Neither ledger records misses directly, so the window
    count is the honest proxy — actions run sequentially, so the window
    is attributable.
    """
    from sqlalchemy import func, select

    from app.database import SessionLocal
    from app.models import CacheCostLog, ProviderCache

    with SessionLocal() as db:
        provider_writes = db.execute(
            select(func.count()).select_from(ProviderCache)
            .where(ProviderCache.fetched_at >= since)
        ).scalar_one()
        rows = db.execute(
            select(CacheCostLog.kind).where(CacheCostLog.generated_at >= since)
        ).all()
    snapshot_writes = sum(1 for (k,) in rows if not str(k).endswith(":hit"))
    snapshot_hits = sum(1 for (k,) in rows if str(k).endswith(":hit"))
    return {
        "provider_cache_writes": int(provider_writes),
        "snapshot_writes": snapshot_writes,
        "snapshot_hits": snapshot_hits,
    }


def _llm_summary(run_id: str) -> dict[str, Any]:
    from app.services.llm_metrics import cost_per_run
    info = cost_per_run(run_id)
    return {
        "n_calls": info["n_calls"],
        "tokens_in": info["tokens_in"],
        "tokens_out": info["tokens_out"],
        "cost_usd": info["cost_usd_total"],
        "n_failures": info["n_failures"],
        "duration_ms": info["duration_ms_total"],
        "models": sorted({c["model"] for c in info["calls"]}),
    }


class _Sample:
    """One measured action. Built incrementally so a crash mid-action still
    leaves a row that says what happened."""

    def __init__(self, feature: str, index: int, ticker: str | None) -> None:
        self.row: dict[str, Any] = {
            "feature": feature, "index": index, "ticker": ticker,
            "run_id": f"audit-{feature}-{index}-{int(time.time())}",
            "status": "pending", "note": "",
            "llm": None, "provider": None, "seconds": None, "worker_seconds": None,
        }

    def run(self, fn) -> dict[str, Any]:
        from app.agents.llm import llm_call_context
        from app.agents.log_safety import safe_exc

        since = _utcnow()
        t0 = time.monotonic()
        try:
            with llm_call_context(agent_name="unit_cost_audit", run_id=self.row["run_id"]):
                extra = fn(self.row["run_id"]) or {}
            self.row["status"] = extra.pop("status", "ok")
            self.row.update(extra)
        except Exception as exc:  # measurement must never abort the sweep
            self.row["status"] = "error"
            self.row["note"] = safe_exc(exc)[:200]
        self.row["seconds"] = round(time.monotonic() - t0, 2)
        self.row["provider"] = _provider_activity(since)
        self.row["llm"] = _llm_summary(self.row["run_id"])
        return self.row


# ---------------------------------------------------------------------------
# Feature runners — each returns a dict merged into the sample row
# ---------------------------------------------------------------------------

def _memo_view(ticker: str):
    def go(_run_id: str) -> dict[str, Any]:
        from app.services import memo_store
        snap = memo_store.latest_memo(ticker)
        if snap is None:
            return {"status": "skipped", "note": "no stored memo; run research_run first"}
        memo_store.memo_to_pydantic(snap)
        return {"memo_version": snap.version}
    return go


def _research_run(ticker: str):
    """The production path: enqueue → claim → execute_job, synchronously.

    Uses the job's own `run_id` (set by `enqueue`) so `LLMCallLog` rows
    join exactly as they do for a user-triggered run; the sample's
    `run_id` is overwritten with it. Worker seconds come from the
    `RegenJob` row, the same field the ops endpoints report.
    """
    def go(_run_id: str) -> dict[str, Any]:
        from app.api.routes_stocks import _ensure_lazy_universe
        from app.database import SessionLocal
        from app.models import RegenJob
        from app.services import regen_worker

        # Same step `POST /analyze` performs before enqueueing today (profile
        # lookup + 5-year backfill for a ticker the DB has never seen). It
        # runs inside the measurement window on purpose: a cold ticker's
        # provider calls are part of what a research run costs. When S2
        # moves this into the worker's `execute_job`, drop this line.
        _ensure_lazy_universe(ticker)
        job, created = regen_worker.enqueue(ticker, "soft_landing", source="unit_cost_audit")
        if not created:
            return {"status": "skipped", "note": f"coalesced onto existing job {job['id']}"}
        # Drain anything queued ahead of ours so timings stay attributable;
        # bounded so a wedged queue cannot spin the audit forever.
        for _ in range(5):
            claimed = regen_worker.claim_next_job()
            if claimed is None:
                return {"status": "error", "note": "queue empty before our job was claimed"}
            regen_worker.execute_job(claimed)
            if claimed == job["id"]:
                break
        else:
            return {"status": "error", "note": "our job was not claimed within 5 attempts"}
        with SessionLocal() as db:
            row = db.get(RegenJob, job["id"])
            if row is None:
                return {"status": "error", "note": "job row vanished"}
            worker_seconds = None
            if row.started_at and row.finished_at:
                worker_seconds = round((row.finished_at - row.started_at).total_seconds(), 1)
            return {
                "run_id": row.run_id,
                "status": "ok" if row.status == "succeeded" else "error",
                "note": row.error_type or "",
                "worker_seconds": worker_seconds,
                "memo_version": row.memo_version,
            }
    return go


_PM_QUESTIONS = (
    ("What are the biggest risks in the committee's {t} memo?", []),
    ("Why does the committee rate {t} the way it does, and what would change its mind?",
     [{"role": "user", "content": "Summarise the {t} thesis."},
      {"role": "assistant", "content": "Here is the stored thesis for {t}."}]),
    ("Which of {t} and {u} has the more durable moat?", []),
)


def _pm_chat(index: int, tickers: list[str]):
    """One Ask-the-PM turn with inline memo generation disabled.

    Under `AUTH_ENABLED` the orchestrator answers from `memo_store.latest_memo`
    and never runs `run_stock_memo` in-request (plan §4, orchestrator
    `allow_inline_memo=False`). Until that flag exists the same behaviour
    is emulated by swapping the module-level `run_stock_memo` for a stored-
    memo lookup — otherwise a `single_stock_analysis` intent would bill a
    5-9 minute memo run against the chat sample and the number would be
    meaningless for pm_chat.
    """
    t = tickers[index % len(tickers)]
    u = tickers[(index + 1) % len(tickers)]
    template, history = _PM_QUESTIONS[index % len(_PM_QUESTIONS)]
    question = template.format(t=t, u=u)
    hist = [{"role": m["role"], "content": m["content"].format(t=t, u=u)} for m in history]

    def go(_run_id: str) -> dict[str, Any]:
        from app.agents import orchestrator as orch_mod
        from app.schemas import ChatMessage
        from app.services import memo_store

        def stored_only(ticker: str, **_kw):
            snap = memo_store.latest_memo(ticker)
            if snap is None:
                raise ValueError(f"no stored memo for {ticker} (inline generation disabled)")
            return memo_store.memo_to_pydantic(snap)

        original = orch_mod.run_stock_memo
        orch_mod.run_stock_memo = stored_only
        try:
            resp = orch_mod.Orchestrator().chat(
                question, [ChatMessage(**m) for m in hist],
            )
        finally:
            orch_mod.run_stock_memo = original
        # Intent + answer length only: the answer text is never printed.
        return {"intent": resp.intent, "answer_chars": len(resp.answer or "")}
    return go


def _dcf(ticker: str):
    def go(_run_id: str) -> dict[str, Any]:
        from app.services.valuation_service import build_dcf
        res = build_dcf(ticker, None, force_refresh=True)
        if res is None:
            return {"status": "skipped", "note": "build_dcf returned None (no financials)"}
        return {"scenarios": len(res.scenarios or [])}
    return go


def _comps(ticker: str):
    def go(_run_id: str) -> dict[str, Any]:
        from app.services.valuation_service import build_comps
        res = build_comps(ticker, force_refresh=True)
        if res is None:
            return {"status": "skipped", "note": "build_comps returned None (no peers/financials)"}
        return {"peers": len(res.peers or [])}
    return go


def _chart_commentary(ticker: str):
    # FEAT-001 (fundamentals explorer + chart commentary) is not built yet.
    # The row exists so the doc's formulas have a slot; the cost is unknown,
    # not zero, and the summary says so.
    def go(_run_id: str) -> dict[str, Any]:
        return {"status": "not_implemented", "note": "FEAT-001 chart commentary not shipped"}
    return go


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [s for s in samples if s["status"] == "ok"]
    return {
        "n_samples": len(samples),
        "n_ok": len(ok),
        "statuses": sorted({s["status"] for s in samples}),
        "mean_llm_cost_usd": _mean([s["llm"]["cost_usd"] for s in ok if s["llm"]]),
        "max_llm_cost_usd": max([s["llm"]["cost_usd"] for s in ok if s["llm"]], default=None),
        "mean_llm_calls": _mean([s["llm"]["n_calls"] for s in ok if s["llm"]]),
        "mean_provider_cache_writes": _mean(
            [s["provider"]["provider_cache_writes"] for s in ok if s["provider"]]),
        "mean_snapshot_writes": _mean(
            [s["provider"]["snapshot_writes"] for s in ok if s["provider"]]),
        "mean_seconds": _mean([s["seconds"] for s in ok]),
        "mean_worker_seconds": _mean([s.get("worker_seconds") for s in ok]),
        "samples": samples,
    }


def _worst_case(features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Monthly variable LLM cost per user if every allowance is exhausted.

    A feature with no `ok` sample (skipped / not_implemented / errored)
    makes the total `None` — an unmeasured term is reported as unknown,
    never silently treated as zero.
    """
    out: dict[str, Any] = {}
    for plan, mult in ALLOWANCES.items():
        terms: dict[str, float | None] = {}
        total: float | None = 0.0
        for feat, qty in mult.items():
            mean = (features.get(feat) or {}).get("mean_llm_cost_usd")
            terms[feat] = None if mean is None else round(qty * mean, 4)
            total = None if (total is None or mean is None) else total + qty * mean
        out[plan] = {"terms": terms, "total_usd": None if total is None else round(total, 4)}
    pro, free = out["pro"]["total_usd"], out["free"]["total_usd"]
    out["verdict"] = {
        "pro_under_threshold": None if pro is None else pro < THRESHOLDS["pro_max_usd"],
        "free_under_threshold": None if free is None else free < THRESHOLDS["free_max_usd"],
        "thresholds": THRESHOLDS,
    }
    return out


def _markdown(features: dict[str, dict[str, Any]]) -> str:
    lines = ["| Feature | n ok | LLM $/action (mean) | LLM calls | provider writes | snapshot writes | seconds | worker s |",
             "|---|---|---|---|---|---|---|---|"]
    for feat in FEATURES:
        a = features.get(feat)
        if a is None:
            continue
        def f(v, fmt="{:.4f}"):
            return "n/a" if v is None else fmt.format(v)
        lines.append(
            f"| {feat} | {a['n_ok']}/{a['n_samples']} | {f(a['mean_llm_cost_usd'])} | "
            f"{f(a['mean_llm_calls'], '{:.1f}')} | {f(a['mean_provider_cache_writes'], '{:.1f}')} | "
            f"{f(a['mean_snapshot_writes'], '{:.1f}')} | {f(a['mean_seconds'], '{:.1f}')} | "
            f"{f(a['mean_worker_seconds'], '{:.1f}')} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", default=os.environ.get("SAMPLE_TICKERS", "NVDA,COST,JPM"))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--research-n", type=int, default=1)
    p.add_argument("--features", default=",".join(FEATURES))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if not RUN_LIVE:
        return _refuse("RUN_LIVE_TESTS is not set to 1")

    from app.config import settings
    # `has_llm` is the app's own gate for every LLM call site: without it the
    # chat / DCF / comps paths take their deterministic fallbacks and the
    # numbers would measure nothing. (Not `has_gemini`: that is true on a
    # bare VERTEX_PROJECT_ID with no credentials, which is how a dev .env
    # once let this script past the gate — harmlessly, but past it.)
    if not settings.has_llm:
        return _refuse("no LLM key configured (OPENAI_API_KEY / ANTHROPIC_API_KEY)")

    from app.database import init_db
    init_db()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    wanted = [f.strip() for f in args.features.split(",") if f.strip()]
    unknown = sorted(set(wanted) - set(FEATURES))
    if unknown:
        print(f"unknown features: {unknown}; choose from {list(FEATURES)}")
        return 2
    if not settings.enable_live_data:
        print("WARNING: ENABLE_LIVE_DATA is false — provider counts reflect the demo "
              "provider, only the LLM numbers are representative.")

    started = _utcnow()
    results: dict[str, dict[str, Any]] = {}
    for feat in FEATURES:
        if feat not in wanted:
            continue
        n = args.research_n if feat == "research_run" else args.n
        samples: list[dict[str, Any]] = []
        for i in range(n):
            ticker = tickers[i % len(tickers)]
            if feat == "memo_view":
                fn, tk = _memo_view(ticker), ticker
            elif feat == "research_run":
                fn, tk = _research_run(ticker), ticker
            elif feat == "pm_chat":
                fn, tk = _pm_chat(i, tickers), None
            elif feat == "dcf":
                fn, tk = _dcf(ticker), ticker
            elif feat == "comps":
                fn, tk = _comps(ticker), ticker
            else:
                fn, tk = _chart_commentary(ticker), ticker
            row = _Sample(feat, i, tk).run(fn)
            print(f"  {feat:<17s} #{i} {row['status']:<15s} "
                  f"llm=${row['llm']['cost_usd']:.4f} calls={row['llm']['n_calls']} "
                  f"provider_writes={row['provider']['provider_cache_writes']} "
                  f"{row['seconds']}s {row['note']}")
            samples.append(row)
        results[feat] = _aggregate(samples)

    summary = {
        "generated_at": started.isoformat() + "Z",
        "finished_at": _utcnow().isoformat() + "Z",
        "environment": {
            "enable_live_data": settings.enable_live_data,
            "use_demo_data": settings.use_demo_data,
            "llm_provider": settings.active_llm_provider,
            "app_env": settings.app_env,
            "database_dialect": os.environ.get("DATABASE_URL", "sqlite").split(":", 1)[0],
        },
        "tickers": tickers,
        "n": args.n,
        "research_n": args.research_n,
        "allowances": ALLOWANCES,
        "features": results,
        "worst_case_monthly_usd": _worst_case(results),
    }

    out = Path(args.out) if args.out else (
        REPO_ROOT / "docs" / "economics" / f"unit-costs-measured-{started.date().isoformat()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(_markdown(results))
    wc = summary["worst_case_monthly_usd"]
    print()
    print(f"Free worst case: ${wc['free']['total_usd']}  (threshold < ${THRESHOLDS['free_max_usd']})")
    print(f"Pro  worst case: ${wc['pro']['total_usd']}  (threshold < ${THRESHOLDS['pro_max_usd']})")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
