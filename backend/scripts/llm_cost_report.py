"""LLM cost report (Wave 1A).

Aggregates LLMCallLog rows over a configurable window and prints a
human-readable breakdown by agent + provider + slowest calls.

Usage from `backend/`:

    python -m scripts.llm_cost_report               # last 7 days
    python -m scripts.llm_cost_report --days 30     # last 30 days
    python -m scripts.llm_cost_report --run-id <X>  # one specific memo run

Costs nothing — only reads the local SQLite log.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm_metrics import (
    cost_per_agent,
    cost_per_provider,
    cost_per_run,
    slowest_calls,
)


def _fmt_int(n: int) -> str:
    return f"{n:>9,d}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7,
                   help="Aggregation window in days (default 7).")
    p.add_argument("--run-id", type=str, default=None,
                   help="Show per-call detail for one memo run.")
    args = p.parse_args()

    if args.run_id:
        info = cost_per_run(args.run_id)
        print(f"\nRun {args.run_id}")
        print(f"  calls={info['n_calls']}  tokens={info['tokens_total']:,d}  "
              f"duration={info['duration_ms_total']:,d}ms  "
              f"failures={info['n_failures']}\n")
        for c in info["calls"]:
            ok = "✓" if c["success"] else "✗"
            print(f"  {ok} {c['agent_name']:<24s} "
                  f"{c['provider']:<10s} {c['model']:<22s}  "
                  f"in={_fmt_int(c['tokens_in'])}  out={_fmt_int(c['tokens_out'])}  "
                  f"{c['duration_ms']:>6d}ms")
        return 0

    since = datetime.utcnow() - timedelta(days=args.days)

    print(f"\nLLM cost report — last {args.days} day(s)\n")
    by_agent = cost_per_agent(since=since)
    print("By agent:")
    print(f"  {'agent':<26s} {'calls':>6s}  {'tok in':>10s}  {'tok out':>10s}  "
          f"{'dur ms':>10s}  fails")
    for agent, agg in sorted(by_agent.items(), key=lambda kv: -(kv[1]["tokens_in"] + kv[1]["tokens_out"])):
        print(f"  {agent:<26s} {agg['n_calls']:>6d}  "
              f"{_fmt_int(agg['tokens_in'])}  {_fmt_int(agg['tokens_out'])}  "
              f"{_fmt_int(agg['duration_ms_total'])}  {agg['n_failures']}")

    print("\nBy provider:")
    by_prov = cost_per_provider(since=since)
    for prov, agg in by_prov.items():
        print(f"  {prov:<10s}  calls={agg['n_calls']:>5d}  "
              f"tok={_fmt_int(agg['tokens_in'] + agg['tokens_out'])}  "
              f"fails={agg['n_failures']}")

    slow = slowest_calls(since=since, n=10)
    if slow:
        print("\nSlowest calls:")
        for c in slow:
            print(f"  {c['duration_ms']:>6d}ms  {c['agent_name']:<24s} "
                  f"{c['provider']:<10s} {c['model']:<22s}  run={c['run_id'] or '-'}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
