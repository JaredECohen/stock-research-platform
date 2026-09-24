"""FMP-primary fundamentals re-pull client (owner decision 2026-09-24, item 3).

The deployed worker runs the same re-pull itself (`services/fmp_repull_ledger`:
dry run by default, execution only after the owner authorizes the reviewed
digest). This client is the admin-endpoint path for the owner, or for a
single ticker after a `repull_plan_mismatch`:

    python -m app.scripts.fundamentals_repull --base-url https://marketmosaic.onrender.com \
        --out <dir> --dry-run [--tickers A,B] [--pace 1.0]
    python -m app.scripts.fundamentals_repull --in-process --out <dir> --dry-run

It GETs `/api/admin/market-data/plan`, POSTs
`/api/admin/market-data/backfill?ticker=T&force_refresh=true&scope=fundamentals[&dry_run=true]`
for every `fundamentals_required` target sequentially, writes `<out>/<T>.json`,
skips tickers already written (resumable), records HTTP failures and
continues, then writes `<out>/summary.json` from `summarize`. The bearer
token comes from `ADMIN_API_TOKEN` and is never printed or written.
`--in-process` calls `sync_ticker` directly; use it only from a Render *web*
shell (the 512 MiB worker must not host a second interpreter).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..services.fmp_repull_ledger import compact_report
from ..services.fmp_repull_ledger import summarize as _summarize_compacts


def summarize(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-ticker compact records, totals and the review list (pure).

    `reports` maps ticker -> the backfill endpoint's JSON response, or
    `{"http_error": status}` for a request that failed.
    """
    compacts = {}
    for ticker, report in reports.items():
        if "http_error" in report:
            compacts[ticker] = {"status": "http_error", "success": False, "fundamentals_success": False,
                                "error": f"HTTP {report['http_error']}", "rows_quarantined": 0}
        else:
            compacts[ticker] = compact_report(report)
    return {"per_ticker": compacts, **_summarize_compacts(compacts)}


def _parse_tickers(raw: str | None) -> set[str] | None:
    return {t.strip().upper() for t in raw.split(",") if t.strip()} if raw else None


def _http(base_url: str) -> tuple[Any, Any]:
    import httpx

    token = os.environ.get("ADMIN_API_TOKEN", "")
    if not token:
        raise SystemExit("ADMIN_API_TOKEN is not set")
    client = httpx.Client(base_url=base_url.rstrip("/"), timeout=180.0,
                          headers={"Authorization": f"Bearer {token}"})

    def plan() -> dict[str, Any]:
        response = client.get("/api/admin/market-data/plan")
        response.raise_for_status()
        return response.json()

    def sync(ticker: str, dry_run: bool) -> dict[str, Any]:
        params = {"ticker": ticker, "force_refresh": "true", "scope": "fundamentals"}
        if dry_run:
            params["dry_run"] = "true"
        response = client.post("/api/admin/market-data/backfill", params=params)
        if response.status_code != 200:
            return {"http_error": response.status_code}
        return response.json()

    return plan, sync


def _in_process() -> tuple[Any, Any]:
    from ..services.market_data_backfill import backfill_plan, sync_ticker

    def sync(ticker: str, dry_run: bool) -> dict[str, Any]:
        return sync_ticker(ticker, force_refresh=True, scope="fundamentals", dry_run=dry_run)

    return backfill_plan, sync


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url")
    source.add_argument("--in-process", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tickers")
    parser.add_argument("--pace", type=float, default=1.0)
    args = parser.parse_args(argv)
    plan, sync = _in_process() if args.in_process else _http(args.base_url)
    wanted = _parse_tickers(args.tickers)
    args.out.mkdir(parents=True, exist_ok=True)
    targets = [t["ticker"] for t in plan()["targets"] if t["fundamentals_required"]
               and (wanted is None or t["ticker"] in wanted)]
    reports: dict[str, dict[str, Any]] = {}
    for ticker in targets:
        path = args.out / f"{ticker.replace('/', '_')}.json"
        if path.exists():
            reports[ticker] = json.loads(path.read_text())
            continue
        try:
            report = sync(ticker, args.dry_run)
        except Exception as exc:  # recorded and continued; text may quote a URL
            report = {"http_error": type(exc).__name__}
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
        reports[ticker] = report
        print(f"{ticker}: {report.get('status', report.get('http_error'))}", flush=True)
        time.sleep(args.pace)
    summary = summarize(reports)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_ticker"}, indent=2, sort_keys=True, default=str))
    return 0 if not summary["review_list"] and not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
