"""QA harness — offline evaluation rig for single-stock memos.

Loops every (ticker, question-rubric) pair, generates a memo via the
FastAPI TestClient, and scores each rubric expectation against the
memo's actual shape. Outputs:

    out/<run-id>/results.jsonl            one row per attempt
    out/<run-id>/summary.csv              flat per-attempt scores
    out/<run-id>/failure_taxonomy.json    issue counter + axis averages
    out/<run-id>/improvement_report.md    human-readable digest
    out/<run-id>/improvement_backlog.json prioritized work list
    out/<run-id>/run_state.json           resume metadata

Usage:
    python -m qa.run_matrix --smoke
    python -m qa.run_matrix --full
    python -m qa.run_matrix --resume       # pick up after the last checkpoint
    python -m qa.run_matrix --tickers AAPL,MSFT --questions q_thesis,q_dcf_signal

Tests should run this against the in-process DemoProvider fixture to
keep it deterministic + cheap. Live runs are gated by `--live` and need
real provider keys.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "out"

log = logging.getLogger("qa.run_matrix")

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Memo fetch
# ---------------------------------------------------------------------------

def fetch_memo(client, ticker: str, ondemand: bool = True) -> Tuple[int, Dict[str, Any]]:
    """Fetch a memo via the FastAPI TestClient.

    Returns (status_code, payload_dict). Payload includes the raw memo
    on 200, or an error envelope otherwise.
    """
    params: Dict[str, Any] = {}
    if ondemand:
        params["ondemand"] = "true"
    try:
        resp = client.get(f"/api/stocks/{ticker.upper()}/memo", params=params, timeout=120)
    except Exception as exc:  # pragma: no cover
        return 599, {"error": "client_raised", "detail": str(exc)}
    try:
        body = resp.json()
    except Exception:
        body = {"error": "non_json_response", "raw": resp.text[:500]}
    return resp.status_code, body


# ---------------------------------------------------------------------------
# Rubric scoring
# ---------------------------------------------------------------------------

@dataclass
class RubricResult:
    expectation: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"expectation": self.expectation, "passed": self.passed, "detail": self.detail}


def score_memo(memo: Dict[str, Any], expectations: List[str]) -> List[RubricResult]:
    """Evaluate a memo against a set of rubric expectations.

    Each expectation is a small predicate; pass/fail + a one-line detail.
    Adding a new rubric tag: register it in `_RUBRIC_FUNCS`.
    """
    results: List[RubricResult] = []
    for tag in expectations:
        fn = _RUBRIC_FUNCS.get(tag)
        if fn is None:
            results.append(RubricResult(tag, False, "no rubric implementation"))
            continue
        try:
            passed, detail = fn(memo)
        except Exception as exc:
            passed, detail = False, f"rubric raised: {exc}"
        results.append(RubricResult(tag, passed, detail))
    return results


def _r_thesis(memo: Dict[str, Any]) -> Tuple[bool, str]:
    text = (memo.get("one_sentence_thesis") or "").strip()
    return (bool(text) and len(text) > 20, f"thesis_len={len(text)}")


def _r_bull_bear_balanced(memo: Dict[str, Any]) -> Tuple[bool, str]:
    bull = (memo.get("bull_case") or {}).get("key_points") or []
    bear = (memo.get("bear_case") or {}).get("key_points") or []
    ok = len(bull) >= 2 and len(bear) >= 2
    return ok, f"bull={len(bull)} bear={len(bear)}"


def _r_rating_valid(memo: Dict[str, Any]) -> Tuple[bool, str]:
    valid = {"Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"}
    rating = memo.get("rating_label")
    return (rating in valid, f"rating={rating}")


def _r_rating_supported(memo: Dict[str, Any]) -> Tuple[bool, str]:
    rating = (memo.get("rating_label") or "").lower()
    thesis = (memo.get("one_sentence_thesis") or "").lower()
    if not rating or not thesis:
        return False, "missing rating or thesis"
    # Lightweight directional check: if rating is bullish, thesis shouldn't
    # be dominated by bear-language and vice versa.
    bull_words = ("growth", "expand", "beat", "upside", "leading", "moat", "tailwind", "catalyst")
    bear_words = ("decline", "miss", "headwind", "pressure", "compressing", "weak", "risk", "loss")
    bull_hits = sum(w in thesis for w in bull_words)
    bear_hits = sum(w in thesis for w in bear_words)
    if "bull" in rating:
        return (bull_hits >= bear_hits, f"bull={bull_hits} bear={bear_hits}")
    if "bear" in rating:
        return (bear_hits >= bull_hits, f"bull={bull_hits} bear={bear_hits}")
    return True, "neutral"


def _agent_finding_present(memo: Dict[str, Any], key: str) -> Tuple[bool, str]:
    fin = memo.get(key) or {}
    summary_len = len((fin.get("summary") or ""))
    kp = fin.get("key_points") or []
    ok = bool(fin) and (summary_len > 30 or len(kp) >= 2)
    return ok, f"summary_len={summary_len} kp={len(kp)}"


def _r_sector_finding(memo: Dict[str, Any]) -> Tuple[bool, str]:
    return _agent_finding_present(memo, "sector_agent_view")


def _r_sector_overlay_cited(memo: Dict[str, Any]) -> Tuple[bool, str]:
    fin = memo.get("sector_agent_view") or {}
    data = fin.get("data") or {}
    ctx = data.get("sector_data_context") or {}
    bundles = (ctx.get("overlays") or {}).get("bundles") or {}
    hints: List[str] = []
    for bundle in bundles.values():
        if bundle.get("available"):
            hints.extend(bundle.get("narrative_hints") or [])
    return (len(hints) >= 1, f"overlay_hints={len(hints)}")


def _r_earnings_finding(memo: Dict[str, Any]) -> Tuple[bool, str]:
    return _agent_finding_present(memo, "earnings_agent_view")


def _r_earnings_non_generic(memo: Dict[str, Any]) -> Tuple[bool, str]:
    text = (memo.get("earnings_agent_view") or {}).get("summary") or ""
    generic = ("management tone was constructive" in text.lower()
               and len(text) < 200)
    return (not generic and len(text) > 50, f"summary_len={len(text)} generic={generic}")


def _r_filing_finding(memo: Dict[str, Any]) -> Tuple[bool, str]:
    return _agent_finding_present(memo, "filing_agent_view")


def _r_dcf_present(memo: Dict[str, Any]) -> Tuple[bool, str]:
    dcf = memo.get("dcf_summary") or {}
    return (bool(dcf), f"keys={len(dcf)}")


def _r_dcf_has_fair_value(memo: Dict[str, Any]) -> Tuple[bool, str]:
    dcf = memo.get("dcf_summary") or {}
    keys = ("fair_value", "target_price", "implied_per_share", "fair_value_per_share", "wacc")
    found = [k for k in keys if dcf.get(k) is not None]
    return (bool(found), f"present={found}")


def _r_comps_finding(memo: Dict[str, Any]) -> Tuple[bool, str]:
    return _agent_finding_present(memo, "comps_agent_view")


def _r_macro_finding(memo: Dict[str, Any]) -> Tuple[bool, str]:
    return _agent_finding_present(memo, "macro_sensitivity")


def _r_risks_listed(memo: Dict[str, Any]) -> Tuple[bool, str]:
    risks = memo.get("key_risks") or []
    return (len(risks) >= 2, f"risks={len(risks)}")


def _r_critic_present(memo: Dict[str, Any]) -> Tuple[bool, str]:
    critic = memo.get("risk_committee_challenge") or {}
    return (bool(critic) and bool(critic.get("overall_assessment")),
            f"assessment_len={len((critic.get('overall_assessment') or ''))}")


def _r_critic_added(memo: Dict[str, Any]) -> Tuple[bool, str]:
    critic = memo.get("risk_committee_challenge") or {}
    challenges = critic.get("challenges") or []
    revisions = critic.get("suggested_revisions") or []
    underweighted = critic.get("underweighted_risks") or []
    total = len(challenges) + len(revisions) + len(underweighted)
    return (total >= 1, f"challenges={len(challenges)} revs={len(revisions)} under={len(underweighted)}")


def _r_catalysts_listed(memo: Dict[str, Any]) -> Tuple[bool, str]:
    cats = memo.get("catalysts") or []
    return (len(cats) >= 1, f"catalysts={len(cats)}")


def _r_sector_data_context_present(memo: Dict[str, Any]) -> Tuple[bool, str]:
    ctx = (memo.get("sector_agent_view") or {}).get("data", {}).get("sector_data_context") or {}
    catalog = ctx.get("discovered_catalog") or {}
    any_hits = any(catalog.values())
    return (bool(ctx) and any_hits, f"axes_with_hits={sum(1 for v in catalog.values() if v)}")


def _r_overlays_ran(memo: Dict[str, Any]) -> Tuple[bool, str]:
    ctx = (memo.get("sector_agent_view") or {}).get("data", {}).get("sector_data_context") or {}
    overlays = (ctx.get("overlays") or {}).get("overlays_run") or []
    return (len(overlays) >= 1, f"overlays_run={overlays}")


def _r_no_degradation(memo: Dict[str, Any]) -> Tuple[bool, str]:
    degraded = memo.get("degraded_agents") or []
    return (len(degraded) == 0, f"degraded={degraded}")


_RUBRIC_FUNCS = {
    "one_sentence_thesis_non_empty":    _r_thesis,
    "bull_bear_balanced":                _r_bull_bear_balanced,
    "rating_label_valid":                _r_rating_valid,
    "rating_supported_by_thesis":        _r_rating_supported,
    "sector_finding_present":            _r_sector_finding,
    "sector_kpi_or_overlay_cited":       _r_sector_overlay_cited,
    "earnings_finding_present":          _r_earnings_finding,
    "earnings_summary_non_generic":      _r_earnings_non_generic,
    "filing_finding_present":            _r_filing_finding,
    "dcf_summary_present":               _r_dcf_present,
    "dcf_has_fair_value_or_wacc":        _r_dcf_has_fair_value,
    "comps_finding_present":             _r_comps_finding,
    "macro_finding_present":             _r_macro_finding,
    "risks_listed_non_empty":            _r_risks_listed,
    "critic_present":                    _r_critic_present,
    "critic_added_challenge_or_revision": _r_critic_added,
    "catalysts_listed_non_empty":        _r_catalysts_listed,
    "sector_data_context_present":       _r_sector_data_context_present,
    "overlays_ran_at_least_one":         _r_overlays_ran,
    "degraded_agents_empty":             _r_no_degradation,
}


# ---------------------------------------------------------------------------
# Backlog generator
# ---------------------------------------------------------------------------

def build_backlog(issue_counter: Counter) -> List[Dict[str, Any]]:
    """Translate the failure histogram into a prioritized backlog."""
    rules = [
        ("one_sentence_thesis_non_empty", 1,
         "Tighten one-sentence thesis generation",
         "An empty / placeholder thesis is the most-visible memo failure."),
        ("rating_label_valid", 1,
         "Validate rating enum at the synthesis boundary",
         "Invalid rating labels break the UI badge + screener."),
        ("rating_supported_by_thesis", 2,
         "Tighten PM synthesis cross-checks",
         "Rating direction must match thesis language."),
        ("sector_finding_present", 1,
         "Harden sector agent fallback path",
         "Empty sector findings break the bull/bear scaffold."),
        ("sector_kpi_or_overlay_cited", 2,
         "Force sector LLM to cite at least one overlay number",
         "Generic sector summaries don't earn the keep."),
        ("earnings_finding_present", 2,
         "Harden earnings agent fallback path",
         "Missing earnings findings reduce memo trust."),
        ("earnings_summary_non_generic", 2,
         "Penalize tone-only earnings summaries in synthesis",
         "Generic 'tone constructive' summaries are a memo smell."),
        ("filing_finding_present", 3,
         "Backfill filings + filing agent fallback",
         "Filings-empty memos are usually a data-pipeline issue."),
        ("dcf_summary_present", 1,
         "Force a DCF summary to render even with sparse fundamentals",
         "Memos without DCF feel half-finished."),
        ("dcf_has_fair_value_or_wacc", 2,
         "DCF must emit at least fair_value OR WACC",
         "Empty DCF dicts cause silent downstream failures."),
        ("comps_finding_present", 3,
         "Backfill comps for analyzed_on_demand tier",
         "Comps empty for long-tail tickers."),
        ("macro_finding_present", 3,
         "Confirm macro_agent path on memo workflow",
         "Macro overlay missing in memo response."),
        ("risks_listed_non_empty", 2,
         "Risk extractor must yield ≥2 items",
         "Empty risk lists undercut the memo's defensiveness."),
        ("critic_present", 2,
         "Critic must always produce an assessment line",
         "Missing critic block hides the cross-family review value."),
        ("critic_added_challenge_or_revision", 3,
         "Encourage critic to surface ≥1 specific challenge",
         "A no-op critic is invisible to readers."),
        ("catalysts_listed_non_empty", 3,
         "Catalysts list should always have ≥1 item",
         "Memo without catalysts feels static."),
        ("sector_data_context_present", 2,
         "Verify sector_data_context attaches in finding.data",
         "Missing context means the smart-sector capability is bypassed."),
        ("overlays_ran_at_least_one", 2,
         "At least one overlay should run per memo",
         "Sectors with no overlay are missing the new value-add."),
        ("degraded_agents_empty", 3,
         "Reduce degraded-agent count under nominal config",
         "Degraded runs indicate provider or config drift."),
    ]
    backlog: List[Dict[str, Any]] = []
    for tag, priority, title, reason in rules:
        count = issue_counter.get(tag, 0)
        if count == 0:
            continue
        backlog.append({
            "expectation": tag,
            "failures": count,
            "priority": priority,
            "title": title,
            "reason": reason,
        })
    backlog.sort(key=lambda r: (r["priority"], -r["failures"]))
    return backlog


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def summarize_run(results: List[Dict[str, Any]], out_dir: Path) -> None:
    issue_counter: Counter = Counter()
    per_category_passes: Dict[str, List[float]] = defaultdict(list)
    per_ticker_passes: Dict[str, List[float]] = defaultdict(list)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "run_id", "ticker", "question_id", "category",
            "status_code", "expectations_total", "expectations_passed",
            "score_pct", "failed_expectations",
        ])
        writer.writeheader()
        for row in results:
            total = len(row.get("rubric_results", []))
            passed = sum(1 for r in row["rubric_results"] if r["passed"])
            failed = [r["expectation"] for r in row["rubric_results"] if not r["passed"]]
            issue_counter.update(failed)
            pct = (passed / total) if total else 0.0
            per_category_passes[row["category"]].append(pct)
            per_ticker_passes[row["ticker"]].append(pct)
            writer.writerow({
                "run_id": row["run_id"],
                "ticker": row["ticker"],
                "question_id": row["question_id"],
                "category": row["category"],
                "status_code": row["status_code"],
                "expectations_total": total,
                "expectations_passed": passed,
                "score_pct": round(pct * 100, 1),
                "failed_expectations": ";".join(failed),
            })

    failure_taxonomy = {
        "total_attempts": len(results),
        "successful_responses": sum(1 for r in results if r.get("status_code") == 200),
        "issue_counts": issue_counter.most_common(),
        "category_pass_rates": {
            cat: round(sum(v) / len(v), 3)
            for cat, v in sorted(per_category_passes.items())
        },
        "ticker_pass_rates": {
            tic: round(sum(v) / len(v), 3)
            for tic, v in sorted(per_ticker_passes.items())
        },
    }
    write_json(out_dir / "failure_taxonomy.json", failure_taxonomy)

    backlog = build_backlog(issue_counter)
    write_json(out_dir / "improvement_backlog.json", backlog)

    avg = (sum(failure_taxonomy["ticker_pass_rates"].values())
           / max(1, len(failure_taxonomy["ticker_pass_rates"])))
    worst_categories = sorted(failure_taxonomy["category_pass_rates"].items(),
                              key=lambda x: x[1])[:5]
    best_categories = sorted(failure_taxonomy["category_pass_rates"].items(),
                             key=lambda x: -x[1])[:5]
    worst_tickers = sorted(failure_taxonomy["ticker_pass_rates"].items(),
                           key=lambda x: x[1])[:5]
    best_tickers = sorted(failure_taxonomy["ticker_pass_rates"].items(),
                          key=lambda x: -x[1])[:5]

    lines = [
        f"# QA Improvement Report",
        f"",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        f"",
        f"## Headline",
        f"- Attempts: {len(results)}",
        f"- Successful memos (HTTP 200): {failure_taxonomy['successful_responses']}",
        f"- Average rubric pass-rate (per-ticker mean): {avg:.1%}",
        f"",
        f"## Best rubric categories",
        *[f"- `{c}` — {v:.1%}" for c, v in best_categories],
        f"",
        f"## Worst rubric categories",
        *[f"- `{c}` — {v:.1%}" for c, v in worst_categories],
        f"",
        f"## Best tickers",
        *[f"- `{t}` — {v:.1%}" for t, v in best_tickers],
        f"",
        f"## Worst tickers",
        *[f"- `{t}` — {v:.1%}" for t, v in worst_tickers],
        f"",
        f"## Top 10 failing expectations",
        *[f"- `{tag}` — failed {count} times" for tag, count in issue_counter.most_common(10)],
        f"",
        f"## Prioritized backlog",
        *[
            f"- **P{item['priority']}**: {item['title']} ({item['failures']} failures) — {item['reason']}"
            for item in backlog[:12]
        ],
        f"",
        f"## Notes",
        f"- Results are checkpointed in `results.jsonl`. Re-run with `--resume` to skip completed pairs.",
        f"- Each rubric expectation is deterministic. Failing one isn't necessarily a bug — it's a regression smell.",
    ]
    (out_dir / "improvement_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test-client setup
# ---------------------------------------------------------------------------

def get_test_client():
    """Lazily construct a TestClient against the running FastAPI app.

    Imported lazily because importing app.main is expensive (pulls in
    every provider, every route, etc.) and we want the CLI to start
    fast for --help / --resume even if the app isn't fully boot-able.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="qa.run_matrix")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only the first 3 tickers × 3 questions.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full matrix (default if neither --smoke nor --resume given).")
    parser.add_argument("--resume", action="store_true",
                        help="Pick up the most recent run-dir and skip completed pairs.")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker filter (overrides smoke/full counts).")
    parser.add_argument("--questions", type=str, default=None,
                        help="Comma-separated question_id filter.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory under qa/out/ (default: timestamped run).")
    parser.add_argument("--live", action="store_true",
                        help="Don't enable ENABLE_LIVE_DATA=false (default forces demo mode).")
    parser.add_argument("--verbose", action="store_true", help="Stream per-pair scores.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s qa: %(message)s")

    # Force demo mode unless --live was passed. Matches conftest behavior.
    if not args.live:
        os.environ.setdefault("ENABLE_LIVE_DATA", "false")
        os.environ.setdefault("USE_DEMO_DATA", "true")

    tickers = load_json(ROOT / "tickers.json")
    questions = load_json(ROOT / "questions.json")

    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        tickers = [t for t in tickers if t["ticker"].upper() in wanted]
    if args.questions:
        wanted = {q.strip() for q in args.questions.split(",") if q.strip()}
        questions = [q for q in questions if q["id"] in wanted]
    if args.smoke and not args.tickers:
        tickers = tickers[:3]
    if args.smoke and not args.questions:
        questions = questions[:3]

    if not tickers or not questions:
        log.error("No tickers or questions to run after filters.")
        return 2

    # Resolve output directory
    if args.resume:
        runs = sorted(DEFAULT_OUT_ROOT.glob("*"))
        if not runs:
            log.error("--resume given but no prior runs found under %s", DEFAULT_OUT_ROOT)
            return 3
        out_dir = runs[-1]
        log.info("Resuming run dir: %s", out_dir)
    else:
        run_id = args.out or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = DEFAULT_OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    # Resume support
    completed_pairs: set[Tuple[str, str]] = set()
    if args.resume:
        for row in load_jsonl(results_path):
            completed_pairs.add((row["ticker"], row["question_id"]))
        log.info("Skipping %d already-completed pairs.", len(completed_pairs))

    # Construct client. This boots the FastAPI app (including the seeder).
    log.info("Booting FastAPI TestClient (this takes a few seconds)...")
    client = get_test_client()
    log.info("Client ready. Running %d tickers × %d questions.", len(tickers), len(questions))

    started = time.time()
    pair_count = 0
    for ticker_row in tickers:
        ticker = ticker_row["ticker"]
        status_code, memo_payload = fetch_memo(client, ticker, ondemand=True)
        for q in questions:
            key = (ticker, q["id"])
            if key in completed_pairs:
                continue
            pair_count += 1
            rubric = score_memo(memo_payload if status_code == 200 else {}, q.get("expects", []))
            row = {
                "run_id": out_dir.name,
                "ticker": ticker,
                "ticker_sector": ticker_row.get("sector"),
                "ticker_archetype": ticker_row.get("archetype"),
                "question_id": q["id"],
                "category": q["category"],
                "status_code": status_code,
                "rubric_results": [r.to_dict() for r in rubric],
                "passed_count": sum(1 for r in rubric if r.passed),
                "total_count": len(rubric),
                "memo_excerpt": _memo_excerpt(memo_payload) if status_code == 200 else None,
                "error": memo_payload.get("error") if status_code != 200 else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(results_path, row)
            if args.verbose:
                log.info("%s/%s: %d/%d passed",
                         ticker, q["id"], row["passed_count"], row["total_count"])

    elapsed = time.time() - started
    log.info("Completed %d pairs in %.1fs", pair_count, elapsed)

    # Reload + summarize (so we include resumed-from rows too)
    all_results = load_jsonl(results_path)
    summarize_run(all_results, out_dir)
    write_json(out_dir / "run_state.json", {
        "completed_pairs": len(all_results),
        "elapsed_seconds": elapsed,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    log.info("Report written to %s", out_dir / "improvement_report.md")
    return 0


def _memo_excerpt(memo: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a memo down to the fields the rubric needs + a couple of
    human-readable snippets so the JSONL stays small enough to grep."""
    return {
        "rating_label": memo.get("rating_label"),
        "one_sentence_thesis": memo.get("one_sentence_thesis"),
        "confidence_score": memo.get("confidence_score"),
        "degraded_agents": memo.get("degraded_agents"),
        "sector_overlays_run": (
            ((memo.get("sector_agent_view") or {}).get("data") or {})
            .get("sector_data_context", {}).get("overlays", {}).get("overlays_run")
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
