"""S14 merge gate — replay the 7(b) evidence rule over stored memo bodies.

Owner decision 7(b) authorizes enforcing "a Bullish rating on overvalued
valuation evidence needs a stated reason, or it becomes Neutral". Before
that ships, this replay answers the owner's stop question on real memos:
would the centred two-signal rule downgrade more than 25% of recent Bullish
full runs, or make `overvalued` the majority verdict? Either one means
stop and ask instead of enforcing.

It is READ-ONLY and offline: it reads JSON bodies from a directory (the W3
evidence bundle's `bodies/`, fetched earlier with `?version=N` only, or any
directory of memo JSON), recomputes the evidence verdict with the SAME
function the pipeline uses (`memo_quality.valuation_evidence_verdict`), and
writes a report. No database, no network, no LLM, no memo prose in the
output — only tickers, versions, ratings and the numeric signals.

Two sets are reported:
  * full runs: live-mode `first_run` / `full_reanalysis` versions generated
    inside the window (default 2026-08-09 .. 2026-09-22, the W3 sample of
    25 live full runs);
  * stored live: the latest version per ticker whose mode is live (what a
    reader is served; patches included).

Metrics per set: verdict distribution; the UPPER-BOUND Bullish downgrade
rate (every divergent Bullish assumed to lack an accepted reason — stored
memos predate the reason field); the Bearish mirror; the thesis-rewrite
rate under the new guard; and the accept rate (divergent memos whose
stored reason would pass — zero by construction on pre-W2b bodies).

Usage from `backend/`:

    python -m scripts.rating_reconciliation_replay --bodies <dir> \\
        [--json-out report.json] [--md-out receipt.md]

Exit status 0 when below the stop threshold, 3 when the stop condition
holds (so a pipeline can gate on it), 2 on unusable input.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents import memo_quality  # noqa: E402
from app.schemas import CriticReview, RatingReconciliation  # noqa: E402

STOP_DOWNGRADE_RATE = 0.25
DEFAULT_WINDOW = (date(2026, 8, 9), date(2026, 9, 22))
FULL_RUN_TRIGGERS = frozenset({"first_run", "full_reanalysis"})
_VERDICT_WORDS = ("undervalued", "overvalued", "fairly priced")


def _unwrap(raw: Any, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(headers, memo) from a W3 fetch record {status, headers, body} or a
    bare memo dict. Raises ValueError on anything else."""
    if isinstance(raw, dict) and "body" in raw and isinstance(raw.get("headers"), dict):
        body = raw["body"]
        if isinstance(body, str):
            body = json.loads(body)
        if not isinstance(body, dict):
            raise ValueError(f"{name}: body is not a memo object")
        return raw["headers"], body
    if isinstance(raw, dict) and "ticker" in raw and "rating_label" in raw:
        return {}, raw
    raise ValueError(f"{name}: not a memo body")


def load_bodies(directory: Path) -> list[dict[str, Any]]:
    """Every `*.json` under `directory` as a normalized record. Files that
    are not memo bodies raise: a silently skipped file would shrink the
    sample the gate is judged on."""
    records = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        headers, memo = _unwrap(raw, path.name)
        records.append(normalize(headers, memo, name=path.name))
    if not records:
        raise ValueError(f"no memo bodies in {directory}")
    return records


def normalize(headers: dict[str, Any], memo: dict[str, Any], *, name: str = "") -> dict[str, Any]:
    """The fields the replay needs, from a stored memo payload."""
    version = headers.get("x-memo-version") or memo.get("version")
    if version is None and "_v" in name:
        version = name.rsplit("_v", 1)[1].split(".")[0]
    trigger = headers.get("x-memo-trigger") or memo.get("trigger") or ""
    sc = memo.get("scorecard") or {}
    cat = ((sc.get("categories") or {}).get("valuation") or {}) if isinstance(sc, dict) else {}
    fam_cov = cat.get("coverage")
    if fam_cov is None and isinstance(sc, dict):
        fam_cov = sc.get("coverage")
    vv = memo.get("valuation_verdict") or {}
    dcf = memo.get("dcf_summary") or {}
    # No PM adjustment fired when `dcf_initial_summary` is empty: the
    # initial model IS the final one.
    initial = memo.get("dcf_initial_summary") or dcf
    quality = memo.get("quality") or {}
    rec = (quality.get("rating_reconciliation") or {}) if isinstance(quality, dict) else {}
    critic = memo.get("risk_committee_challenge") or {}
    return {
        "ticker": str(memo.get("ticker") or ""),
        "version": int(version) if version is not None else None,
        "trigger": str(trigger),
        "generated_at": str(memo.get("generated_at") or headers.get("x-memo-generated-at") or ""),
        "mode": str(memo.get("generation_mode") or ""),
        # The comps vote is withheld for sectors where EV is not meaningful.
        # Stored bodies carry the premium but not the multiples, so the
        # negative-multiple guard can only use the premium's own bound.
        "sector": str(memo.get("sector") or ""),
        "rating": str(memo.get("rating_label") or ""),
        "thesis": str(memo.get("one_sentence_thesis") or ""),
        "family_pct": cat.get("percentile"),
        "family_coverage": fam_cov,
        "comps_premium": vv.get("comps_ev_ebitda_premium"),
        "dcf_initial": initial.get("base_upside"),
        "dcf_tv_clamped": bool(initial.get("tv_clamped")),
        "dcf_final": dcf.get("base_upside"),
        "reason": str(rec.get("reason") or ""),
        "critic_mode": str(critic.get("review_mode") or "unknown"),
        "critic_assessment": str(critic.get("valuation_divergence_assessment") or "not_assessed"),
    }


def _looks_like_anti_pattern(text: str) -> bool:
    # The guard's own detector; imported lazily because graph.py pulls in
    # the whole memo pipeline (still no I/O at import).
    from app.agents.graph import _looks_like_anti_pattern_thesis
    return _looks_like_anti_pattern_thesis(text)


def replay_one(r: dict[str, Any]) -> dict[str, Any]:
    """The rule applied to one stored memo. The published rating is the
    post-blend rating the rule binds."""
    vv = memo_quality.valuation_evidence_verdict(
        family_pct=r["family_pct"], family_coverage=r["family_coverage"],
        comps_premium=r["comps_premium"], dcf_initial_upside=r["dcf_initial"],
        dcf_initial_tv_clamped=r["dcf_tv_clamped"], dcf_final_upside=r["dcf_final"],
        sector=r.get("sector") or None,
    )
    critic = CriticReview(
        overall_assessment="replay", review_mode=r["critic_mode"]
        if r["critic_mode"] in ("unknown", "pending", "live", "rule_based", "unavailable") else "unknown",
        valuation_divergence_assessment=r["critic_assessment"]
        if r["critic_assessment"] in ("supported", "unsupported") else "not_assessed",
    )
    rec = memo_quality.reconcile_rating(
        blended_rating=r["rating"], verdict=vv,
        pm=RatingReconciliation(pm_rating=r["rating"], reason=r["reason"]), critic=critic,
    )
    thesis = r["thesis"]
    stated = next((w for w in _VERDICT_WORDS if w in thesis.lower()), None)
    rewrite, _word = memo_quality.thesis_rewrite_word(
        stated=stated, rating=rec.final_rating, verdict=vv,
        accepted=rec.outcome == "accepted", anti_pattern=_looks_like_anti_pattern(thesis),
    )
    return {
        "ticker": r["ticker"], "version": r["version"], "trigger": r["trigger"],
        "generated_at": r["generated_at"][:10], "mode": r["mode"], "rating": r["rating"],
        "family_pct": _round(r["family_pct"], 1), "family_coverage": _round(r["family_coverage"], 2),
        "comps_premium": _round(r["comps_premium"], 3), "dcf_initial": _round(r["dcf_initial"], 3),
        "dcf_tv_clamped": r["dcf_tv_clamped"], "dcf_final": _round(r["dcf_final"], 3),
        "votes": dict(vv.signals.get("votes") or {}), "verdict": vv.verdict,
        "diverges": rec.divergence, "outcome": rec.outcome, "final_rating": rec.final_rating,
        "reason_present": bool(r["reason"].strip()), "thesis_rewrite": bool(rewrite),
    }


def _round(v: Any, n: int) -> float | None:
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    bull = [r for r in rows if memo_quality.rating_direction(r["rating"]) > 0]
    bear = [r for r in rows if memo_quality.rating_direction(r["rating"]) < 0]
    divergent = [r for r in rows if r["diverges"]]
    bull_dg = [r for r in bull if r["diverges"]]
    bear_dg = [r for r in bear if r["diverges"]]
    accepted = [r for r in divergent if r["outcome"] == "accepted"]
    verdicts = Counter(r["verdict"] for r in rows)
    return {
        "n": n,
        "verdicts": {k: verdicts.get(k, 0) for k in ("undervalued", "fairly_priced", "overvalued", "mixed")},
        "ratings": dict(Counter(r["rating"] for r in rows)),
        "bullish": len(bull),
        "bullish_downgrade_upper_bound": len(bull_dg),
        "bullish_downgrade_rate": (len(bull_dg) / len(bull)) if bull else 0.0,
        "bearish": len(bear),
        "bearish_downgrade_upper_bound": len(bear_dg),
        "divergent": len(divergent),
        "accepted": len(accepted),
        "accept_rate": (len(accepted) / len(divergent)) if divergent else 0.0,
        "reasons_present": sum(1 for r in divergent if r["reason_present"]),
        "thesis_rewrites": sum(1 for r in rows if r["thesis_rewrite"]),
        "thesis_rewrite_rate": (sum(1 for r in rows if r["thesis_rewrite"]) / n) if n else 0.0,
        "overvalued_majority": verdicts.get("overvalued", 0) > n / 2 if n else False,
    }


def replay(records: list[dict[str, Any]], *, window: tuple[date, date] = DEFAULT_WINDOW) -> dict[str, Any]:
    """The report over normalized records (`normalize`)."""
    lo, hi = window
    full = [
        r for r in records
        if r["mode"] == "live" and r["trigger"] in FULL_RUN_TRIGGERS
        and r["generated_at"][:10] and lo.isoformat() <= r["generated_at"][:10] <= hi.isoformat()
    ]
    latest: dict[str, dict[str, Any]] = {}
    for r in records:
        cur = latest.get(r["ticker"])
        if cur is None or (r["version"] or 0) > (cur["version"] or 0):
            latest[r["ticker"]] = r
    stored = [r for r in latest.values() if r["mode"] == "live"]
    full_rows = [replay_one(r) for r in sorted(full, key=lambda x: (x["ticker"], x["version"] or 0))]
    stored_rows = [replay_one(r) for r in sorted(stored, key=lambda x: x["ticker"])]
    fs, ss = _stats(full_rows), _stats(stored_rows)
    reasons = []
    if fs["bullish_downgrade_rate"] > STOP_DOWNGRADE_RATE:
        reasons.append(
            f"upper-bound Bullish downgrade rate {fs['bullish_downgrade_rate']:.0%} of full runs "
            f"exceeds {STOP_DOWNGRADE_RATE:.0%}")
    if fs["overvalued_majority"]:
        reasons.append("overvalued is the majority verdict over the full runs")
    if ss["overvalued_majority"]:
        reasons.append("overvalued is the majority verdict over the stored live memos")
    return {
        "method": memo_quality.METHOD,
        "window": [lo.isoformat(), hi.isoformat()],
        "bodies_read": len(records),
        "stop": bool(reasons),
        "stop_reasons": reasons,
        "full_runs": fs,
        "stored_live": ss,
        "full_run_rows": full_rows,
        "stored_live_rows": stored_rows,
    }


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{v * 100:+.0f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """The receipt: counts plus a per-memo table, no memo prose."""
    fs, ss = report["full_runs"], report["stored_live"]
    lines = [
        "# S14 rating-reconciliation replay receipt",
        "",
        f"Method `{report['method']}`; full-run window {report['window'][0]} .. {report['window'][1]}; "
        f"{report['bodies_read']} bodies read (read-only, offline).",
        "",
        f"**Decision input: {'STOP — ask the owner' if report['stop'] else 'below threshold — enforce'}.**"
        + (" " + "; ".join(report["stop_reasons"]) + "." if report["stop_reasons"] else ""),
        "",
        "| Set | n | undervalued | fairly priced | overvalued | mixed | Bullish | Bullish downgrade (upper bound) | "
        "Bearish downgrade (upper bound) | accepted / divergent | thesis rewrites |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, st in (("Live full runs", fs), ("Stored live memos", ss)):
        v = st["verdicts"]
        lines.append(
            f"| {label} | {st['n']} | {v['undervalued']} | {v['fairly_priced']} | {v['overvalued']} | {v['mixed']} "
            f"| {st['bullish']} | {st['bullish_downgrade_upper_bound']} ({st['bullish_downgrade_rate']:.0%}) "
            f"| {st['bearish_downgrade_upper_bound']} | {st['accepted']} / {st['divergent']} "
            f"| {st['thesis_rewrites']} ({st['thesis_rewrite_rate']:.0%}) |"
        )
    lines += [
        "",
        "Accept rate: no stored body carries a divergence reason (the field postdates them), so every divergent "
        "Bullish/Bearish memo is counted as downgraded — the downgrade rates are upper bounds.",
    ]
    for label, key in (("Live full runs", "full_run_rows"), ("Stored live memos", "stored_live_rows")):
        lines += [
            "", f"## {label}", "",
            "| Ticker | v | trigger | generated | rating | family pct (cov) | EV/EBITDA prem | DCF initial | "
            "DCF final | votes | verdict | outcome | final | thesis rewrite |",
            "|---|---:|---|---|---|---|---:|---:|---:|---|---|---|---|---|",
        ]
        for r in report[key]:
            fam = "n/a" if r["family_pct"] is None else f"{r['family_pct']:.0f} ({r['family_coverage']})"
            votes = ", ".join(f"{k}:{v:+d}" for k, v in r["votes"].items()) or "none"
            clamp = " (TV clamped)" if r["dcf_tv_clamped"] else ""
            lines.append(
                f"| {r['ticker']} | {r['version']} | {r['trigger']} | {r['generated_at']} | {r['rating']} | {fam} "
                f"| {_pct(r['comps_premium'])} | {_pct(r['dcf_initial'])}{clamp} | {_pct(r['dcf_final'])} "
                f"| {votes} | {r['verdict']} | {r['outcome']} | {r['final_rating']} "
                f"| {'yes' if r['thesis_rewrite'] else 'no'} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bodies", required=True, type=Path, help="directory of memo JSON bodies")
    ap.add_argument("--window-start", type=date.fromisoformat, default=DEFAULT_WINDOW[0])
    ap.add_argument("--window-end", type=date.fromisoformat, default=DEFAULT_WINDOW[1])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    args = ap.parse_args(argv)
    try:
        records = load_bodies(args.bodies)
    except (OSError, ValueError) as exc:
        print(f"replay: {exc}", file=sys.stderr)
        return 2
    report = replay(records, window=(args.window_start, args.window_end))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1, sort_keys=True))
    md = render_markdown(report)
    if args.md_out:
        args.md_out.write_text(md)
    fs = report["full_runs"]
    print(f"full runs: n={fs['n']} verdicts={fs['verdicts']} bullish={fs['bullish']} "
          f"downgrade_upper_bound={fs['bullish_downgrade_upper_bound']} ({fs['bullish_downgrade_rate']:.0%}) "
          f"rewrites={fs['thesis_rewrites']} accepted={fs['accepted']}/{fs['divergent']}")
    ss = report["stored_live"]
    print(f"stored live: n={ss['n']} verdicts={ss['verdicts']} bullish={ss['bullish']} "
          f"downgrade_upper_bound={ss['bullish_downgrade_upper_bound']} ({ss['bullish_downgrade_rate']:.0%})")
    print("STOP: " + "; ".join(report["stop_reasons"]) if report["stop"] else "below threshold: enforce")
    return 3 if report["stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
