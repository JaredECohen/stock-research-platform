"""Build the minimized W2a memo-section fixtures from the local evidence files.

    python -m app.scripts.trim_memo_section_fixtures [--source-dir DIR] [--check]

The presenter's scenario tests (`test_memo_sections.py`) run over five saved
production memos. Those evidence files are gitignored (`.gitignore:72`) and
this repository is PUBLIC, so the committed copies under
`app/tests/fixtures/memo_sections/` are minimized (integration plan S11,
critique delta 9):

* template text — the signature-bearing fallback prose our own code wrote,
  which is exactly what the classifier reads — is kept verbatim;
* computed numbers and flags are kept;
* every piece of analyst (LLM) prose, the provider's business description,
  filing and transcript excerpts, PM questions and intake rationales are
  replaced with numbered synthetic sentences, consistently (a thesis quoted
  inside the final verdict is replaced by the same synthetic text there);
* citations, `data.structured`, `data.narrative`, source lists and the PM DCF
  adjustment audit trail are dropped; long-form reports are replaced by a
  synthetic body that keeps only whether an analyst expansion existed.

The script refuses to write a fixture whose section verdicts differ from the
full memo's (`--check` only compares), so minimizing can never change what
the tests prove. `test_memo_sections.py::test_minimized_fixtures_classify_like_evidence`
repeats that comparison when the evidence files are present locally.

Sources (the SHA-256 of each file when these fixtures were cut, 2026-09-24):

* `docs/reviews/2026-09-13-baseline-memo-evidence.json` (AAPL, GOOGL, MSFT)
  bd3a76a1781f8d41909d62a9e423513cae66b44ecfca74c59035679a1ed4f154
* `docs/reviews/2026-09-13-META-v1.json` (META v1)
  eae9369dc33898fe446138241b7129193a4d78e1643d59bb1f78ecef02344b5f
* `docs/reviews/2026-09-13-ABBV-v7-compatibility.json` (`post_deploy_response`, ABBV v7)
  d6abb19c6c98481a6b39338b974851d3a78b57692638094c76308ce498e975fd

Reads local files and writes fixture files only: no database, no provider,
no LLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from ..schemas import StockMemoOut
from ..services.memo_sections import (
    COMPUTED_ROSTER_KEYS,
    UNAVAILABLE_TEXT,
    compute_availability,
    present_memo,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = REPO_ROOT / "docs" / "reviews"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "memo_sections"

BASELINE = "2026-09-13-baseline-memo-evidence.json"
META = "2026-09-13-META-v1.json"
ABBV = "2026-09-13-ABBV-v7-compatibility.json"

EXPECTED_SHA256 = {
    BASELINE: "bd3a76a1781f8d41909d62a9e423513cae66b44ecfca74c59035679a1ed4f154",
    META: "eae9369dc33898fe446138241b7129193a4d78e1643d59bb1f78ecef02344b5f",
    ABBV: "d6abb19c6c98481a6b39338b974851d3a78b57692638094c76308ce498e975fd",
}

# fixture name -> (source file, how to pull the memo out of it)
FIXTURES: dict[str, tuple[str, str]] = {
    "aapl_demo": (BASELINE, "memos.AAPL.memo"),
    "googl_live_prepflag": (BASELINE, "memos.GOOGL.memo"),
    "msft_live": (BASELINE, "memos.MSFT.memo"),
    "meta_v1": (META, ""),
    "abbv_v7_patch": (ABBV, "post_deploy_response"),
}

_FINDING_FIELDS = (
    "sector_agent_view", "earnings_agent_view", "filing_agent_view", "valuation_agent_view",
    "comps_agent_view", "macro_sensitivity", "technical_agent_view", "earnings_qoq_delta",
)
# Sector `data` keys worth keeping: the template bull/bear block, computed
# placements and labels, and the provenance flags.
_SECTOR_DATA_KEEP = {
    "bull_bear_analysis", "kpi_placements", "regime", "sector", "sub_industry",
    "cross_sector_relevance", "macro_alignment",
    "deterministic_fallback", "bull_bear_parse_failed", "degraded", "error",
    "intake_skipped", "intake_rationale", "retrieval_failed", "no_mapping",
}
_FLAG_KEYS = {
    "deterministic_fallback", "bull_bear_parse_failed", "degraded", "error", "intake_skipped",
    "intake_rationale", "retrieval_failed", "no_mapping", "signals",
}
_KEEP_LINE = re.compile(r"^(?:DCF (?:bull|bear) case implies |Risk lens: |Next earnings: )")
_EXPANSION = "### Analyst expansion"


def load_evidence(source_dir: Path) -> dict[str, dict[str, Any]]:
    """The five full memos, keyed by fixture name. Refuses a changed file."""
    out: dict[str, dict[str, Any]] = {}
    for name, (fname, path) in FIXTURES.items():
        blob = (source_dir / fname).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != EXPECTED_SHA256[fname]:
            raise SystemExit(f"{fname}: sha256 {digest} does not match the recorded source")
        node: Any = json.loads(blob)
        for part in [p for p in path.split(".") if p]:
            node = node[part]
        out[name] = node
    return out


class _Synth:
    """Numbered synthetic replacements, applied consistently everywhere."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.n = 0
        self.map: dict[str, str] = {}

    def text(self, orig: str, kind: str = "analyst text") -> str:
        orig = orig or ""
        if not orig.strip():
            return orig
        if orig in self.map:
            return self.map[orig]
        self.n += 1
        new = f"Synthetic {kind} {self.n} for the {self.name} fixture."
        # Keep the opening an LLM patch imitated ("Research view: Bullish.")
        # and the self-labelling DCF driver prefix: the classifier reads both.
        m = re.match(r"^(Research view: [A-Za-z ]+\. )", orig)
        if m:
            new = m.group(1) + new
        if orig.startswith("DCF driver — "):
            new = f"DCF driver — Synthetic driver {self.n}: synthetic rationale for the {self.name} fixture."
        self._register(orig, new)
        return new

    def _register(self, orig: str, new: str) -> None:
        self.map[orig] = new
        # Builders truncate what they copy (risk titles [:80], signal lines
        # [:240]); map the truncations too so copies stay consistent.
        for cut in (80, 240):
            if len(orig) > cut:
                self.map.setdefault(orig[:cut], new[:cut])

    def apply(self, value: Any) -> Any:
        if isinstance(value, str):
            if value in self.map:
                return self.map[value]
            out = value
            for orig in sorted(self.map, key=len, reverse=True):
                if len(orig) >= 24 and orig in out:
                    out = out.replace(orig, self.map[orig])
            return out
        if isinstance(value, list):
            return [self.apply(v) for v in value]
        if isinstance(value, dict):
            return {k: self.apply(v) for k, v in value.items()}
        return value


def _synthetic_long_form(report: str | None, synth: _Synth) -> str | None:
    if not report:
        return report
    body = f"Synthetic deterministic drill-down body for the {synth.name} fixture."
    if _EXPANSION in report:
        return f"{body}\n\n{_EXPANSION}\nSynthetic analyst expansion for the {synth.name} fixture."
    return body


def _minimize_finding(f: dict[str, Any], status: str, synth: _Synth, *, computed: bool,
                      is_sector: bool) -> dict[str, Any]:
    out = dict(f)
    data = dict(f.get("data") or {})
    hidden = status == "unavailable"
    headline, summary = f.get("headline") or "", f.get("summary") or ""
    if data.get("intake_skipped"):
        out["summary"] = synth.text(summary, "intake rationale")
        if data.get("intake_rationale"):
            data["intake_rationale"] = synth.text(data["intake_rationale"], "intake rationale")
    elif not hidden and not computed:
        out["headline"] = synth.text(headline, "analyst headline")
        out["summary"] = synth.text(summary, "analyst summary")
        out["key_points"] = [synth.text(p, "analyst point") for p in f.get("key_points") or []]
    elif hidden:
        # Template read-outs stay, minus third-party excerpts they quote.
        m = re.match(r"^([\w/-]+ dated [\w/—-]+[.:])", summary)
        if m:
            out["summary"] = f"{m.group(1)} Synthetic filing excerpt for the {synth.name} fixture."
            out["key_points"] = [
                f"Risk: synthetic risk factor {i + 1}." if str(p).startswith("Risk: ") else p
                for i, p in enumerate(f.get("key_points") or [])
            ]
        elif summary.startswith("Scenario read: "):
            out["summary"] = f"Scenario read: Synthetic scenario narrative for the {synth.name} fixture."
    out["evidence"] = []
    out["sources"] = []
    out["long_form_report"] = _synthetic_long_form(f.get("long_form_report"), synth)
    data.pop("structured", None)
    data.pop("narrative", None)
    if is_sector:
        data = {k: v for k, v in data.items() if k in _SECTOR_DATA_KEEP}
        bb = data.get("bull_bear_analysis")
        if isinstance(bb, dict) and not hidden_bb(bb):
            data["bull_bear_analysis"] = _synth_bb(bb, synth)
        if isinstance(data.get("macro_alignment"), str) and len(data["macro_alignment"]) > 24:
            data["macro_alignment"] = synth.text(data["macro_alignment"], "macro alignment")
    else:
        data = {k: v for k, v in data.items() if k in _FLAG_KEYS or k == "causal_chain"}
    out["data"] = data
    return out


def hidden_bb(bb: dict[str, Any]) -> bool:
    return (bb.get("key_disagreement") or "").startswith("Bears price in cohort margin compression")


def _synth_bb(bb: dict[str, Any], synth: _Synth) -> dict[str, Any]:
    out = dict(bb)
    for side in ("bull_case", "bear_case"):
        case = dict(bb.get(side) or {})
        case["headline"] = synth.text(case.get("headline") or "", f"{side} headline")
        case["key_points"] = [synth.text(p, f"{side} point") for p in case.get("key_points") or []]
        out[side] = case
    out["key_disagreement"] = synth.text(bb.get("key_disagreement") or "", "key disagreement")
    out["sector_synthesis"] = synth.text(bb.get("sector_synthesis") or "", "sector synthesis")
    out["falsifiable_tests"] = [
        {**t, "statement": synth.text(t.get("statement") or "", "falsifier")}
        for t in bb.get("falsifiable_tests") or [] if isinstance(t, dict)
    ]
    return out


def minimize(name: str, memo: dict[str, Any]) -> dict[str, Any]:
    full = StockMemoOut.model_validate(memo)
    av = compute_availability(full)
    synth = _Synth(name)
    m = json.loads(full.model_dump_json(exclude={"section_availability"}))

    # Findings first, so copies of their text elsewhere map consistently.
    sector_bb = (m["sector_agent_view"].get("data") or {}).get("bull_bear_analysis")
    if isinstance(sector_bb, dict) and not hidden_bb(sector_bb):
        _synth_bb(sector_bb, synth)
    for key in _FINDING_FIELDS:
        if m.get(key) is None:
            continue
        m[key] = _minimize_finding(m[key], av[key].status, synth,
                                   computed=key == "comps_agent_view",
                                   is_sector=key == "sector_agent_view")
    for key, f in list((m.get("extra_agent_views") or {}).items()):
        m["extra_agent_views"][key] = _minimize_finding(
            f, av[f"extra_agent_views.{key}"].status, synth,
            computed=key in COMPUTED_ROSTER_KEYS, is_sector=False)
    shown = present_memo(full)
    for r_i, rnd in enumerate(m.get("round_findings") or []):
        rnd["pm_questions"] = [
            {**q, "question": synth.text(q.get("question") or "", "PM question"),
             "why_it_matters": synth.text(q.get("why_it_matters") or "", "PM rationale")}
            for q in rnd.get("pm_questions") or []
        ]
        rnd["pm_rationale"] = synth.text(rnd.get("pm_rationale") or "", "PM rationale")
        for k, f in list((rnd.get("findings") or {}).items()):
            blanked = shown.round_findings[r_i].findings[k].headline == UNAVAILABLE_TEXT
            status = "unavailable" if blanked else "available"
            rnd["findings"][k] = _minimize_finding(f, status, synth,
                                                   computed=k in COMPUTED_ROSTER_KEYS,
                                                   is_sector=k == "sector")

    # Top-level prose the committee (LLM) or the provider wrote.
    if av["final_pm_view"].status != "unavailable":
        m["final_pm_view"] = synth.text(m["final_pm_view"], "PM view")
    if av["one_sentence_thesis"].status == "available":
        m["one_sentence_thesis"] = synth.text(m["one_sentence_thesis"], "thesis")
    if av["mispricing_thesis"].status == "available":
        mt = m["mispricing_thesis"]
        for k in ("consensus_view", "our_view", "gap"):
            mt[k] = synth.text(mt.get(k) or "", f"mispricing {k}")
        mt["falsifiers"] = [synth.text(x, "falsifier") for x in mt.get("falsifiers") or []]
    m["business_summary"] = synth.text(m.get("business_summary") or "", "business description")
    for key in ("bull_case", "bear_case"):
        case = m[key]
        hidden_head = av[key].headline_hidden
        if case.get("headline") and not hidden_head:
            case["headline"] = synth.text(case["headline"], f"{key} headline")
    templ_items = _template_items(full, av)
    for key in ("bull_case", "bear_case"):
        m[key]["key_points"] = [
            p if (p in templ_items or _KEEP_LINE.match(p)) else synth.text(p, f"{key} point")
            for p in m[key]["key_points"]
        ]
    for key in ("key_risks", "thesis_breakers"):
        for r in m[key]:
            if r["detail"] not in templ_items:
                r["detail"] = synth.text(r["detail"], "risk")
                r["title"] = r["detail"][:80]
    for c in m["catalysts"]:
        if not _KEEP_LINE.match(c["title"]) and c["detail"] not in templ_items:
            c["detail"] = synth.text(c["detail"], "catalyst")
            c["title"] = c["detail"][:80]
    rc = m["risk_committee_challenge"]
    if rc.get("review_mode") == "live":
        rc["overall_assessment"] = synth.text(rc["overall_assessment"], "critic assessment")
        for k in ("challenges", "underweighted_risks", "suggested_revisions"):
            rc[k] = [synth.text(x, "critic point") for x in rc.get(k) or []]
    if isinstance(m.get("scorecard"), dict) and m["scorecard"].get("reconciliation"):
        m["scorecard"]["reconciliation"] = synth.text(m["scorecard"]["reconciliation"],
                                                      "scorecard reconciliation")
    if m.get("intake_decision", {}).get("rationale"):
        m["intake_decision"]["rationale"] = synth.text(m["intake_decision"]["rationale"], "intake rationale")

    # Drop what nothing reads and trim bulky computed blobs.
    m["dcf_initial_summary"] = {}
    m["dcf_pm_adjustments"] = []
    m["dcf_pm_adjustment_headline"] = ""
    m["sources_used"] = []
    m["macro_snapshot_at_memo"] = {}
    m["forward_catalysts"] = []

    # Final consistent pass: every remaining copy (final verdict, patched
    # legacy event, mispricing falsifiers) of replaced text follows suit.
    m = synth.apply(m)
    minimized = StockMemoOut.model_validate(m)
    if not same_verdicts(compute_availability(full), compute_availability(minimized)):
        raise SystemExit(f"{name}: minimizing changed a section verdict; refusing to write")
    return json.loads(minimized.model_dump_json(exclude={"section_availability"}))


def _template_items(full: StockMemoOut, av: dict[str, Any]) -> set[str]:
    """Case/risk/catalyst texts the presenter hides (template): kept verbatim."""

    shown = present_memo(full)
    kept: set[str] = set()
    for key in ("bull_case", "bear_case"):
        kept |= set(getattr(shown, key).key_points)
    for r in shown.key_risks + shown.thesis_breakers:
        kept.add(r.detail)
    for c in shown.catalysts:
        kept.add(c.detail)
    everything: set[str] = set()
    for key in ("bull_case", "bear_case"):
        everything |= set(getattr(full, key).key_points)
    for r in full.key_risks + full.thesis_breakers:
        everything.add(r.detail)
    for c in full.catalysts:
        everything.add(c.detail)
    return everything - kept


def same_verdicts(a: dict[str, Any], b: dict[str, Any]) -> bool:
    def norm(av: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
        return {k: (v.status, v.reason, v.hidden_items, v.headline_hidden, tuple(v.basis))
                for k, v in av.items()}
    return norm(a) == norm(b)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source-dir", type=Path,
                        default=Path(os.environ.get("MM_EVIDENCE_DIR") or DEFAULT_SOURCE_DIR))
    parser.add_argument("--check", action="store_true",
                        help="compare the committed fixtures instead of writing them")
    args = parser.parse_args(argv)
    evidence = load_evidence(args.source_dir)
    status = 0
    for name, memo in evidence.items():
        text = json.dumps(minimize(name, memo), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        path = FIXTURE_DIR / f"{name}.json"
        if args.check:
            if not path.exists() or path.read_text() != text:
                print(f"{name}: committed fixture differs from the trimmed evidence")
                status = 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(text)} bytes)")
    return status


if __name__ == "__main__":
    sys.exit(main())
