"""Normalise legacy Industry Analysis edition flags to the display rule.

Owner decision 1 (2026-09-24): template editions are stored for audit only
and never displayed. Every reader already applies that rule
(``industry_report_store.is_publishable``) and none reads the flags, so the
public site is correct the moment the code deploys. What this changes is
the ROW state the pre-rule code left behind — a template still marked
``succeeded`` + ``is_latest_good``, the analyst edition it marked
``superseded`` — so admin views and the flags agree with the rule:

* template (or below-minimum analyst) rows in ``succeeded``/``superseded``
  → ``audit_only``, not latest-good;
* each group's newest publishable row → ``succeeded``, latest-good;
* every other publishable row → ``superseded``;
* ``pending_review`` rows are left alone.

Usage — the OWNER's path, from the WEB service shell (never run by an
agent against production)::

    cd /app/backend && python -m app.scripts.reclassify_industry_editions            # dry run
    cd /app/backend && python -m app.scripts.reclassify_industry_editions --apply    # write
    cd /app/backend && python -m app.scripts.reclassify_industry_editions --revert manifest.json [--apply]
    cd /app/backend && python -m app.scripts.reclassify_industry_editions --revert-from-ledger [--apply]

The deployed worker ALSO runs the apply once by itself
(``industry_report_worker.reclassify_legacy_editions_once``, switch
``INDUSTRY_RECLASSIFY_LEGACY_EDITIONS``); a ledger row makes the two paths
one operation, so whichever runs first is the only one that writes.

Safety, all enforced in ``industry_report_store`` (this file only parses
arguments and prints):

* dry run by default; prints the full manifest (row id, group, version,
  before/after status and flag) and counts as JSON;
* refuses (exit 2) while any group-report job is queued or running,
  checked ``SELECT … FOR UPDATE`` inside the same transaction as the writes;
* every write is a compare-and-set against the planned before-state, and
  the plan is recomputed after the writes and must be empty — either
  failing rolls back the whole transaction (exit 3);
* one transaction; the manifest is written to the ledger row in the same
  commit (the audit trail, and what ``--revert-from-ledger`` replays);
* never deletes a row; spends nothing; idempotent (a second ``--apply``
  reports ``changed: 0``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_ABORTED = 3


def _load_manifest(path: str) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    manifest = raw.get("manifest") if isinstance(raw, dict) else raw
    if not isinstance(manifest, list):
        raise ValueError(f"{path} holds no manifest list (expected the JSON this script printed)")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    parser.add_argument("--taxonomy", default=None, help="taxonomy version key (default: the active one)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--revert", metavar="MANIFEST_JSON",
                       help="replay a printed manifest backwards (after -> before)")
    group.add_argument("--revert-from-ledger", action="store_true",
                       help="replay the manifest stored on the apply's ledger row backwards")
    args = parser.parse_args(argv)

    from app.services import industry_report_store as store

    try:
        if args.revert or args.revert_from_ledger:
            if args.revert:
                manifest = _load_manifest(args.revert)
            else:
                ledger = store.reclassification_ledger(version=args.taxonomy)
                if ledger is None:
                    print(json.dumps({"error": "no reclassification ledger row to revert"}), file=sys.stderr)
                    return EXIT_REFUSED
                manifest = ledger["manifest"]
            result = store.revert_reclassification(manifest, apply=args.apply, source="web_shell_cli",
                                                   version=args.taxonomy)
        else:
            result = store.reclassify_legacy_editions(apply=args.apply, source="web_shell_cli",
                                                      version=args.taxonomy)
    except store.ReclassifyRefused as exc:
        print(json.dumps({"refused": str(exc)}), file=sys.stderr)
        return EXIT_REFUSED
    except store.ReclassifyAborted as exc:
        print(json.dumps({"aborted": str(exc), "written": False}), file=sys.stderr)
        return EXIT_ABORTED
    result["mode"] = "apply" if args.apply else "dry_run"
    print(json.dumps(result, indent=1, sort_keys=True, default=str))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
