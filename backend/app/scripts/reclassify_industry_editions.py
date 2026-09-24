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
``INDUSTRY_RECLASSIFY_LEGACY_EDITIONS``). The ledger makes the two paths
one operation: ``--apply`` after the worker (or after an earlier
``--apply``) reports ``status: already_done`` and writes nothing, so the
ledger's manifest stays the one ``--revert-from-ledger`` undoes. After a
revert, ``--apply`` may run again; ``--force`` applies on top of a live
apply (its own manifest, which the next ``--revert-from-ledger`` undoes).
The worker never re-applies after a revert.

Safety, all enforced in ``industry_report_store`` (this file only parses
arguments and prints):

* dry run by default; prints the full manifest (row id, group, version,
  period, before/after status and flag) and counts as JSON;
* the plan must be the pinned expected population
  (``industry_report_store.RECLASSIFY_*``: legacy weeks
  2026-W37..``--last-legacy-period``, only the three legacy transitions, at
  most 125 rows) — otherwise nothing is written (exit 3; a dry run prints
  the ``violations`` and exits 3 too). Widen the window only after reading
  the dry run;
* refuses (exit 2) while any group-report job is queued or running,
  checked ``SELECT … FOR UPDATE`` inside the same transaction as the writes;
* every write is a compare-and-set against the planned before-state, and
  the plan is recomputed after the writes and must be empty — either
  failing rolls back the whole transaction (exit 3);
* one transaction; the manifest is written to the ledger row in the same
  commit (the audit trail, and what ``--revert-from-ledger`` replays);
* never deletes a row; spends nothing; idempotent (a second ``--apply``
  reports ``already_done`` / ``changed: 0`` and writes no ledger row);
* a revert refuses an empty manifest and a ledger apply already reverted.
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
    parser.add_argument("--force", action="store_true",
                        help="apply even though an un-reverted apply ledger exists")
    parser.add_argument("--last-legacy-period", default=None, metavar="YYYY-Www",
                        help="widen the pinned legacy window's last week (default: the pinned one)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--revert", metavar="MANIFEST_JSON",
                       help="replay a printed manifest backwards (after -> before)")
    group.add_argument("--revert-from-ledger", action="store_true",
                       help="replay the manifest stored on the apply's ledger row backwards")
    args = parser.parse_args(argv)

    from app.services import industry_report_store as store

    last_legacy_period = args.last_legacy_period or store.RECLASSIFY_LAST_LEGACY_PERIOD
    try:
        if args.revert or args.revert_from_ledger:
            if args.revert:
                manifest = _load_manifest(args.revert)
            else:
                ledger = store.reclassification_ledger(version=args.taxonomy)
                if ledger is None:
                    print(json.dumps({"error": "no reclassification ledger row to revert"}), file=sys.stderr)
                    return EXIT_REFUSED
                if ledger["reverted"]:
                    print(json.dumps({"error": f"apply ledger {ledger['ledger_id']} was already reverted"}),
                          file=sys.stderr)
                    return EXIT_REFUSED
                manifest = ledger["manifest"]
            result = store.revert_reclassification(manifest, apply=args.apply, source="web_shell_cli",
                                                   version=args.taxonomy)
        else:
            result = store.reclassify_legacy_editions(
                apply=args.apply, source="web_shell_cli", version=args.taxonomy,
                once=None if args.force else store.ONCE_UNLESS_REVERTED,
                last_legacy_period=last_legacy_period,
            )
    except store.ReclassifyRefused as exc:
        print(json.dumps({"refused": str(exc)}), file=sys.stderr)
        return EXIT_REFUSED
    except store.ReclassifyAborted as exc:
        print(json.dumps({"aborted": str(exc), "written": False}), file=sys.stderr)
        return EXIT_ABORTED
    result["mode"] = "apply" if args.apply else "dry_run"
    print(json.dumps(result, indent=1, sort_keys=True, default=str))
    if result.get("violations") and not args.apply:
        # The apply would abort; say so in the exit code, not only in JSON.
        print(json.dumps({"would_abort": result["violations"][:20]}), file=sys.stderr)
        return EXIT_ABORTED
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
