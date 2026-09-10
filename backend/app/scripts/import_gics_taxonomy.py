"""Import (and optionally activate) the GICS taxonomy from the knowledge JSON.

Usage (from ``backend/``)::

    python -m app.scripts.import_gics_taxonomy                 # import bundled JSON
    python -m app.scripts.import_gics_taxonomy --activate      # …and make it active
    python -m app.scripts.import_gics_taxonomy --path other.json --version-key gics-2027-01
    python -m app.scripts.import_gics_taxonomy --classify      # then classify every company
    python -m app.scripts.import_gics_taxonomy --status        # print versions + counts

Idempotent by checksum: re-running against the same file is a no-op; a
different node set under an existing key is refused (exit 2) because a
changed structure is a new version, never a rewrite. The registry reads
exactly one file — the JSON that ``build_industry_knowledge`` writes —
so there is no other place a code or name could come from.

This is what the admin import route will call too; the CLI exists so a
fresh database can be prepared before the worker's first tick.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..database import init_db
from ..services import gics_registry, industry_classification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import the GICS taxonomy into the registry.")
    parser.add_argument("--path", type=Path, default=None,
                        help="knowledge JSON to import (default: the bundled gics_industries_2026.json)")
    parser.add_argument("--version-key", default=None,
                        help="override the payload's taxonomy_version as the version key")
    parser.add_argument("--activate", action="store_true", help="make this version the active one")
    parser.add_argument("--classify", action="store_true",
                        help="after import, classify every company against the active version")
    parser.add_argument("--status", action="store_true",
                        help="print the imported versions and the active version's counts, then exit")
    parser.add_argument("--notes", default="", help="free-text note stored on the version row")
    args = parser.parse_args(argv)

    init_db()
    if args.status:
        out = {
            "versions": [v.as_dict() for v in gics_registry.list_versions()],
            "active": None,
        }
        active = gics_registry.active_version()
        if active is not None:
            out["active"] = {**active.as_dict(), "counts": gics_registry.counts(active)}
            out["classification"] = industry_classification.audit(version=active)
            # The registry vs the bundled JSON — a mismatch is what the
            # daily loop reports as taxonomy_drift=1.
            out["bundled_drift"] = gics_registry.bundled_drift(active)
        print(json.dumps(out, indent=2, default=str))
        return 0

    try:
        result = gics_registry.import_from_knowledge_json(
            args.path, version_key=args.version_key, activate=args.activate, notes=args.notes,
        )
    except gics_registry.TaxonomyChecksumMismatch as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    if args.classify:
        active = gics_registry.active_version()
        if active is None:
            print("no active taxonomy — pass --activate (or activate one) before --classify", file=sys.stderr)
            return 3
        summary = industry_classification.classify_all(version=active)
        summary.pop("changed", None)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
