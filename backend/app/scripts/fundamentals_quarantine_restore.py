"""Reverse one FMP-primary quarantine, adoption or restatement audit (FIX-006).

    python -m app.scripts.fundamentals_quarantine_restore <repair_id>

Run from a Render *web* shell by the owner (it needs the production database).
The restore is fenced: every audited row must still match its recorded image,
or nothing changes. For a quarantine, the rows now holding the restored keys
(FMP's replacements) are themselves quarantined with their own audit first,
so nothing is deleted in either direction. Prints the result as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys

from ..services.fundamental_quarantine import restore_repair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repair_id")
    args = parser.parse_args(argv)
    try:
        result = restore_repair(args.repair_id)
    except LookupError:
        print(json.dumps({"repair_id": args.repair_id, "error": "unknown repair"}))
        return 2
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"repair_id": args.repair_id, "error": type(exc).__name__, "detail": str(exc)}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
