"""Operator CLI for the W7 corpus census and targeted vector repair.

    python -m scripts.corpus_repair --inventory
    python -m scripts.corpus_repair --plan [--class reembed|index-missing|all]   # default; writes nothing
    python -m scripts.corpus_repair --apply --class reembed \\
        --max-usd 3 --max-rows 100000 --max-added-mb 300 --max-db-mb <ceiling>
    python -m scripts.corpus_repair --apply --class index-missing \\
        --max-sources 200 --max-usd 3 --max-added-mb 300 --max-db-mb <ceiling>

Run from `backend/`, by the owner or as a Render one-off job — never by an
agent against production. Always `--plan` first and read the receipt.

`--apply` refuses (exit 2) unless OpenAI embeddings are live in this process
(a key, and not demo-only mode), and unless `--max-db-mb` is given: storage,
not money, is the binding constraint on `basic-256mb` Postgres, so every
write run carries an absolute database ceiling, measured before each batch.
Run `--inventory` between `index-missing` runs to watch it.

Prints exactly one machine-readable line, `CORPUS_REPAIR_RECEIPT {json}` (or
`CORPUS_INVENTORY {json}`), and exits 0 (done), 1 (stopped early by a cap,
the database ceiling or the provider — resumable), or 2 (refused).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from app.services import corpus_repair  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.corpus_repair", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--inventory", action="store_const", dest="mode", const="inventory")
    mode.add_argument("--plan", action="store_const", dest="mode", const="plan")
    mode.add_argument("--apply", action="store_const", dest="mode", const="apply")
    p.set_defaults(mode="plan")
    p.add_argument("--class", dest="klass", choices=("reembed", "index-missing", "all"), default="all")
    p.add_argument("--max-usd", type=float, default=3.0)
    p.add_argument("--max-rows", type=int, default=100_000)
    p.add_argument("--max-sources", type=int, default=200)
    p.add_argument("--max-added-mb", type=float, default=300.0)
    p.add_argument("--max-db-mb", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = corpus_repair.run(
        mode=args.mode, klass=args.klass, max_usd=args.max_usd, max_rows=args.max_rows,
        max_sources=args.max_sources, max_added_mb=args.max_added_mb, max_db_mb=args.max_db_mb,
    )
    label = "CORPUS_INVENTORY" if args.mode == "inventory" else "CORPUS_REPAIR_RECEIPT"
    print(f"{label} {json.dumps(receipt, sort_keys=True, default=str)}")
    return int(receipt["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
