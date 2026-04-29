"""Live-mode token cost measurement (Phase C).

Runs the full smoke suite twice — once cold (empty cache), once warm — and
prints a breakdown of how many tokens each cache write recorded. With real
provider keys configured, the captured tokens reflect *actual* OpenAI /
Anthropic / Gemini usage (via `resolved_cost_tokens` reading
`agents.llm.last_usage()` after every provider call).

Run from `backend/`:

    python -m scripts.measure_live_cost

Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` in `.env` and
flip `USE_DEMO_DATA=false` + `ENABLE_LIVE_DATA=true` if you want to exercise
the live providers as well as live LLMs.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.cache.snapshots import CacheCostLog  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402


def _bucket_writes(rows: List[CacheCostLog]) -> Dict[str, Dict[str, int]]:
    """Group cost rows by `kind` (writes only, hits excluded)."""
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        if r.kind.endswith(":hit"):
            continue
        b = out.setdefault(r.kind, {"count": 0, "tokens": 0})
        b["count"] += 1
        b["tokens"] += int(r.cost_tokens or 0)
    return out


def _query_since(since: datetime) -> List[CacheCostLog]:
    with SessionLocal() as db:
        return list(db.execute(
            select(CacheCostLog).where(CacheCostLog.generated_at >= since)
        ).scalars().all())


def _hit_count_since(since: datetime) -> int:
    with SessionLocal() as db:
        rows = list(db.execute(
            select(CacheCostLog).where(CacheCostLog.generated_at >= since)
        ).scalars().all())
    return sum(1 for r in rows if r.kind.endswith(":hit"))


def main() -> int:
    db_file = Path("./marketmosaic.db")
    if db_file.exists():
        db_file.unlink()
    init_db()

    # Seed before timing so seed cost doesn't pollute the measurement.
    from app.seed_demo_data import run_full_seed
    run_full_seed()

    # Lazy import smoke so the seed runs first.
    import scripts.smoke_test as smoke

    print("Running cold smoke suite...")
    t0 = datetime.utcnow()
    with contextlib.redirect_stdout(io.StringIO()):
        smoke.main()

    print("Running warm smoke suite...")
    # Tiny pause so timestamp ordering is unambiguous between cold and warm.
    import time as _t; _t.sleep(0.25)
    t1 = datetime.utcnow()
    with contextlib.redirect_stdout(io.StringIO()):
        smoke.main()
    t2 = datetime.utcnow()

    cold_rows = [r for r in _query_since(t0) if r.generated_at < t1]
    warm_rows = [r for r in _query_since(t1) if r.generated_at < t2]
    cold = _bucket_writes(cold_rows)
    warm = _bucket_writes(warm_rows)
    cold_hits = sum(1 for r in cold_rows if r.kind.endswith(":hit"))
    warm_hits = sum(1 for r in warm_rows if r.kind.endswith(":hit"))

    cold_total = sum(b["tokens"] for b in cold.values())
    warm_total = sum(b["tokens"] for b in warm.values())
    ratio = (warm_total / cold_total) if cold_total else 0.0

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    mode_label = "LIVE" if (has_openai or has_anthropic or has_gemini) else "DEMO"

    print()
    print(f"=== {mode_label} mode  (openai={has_openai} anthropic={has_anthropic} gemini={has_gemini}) ===")
    print()
    print(f"COLD: total_tokens={cold_total:>8d}  hits={cold_hits}")
    for k in sorted(cold, key=lambda x: -cold[x]["tokens"]):
        print(f"   {k:<32s} count={cold[k]['count']:>3d}  tokens={cold[k]['tokens']:>8d}")
    print()
    print(f"WARM: total_tokens={warm_total:>8d}  hits={warm_hits}")
    for k in sorted(warm, key=lambda x: -warm[x]["tokens"]):
        print(f"   {k:<32s} count={warm[k]['count']:>3d}  tokens={warm[k]['tokens']:>8d}")
    print()
    print(f"warm/cold = {100 * ratio:.1f}%  (target <30%)")
    return 0 if ratio < 0.30 else 1


if __name__ == "__main__":
    sys.exit(main())
