"""Tools the sector analyst calls to discover & analyze relevant data.

Two surfaces:

1. **Thin wrappers** that mirror the catalog/overlay services. Sector
   prompts reference these as documented capabilities; deterministic
   call sites use them too:
     - `discover_relevant_data(ticker, profile, extra_categories=...)`
     - `fetch_data_snapshot(series_ids)`
     - `compute_overlays(ticker, profile, names=...)`

2. **Composite** `prepare_sector_context(ticker, profile)` — the one-call
   entry point for the sector agent. It:
     - discovers which catalog entries are most relevant
     - pre-fetches the top-K snapshots so the LLM sees real numbers
     - runs the sector-appropriate overlays
     - assembles a single dict that's safe to embed in the prompt

The composite is deliberately defensive: every step that touches network
or DB is wrapped so a single failed snapshot doesn't degrade the rest of
the context.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from ..services import data_catalog_service, sector_overlays

log = logging.getLogger(__name__)

# How many discovered series we eagerly snapshot before handing context
# to the LLM. The agent can request more by series_id if it wants to.
DEFAULT_PREFETCH_PER_AXIS = 4
DEFAULT_MAX_PREFETCH_TOTAL = 16


# ---------------------------------------------------------------------------
# Thin wrappers
# ---------------------------------------------------------------------------

def discover_relevant_data(
    ticker: str,
    *,
    profile: Optional[Dict[str, Any]] = None,
    extra_categories: Optional[Iterable[str]] = None,
    max_per_axis: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    return data_catalog_service.discover_for_ticker(
        ticker, profile=profile,
        extra_categories=extra_categories, max_per_axis=max_per_axis,
    )


def fetch_data_snapshot(
    series_ids: Iterable[str], *, force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    return [
        snap.to_dict() for snap in
        data_catalog_service.fetch_snapshots(series_ids, force_refresh=force_refresh)
    ]


def compute_overlays(
    ticker: str,
    *,
    profile: Optional[Dict[str, Any]] = None,
    names: Optional[List[str]] = None,
    max_overlays: int = 3,
) -> Dict[str, Any]:
    return sector_overlays.compute_sector_overlays(
        ticker, profile=profile,
        explicit_overlays=names, max_overlays=max_overlays,
    )


# ---------------------------------------------------------------------------
# Composite entry point
# ---------------------------------------------------------------------------

def prepare_sector_context(
    ticker: str,
    *,
    profile: Optional[Dict[str, Any]] = None,
    prefetch_per_axis: int = DEFAULT_PREFETCH_PER_AXIS,
    max_prefetch_total: int = DEFAULT_MAX_PREFETCH_TOTAL,
    overlay_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a one-shot context payload the sector LLM can read directly.

    Shape:
        {
          "ticker": "PLD",
          "discovered_catalog": {
              "sector_relevant":     [SeriesSpec.to_dict(), ...],
              "sub_industry_relevant": [...],
              "geography_relevant": [...],
              "category_relevant":  [...]
          },
          "prefetched_snapshots": [SeriesSnapshot.to_dict(), ...],
          "overlays": {
              "ticker": "...", "sector": "...", "overlays_run": [...],
              "bundles": {...}
          },
          "errors": ["..."]
        }
    """
    errors: List[str] = []

    try:
        discovered = discover_relevant_data(
            ticker, profile=profile,
            extra_categories=("inflation", "credit"),
        )
    except Exception as exc:  # pragma: no cover
        discovered = {}
        errors.append(f"discover_relevant_data: {exc}")

    # Pick the top-K series across axes (sector_relevant first), capped.
    prefetch_ids: List[str] = []
    seen: set[str] = set()
    for axis in ("sector_relevant", "sub_industry_relevant",
                 "geography_relevant", "category_relevant"):
        for spec in (discovered.get(axis) or [])[:prefetch_per_axis]:
            sid = spec.get("series_id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            prefetch_ids.append(sid)
            if len(prefetch_ids) >= max_prefetch_total:
                break
        if len(prefetch_ids) >= max_prefetch_total:
            break

    try:
        snapshots = fetch_data_snapshot(prefetch_ids) if prefetch_ids else []
    except Exception as exc:  # pragma: no cover
        snapshots = []
        errors.append(f"fetch_data_snapshot: {exc}")

    try:
        overlays = compute_overlays(
            ticker, profile=profile, names=overlay_names,
        )
    except Exception as exc:  # pragma: no cover
        overlays = {"ticker": ticker, "overlays_run": [], "bundles": {}}
        errors.append(f"compute_overlays: {exc}")

    return {
        "ticker": ticker,
        "discovered_catalog": discovered,
        "prefetched_snapshots": snapshots,
        "overlays": overlays,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Compact prompt block (used by sector_agents.py)
# ---------------------------------------------------------------------------

def render_prompt_block(context: Dict[str, Any], *, max_chars: int = 5000) -> str:
    """Render the composite context as a markdown block for the prompt.

    Keeps the prompt readable instead of dumping raw JSON. The LLM can
    still re-request anything by series_id via the regular tool surface,
    but for a one-shot pass this gives it the highlights inline.
    """
    if not context:
        return ""
    lines: List[str] = ["## Sector data context"]
    discovered = context.get("discovered_catalog") or {}

    def _axis_block(label: str, key: str) -> None:
        entries = discovered.get(key) or []
        if not entries:
            return
        lines.append(f"\n**{label}** ({len(entries)} relevant series):")
        for spec in entries[:6]:
            lines.append(
                f"- `{spec['series_id']}` ({spec['source']}, {spec['frequency']}) — "
                f"{spec['name']}: {spec['description'][:120]}"
            )

    _axis_block("Sector-relevant", "sector_relevant")
    _axis_block("Sub-industry-relevant", "sub_industry_relevant")
    _axis_block("Geography-relevant (footprint metros)", "geography_relevant")
    _axis_block("Other categories surfaced", "category_relevant")

    snapshots = context.get("prefetched_snapshots") or []
    if snapshots:
        lines.append("\n**Pre-fetched readings**:")
        for snap in snapshots[:12]:
            if snap.get("error"):
                continue
            latest = snap.get("latest") or {}
            extras = []
            if snap.get("yoy_pct") is not None:
                extras.append(f"YoY {snap['yoy_pct']:+.1%}")
            if snap.get("change_3m") is not None:
                extras.append(f"3m {snap['change_3m']:+.2f}")
            if snap.get("z_score_5y") is not None:
                extras.append(f"5y z={snap['z_score_5y']:+.2f}")
            tail = " | ".join(extras) if extras else ""
            lines.append(
                f"- `{snap['series_id']}` {snap['name']} — "
                f"latest {latest.get('value')} ({latest.get('date')})"
                + (f" — {tail}" if tail else "")
            )

    overlays = (context.get("overlays") or {}).get("bundles") or {}
    if overlays:
        lines.append("\n**Sector overlays**:")
        for overlay_name, bundle in overlays.items():
            if not bundle.get("available"):
                continue
            hints = bundle.get("narrative_hints") or []
            if not hints:
                continue
            lines.append(f"- {overlay_name}:")
            for hint in hints[:5]:
                lines.append(f"  - {hint}")

    block = "\n".join(lines).strip()
    if len(block) > max_chars:
        block = block[:max_chars] + "\n...(truncated)"
    return block
