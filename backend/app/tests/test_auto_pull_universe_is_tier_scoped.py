"""The scheduled pollers pull for the curated tier only — and say so.

`transcripts_poller`'s module docstring claimed, verbatim, that it "iterates
every ticker in `Company.universe_tier == 'auto_analysis'`". It did not. Line
64 was `ds.list_tickers()`, which is `db.query(Company.ticker).all()` — every
row in `companies`, no tier filter, same as `edgar_poller`. The documented
constraint and the executed one had diverged, and nothing in the suite noticed
because every existing poller test passes `tickers=["NVDA"]` explicitly and so
never exercises the selection path at all.

That gap matters more than the three tickers it leaked today. The tier exists
so on-demand research does not enlarge the automatic pull universe: each manual
search inserts an `analyzed_on_demand` row, and an unfiltered poller adopts it
into a 30-minute cron forever. The set only grows.

So these tests pin three things at once, because pinning fewer is what let the
drift happen:

1. Behaviour — each poller, run with no argument against a mixed-tier
   universe, touches the curated tier and nothing else.
2. The escape hatch — an explicit `tickers=` still bypasses the filter, which
   is how admin re-runs and the rest of the suite drive one name through.
3. The prose — the tier a poller's docstring *claims* to iterate must equal
   the tier it actually selects. The claim is the first paragraph that names
   any tier literal; later paragraphs are free to discuss the excluded tiers,
   which is where the reasoning belongs.

`list_tickers()` with no argument stays unfiltered and is tested here too. That
is not an oversight in the pollers' favour — it is the load-bearing half of the
user's constraint. Manual research on ANY ticker must keep working, and the
resolvers in `agents/orchestrator.py`, `agents/sector_agents.py` and
`services/sector_research_service.py` all resolve against that unfiltered set.
"""
from __future__ import annotations

import pytest

from app.database import SessionLocal
from app.models import Company
from app.monitoring import edgar_poller, transcripts_poller
from app.services.data_service import AUTO_PULL_TIERS, get_data_service

# Seeded per test, deleted on teardown. `ZZ` prefix keeps them out of the way
# of the demo universe and sorts them last, so the ordering assertion below
# fails loudly if the ORDER BY ever goes missing.
_SEEDED: dict[str, str] = {
    "ZZAUTO": "auto_analysis",
    "ZZDEMAND": "analyzed_on_demand",
    "ZZDATA": "data_only",
}

_TIER_LITERALS = ("auto_analysis", "analyzed_on_demand", "data_only")


@pytest.fixture()
def mixed_tier_universe():
    """Put one company of each tier in `companies`; remove them after."""
    with SessionLocal() as db:
        for ticker, tier in _SEEDED.items():
            db.merge(Company(
                ticker=ticker, company_name=f"{ticker} Test Co",
                sector="Test", industry="Test", universe_tier=tier,
            ))
        db.commit()
    yield _SEEDED
    with SessionLocal() as db:
        db.query(Company).filter(
            Company.ticker.in_(tuple(_SEEDED))
        ).delete(synchronize_session=False)
        db.commit()


def _tier_of() -> dict[str, str]:
    with SessionLocal() as db:
        return dict(db.query(Company.ticker, Company.universe_tier).all())


def _drive(monkeypatch, module, tickers=None) -> tuple[list[str], str]:
    """Run a poller with its provider + bookkeeping stubbed out.

    Returns `(tickers_polled, record_run_note)`. The provider call is the
    observation point: what a poller asks the provider about IS its pull
    universe, which is the property under test.
    """
    polled: list[str] = []
    notes: list[str] = []
    if module is edgar_poller:
        monkeypatch.setattr(module, "get_filings", lambda t: polled.append(t) or [])
        monkeypatch.setattr(module, "_seen_accessions", lambda t: set())
        monkeypatch.setattr(module, "_save_seen_accessions", lambda t, acc: None)
    else:
        monkeypatch.setattr(module, "get_transcripts", lambda t: polled.append(t) or [])
        monkeypatch.setattr(module, "_seen_periods", lambda t: set())
        monkeypatch.setattr(module, "_save_seen_periods", lambda t, p: None)
    monkeypatch.setattr(module, "record_run", lambda *a, **k: notes.append(k.get("note", "")))
    module.run_once(tickers)
    (note,) = notes
    return polled, note


# ---------------------------------------------------------------------------
# list_tickers
# ---------------------------------------------------------------------------

def test_list_tickers_unfiltered_returns_every_tier_in_ticker_order(mixed_tier_universe):
    tickers = get_data_service().list_tickers()

    assert set(_SEEDED) <= set(tickers), (
        "the default call must still see analyzed_on_demand and data_only — "
        "manual research resolves against this set"
    )
    assert tickers == sorted(tickers), "list_tickers() must be ordered by ticker"


def test_list_tickers_with_tiers_excludes_on_demand_and_data_only(mixed_tier_universe):
    ds = get_data_service()
    curated = ds.list_tickers(tiers=AUTO_PULL_TIERS)

    assert "ZZAUTO" in curated
    assert "ZZDEMAND" not in curated
    assert "ZZDATA" not in curated
    assert curated == sorted(curated)

    tier_of = _tier_of()
    assert {tier_of[t] for t in curated} == set(AUTO_PULL_TIERS)
    # And the filter is a subset of the default, never a different set.
    assert set(curated) <= set(ds.list_tickers())


def test_list_tickers_with_an_empty_tier_set_selects_nothing(mixed_tier_universe):
    # `tiers=()` is an explicit empty selection, distinct from `tiers=None`
    # meaning "no filter". A caller that computes its tier list and gets an
    # empty one must not silently receive the whole universe.
    assert get_data_service().list_tickers(tiers=()) == []


# ---------------------------------------------------------------------------
# The pollers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [edgar_poller, transcripts_poller])
def test_poller_with_no_argument_polls_only_the_curated_tier(
    monkeypatch, mixed_tier_universe, module,
):
    polled, _ = _drive(monkeypatch, module)

    assert "ZZAUTO" in polled
    assert "ZZDEMAND" not in polled, "an on-demand name must not join the cron"
    assert "ZZDATA" not in polled

    tier_of = _tier_of()
    assert {tier_of[t] for t in polled} == set(AUTO_PULL_TIERS), (
        "asserted across the whole seeded universe, not just the fixture rows"
    )


@pytest.mark.parametrize("module", [edgar_poller, transcripts_poller])
def test_explicit_tickers_bypass_the_tier_filter(monkeypatch, mixed_tier_universe, module):
    # Admin re-runs and the rest of the suite drive one name through here and
    # must not have to care what tier it sits in.
    polled, note = _drive(monkeypatch, module, ["ZZDEMAND", "ZZDATA"])

    assert polled == ["ZZDEMAND", "ZZDATA"]
    assert "out-of-tier" not in note, (
        "no tier filter ran, so the note must not claim one did"
    )


@pytest.mark.parametrize("module", [edgar_poller, transcripts_poller])
def test_poller_note_reports_what_the_tier_filter_skipped(
    monkeypatch, mixed_tier_universe, module,
):
    # Standing rule: no silent caps. /api/admin/cron-health should show the
    # constraint is active rather than just a smaller number of pulls.
    ds = get_data_service()
    curated = len(ds.list_tickers(tiers=AUTO_PULL_TIERS))
    excluded = len(ds.list_tickers()) - curated
    assert excluded >= 2, "fixture should leave at least the two out-of-tier rows"

    _, note = _drive(monkeypatch, module)

    assert f"polled {curated} in-tier" in note
    assert f"skipped {excluded} out-of-tier" in note


# ---------------------------------------------------------------------------
# Prose vs. behaviour
# ---------------------------------------------------------------------------

def _claimed_tiers(doc: str) -> set[str]:
    """Tiers named by a module docstring's claim.

    The claim is the first paragraph that names any tier literal. Everything
    after it is rationale, which has to be free to name the tiers it is
    explaining the exclusion of.
    """
    for paragraph in (doc or "").strip().split("\n\n"):
        named = {t for t in _TIER_LITERALS if t in paragraph}
        if named:
            return named
    return set()


@pytest.mark.parametrize("module", [edgar_poller, transcripts_poller])
def test_docstring_claim_matches_the_tier_run_once_actually_selects(
    monkeypatch, mixed_tier_universe, module,
):
    claimed = _claimed_tiers(module.__doc__)
    assert claimed, (
        f"{module.__name__}'s docstring no longer names the tier it polls; "
        "state the constraint so this guard can check it"
    )

    polled, _ = _drive(monkeypatch, module)
    tier_of = _tier_of()
    selected = {tier_of[t] for t in polled}

    assert claimed == set(AUTO_PULL_TIERS), (
        f"{module.__name__} claims it polls {sorted(claimed)}, but the constant "
        f"is {sorted(AUTO_PULL_TIERS)}"
    )
    assert selected == claimed, (
        f"{module.__name__} claims it polls {sorted(claimed)} but actually "
        f"polled {sorted(selected)} — the exact divergence this file exists for"
    )
