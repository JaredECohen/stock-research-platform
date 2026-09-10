"""Scorecard evaluation job (Phase 6): quintile long/short, FF5+momentum
regression and the double-selection LASSO, persisted with caveats.

Worker-only. The panel is assembled from what the scoring runs already
wrote — month-end `scorecard_scores` rows from succeeded runs joined to
`price_month_ends` for the next month's return — and handed to the pure
arithmetic in `finance.scorecard_evaluation_math` and
`finance.double_selection_lasso`. The Ken French monthly series come
through `factor_analytics._full_series_points` (provider-cached 7 days);
tests monkeypatch `_load_factor_points` and never reach a provider.

Every result row carries `EVALUATION_CAVEATS` verbatim (unadjusted prices,
current constituents, current sector labels, restated values at original
availability, no costs) and the minimums actually applied (`min_leg`,
`min_months`, `min_obs`, `min_coverage`), so a reader can see both what
was measured and what the measurement cannot tell them. Statistics that
cannot be computed are None with a reason, never 0. Model output for
research and education — a scenario read of one sample, not a
recommendation.

Control set deviation from plan §5.4 (recorded, not hidden): fs-v1 runs
the LASSO with `LASSO_CONTROLS` + sector one-hots only; `book_to_market`,
`asset_growth_1y` and `net_debt_to_assets` are deferred
(`LASSO_CONTROLS_DEFERRED`, persisted on every LASSO row) because the
score row does not keep the lines they need. And because the price store
is fed from the 252-day cached series, forward returns and 12-1 momentum
exist only for stored months — the LASSO reports `insufficient_data`,
with `PRICE_DEPTH_NOTE` in its reasons, until the store has deepened.

numpy is imported inside functions (the module must stay importable on
the web process, which only reads the persisted rows).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from sqlalchemy import select

from ..agents.log_safety import safe_exc
from ..config import settings
from ..database import SessionLocal
from ..finance import double_selection_lasso as dsl
from ..finance import scorecard_evaluation_math as sem
from ..models import Company, PriceMonthEnd, ScorecardEvaluation, ScorecardRun, ScorecardScore
from . import scorecard_queue

log = logging.getLogger(__name__)

KIND_QUINTILE = "quintile_ls"
KIND_FF6 = "ff6_regression"
KIND_LASSO = "double_lasso"
EVAL_KINDS: tuple[str, ...] = (KIND_QUINTILE, KIND_FF6, KIND_LASSO)

# Controls the double-selection test can build from what the scorecard
# already stores plus the month-end price store and `companies.beta`:
# size, 12-1 momentum, 1-month reversal, market beta, return on assets,
# then sector one-hots (today's labels — a documented residue).
LASSO_CONTROLS: tuple[str, ...] = ("log_mktcap", "momentum_12_1", "reversal_1m", "beta", "roa")
# Plan §5.4 also named these three. They are NOT in fs-v1: the score row
# keeps feature values, not the balance-sheet lines they need, and the
# spec is frozen under `spec_hash`. Persisted on every LASSO row
# (`params.controls_deferred`) so a reader of the verdict sees the
# narrower control set rather than inferring it from the code. fs-v2 hook:
# store `shareholders_equity`, `total_assets` (t, t-1) and net debt on the
# row's `_context` and add them here.
LASSO_CONTROLS_DEFERRED: dict[str, str] = {
    "book_to_market": "needs shareholders_equity on the score row (only the derived ratio is stored)",
    "asset_growth_1y": "needs total_assets for two fiscal years on the score row",
    "net_debt_to_assets": "needs net debt and total_assets on the score row; null for Financials by the sector mask",
}
# Why the panel is thin for a long time after launch, stated where the
# verdict is read: the store is fed from the app's 252-day cached series
# (a Phase 6 decision, no new provider key), so it starts ~12 months deep
# and gains one month per `pit_prepare`.
PRICE_DEPTH_NOTE = (
    "price_month_ends is fed from the 252-day cached price series, so it starts about 12 months deep and "
    "deepens by one month per month: a forward return exists only for months whose next month-end close is "
    "stored, and momentum_12_1 needs 13 stored month ends per ticker. The LASSO's min_obs (2000) and "
    "min_months (24) are not reachable until the store has grown, and the quintile / FF6 samples cover only "
    "the stored months — expect insufficient_data here for well over a year after launch; the rows are honest, "
    "not broken."
)
DEFAULT_MIN_MONTHS = dsl.DEFAULT_MIN_MONTHS
DEFAULT_MIN_OBS = dsl.DEFAULT_MIN_OBS


def _utcnow() -> datetime:
    return datetime.utcnow()


def _ensure_tables(db) -> None:
    bind = db.get_bind()
    for model in (ScorecardEvaluation, ScorecardRun, ScorecardScore, PriceMonthEnd, Company):
        model.__table__.create(bind=bind, checkfirst=True)


def _finite(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _load_factor_points() -> dict[str, list[dict[str, Any]]]:
    """`{series_id: [{date, value}, ...]}` for FF5 + MOM + RF, monthly.
    Raises when a series is unavailable; the caller turns that into an
    `insufficient` regression row with the reason, not a crash."""
    from . import factor_analytics
    out: dict[str, list[dict[str, Any]]] = {}
    for sid in list(sem.KFR_MONTHLY_FACTOR_IDS.values()) + [sem.KFR_MONTHLY_RF_ID]:
        points = factor_analytics._full_series_points(sid)
        if not points:
            raise RuntimeError(f"factor series {sid} unavailable")
        out[sid] = points
    return out


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def _next_month_end(d: date) -> date:
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    from calendar import monthrange
    return date(y, m, monthrange(y, m)[1])


def _prev_month_end(d: date, back: int = 1) -> date:
    y, m = d.year, d.month
    for _ in range(back):
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    from calendar import monthrange
    return date(y, m, monthrange(y, m)[1])


def build_panel(version_key: str, *, db=None) -> dict[str, Any]:
    """Month-end observations for the evaluation.

    Returns `{observations, controls, months, n_rows, n_missing_return}`
    where `observations` are `PanelObservation`s (score = overall z,
    forward return = month-end close to the next month-end close using
    `adjusted_close` when both ends have one, else `close`) and
    `controls` is a parallel list of dicts for the LASSO. Only rows from
    SUCCEEDED month-end runs count; where a month has several succeeded
    runs the latest one wins (same rule the readers use).
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_tables(db)
        rows = db.execute(
            select(ScorecardScore, ScorecardRun.id)
            .join(ScorecardRun, ScorecardRun.id == ScorecardScore.run_id)
            .where(
                ScorecardScore.version_key == version_key, ScorecardScore.is_month_end.is_(True),
                ScorecardRun.status == scorecard_queue.STATUS_SUCCEEDED,
            )
            .order_by(ScorecardScore.as_of, ScorecardScore.run_id.desc(), ScorecardScore.ticker)
            .execution_options(yield_per=500)
        )
        latest_run_for_month: dict[date, int] = {}
        picked: list[ScorecardScore] = []
        for score, run_row_id in rows:
            best = latest_run_for_month.get(score.as_of)
            if best is None:
                latest_run_for_month[score.as_of] = run_row_id
                best = run_row_id
            if run_row_id != best:
                continue
            picked.append(score)
        tickers = sorted({s.ticker for s in picked})
        months = sorted(latest_run_for_month)
        if not picked:
            return {"observations": [], "controls": [], "months": [], "n_rows": 0, "n_missing_return": 0}

        needed: set[date] = set()
        for m in months:
            needed.update({m, _next_month_end(m), _prev_month_end(m, 1), _prev_month_end(m, 12)})
        price_rows = db.execute(
            select(PriceMonthEnd.ticker, PriceMonthEnd.month_end, PriceMonthEnd.close, PriceMonthEnd.adjusted_close)
            .where(PriceMonthEnd.ticker.in_(tickers), PriceMonthEnd.month_end.in_(sorted(needed)))
        ).all()
        prices: dict[tuple[str, date], tuple[float | None, float | None]] = {
            (t, me): (_finite(c), _finite(a)) for t, me, c, a in price_rows
        }
        betas = {t: _finite(b) for t, b in db.execute(
            select(Company.ticker, Company.beta).where(Company.ticker.in_(tickers))
        ).all()}
    finally:
        if own:
            db.close()

    def _ret(t: str, start: date, end: date) -> float | None:
        p0, p1 = prices.get((t, start)), prices.get((t, end))
        if p0 is None or p1 is None:
            return None
        # Adjusted closes only when BOTH ends carry one — mixing bases
        # manufactures a return out of a corporate action.
        if p0[1] is not None and p1[1] is not None and p0[1] > 0:
            return p1[1] / p0[1] - 1.0
        if p0[0] is not None and p1[0] is not None and p0[0] > 0:
            return p1[0] / p0[0] - 1.0
        return None

    observations: list[sem.PanelObservation] = []
    controls: list[dict[str, Any]] = []
    n_missing = 0
    for s in picked:
        fwd = _ret(s.ticker, s.as_of, _next_month_end(s.as_of))
        if fwd is None:
            n_missing += 1
        observations.append(sem.PanelObservation(
            as_of=s.as_of.isoformat(), ticker=s.ticker, score=_finite(s.overall_z), forward_return=fwd,
            coverage=_finite(s.coverage),
        ))
        raw = s.feature_raw or {}
        ctx = raw.get("_context") or {}
        mktcap = _finite(ctx.get("market_cap"))
        controls.append({
            "log_mktcap": math.log(mktcap) if mktcap is not None and mktcap > 0 else None,
            "momentum_12_1": _ret(s.ticker, _prev_month_end(s.as_of, 12), _prev_month_end(s.as_of, 1)),
            "reversal_1m": _ret(s.ticker, _prev_month_end(s.as_of, 1), s.as_of),
            "beta": betas.get(s.ticker),
            "roa": _finite(raw.get("roa")),
            "sector": s.sector or "",
        })
    return {"observations": observations, "controls": controls, "months": [m.isoformat() for m in months],
            "n_rows": len(picked), "n_missing_return": n_missing}


# ---------------------------------------------------------------------------
# The three evaluations
# ---------------------------------------------------------------------------

def _quintile(panel: dict[str, Any], *, min_leg: int, min_coverage: float) -> dict[str, Any]:
    return sem.quintile_long_short(panel["observations"], min_leg=min_leg, min_coverage=min_coverage)


def _ff6(quintile: dict[str, Any], *, min_months: int, loader: Callable[[], dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    months = quintile.get("months") or []
    try:
        factor_points = loader()
    except Exception as exc:
        log.warning("scorecard evaluation: factor series unavailable: %s", safe_exc(exc))
        return {
            "series": "spread", "alpha_monthly": None, "alpha_annualized": None, "alpha_t": None, "alpha_se": None,
            "betas": {n: None for n in sem.FF6_FACTOR_NAMES}, "beta_t": {n: None for n in sem.FF6_FACTOR_NAMES},
            "r_squared": None, "n_months": 0, "start": None, "end": None, "insufficient": True,
            "reasons": [f"factor series unavailable: {type(exc).__name__}"], "aligned": None,
            "caveats": list(sem.EVALUATION_CAVEATS),
        }
    aligned = sem.align_monthly_factors(months, factor_points)
    if aligned["n_aligned"] == 0:
        # No spread month met a factor month: nothing to regress. Reported as
        # insufficient with the reason rather than fed to the regression as
        # an empty (and mis-shaped) design.
        reg: dict[str, Any] = {
            "series": "spread", "alpha_monthly": None, "alpha_annualized": None, "alpha_t": None, "alpha_se": None,
            "betas": {n: None for n in sem.FF6_FACTOR_NAMES}, "beta_t": {n: None for n in sem.FF6_FACTOR_NAMES},
            "r_squared": None, "n_months": 0, "start": None, "end": None, "insufficient": True,
            "reasons": [f"no spread months aligned to the factor series ({len(months)} spread months, "
                        f"{len(aligned['missing_months'])} unmatched)"],
        }
    else:
        reg = sem.factor_regression(
            aligned["returns"], aligned["factors"], factor_names=aligned["factor_names"], min_obs=min_months,
            dates=aligned["return_months"], series="spread",
        )
    reg["aligned"] = {"n_aligned": aligned["n_aligned"], "missing_months": aligned["missing_months"][:24],
                      "n_missing": len(aligned["missing_months"])}
    reg["caveats"] = list(sem.EVALUATION_CAVEATS)
    return reg


def _lasso(
    panel: dict[str, Any], quintile: dict[str, Any], *, min_months: int, min_obs: int, min_coverage: float,
) -> dict[str, Any]:
    import numpy as np

    ew_by_month = {m["as_of"]: m["universe_ew"] for m in (quintile.get("months") or [])}
    sectors = sorted({c["sector"] for c in panel["controls"] if c.get("sector")})
    names = list(LASSO_CONTROLS) + [f"sector:{s}" for s in sectors]
    y: list[float] = []
    d: list[float] = []
    X: list[list[float]] = []
    month_ids: list[str] = []
    n_skipped = 0
    # Why each row fell out, so an `insufficient_data` verdict can name
    # the cause (a thin price store looks exactly like a broken panel
    # from `n_obs` alone).
    skipped_by_reason: dict[str, int] = {}

    def _skip(reason: str) -> None:
        nonlocal n_skipped
        n_skipped += 1
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1

    for obs, ctl in zip(panel["observations"], panel["controls"]):
        ew = ew_by_month.get(obs.as_of)
        if obs.forward_return is None:
            _skip("no_forward_return")
            continue
        if ew is None:
            _skip("month_not_eligible")
            continue
        if obs.score is None:
            _skip("no_score")
            continue
        if obs.coverage is None or obs.coverage < min_coverage:
            _skip("low_coverage")
            continue
        base = [ctl.get(k) for k in LASSO_CONTROLS]
        missing = [k for k, v in zip(LASSO_CONTROLS, base) if v is None]
        if missing:
            _skip("missing_control:" + missing[0])
            continue
        one_hot = [1.0 if ctl.get("sector") == s else 0.0 for s in sectors]
        y.append(obs.forward_return - ew)
        d.append(obs.score)
        X.append([float(v) for v in base] + one_hot)   # type: ignore[arg-type]
        month_ids.append(obs.as_of)
    n = len(y)
    if n == 0:
        result = dsl._insufficient(n_obs=0, n_months=0, p_controls=len(names), n_dropped=n_skipped,
                                   reasons=["no observations with a score, a forward return and every control"])
        out = result.to_dict()
    else:
        ids = np.asarray(month_ids)
        y_dm = dsl.demean_within_groups(np.asarray(y), ids)
        d_dm = dsl.demean_within_groups(np.asarray(d), ids)
        X_dm = dsl.demean_within_groups(np.asarray(X), ids)
        result = dsl.double_selection(y_dm, d_dm, X_dm, ids, control_names=names,
                                      min_months=min_months, min_obs=min_obs)
        out = result.to_dict()
    out["controls"] = names
    out["controls_deferred"] = dict(LASSO_CONTROLS_DEFERRED)
    out["n_skipped_rows"] = n_skipped
    out["n_skipped_by_reason"] = dict(sorted(skipped_by_reason.items()))
    price_depth_limited = bool(
        skipped_by_reason.get("no_forward_return") or skipped_by_reason.get("missing_control:momentum_12_1")
        or skipped_by_reason.get("missing_control:reversal_1m")
    )
    if out.get("verdict") == dsl.VERDICT_INSUFFICIENT:
        reasons = list(out.get("reasons") or [])
        if skipped_by_reason:
            reasons.append(
                "rows dropped: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped_by_reason.items()))
            )
        if price_depth_limited:
            reasons.append(PRICE_DEPTH_NOTE)
        out["reasons"] = reasons
        out["interpretation"] = "No verdict: " + "; ".join(reasons) + "."
    out["price_depth_limited"] = price_depth_limited
    out["outcome"] = "next-month return minus the equal-weight universe return, demeaned within month"
    out["treatment"] = "scorecard overall z at the formation month end, demeaned within month"
    out["caveats"] = list(sem.EVALUATION_CAVEATS)
    return out


def run_evaluation(
    version_key: str,
    *,
    run_id: str = "",
    run_row_id: int | None = None,
    factor_loader: Callable[[], dict[str, list[dict[str, Any]]]] | None = None,
    min_leg: int | None = None,
    min_coverage: float | None = None,
    min_months: int = DEFAULT_MIN_MONTHS,
    min_obs: int = DEFAULT_MIN_OBS,
) -> dict[str, Any]:
    """Job body for `run_kind="evaluate"`: build the panel, run the three
    evaluations, persist one `scorecard_evaluations` row per kind, and
    finish the run row. Returns `{run, rows: {kind: id}, note}`.

    All three rows are written even when a sample is too thin — the
    `insufficient` / `insufficient_data` markers are the honest answer,
    and a missing row would read as "not evaluated" rather than "cannot
    be evaluated yet".
    """
    min_leg = int(settings.scorecard_min_leg_n if min_leg is None else min_leg)
    min_coverage = float(settings.scorecard_min_coverage if min_coverage is None else min_coverage)
    loader = factor_loader or _load_factor_points
    if run_row_id is None:
        run_row_id = scorecard_queue.create_running_row(
            version_key=version_key, as_of=_utcnow().date(), kind=scorecard_queue.KIND_EVALUATE,
        )
        run_id = run_id or (scorecard_queue.get_run(run_row_id) or {}).get("run_id", "")
    try:
        panel = build_panel(version_key)
        quintile = _quintile(panel, min_leg=min_leg, min_coverage=min_coverage)
        ff6 = _ff6(quintile, min_months=min_months, loader=loader)
        lasso = _lasso(panel, quintile, min_months=min_months, min_obs=min_obs, min_coverage=min_coverage)
        months = panel["months"]
        sample_start = date.fromisoformat(months[0]) if months else None
        sample_end = date.fromisoformat(months[-1]) if months else None
        params_common = {
            "min_leg": min_leg, "min_months": min_months, "min_obs": min_obs, "min_coverage": min_coverage,
            "n_quantiles": sem.DEFAULT_N_QUANTILES, "n_panel_rows": panel["n_rows"],
            "n_missing_forward_return": panel["n_missing_return"], "price_basis": "adjusted_close when both "
            "ends carry one, else close (FMP-sourced rows are unadjusted)",
            # The sample can only be as deep as the price store; say so
            # next to the counts a reader will otherwise misread.
            "price_store_depth": PRICE_DEPTH_NOTE,
        }
        results = {
            KIND_QUINTILE: (quintile, quintile.get("n_months", 0) and sum(m["n_eligible"] for m in quintile["months"])),
            KIND_FF6: (ff6, ff6.get("n_months", 0)),
            KIND_LASSO: (lasso, lasso.get("n_obs", 0)),
        }
        ids: dict[str, int] = {}
        now = _utcnow()
        with SessionLocal() as db:
            _ensure_tables(db)
            for kind, (result, n_obs) in results.items():
                params = dict(params_common)
                if kind == KIND_LASSO:
                    params["controls"] = list(LASSO_CONTROLS)
                    params["controls_deferred"] = dict(LASSO_CONTROLS_DEFERRED)
                row = ScorecardEvaluation(
                    version_key=version_key, eval_kind=kind, params=params, result=_jsonable(result),
                    sample_start=sample_start, sample_end=sample_end, n_obs=int(n_obs or 0), run_id=run_id or "",
                    created_at=now,
                )
                db.add(row)
                db.flush()
                ids[kind] = int(row.id)
            db.commit()
        note = (
            f"version={version_key} panel_rows={panel['n_rows']} months={len(months)} "
            f"quintile_months={quintile.get('n_months', 0)} skipped_months={quintile.get('n_skipped', 0)} "
            f"mean_spread={_fmt(quintile.get('mean_spread'))} ff6_months={ff6.get('n_months', 0)} "
            f"alpha_t={_fmt(ff6.get('alpha_t'))} lasso_verdict={lasso.get('verdict')} lasso_n={lasso.get('n_obs', 0)} "
            f"written={len(ids)}"
        )
        run = scorecard_queue.finish_run(
            run_row_id, status=scorecard_queue.STATUS_SUCCEEDED, note=note, scored_count=len(ids),
            params_update={"evaluation_ids": ids},
        )
        log.info("scorecard evaluation run %d succeeded: %s", run_row_id, note)
        return {"run": run or {}, "rows": ids, "note": note}
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        log.error("scorecard evaluation run %d failed: %s", run_row_id, safe_exc(exc))
        return {"run": scorecard_queue.mark_failed(run_row_id, exc) or {}, "rows": {}, "note": ""}


def _fmt(v: Any) -> str:
    f = _finite(v)
    return "n/a" if f is None else f"{f:.4f}"


def _jsonable(obj: Any) -> Any:
    """Plain JSON types only: numpy scalars → Python, non-finite → None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return _jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return [_jsonable(v) for v in obj.tolist()]
    except Exception:  # pragma: no cover
        pass
    return str(obj)


__all__ = [
    "DEFAULT_MIN_MONTHS",
    "DEFAULT_MIN_OBS",
    "EVAL_KINDS",
    "KIND_FF6",
    "KIND_LASSO",
    "KIND_QUINTILE",
    "LASSO_CONTROLS",
    "LASSO_CONTROLS_DEFERRED",
    "PRICE_DEPTH_NOTE",
    "build_panel",
    "run_evaluation",
]
