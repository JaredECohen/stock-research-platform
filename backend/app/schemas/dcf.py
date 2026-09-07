"""DCF schemas — assumptions, scenarios, sensitivities, guardrails, result."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DCFAssumptions(BaseModel):
    revenue_growth: List[float] = Field(default_factory=lambda: [0.10, 0.09, 0.08, 0.07, 0.06])
    operating_margin: List[float] = Field(default_factory=lambda: [0.25, 0.26, 0.27, 0.27, 0.27])
    tax_rate: float = 0.21
    da_pct_revenue: float = 0.04
    capex_pct_revenue: float = 0.05
    nwc_pct_revenue: float = 0.02
    terminal_growth: float = 0.025
    exit_ebitda_multiple: float = 15.0
    wacc: float = 0.085

    base_revenue: float = 0.0
    net_debt: float = 0.0
    diluted_shares: float = 0.0
    current_price: float = 0.0


class DCFYearProjection(BaseModel):
    year: int
    revenue: float
    ebit: float
    nopat: float
    da: float
    capex: float
    change_nwc: float
    fcff: float
    discount_factor: float
    pv_fcff: float


class ScenarioDriver(BaseModel):
    """Wave 10k — a concrete bull / bear driver for a DCF scenario.

    `name` is short ("AI revenue ramps", "China consumer slowdown");
    `assumption_changes` is the list of fields that this driver
    moved (e.g. ["revenue_growth", "operating_margin"]); `rationale`
    is one sentence explaining why this driver belongs in the
    scenario.
    """
    name: str = ""
    rationale: str = ""
    assumption_changes: List[str] = Field(default_factory=list)


class DCFScenario(BaseModel):
    name: Literal["base", "bull", "bear"]
    label: str
    assumptions: DCFAssumptions
    projections: List[DCFYearProjection]
    pv_explicit: float
    terminal_value_gordon: float
    terminal_value_exit_multiple: float
    pv_terminal_gordon: float
    pv_terminal_exit: float
    enterprise_value_gordon: float
    enterprise_value_exit: float
    enterprise_value_blended: float
    equity_value: float
    # None when the number genuinely cannot be computed — no diluted share
    # count for the implied price, no (positive) current price for the
    # upside. A 0.0 here used to flow into memos as "+0.0%" and into the
    # valuation verdict as a neutral signal, which is a lie, not a value.
    # Renderers print "n/a"; verdict logic treats None as "DCF unavailable".
    implied_share_price: Optional[float] = None
    upside_pct: Optional[float] = None
    # True when WACC − terminal growth was ≤ 0.5% and the Gordon denominator
    # was floored. The terminal value is then a cap, not a valuation, so the
    # flag rides on the scenario for the UI badge + a `check_dcf_realism`
    # warning. Defaults False so memos that pre-date the field validate.
    tv_clamped: bool = False
    # Wave 10k — named drivers for bull / bear scenarios. Empty for
    # the base case + on memos that pre-date the field. Bull / bear
    # scenarios populated via `services/scenario_assumptions.py` so
    # the prose drivers and the assumption changes are tied — no more
    # symmetric ±400bps bumps with no narrative connection.
    drivers: List[ScenarioDriver] = Field(default_factory=list)


class SensitivityCell(BaseModel):
    row_label: str
    col_label: str
    # None mirrors `DCFScenario.implied_share_price` — a grid cell has no
    # implied price when the share count is missing.
    value: Optional[float] = None


class DCFSensitivity(BaseModel):
    name: str
    row_axis: str
    col_axis: str
    rows: List[float]
    cols: List[float]
    cells: List[SensitivityCell]


class DCFGuardrail(BaseModel):
    """Wave 10 — sanity-check flag emitted by `check_dcf_realism`.

    `severity`: warn | error. Warn = 'consider revising'; error =
    'the model is internally inconsistent'.
    """
    severity: Literal["warn", "error"] = "warn"
    message: str = ""
    metric: str = ""  # e.g. "implied_y5_ev_ebitda", "terminal_disagreement"
    value: Optional[float] = None
    cohort_p90: Optional[float] = None


class DCFResult(BaseModel):
    ticker: str
    # None when no quote reached the model (off-universe name, quote chain
    # down). Historical payloads carry 0.0 here; `run_dcf` treats both as
    # "no price" so the upside comes back None rather than -100%.
    current_price: Optional[float] = None
    base: DCFScenario
    bull: DCFScenario
    bear: DCFScenario
    sensitivities: List[DCFSensitivity] = Field(default_factory=list)
    summary: str = ""
    # Wave 10 — reality-check flags. Empty list when nothing tripped.
    # The PM sees these and decides whether to defend or revise the
    # model; the UI surfaces them as a "model warnings" block.
    guardrails: List[DCFGuardrail] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
