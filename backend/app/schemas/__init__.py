"""Pydantic schemas used at the API boundary and as agent structured outputs.

These are deliberately verbose: agents emit Pydantic-validated JSON so the
frontend can render rich, deterministic memos even when the LLM is offline.

This package replaced the single `schemas.py` module (RP-004). Every public
name is re-exported here so `from app.schemas import X` keeps working
unchanged; the submodules form a DAG (common -> agents -> dcf / comps /
macro / portfolio / screener -> memo -> chat) so each can be imported on
its own without a cycle.
"""
from __future__ import annotations

from .accounts import (
    AccountOut,
    BillingOut,
    BootstrapOut,
    EntitlementOut,
    PlanStateOut,
    PublicConfigOut,
    StructuredError,
    UsageHistoryItem,
    UsageOut,
    UserOut,
)
from .agents import (
    AgentFinding,
    BullBearAnalysis,
    BullBearCase,
    CatalystItem,
    Citation,
    CriticReview,
    CritiqueOutput,
    CritiqueQuestion,
    EarningsStructured,
    FalsifiableTest,
    GuidanceChange,
    MacroBroadcast,
    NewsAlert,
    NewsSeverity,
    QAThemeAnalysis,
    RiskItem,
    RiskRecommendation,
    RoundFindings,
    SectorQuery,
    SectorReport,
    TechnicalSignals,
    ToneSignal,
    ToolFinding,
)
from .chat import AgentTrace, ChatMessage, ChatRequest, ChatResponse
from .common import (
    CompanyOut,
    IntentType,
    RatingLabel,
    rating_from_stock_score,
    score_from_rating_label,
)
from .comps import CompsHistoryStats, CompsResult, CompsRow
from .dcf import (
    DCFAssumptions,
    DCFGuardrail,
    DCFResult,
    DCFScenario,
    DCFSensitivity,
    DCFYearProjection,
    ScenarioDriver,
    SensitivityCell,
)

# FEAT-001 (`schemas/fundamentals.py`): the catalog, series and
# commentary contracts. Pinned in `test_schema_package_reexports`
# alongside everything else here.
from .fundamentals import (
    AppliedLimits,
    CatalogOut,
    CommentaryOut,
    CommentaryRef,
    CommentaryRequest,
    MemoViewItem,
    MetricSeries,
    MetricSpecOut,
    ObservedItem,
    SeriesCoverage,
    SeriesLimits,
    SeriesPoint,
    SeriesProvenance,
    SeriesRequest,
    SeriesResponse,
    UnavailableTicker,
)

# FEAT-003 (`schemas/industry.py`). Reachable as
# `app.schemas.IndustryReportOut` etc. NOT listed in `__all__` below:
# `test_schema_package_reexports.test_all_lists_match_the_frozen_name_sets`
# pins `__all__` to a frozen name list that lives in a test file this
# slice does not own. The re-export is what callers need; the `__all__`
# entries and the matching `SCHEMA_NAMES` rows must be added together, in
# one commit, or the frozen-list test fails for a cosmetic reason.
from .industry import (
    ClassifyOut,
    ClassifyRequest,
    FactDeltaOut,
    IndustryAccessOut,
    IndustryChangesOut,
    IndustryCompaniesOut,
    IndustryCompanyRowOut,
    IndustryHistoryOut,
    IndustryJobOut,
    IndustryJobsOut,
    IndustryReportHistoryItemOut,
    IndustryReportOut,
    IndustrySnapshotOut,
    IndustryStatsOut,
    RegenerateOut,
    RegenerateRequest,
    TaxonomyImportOut,
    TaxonomyImportRequest,
    TaxonomyOut,
)
from .macro import MacroScenarioRequest, MacroScenarioResult, MacroSeries, MacroSeriesPoint
from .memo import MispricingThesis, StockMemoOut, ValuationVerdict
from .portfolio import ModelPortfolio, PortfolioBrief, PortfolioHolding, PortfolioRequest

# Phase 6 (`schemas/scorecard.py`). Reachable as `app.schemas.ScorecardSummary`
# etc.; deliberately NOT added to `__all__` below, which
# `test_schema_package_reexports.SCHEMA_NAMES` freezes — extend both together.
from .scorecard import (
    ScorecardBackfillOut,
    ScorecardBackfillRequest,
    ScorecardCategory,
    ScorecardContribution,
    ScorecardDetailOut,
    ScorecardDisagreementFlag,
    ScorecardEnqueueOut,
    ScorecardEvaluateRequest,
    ScorecardEvaluationItem,
    ScorecardEvaluationOut,
    ScorecardFeatureOut,
    ScorecardHistoryPoint,
    ScorecardRefreshRequest,
    ScorecardRunOut,
    ScorecardSummary,
    ScorecardUniverseOut,
    ScorecardUniverseRow,
)
from .screener import (
    CustomScreenRequest,
    CustomScreenResult,
    CustomScreenRow,
    ScreenerMetricName,
    ScreenerOp,
    ScreenerRequest,
    ScreenerResult,
    ScreenerRow,
    ScreenerRule,
)

__all__ = [
    # common
    "RatingLabel",
    "rating_from_stock_score",
    "score_from_rating_label",
    "IntentType",
    "CompanyOut",
    # agents
    "CatalystItem",
    "RiskItem",
    "BullBearCase",
    "CritiqueQuestion",
    "CritiqueOutput",
    "RoundFindings",
    "RiskRecommendation",
    "FalsifiableTest",
    "BullBearAnalysis",
    "Citation",
    "AgentFinding",
    "GuidanceChange",
    "ToneSignal",
    "QAThemeAnalysis",
    "EarningsStructured",
    "CriticReview",
    "NewsSeverity",
    "NewsAlert",
    "MacroBroadcast",
    "SectorQuery",
    "SectorReport",
    "ToolFinding",
    "TechnicalSignals",
    # dcf
    "DCFAssumptions",
    "DCFYearProjection",
    "ScenarioDriver",
    "DCFScenario",
    "SensitivityCell",
    "DCFSensitivity",
    "DCFGuardrail",
    "DCFResult",
    # comps
    "CompsRow",
    "CompsHistoryStats",
    "CompsResult",
    # macro
    "MacroSeriesPoint",
    "MacroSeries",
    "MacroScenarioRequest",
    "MacroScenarioResult",
    # portfolio
    "PortfolioRequest",
    "PortfolioBrief",
    "PortfolioHolding",
    "ModelPortfolio",
    # screener
    "ScreenerRow",
    "ScreenerRequest",
    "ScreenerResult",
    "ScreenerMetricName",
    "ScreenerOp",
    "ScreenerRule",
    "CustomScreenRequest",
    "CustomScreenRow",
    "CustomScreenResult",
    # memo
    "MispricingThesis",
    "ValuationVerdict",
    "StockMemoOut",
    # chat
    "ChatMessage",
    "ChatRequest",
    "AgentTrace",
    "ChatResponse",
    # accounts (FEAT-002)
    "StructuredError",
    "EntitlementOut",
    "PlanStateOut",
    "UserOut",
    "BillingOut",
    "AccountOut",
    "BootstrapOut",
    "UsageHistoryItem",
    "UsageOut",
    "PublicConfigOut",
    # fundamentals (FEAT-001)
    "MetricSpecOut",
    "CatalogOut",
    "SeriesRequest",
    "SeriesPoint",
    "SeriesCoverage",
    "SeriesProvenance",
    "MetricSeries",
    "AppliedLimits",
    "SeriesLimits",
    "UnavailableTicker",
    "SeriesResponse",
    "CommentaryRequest",
    "CommentaryRef",
    "ObservedItem",
    "MemoViewItem",
    "CommentaryOut",
    "ScorecardBackfillOut",
    "ScorecardBackfillRequest",
    "ScorecardCategory",
    "ScorecardContribution",
    "ScorecardDetailOut",
    "ScorecardDisagreementFlag",
    "ScorecardEnqueueOut",
    "ScorecardEvaluateRequest",
    "ScorecardEvaluationItem",
    "ScorecardEvaluationOut",
    "ScorecardFeatureOut",
    "ScorecardHistoryPoint",
    "ScorecardRefreshRequest",
    "ScorecardRunOut",
    "ScorecardSummary",
    "ScorecardUniverseOut",
    "ScorecardUniverseRow",
]
