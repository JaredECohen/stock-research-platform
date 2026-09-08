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
from .macro import MacroScenarioRequest, MacroScenarioResult, MacroSeries, MacroSeriesPoint
from .memo import MispricingThesis, StockMemoOut, ValuationVerdict
from .portfolio import ModelPortfolio, PortfolioBrief, PortfolioHolding, PortfolioRequest
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
]
