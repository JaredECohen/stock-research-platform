"""SQLAlchemy ORM models for MarketMosaic.

We persist the master security universe, daily prices with provider and
adjustment provenance, annual/quarterly financial periods, generated agent
memos and screener score history. Expiring provider responses are separate
from these durable historical records.

This package replaced the single `models.py` module (RP-004). Every ORM
class is re-exported here so `from app.models import X` keeps working, and
`database.init_db` / `reconcile_missing_columns` still do `from . import
models` to make sure every table is registered on `Base.metadata` before
`create_all`. String foreign keys ("companies.ticker", "memo_snapshots.id")
resolve through the shared metadata, so the module boundaries below do not
matter to SQLAlchemy — they exist for readers.
"""
from __future__ import annotations

# Import the cache models so init_db()'s create_all picks them up. Imported
# for the side-effect of registering on Base.metadata; the symbols themselves
# are re-exported via app.cache.
from ..cache.snapshots import CacheCostLog, ResearchSnapshot  # noqa: F401
from .accounts import (
    ActiveAction,
    AdminOverride,
    BillingWebhookEvent,
    RateLimitWindow,
    Subscription,
    UsageCounter,
    UsageEvent,
    User,
)
from .dcf import DCFModel
from .documents import (
    CachedDocument,
    DocChunk,
    EarningsTranscript,
    FilingDoc,
    FinancialPeriod,
    ProviderCache,
)
from .fundamentals import ChartCommentary
from .industry import (
    CompanyClassification,
    CompanyIndustryClassification,
    CrossIndustrySnapshot,
    GicsNode,
    IndustryReport,
    IndustryReportJob,
    IndustryStatSnapshot,
    TaxonomyVersion,
)
from .jobs import CatalystEvent, CronLoopRun, RegenJob, ThemeExposure
from .market_data import DailyPrice, FinancialDataRepair, MarketDataSync
from .memo import (
    MemoOutcome,
    MemoOutcomeEligibility,
    MemoPostmortem,
    MemoRunCheckpoint,
    MemoSnapshot,
    MispricingAudit,
)
from .portfolio import PortfolioRun
from .public import AnalyticsEvent, PublicSample
from .scorecard import (
    PriceMonthEnd,
    ScorecardDisagreement,
    ScorecardEvaluation,
    ScorecardRun,
    ScorecardScore,
    ScorecardVersion,
)
from .telemetry import LLMCallLog, SDKTrace, UILog
from .universe import Company, ScreenerMetric, ScreenerScore, StockMemo

__all__ = [
    # accounts (FEAT-002)
    "User",
    "Subscription",
    "UsageCounter",
    "UsageEvent",
    "AdminOverride",
    "BillingWebhookEvent",
    "RateLimitWindow",
    "ActiveAction",
    # public (FEAT-002)
    "PublicSample",
    "AnalyticsEvent",
    # universe
    "Company",
    "StockMemo",
    "ScreenerScore",
    "ScreenerMetric",
    # documents
    "CachedDocument",
    "ProviderCache",
    "FinancialPeriod",
    "DailyPrice",
    "MarketDataSync",
    "FinancialDataRepair",
    "FilingDoc",
    "EarningsTranscript",
    "DocChunk",
    # memo
    "MemoSnapshot",
    "MemoRunCheckpoint",
    "MemoOutcome",
    "MemoOutcomeEligibility",
    "MemoPostmortem",
    "MispricingAudit",
    # dcf
    "DCFModel",
    # portfolio
    "PortfolioRun",
    # telemetry
    "LLMCallLog",
    "SDKTrace",
    "UILog",
    # jobs
    "ThemeExposure",
    "CatalystEvent",
    "RegenJob",
    "CronLoopRun",
    # fundamentals (FEAT-001)
    "ChartCommentary",
    # scorecard (Phase 6)
    "ScorecardVersion",
    "ScorecardRun",
    "ScorecardScore",
    "PriceMonthEnd",
    "ScorecardEvaluation",
    "ScorecardDisagreement",
    # industry (FEAT-003)
    "TaxonomyVersion",
    "GicsNode",
    "CompanyIndustryClassification",
    "CompanyClassification",
    "IndustryStatSnapshot",
    "CrossIndustrySnapshot",
    "IndustryReport",
    "IndustryReportJob",
]
