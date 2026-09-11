"""RP-004 — `app.schemas` and `app.models` are packages, not modules.

The split moved every class into a submodule and re-exports it from the
package `__init__`, so the 65 + 54 existing `from app.schemas import X` /
`from app.models import Y` sites keep working unchanged. These tests pin
the three things a move can silently break:

  1. a name that used to be importable no longer is (or resolves to a
     different object than the package advertises);
  2. an ORM class stops registering on `Base.metadata`, so `create_all`
     never creates its table (the cache tables are registered purely by
     a side-effect import in `models/__init__.py`, which is easy to lose);
  3. a pydantic forward reference stops resolving. Under
     `from __future__ import annotations` pydantic resolves annotations
     from the *defining module's* globals, so a class that is only
     reachable through the package `__init__` leaves the model incomplete
     and validation fails at first use, not at import.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys

import pytest
from pydantic import BaseModel

import app.models as models_pkg
import app.schemas as schemas_pkg
from app.database import Base

# Every public name the pre-split `schemas.py` defined (classes, the two
# rating helpers and the Literal aliases). Hardcoded on purpose: the point
# is to notice when the package stops exporting one of them.
SCHEMA_NAMES = [
    "RatingLabel", "rating_from_stock_score", "score_from_rating_label",
    "IntentType", "CompanyOut",
    "CatalystItem", "RiskItem", "BullBearCase", "CritiqueQuestion",
    "CritiqueOutput", "RoundFindings", "RiskRecommendation",
    "FalsifiableTest", "BullBearAnalysis", "Citation", "AgentFinding",
    "GuidanceChange", "ToneSignal", "QAThemeAnalysis", "EarningsStructured",
    "CriticReview", "NewsSeverity", "NewsAlert", "MacroBroadcast",
    "SectorQuery", "SectorReport", "ToolFinding", "TechnicalSignals",
    "DCFAssumptions", "DCFYearProjection", "ScenarioDriver", "DCFScenario",
    "SensitivityCell", "DCFSensitivity", "DCFGuardrail", "DCFResult",
    "CompsRow", "CompsHistoryStats", "CompsResult",
    "MacroSeriesPoint", "MacroSeries", "MacroScenarioRequest",
    "MacroScenarioResult",
    "PortfolioRequest", "PortfolioBrief", "PortfolioHolding", "ModelPortfolio",
    "ScreenerRow", "ScreenerRequest", "ScreenerResult", "ScreenerMetricName",
    "ScreenerOp", "ScreenerRule", "CustomScreenRequest", "CustomScreenRow",
    "CustomScreenResult",
    "MispricingThesis", "ValuationVerdict", "StockMemoOut",
    "ChatMessage", "ChatRequest", "AgentTrace", "ChatResponse",
    # FEAT-002 (additive, `schemas/accounts.py`).
    "StructuredError", "EntitlementOut", "PlanStateOut", "UserOut", "BillingOut",
    "AccountOut", "BootstrapOut", "UsageHistoryItem", "UsageOut", "PublicConfigOut",
    # FEAT-001 (additive, `schemas/fundamentals.py`).
    "MetricSpecOut", "CatalogOut", "SeriesRequest", "SeriesPoint", "SeriesCoverage",
    "SeriesProvenance", "MetricSeries", "AppliedLimits", "SeriesLimits", "UnavailableTicker",
    "SeriesResponse", "CommentaryRequest", "CommentaryRef", "ObservedItem", "MemoViewItem",
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
    # FEAT-003 (`schemas/industry.py`).
    "ClassifyOut",
    "ClassifyRequest",
    "FactDeltaOut",
    "IndustryAccessOut",
    "IndustryChangesOut",
    "IndustryCompaniesOut",
    "IndustryCompanyRowOut",
    "IndustryHistoryOut",
    "IndustryJobOut",
    "IndustryJobsOut",
    "IndustryReportHistoryItemOut",
    "IndustryReportOut",
    "IndustrySnapshotOut",
    "IndustryStatsOut",
    "RegenerateOut",
    "RegenerateRequest",
    "TaxonomyImportOut",
    "TaxonomyImportRequest",
    "TaxonomyOut",
]

# `typing.Literal` aliases are not classes and carry no `__module__`.
_LITERAL_ALIASES = {
    "RatingLabel", "IntentType", "NewsSeverity", "ScreenerMetricName", "ScreenerOp",
}

MODEL_NAMES = [
    "Company", "StockMemo", "ScreenerScore", "CachedDocument", "ProviderCache",
    "ScreenerMetric", "MemoSnapshot", "PortfolioRun", "LLMCallLog", "SDKTrace",
    "UILog", "FinancialPeriod", "FilingDoc", "MemoRunCheckpoint", "DCFModel",
    "MemoOutcome", "EarningsTranscript", "DocChunk", "MemoPostmortem",
    "ThemeExposure", "CatalystEvent", "RegenJob", "MispricingAudit",
    "CronLoopRun",
    # FEAT-002 (`models/accounts.py`, `models/public.py`).
    "User", "Subscription", "UsageCounter", "UsageEvent", "AdminOverride",
    "BillingWebhookEvent", "RateLimitWindow", "ActiveAction",
    "PublicSample", "AnalyticsEvent",
    # FEAT-001 (`models/fundamentals.py`).
    "ChartCommentary",
    # Phase 6 (`models/scorecard.py`).
    "ScorecardVersion", "ScorecardRun", "ScorecardScore", "PriceMonthEnd",
    "ScorecardEvaluation", "ScorecardDisagreement",
    # FEAT-003 (`models/industry.py`). `CompanyClassification` is the
    # plan's frozen-contract alias of `CompanyIndustryClassification`.
    "TaxonomyVersion", "GicsNode", "CompanyIndustryClassification",
    "CompanyClassification", "IndustryStatSnapshot", "CrossIndustrySnapshot",
    "IndustryReport", "IndustryReportJob",
]

# Frozen at the split. `research_snapshots` / `cache_cost_logs` live in
# `app.cache.snapshots` and are only on the metadata because
# `models/__init__.py` imports them for the side effect.
TABLE_NAMES = sorted([
    "cache_cost_logs", "cached_documents", "catalyst_events", "companies",
    "cron_loop_runs", "dcf_models", "doc_chunks", "earnings_transcripts",
    "filing_docs", "financial_periods", "llm_call_logs", "memo_outcomes",
    "memo_postmortems", "memo_run_checkpoints", "memo_snapshots",
    "mispricing_audits", "portfolio_runs", "provider_cache", "regen_jobs",
    "research_snapshots", "screener_metrics", "screener_scores", "sdk_traces",
    "stock_memos", "theme_exposure", "ui_logs",
    # FEAT-002 (`models/accounts.py`, `models/public.py`).
    "users", "subscriptions", "usage_counters", "usage_events", "admin_overrides",
    "billing_webhook_events", "rate_limit_windows", "active_actions",
    "public_samples", "analytics_events",
    # FEAT-001 (`models/fundamentals.py`).
    "chart_commentaries",
    # Phase 6 (`models/scorecard.py`).
    "scorecard_versions", "scorecard_runs", "scorecard_scores", "price_month_ends",
    "scorecard_evaluations", "scorecard_disagreements",
    # FEAT-003 (`models/industry.py`).
    "gics_taxonomy_versions", "gics_nodes", "company_industry_classifications",
    "industry_stats", "industry_snapshots", "industry_reports", "industry_report_jobs",
])

SCHEMA_SUBMODULES = [
    "common", "agents", "dcf", "comps", "macro", "portfolio", "screener",
    "memo", "chat", "accounts", "fundamentals", "industry",
]
MODEL_SUBMODULES = [
    "universe", "documents", "memo", "dcf", "portfolio", "telemetry", "jobs",
    "accounts", "public", "fundamentals", "scorecard", "industry",
]

# sha256 of `StockMemoOut.model_json_schema()` (sorted keys, compact
# separators). The split itself did not move this hash (it was taken on
# the pre-split `schemas.py` and matched after the move); RP-001 then
# added `degradation_events` + `extra_agent_views`; Phase 6 (slice D) then
# added the optional `scorecard: ScorecardSummary | None`. FEAT-003 then
# widened `CritiqueQuestion.target_agent` from a frozen 8-value `Literal`
# to `str`, because the roster — not this schema — is the authority on
# which specialists exist, and `schemas` cannot import `agents.roster`
# without a cycle (roster imports `..schemas`). The frozen enum is what
# made the Industry Group Analyst unreachable from the PM dialog;
# `deep_research._addressable` validates against the live roster instead.
# That widened value is what is pinned here. Update this deliberately, in
# the same commit, whenever `StockMemoOut` itself changes — never to make
# an accidental drift pass.
STOCK_MEMO_OUT_SCHEMA_SHA256 = (
    "c259e5b6e1901f1146b6c3450a012b4816fb5b14d7b0131fbf37becb4f91a562"
)


def _schema_hash(model: type[BaseModel]) -> str:
    blob = json.dumps(model.model_json_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Every old name resolves through the package, from a package submodule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_name_reexported(name):
    obj = getattr(schemas_pkg, name)
    assert name in schemas_pkg.__all__
    if name not in _LITERAL_ALIASES:
        assert obj.__module__.startswith("app.schemas."), obj.__module__
        # The re-export is the defining object, not a copy.
        assert getattr(importlib.import_module(obj.__module__), name) is obj


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_name_reexported(name):
    obj = getattr(models_pkg, name)
    assert name in models_pkg.__all__
    assert obj.__module__.startswith("app.models."), obj.__module__
    assert getattr(importlib.import_module(obj.__module__), name) is obj


def test_all_lists_match_the_frozen_name_sets():
    """`__all__` must not quietly grow or shrink relative to the old module."""
    assert sorted(schemas_pkg.__all__) == sorted(SCHEMA_NAMES)
    assert sorted(models_pkg.__all__) == sorted(MODEL_NAMES)


# ---------------------------------------------------------------------------
# 2. Base.metadata still sees every table
# ---------------------------------------------------------------------------

def test_metadata_table_names_frozen():
    assert sorted(t.name for t in Base.metadata.sorted_tables) == TABLE_NAMES


def test_cache_tables_registered_by_models_package_side_effect():
    """`models/__init__.py` imports `app.cache.snapshots` purely so
    `create_all` picks the cache tables up. Losing that import would not
    break any import site — only production `init_db`."""
    assert "research_snapshots" in Base.metadata.tables
    assert "cache_cost_logs" in Base.metadata.tables


# ---------------------------------------------------------------------------
# 3. Forward references resolve inside each defining module
# ---------------------------------------------------------------------------

def test_every_schema_model_resolves_its_forward_refs():
    """`model_rebuild(raise_errors=True)` re-resolves annotations from the
    defining module's namespace and raises on a name it cannot find. A
    model that references a class defined later in the same module (e.g.
    `RoundFindings` -> `AgentFinding`) is legitimately deferred at import
    and completes here; one that references a class living only in the
    package `__init__` would not."""
    for name in SCHEMA_NAMES:
        cls = getattr(schemas_pkg, name)
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            cls.model_rebuild(raise_errors=True)
            assert cls.__pydantic_complete__, name


@pytest.mark.parametrize("submodule", SCHEMA_SUBMODULES)
def test_schema_submodule_imports_standalone(submodule):
    """Fresh interpreter, one submodule: proves the DAG has no cycle and the
    module does not depend on import order elsewhere in the process."""
    _import_in_subprocess(f"app.schemas.{submodule}")


@pytest.mark.parametrize("submodule", MODEL_SUBMODULES)
def test_model_submodule_imports_standalone(submodule):
    """Also proves no circular import through `app.database`."""
    _import_in_subprocess(f"app.models.{submodule}")


def _import_in_subprocess(dotted: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {dotted}"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


# ---------------------------------------------------------------------------
# 4. The API contract did not move
# ---------------------------------------------------------------------------

def test_stock_memo_out_json_schema_unchanged():
    assert _schema_hash(schemas_pkg.StockMemoOut) == STOCK_MEMO_OUT_SCHEMA_SHA256
