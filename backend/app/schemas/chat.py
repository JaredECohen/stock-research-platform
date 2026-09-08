"""Chat schemas — request/response envelope for the research chat.

`ChatResponse` carries every other result type as an optional payload, so
this module sits at the leaf of the schema import DAG (see `memo.py` for
why the referenced classes must be imported here rather than re-exported).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .common import IntentType
from .comps import CompsResult
from .dcf import DCFResult
from .macro import MacroScenarioResult
from .memo import StockMemoOut
from .portfolio import ModelPortfolio
from .screener import ScreenerResult


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class AgentTrace(BaseModel):
    agent: str
    status: Literal["queued", "running", "done"] = "done"
    detail: str = ""


class ChatResponse(BaseModel):
    intent: IntentType
    answer: str
    agent_trace: list[AgentTrace] = Field(default_factory=list)
    memo: StockMemoOut | None = None
    portfolio: ModelPortfolio | None = None
    macro: MacroScenarioResult | None = None
    dcf: DCFResult | None = None
    comps: CompsResult | None = None
    screener: ScreenerResult | None = None
    sources: list[str] = Field(default_factory=list)
    # FEAT-002: when the login wall is on, chat never generates a memo
    # inside the request. Tickers the answer needed but had no stored memo
    # for are listed here so the UI can offer a (charged) research run.
    # Empty in every other case, including with auth disabled.
    needs_analysis: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "MarketMosaic is for research and education only and does not provide personalized financial advice."
    )
