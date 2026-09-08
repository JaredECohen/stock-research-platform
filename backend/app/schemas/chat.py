"""Chat schemas — request/response envelope for the research chat.

`ChatResponse` carries every other result type as an optional payload, so
this module sits at the leaf of the schema import DAG (see `memo.py` for
why the referenced classes must be imported here rather than re-exported).
"""
from __future__ import annotations

from typing import List, Literal, Optional

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
    history: List[ChatMessage] = Field(default_factory=list)


class AgentTrace(BaseModel):
    agent: str
    status: Literal["queued", "running", "done"] = "done"
    detail: str = ""


class ChatResponse(BaseModel):
    intent: IntentType
    answer: str
    agent_trace: List[AgentTrace] = Field(default_factory=list)
    memo: Optional[StockMemoOut] = None
    portfolio: Optional[ModelPortfolio] = None
    macro: Optional[MacroScenarioResult] = None
    dcf: Optional[DCFResult] = None
    comps: Optional[CompsResult] = None
    screener: Optional[ScreenerResult] = None
    sources: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "MarketMosaic is for research and education only and does not provide personalized financial advice."
    )
