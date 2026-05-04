"""Chat endpoint — main entry point for the 'Ask the PM' interface."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..agents.orchestrator import Orchestrator
from ..rate_limit import LIMITS, limiter
from ..schemas import ChatRequest, ChatResponse

router = APIRouter()
_orch = Orchestrator()


@router.post("/api/chat", response_model=ChatResponse)
@limiter.limit(LIMITS["chat"])
def chat(request: Request, response: Response, req: ChatRequest) -> ChatResponse:
    return _orch.chat(req.message, req.history)
