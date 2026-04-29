"""Chat endpoint — main entry point for the 'Ask the PM' interface."""
from __future__ import annotations

from fastapi import APIRouter

from ..agents.orchestrator import Orchestrator
from ..schemas import ChatRequest, ChatResponse

router = APIRouter()
_orch = Orchestrator()


@router.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return _orch.chat(req.message, req.history)
