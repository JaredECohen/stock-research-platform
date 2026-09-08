"""Chat endpoint — main entry point for the 'Ask the PM' interface."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from ..agents.llm import llm_call_context
from ..agents.orchestrator import Orchestrator
from ..auth.entitlements import Grant, require_feature
from ..auth.principal import current_principal
from ..config import settings
from ..rate_limit import LIMITS, limiter
from ..schemas import ChatRequest, ChatResponse
from .gating import rate_scope

router = APIRouter()
_orch = Orchestrator()


@router.post("/api/chat", response_model=ChatResponse)
@limiter.limit(LIMITS["chat"])
def chat(
    request: Request,
    response: Response,
    req: ChatRequest,
    _rate: None = Depends(rate_scope("llm_light")),
    grant: Grant = Depends(require_feature("pm_chat", resource_param=None)),
) -> ChatResponse:
    """One Ask-the-PM turn.

    FEAT-002: metered as `pm_chat` (Free 10 / Pro 300 a month, two in
    flight per user, 10/min). With the login wall on the orchestrator may
    not start a memo run inside the request — `allow_inline_memo` follows
    `settings.memo_inline_generation_effective`, which is `not
    AUTH_ENABLED` unless forced — and answers from stored memos instead,
    naming the tickers that still need a research run in
    `needs_analysis`. The LLM calls the turn makes are attributed to the
    customer and the feature so per-plan margin can be read from
    `llm_call_logs` without any prompt text.
    """
    principal = current_principal(request)
    with llm_call_context(user_id=principal.user_id, feature="pm_chat"):
        return _orch.chat(
            req.message, req.history,
            allow_inline_memo=settings.memo_inline_generation_effective,
        )
