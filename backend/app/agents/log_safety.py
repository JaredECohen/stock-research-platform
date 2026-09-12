"""Secret-safe exception logging for the agent and provider layers.

Provider SDK exceptions routinely embed the request that failed: OpenAI's
`AuthenticationError` quotes the bearer header, `httpx` errors carry the
full URL including `?apikey=…`, and Google's client echoes `key=AIza…`.
Every `log.warning("... failed: %s", exc)` in this codebase was therefore
one bad key away from writing a credential into Render's log stream, which
is retained, searchable, and shared with anyone who has dashboard access.

The policy here is deliberately asymmetric:

  - The operational level (WARNING by default) gets the message plus the
    exception *type* only. That is enough to alert on and to correlate
    with `LLMCallLog` / `CacheCostLog`, and it can never leak.
  - The redacted, length-bounded detail goes to DEBUG, which production
    does not emit. Flip a logger to DEBUG locally to see what actually
    went wrong; the redaction still applies there.

`redact` is pattern-based rather than value-based (we don't look up
`settings.*_api_key` and replace it) so it also catches keys that reach us
from somewhere other than our own config — a partner's token echoed in a
response body, for instance.
"""
from __future__ import annotations

import logging
import re
from typing import Any

# Anything longer than this is not a useful log line, and a truncated
# secret is still a secret — so redact first, then cut.
MAX_DETAIL_CHARS = 300

# The mask carries no trace of the original — not even the `sk-` prefix —
# so a downstream "does this payload contain `sk-`?" check stays a valid
# leak detector rather than tripping on our own redaction marker.
MASK = "<redacted>"

# Ordered: the more specific shapes run before the generic ones so the
# generic base64 sweep doesn't half-mask a key that has a known prefix.
_PATTERNS = [
    # `Authorization: Bearer <token>` in any casing.
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-+/=]{6,}"), f"Bearer {MASK}"),
    # OpenAI / Anthropic keys (`sk-…`, `sk-ant-…`).
    (re.compile(r"sk-[A-Za-z0-9_\-]{6,}"), MASK),
    # Google API keys.
    (re.compile(r"AIza[0-9A-Za-z_\-]{10,}"), MASK),
    # Query-string / kwarg / dict-repr credentials: apikey=…, api_key=…,
    # token=…, registrationkey=… (BLS), 'access_token': '…'. Stops at `&`,
    # whitespace, or a closing quote so the rest of the URL stays readable.
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|token|access[_-]?token|registrationkey)"
            r"(['\"]?\s*[=:]\s*['\"]?)[A-Za-z0-9._\-+/=]{6,}"
        ),
        rf"\1\2{MASK}",
    ),
    # Long unbroken base64-ish runs — JWTs, opaque session tokens, raw
    # key material without a recognisable prefix. 40 chars is above any
    # ticker/CIK/accession number and below every real token we handle.
    (re.compile(r"[A-Za-z0-9+/_\-]{40,}={0,2}"), MASK),
]


def redact(text: Any) -> str:
    """Mask credential-shaped substrings and bound the length.

    Accepts anything — callers pass exceptions, response bodies, or
    dicts — and always returns a `str`. Never raises: a logging helper
    that throws inside an `except` block hides the original failure.
    """
    try:
        s = text if isinstance(text, str) else str(text)
    except Exception:  # pragma: no cover — pathological __str__
        s = f"<unprintable {type(text).__name__}>"
    for pattern, replacement in _PATTERNS:
        s = pattern.sub(replacement, s)
    if len(s) > MAX_DETAIL_CHARS:
        s = s[:MAX_DETAIL_CHARS] + "…"
    return s


def safe_exc(exc: BaseException) -> str:
    """`ExcType: <redacted, bounded message>` — safe to log or persist."""
    return f"{type(exc).__name__}: {redact(exc)}"


def log_safely(
    log: logging.Logger,
    msg: str,
    exc: BaseException | None,
    *,
    level: int = logging.WARNING,
) -> None:
    """Log `msg` with the exception type at `level`; redacted detail at DEBUG.

    `msg` is a finished string (f-string at the call site), not a
    %-format template — the two records must not diverge in what they
    interpolate, and the operational line must never receive `exc`.

    `exc=None` is for operational events that have no exception to show
    (a failover whose cause the wrapper already swallowed): the message
    alone is logged at `level`, redacted, so every line that leaves the
    agent layer goes through the same mask.
    """
    if exc is None:
        log.log(level, "%s", redact(msg))
        return
    log.log(level, "%s: %s", msg, type(exc).__name__)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("%s — %s", msg, safe_exc(exc))
