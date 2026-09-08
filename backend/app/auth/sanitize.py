"""Keep credentials out of the auth/billing log stream.

`agents/log_safety.redact` already masks provider API keys, `Bearer …`
and long base64 runs. This layer adds the shapes FEAT-002 introduces —
Stripe keys (`sk_live_…`, `sk_test_…`, `rk_…`), webhook signing secrets
(`whsec_…`) and JWTs (`eyJ….eyJ….sig`) — and offers a logging `Filter`
that applies the mask to every record a module emits, so a careless
`log.warning("bad token %s", token)` cannot leak even if it slips past
review. Belt and braces: the middleware never interpolates a token in
the first place; the filter is for the day someone does.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..agents.log_safety import MASK
from ..agents.log_safety import redact as _base_redact

_PATTERNS = [
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-+/=]{6,}"), f"Bearer {MASK}"),
    # Stripe secret / restricted / publishable keys and webhook secrets.
    (re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{4,}"), MASK),
    (re.compile(r"\bsk_[A-Za-z0-9_]{4,}"), MASK),
    (re.compile(r"\bwhsec_[A-Za-z0-9_]{4,}"), MASK),
    # A JWT: three base64url segments, the first of which decodes to a
    # JSON header and therefore always starts with `eyJ`.
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}"), MASK),
    # A lone header/payload fragment is still a partial credential.
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}"), MASK),
]


def redact(text: Any) -> str:
    """Mask auth/billing credential shapes, then everything log_safety masks.

    Never raises — it is called from `except` blocks and from a logging
    filter, and a helper that throws there hides the original failure.
    """
    try:
        s = text if isinstance(text, str) else str(text)
    except Exception:  # pragma: no cover — pathological __str__
        s = f"<unprintable {type(text).__name__}>"
    for pattern, replacement in _PATTERNS:
        s = pattern.sub(replacement, s)
    return _base_redact(s)


class RedactingFilter(logging.Filter):
    """Masks the message template and every string argument of a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            args = record.args
            if isinstance(args, dict):
                record.args = {k: (redact(v) if isinstance(v, str) else v) for k, v in args.items()}
            elif isinstance(args, tuple):
                record.args = tuple(redact(a) if isinstance(a, str) else a for a in args)
        except Exception:  # pragma: no cover — a filter must never raise
            pass
        return True


def safe_logger(name: str) -> logging.Logger:
    """`logging.getLogger(name)` with the redacting filter attached once.

    Filters on a logger apply only to records created on that logger (not
    to propagated children), so every auth module calls this for its own
    `__name__` rather than relying on a parent.
    """
    log = logging.getLogger(name)
    if not any(isinstance(f, RedactingFilter) for f in log.filters):
        log.addFilter(RedactingFilter())
    return log
