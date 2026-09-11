"""Central logging configuration for both entrypoints.

Why this module exists: `app/main.py` and `app/worker.py` each called
`logging.basicConfig(level=logging.INFO)` directly, which turns on
**httpx's** request logger. httpx logs the full request URL including the
query string, and four providers pass their credential as a query
parameter — `fmp_provider` and `alpha_vantage_provider` (`apikey`),
`polygon_provider` (`apiKey`), `census_provider` (`key`). Every provider
call therefore wrote a live API key into Render's log retention in
plaintext, readable by anyone with dashboard access. Confirmed in
production logs on 2026-09-10.

The fix belongs here rather than in each provider, for two reasons:

  - The leak is a property of the logger, not of any one call site. A
    provider added later that puts a credential in a query string would
    silently reintroduce it.
  - Silencing the request logger alone is not enough. An httpx exception
    carries the URL in its own message, so a perfectly ordinary
    `log.warning("provider failed: %s", exc)` elsewhere in the codebase
    leaks the same secret. `SecretScrubbingFilter` closes that path.

Defence in depth, so both are applied: httpx's per-request INFO logging is
turned off, *and* anything that still reaches a handler is scrubbed.
"""
from __future__ import annotations

import logging
import re

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Third-party loggers that emit one INFO line per network call. Their
# volume is noise; their content is a credential leak.
NOISY_LOGGERS: tuple[str, ...] = ("httpx", "httpcore")

# Matches a secret carried in a URL query string. The leading `?` or `&` is
# required on purpose: without it, `key=` would also rewrite ordinary prose
# like "cache key=NVDA:company_cold", making logs harder to read while
# protecting nothing.
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[-_]?key|apikey|key|token|auth|access[-_]?token|secret)=)[^&\s\"'<>]+"
)

REDACTED = "<redacted>"


def redact_secrets(text: str) -> str:
    """Replace query-string credentials in `text` with a placeholder."""
    return _QUERY_SECRET.sub(lambda m: f"{m.group(1)}{REDACTED}", text)


class SecretScrubbingFilter(logging.Filter):
    """Scrub query-string credentials out of every record that is emitted.

    Formats the record early (via `record.getMessage()`) and replaces
    `msg`/`args`, which is the only way a `logging.Filter` can reach text
    that lives in the arguments rather than the format string — and the
    leak does live in the arguments, since the URL arrives as `%s`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover — a broken record must still log
            return True
        cleaned = redact_secrets(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def configure_logging(level: int = logging.INFO, *, fmt: str = DEFAULT_FORMAT) -> None:
    """Configure root logging, quiet the per-request loggers, scrub secrets.

    Idempotent: safe to call from more than one entrypoint, and from tests.
    """
    logging.basicConfig(level=level, format=fmt)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    scrubber = SecretScrubbingFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, SecretScrubbingFilter) for f in handler.filters):
            handler.addFilter(scrubber)
