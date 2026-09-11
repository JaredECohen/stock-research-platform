"""Provider credentials must never reach a log handler.

Both entrypoints used to call `logging.basicConfig(level=INFO)` directly,
which enables httpx's per-request logger. httpx logs the full URL including
the query string, and four providers pass their key as a query parameter,
so every provider call wrote a live credential into Render's log retention
in plaintext. Confirmed in production logs on 2026-09-10.

Two independent guards, tested separately: httpx's request logging is off,
and anything that still reaches a handler is scrubbed (an httpx exception
carries the URL in its own message, so silencing the request logger alone
would not be enough).
"""
from __future__ import annotations

import logging

import pytest

from app.log_setup import (
    NOISY_LOGGERS,
    REDACTED,
    SecretScrubbingFilter,
    configure_logging,
    redact_secrets,
)

SECRET = "fMGNNRbZ2oJfiHS9XO9Gypi3qHBa7rqv"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    f"https://financialmodelingprep.com/stable/profile?symbol=BRK.B&apikey={SECRET}",
    f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=AAPL&apikey={SECRET}",
    f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={SECRET}",
    f"https://api.census.gov/data/2023/cbp?get=EMP&key={SECRET}",
    f"https://example.com/x?token={SECRET}&other=1",
    f"https://example.com/x?access_token={SECRET}",
    f"https://example.com/x?api_key={SECRET}",
])
def test_query_string_credentials_are_redacted(url):
    cleaned = redact_secrets(url)
    assert SECRET not in cleaned, cleaned
    assert REDACTED in cleaned


def test_redaction_keeps_the_rest_of_the_url_readable():
    """The URL is the diagnostic value; only the secret goes."""
    cleaned = redact_secrets(
        f"https://financialmodelingprep.com/stable/profile?symbol=BRK.B&apikey={SECRET}"
    )
    assert cleaned == (
        f"https://financialmodelingprep.com/stable/profile?symbol=BRK.B&apikey={REDACTED}"
    )


@pytest.mark.parametrize("text", [
    "cache key=NVDA:company_cold written",
    "token=cheap route selected",
    "the api_key setting is missing",
    "secret=None",
])
def test_prose_that_merely_mentions_a_key_is_left_alone(text):
    """Over-redaction makes logs useless and protects nothing: the pattern
    only fires inside a URL query string (after `?` or `&`)."""
    assert redact_secrets(text) == text


# ---------------------------------------------------------------------------
# The filter, on the real logging path
# ---------------------------------------------------------------------------

def test_filter_scrubs_a_secret_passed_as_a_log_argument(caplog):
    """The leak arrives as `%s`, not in the format string, so the filter has
    to reach into the formatted message."""
    logger = logging.getLogger("test.redaction.args")
    logger.addFilter(SecretScrubbingFilter())
    with caplog.at_level(logging.INFO, logger="test.redaction.args"):
        logger.info("HTTP Request: GET %s 200 OK",
                    f"https://financialmodelingprep.com/x?apikey={SECRET}")
    assert SECRET not in caplog.text
    assert REDACTED in caplog.text


def test_filter_scrubs_an_exception_message_carrying_the_url(caplog):
    """An httpx error embeds the URL, so ordinary error logging leaks too."""
    logger = logging.getLogger("test.redaction.exc")
    logger.addFilter(SecretScrubbingFilter())
    exc = RuntimeError(f"Server error '500' for url 'https://x.io/y?apikey={SECRET}'")
    with caplog.at_level(logging.WARNING, logger="test.redaction.exc"):
        logger.warning("Provider fmp.get_company_profile failed: %s", exc)
    assert SECRET not in caplog.text


def test_filter_never_drops_a_record():
    f = SecretScrubbingFilter()
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "clean message", None, None)
    assert f.filter(record) is True


def test_filter_survives_a_record_it_cannot_format():
    """Diagnostics must not be able to raise inside the logging path."""
    f = SecretScrubbingFilter()
    bad = logging.LogRecord("n", logging.INFO, __file__, 1, "%d items", ("not-an-int",), None)
    assert f.filter(bad) is True


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_configure_logging_quiets_the_per_request_loggers():
    configure_logging()
    for name in NOISY_LOGGERS:
        assert logging.getLogger(name).level >= logging.WARNING, name
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO), (
        "httpx INFO logging is what wrote provider keys to the logs"
    )


def test_configure_logging_attaches_exactly_one_scrubber_when_called_twice():
    configure_logging()
    configure_logging()
    for handler in logging.getLogger().handlers:
        scrubbers = [f for f in handler.filters if isinstance(f, SecretScrubbingFilter)]
        assert len(scrubbers) <= 1, "idempotent: entrypoints and tests both call it"


def test_root_handlers_scrub_after_configure(caplog):
    configure_logging()
    root_scrubbed = any(
        isinstance(f, SecretScrubbingFilter)
        for h in logging.getLogger().handlers for f in h.filters
    )
    assert root_scrubbed, "configure_logging must attach the scrubber to the root handler"
