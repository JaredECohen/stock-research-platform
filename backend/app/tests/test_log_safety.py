"""Credential-shaped text must never reach the operational log level.

Provider SDK exceptions quote the failing request — bearer headers,
`?apikey=` URLs, `key=AIza…` — so a plain `log.warning("...: %s", exc)`
writes the credential into Render's retained log stream. `log_safety`
puts only the exception *type* at WARNING and the redacted, bounded
detail at DEBUG; `safe_runner` and `DegradationLog` follow the same rule
because their output is persisted with the memo.
"""
from __future__ import annotations

import logging

import pytest

from app.agents import log_safety, safe_runner
from app.agents.log_safety import log_safely, redact, safe_exc
from app.agents.safe_runner import DegradationLog, safe_call, safe_finding

OPENAI_KEY = "sk-proj-abc123def456ghi789jkl012mno345pqr678"
ANTHROPIC_KEY = "sk-ant-api03-zyx987wvu654tsr321qpo098"
GOOGLE_KEY = "AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBxY"
BEARER = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc"


class _ProviderError(Exception):
    """Stands in for an SDK exception whose message quotes the request."""


def _boom_with_secrets():
    raise _ProviderError(
        f"401 from https://api.example.com/v1?apikey={OPENAI_KEY} "
        f"(Authorization: {BEARER}; alt key {ANTHROPIC_KEY})"
    )


# ---------------------------------------------------------------------------
# redact()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("secret", [OPENAI_KEY, ANTHROPIC_KEY, GOOGLE_KEY, BEARER])
def test_redact_masks_every_key_shape(secret):
    out = redact(f"request failed: {secret} (retrying)")
    assert secret not in out
    assert "retrying" in out, "redaction must not eat the surrounding message"


def test_redact_masks_query_and_kwarg_credentials():
    out = redact("GET https://x.test/v3/profile/AAPL?apikey=abcdef123456&limit=5")
    assert "abcdef123456" not in out
    assert "limit=5" in out
    out = redact("body={'registrationkey': 'zz11yy22xx33', 'seriesid': ['CUUR0000SA0']}")
    assert "zz11yy22xx33" not in out
    assert "CUUR0000SA0" in out, "a BLS series id is not a credential"
    out = redact("token=abcdefghijk1234 api_key: 'qwertyuiop9876'")
    assert "abcdefghijk1234" not in out and "qwertyuiop9876" not in out


def test_redact_masks_long_opaque_runs_but_keeps_identifiers():
    blob = "A" * 64
    out = redact(f"session {blob} expired for 0001045810-24-000012")
    assert blob not in out
    # Accession numbers / CIKs / tickers are short enough to survive.
    assert "0001045810-24-000012" in out


def test_redact_bounds_length_after_masking():
    out = redact("x" * 5000)
    assert len(out) <= log_safety.MAX_DETAIL_CHARS + 1
    # A key sitting past the cut still gets masked — redact runs first.
    out = redact("y " * 400 + OPENAI_KEY)
    assert OPENAI_KEY not in out


def test_redact_never_raises_on_odd_input():
    class _Bad:
        def __str__(self):
            raise RuntimeError("no str for you")

    assert isinstance(redact(_Bad()), str)
    assert redact(None) == "None"
    # The mask itself must not look like a key, or "sk-" stops being a
    # usable leak check downstream.
    assert "sk-" not in redact({"apikey": OPENAI_KEY})


def test_safe_exc_carries_type_and_redacted_message():
    try:
        _boom_with_secrets()
    except _ProviderError as exc:
        s = safe_exc(exc)
    assert s.startswith("_ProviderError: ")
    for secret in (OPENAI_KEY, ANTHROPIC_KEY, BEARER):
        assert secret not in s


# ---------------------------------------------------------------------------
# log_safely(): WARNING has the type only, DEBUG has the redacted body
# ---------------------------------------------------------------------------

def _warning_text(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def _debug_text(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


def test_log_safely_keeps_secrets_out_of_warning_and_redacts_debug(caplog):
    log = logging.getLogger("test.log_safety")
    caplog.set_level(logging.DEBUG, logger="test.log_safety")
    try:
        _boom_with_secrets()
    except _ProviderError as exc:
        log_safely(log, "provider call failed", exc)

    warn = _warning_text(caplog)
    assert "provider call failed: _ProviderError" in warn
    for secret in (OPENAI_KEY, ANTHROPIC_KEY, BEARER):
        assert secret not in warn
    assert "401" not in warn, "the operational line carries the type only, no body"

    debug = _debug_text(caplog)
    assert "401" in debug, "the redacted body is available at DEBUG"
    assert log_safety.MASK in debug and "sk-" not in debug
    for secret in (OPENAI_KEY, ANTHROPIC_KEY, BEARER):
        assert secret not in debug


def test_log_safely_honours_the_requested_level(caplog):
    log = logging.getLogger("test.log_safety.level")
    caplog.set_level(logging.DEBUG, logger="test.log_safety.level")
    log_safely(log, "telemetry write failed", ValueError("x"), level=logging.DEBUG)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("telemetry write failed: ValueError" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# safe_runner: the memo pipeline's except-sites follow the same rule
# ---------------------------------------------------------------------------

def test_safe_finding_logs_type_at_warning_and_traceback_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger=safe_runner.__name__)
    dlog = DegradationLog()
    finding = safe_finding("Sector Analyst", _boom_with_secrets, log_to=dlog)

    warn = _warning_text(caplog)
    assert "Agent Sector Analyst failed: _ProviderError" in warn
    for secret in (OPENAI_KEY, ANTHROPIC_KEY, BEARER):
        assert secret not in warn

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records and debug_records[0].exc_info is not None, (
        "the traceback must still be recoverable at DEBUG"
    )

    # Persisted surfaces — the degradation banner and the fallback finding
    # — are redacted too.
    assert dlog.failures[0]["error_type"] == "_ProviderError"
    for secret in (OPENAI_KEY, ANTHROPIC_KEY, BEARER):
        assert secret not in dlog.failures[0]["message"]
        assert secret not in finding.key_points[0]
        assert secret not in str(finding.data)


def test_safe_call_does_not_log_the_exception_body(caplog):
    caplog.set_level(logging.DEBUG, logger=safe_runner.__name__)
    out = safe_call(_boom_with_secrets, fallback="fb", name="Thing", log_to=DegradationLog())
    assert out == "fb"
    assert OPENAI_KEY not in _warning_text(caplog)
    assert "Safe call Thing failed: _ProviderError" in _warning_text(caplog)


def test_degradation_log_record_redacts_the_message():
    dlog = DegradationLog()
    dlog.record("Filings Service", RuntimeError(f"denied for {BEARER}"))
    msg = dlog.failures[0]["message"]
    assert BEARER not in msg and f"Bearer {log_safety.MASK}" in msg
    assert len(msg) <= log_safety.MAX_DETAIL_CHARS + 1
