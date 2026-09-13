# JSON call accounting

The September 13 META observation exposed an accounting mismatch: an
Anthropic-to-OpenAI failover could be logged as `call_failed` while that run's
LLM metrics reported zero failed calls. Anthropic recorded nonempty response
text as success, then the JSON wrapper parsed it and could return `None`.
Its breaker was also reset before parsing. Gemini had the same ordering;
OpenAI marked parse exceptions as failed but discarded already-reported usage.

JSON attempts now run the existing parser before their single accounting
write. A result of `None` records failure and advances the provider's existing
breaker. Recovered JSON and previously accepted falsey JSON values keep their
existing outputs and success behavior. Text callers continue to accept text.

The existing `LLMCallLog.error` field carries safe categories:

| Category | Meaning |
| --- | --- |
| `invalid_json_response` | The existing JSON parser produced no usable result, including JSON null. |
| `empty_response` | No response text was available. |
| `provider_error:<exception class>` | The provider call raised before a response was returned; HTTP errors are distinguishable by their class. |
| `response_error:<exception class>` | A response was returned but subsequent processing raised. |

Errors contain neither response bodies nor prompts. Reported input/output
tokens remain attached to failed parses and processing failures. Zero usage
on a provider exception means no usage was available to this wrapper; it does
not independently establish that the provider charged nothing. Each actual
provider attempt writes one call row. A bounded failover writes two rows only
when two providers were actually called, retaining each provider's usage.

When already present on the response, a failed attempt also records a safe
`stop_reason` or `finish_reason` suffix, such as `max_tokens` or `length`.
Only an allowlist of categorical values is retained; unknown metadata becomes
`unrecognized`. This helps distinguish output-limit termination from ordinary
completion without logging model text or making another request.

The repair changes no request prompt, model, routing choice, retry limit,
failover policy, or JSON recovery rule. Correctly counting failures means the
existing breaker can now open on consecutive unusable JSON responses. It
does not enable Gemini. Existing historical call rows, including the observed
META run, are not rewritten. A successful parse does not certify factual
accuracy, completeness, or conformance to an agent's downstream schema.

Offline regressions exercise all three providers through mocked SDK response
objects and the real database writer, metrics aggregation, parser, breaker,
and bounded failover path. They cover malformed/empty/null responses, accepted
falsey values, existing recovery, transport versus processing errors, token
retention, one row per attempt, and unchanged text behavior.
