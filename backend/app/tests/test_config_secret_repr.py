"""Settings never renders a credential in its repr.

pytest prints the operands of a failing assert, so a guard written as
`assert not settings.has_llm` printed `Settings(...)`, configured keys
included, exactly when a developer `.env` was loaded. The same repr is
what `--showlocals`, `%r` logging and frame-capturing error reporters
show. Credential fields are declared `repr=False`; these tests pin that
for every field shaped like a credential, including ones added later.

Sentinel values only, and every assertion compares names, never values:
a failure here must not become the leak it guards against.
"""
from __future__ import annotations

import re

from app.config import Settings

# Name shapes that carry a credential. A new field matching one must be
# declared `Field(..., repr=False)`. `clerk_publishable_key` is public by
# design and hidden anyway: one uniform rule beats an exemption list.
_SECRET_NAME = re.compile(r"(_key|_token|_secret|_salt)$")
# URLs that embed credentials in production (postgres user:password@host,
# redis://:password@host). Listed by name: most *_url fields are public.
_SECRET_URLS = {"database_url", "rate_limit_storage_url"}


def _secret_fields() -> list[str]:
    return sorted(n for n in Settings.model_fields if _SECRET_NAME.search(n) or n in _SECRET_URLS)


def _sentinels() -> dict[str, str]:
    return {name: f"SENTINEL-{name}-do-not-print" for name in _secret_fields()}


def test_the_credential_rule_still_matches_the_known_secrets():
    # If the pattern stopped matching, every check below would pass vacuously.
    known = {"openai_api_key", "anthropic_api_key", "fmp_api_key", "stripe_secret_key",
             "stripe_webhook_secret", "admin_api_token", "abuse_hash_salt", "database_url"}
    missing = sorted(known - set(_secret_fields()))
    assert missing == []


def test_every_credential_field_is_excluded_from_repr():
    shown = [name for name in _secret_fields() if Settings.model_fields[name].repr is not False]
    assert shown == [], "declare these Field(..., repr=False)"


def test_repr_and_str_carry_no_credential_value():
    # `_env_file=None` keeps a developer .env out of the strings under test.
    s = Settings(_env_file=None, **_sentinels())
    rendered = f"{s!r}\n{s!s}"
    leaked = sorted(name for name, value in _sentinels().items() if value in rendered)
    assert leaked == []


def test_repr_false_changes_nothing_but_the_repr():
    # The app reads these attributes and model_dump still carries them.
    sentinels = _sentinels()
    s = Settings(_env_file=None, **sentinels)
    dumped = s.model_dump()
    wrong = sorted(n for n, v in sentinels.items() if getattr(s, n) != v or dumped.get(n) != v)
    assert wrong == []
