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
_SECRET_NAME = re.compile(r"(_key|_token|_secret|_salt|_password|_dsn)$")
# A URL can embed a credential (postgres user:password@host,
# redis://:password@host), so every *_url / *_uri counts as one unless it is
# listed here as public. The list names the public ones, not the secret ones,
# so it fails closed: a new cache or broker URL is caught without anyone
# remembering to list it, and a new public URL is exempted only on purpose.
_URL_NAME = re.compile(r"(_url|_uri)$")
_PUBLIC_URLS = {"frontend_url", "clerk_jwks_url", "public_base_url"}


def _is_credential(name: str) -> bool:
    if _SECRET_NAME.search(name):
        return True
    return bool(_URL_NAME.search(name)) and name not in _PUBLIC_URLS


def _secret_fields(model: type[Settings] = Settings) -> list[str]:
    return sorted(n for n in model.model_fields if _is_credential(n))


def _shown_in_repr(model: type[Settings] = Settings) -> list[str]:
    return [name for name in _secret_fields(model) if model.model_fields[name].repr is not False]


def _sentinels() -> dict[str, str]:
    return {name: f"SENTINEL-{name}-do-not-print" for name in _secret_fields()}


def test_the_credential_rule_still_matches_the_known_secrets():
    # If the pattern stopped matching, every check below would pass vacuously.
    known = {"openai_api_key", "anthropic_api_key", "fmp_api_key", "stripe_secret_key",
             "stripe_webhook_secret", "admin_api_token", "abuse_hash_salt", "database_url",
             "rate_limit_storage_url"}
    missing = sorted(known - set(_secret_fields()))
    assert missing == []


def test_every_credential_field_is_excluded_from_repr():
    assert _shown_in_repr() == [], (
        "declare these Field(..., repr=False); a URL that can never carry a "
        "credential may instead be added to _PUBLIC_URLS")


def test_a_new_unhidden_credential_field_is_caught_by_its_shape():
    # The rule has to catch fields nobody has listed yet, or the check above
    # only re-verifies today's list. These are the shapes a cache, a broker,
    # an error reporter or a mail relay would plausibly add, all left visible.
    class _Future(Settings):
        cache_redis_url: str = ""
        broker_uri: str = ""
        sentry_dsn: str = ""
        smtp_password: str = ""

    assert _shown_in_repr(_Future) == ["broker_uri", "cache_redis_url", "sentry_dsn", "smtp_password"]


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
