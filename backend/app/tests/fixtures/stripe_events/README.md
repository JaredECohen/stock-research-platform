# Synthetic Stripe events

Hand-written, Stripe-shaped payloads for `test_webhooks.py` and friends.
They are **not** recordings — no live Stripe account was involved — and
they carry no real ids, emails, cards or addresses. Placeholders
(`cus_TEST`, `sub_TEST`, `evt_TEST`, `USER_TEST`, `in_TEST`) are swapped
per test by `app/tests/billing_helpers.load_event`, and the body is
signed at test time with a throwaway secret (`stripe_client.sign_payload`),
so the signature check exercised in CI is the real one.

Shapes follow the 2024-06-20 API (`current_period_*` on the subscription;
`subscription` as a string on the invoice). `billing_service` also reads
the 2025+ placement (`items.data[0].current_period_*`,
`invoice.parent.subscription_details.subscription`), covered by a test
that rewrites the fixture in place.
