# FEAT-002 — owner setup and go-live checklist

Accounts (Clerk), the card-less 7-day Pro trial, usage meters and Pro
subscriptions (Stripe) ship **dark**: `AUTH_ENABLED`, `USAGE_LIMITS_ENABLED`
and `BILLING_ENABLED` are `"false"` in `render.yaml` and nothing below
turns on until you flip them in the Render dashboard, in the order in §5.
Everything in this file is an action only the owner can take — dashboard
clicks, secrets, DNS — and none of it is in the repo.

The product these steps expose is the research process in
`docs/research/README.md`: Free "explores the committee's stored work"
(stored memos, screener, follows-memo DCF/comps), Pro "underwrites" (runs
research, DCF, comps, portfolios, macro). Model outputs are not
recommendations; nothing here changes that.

---

## 1. Decisions already made (do not re-decide in the dashboard)

| Topic | Decision |
|---|---|
| Auth | Clerk, verified by the backend from a JWT template named **`marketmosaic`** carrying `email` + `email_verified`. RS256 only. No Clerk backend SDK, no per-request Clerk API call. |
| Sign-in failure limits | **Clerk attack protection** owns them (the backend never sees a sign-in attempt). Enable it — §2 step 6. |
| Billing | Stripe Checkout + Customer Portal via a thin httpx client. **No Stripe SDK.** Webhooks are authoritative; `POST /api/billing/reconcile` is the customer's self-service fix. |
| Trial → paid | Checkout during a trial with ≥48h left passes the local trial end to Stripe (`subscription_data[trial_end]`), so the customer keeps the rest of the trial and Stripe starts `trialing`; under 48h billing starts at checkout and the UI says so. Webhooks never touch `users.trial_*`. |
| Meters | UTC calendar months. Free memo allowance = **3 distinct tickers/month** (opening one memo twice is one). Free 1 research run / 10 chat turns; Pro 20 / 300. `POST /api/macro/analyze` and `POST /api/screener/nl` count as chat turns. |
| Worker secrets | The worker carries **no** `CLERK_*` / `STRIPE_*` values (`test_deploy_config` enforces it). Reconciliation runs on web (webhooks + reconcile endpoint). |
| Reminder emails | Deferred (no email provider in the repo; `users.trial_reminder_sent_at` reserved). |
| Legal pages | Ship as clearly labelled drafts until `LEGAL_REVIEWED_AT` is set. |
| Google sign-in, custom domain | Owner actions (§2 step 2, §4). |

## 2. Clerk

1. **Create the application** at dashboard.clerk.com (instance: *Development* first, *Production* later — each has its own keys and Frontend API).
2. **Sign-in methods**: enable Email (code or magic link). Enable **Google** by supplying a Google OAuth client id/secret (Google Cloud Console → Credentials → OAuth 2.0 client, authorized redirect from the Clerk dashboard). Require a verified email before a session is created if the option is offered; the backend refuses cost-bearing routes to unverified addresses regardless.
3. **JWT template** — Configure → Sessions → *JWT templates* → **New template**, name exactly `marketmosaic`, claims:
   ```json
   {
     "email": "{{user.primary_email_address}}",
     "email_verified": "{{user.email_verified}}"
   }
   ```
   Leave the signing algorithm RS256 (default). The frontend requests tokens with `getToken({ template: "marketmosaic" })`; a token without these claims is treated as an unverified user.
4. **Note three values**: the *Frontend API URL* (`https://<slug>.clerk.accounts.dev` on dev; your custom `clerk.<domain>` on production), the **publishable key** (`pk_…`, public), and the JWKS URL, which is `<Frontend API URL>/.well-known/jwks.json`. The secret key is **not** needed by this backend.
5. **Allowed origins / redirect URLs**: add `PUBLIC_BASE_URL` (§4). The backend checks the token's `azp` against `CLERK_AUTHORIZED_PARTIES`, so a mismatch here is a hard 401.
6. **Attack protection** — Configure → *Attack protection*: enable **bot protection** on sign-up and **brute-force / lockout** on sign-in (this is the "5 failures per 15 minutes" requirement; the backend cannot enforce it). Also enable *Email address verification on sign-up*.
7. Set on the **web** service (Render dashboard, never in `render.yaml`):

   | Variable | Value |
   |---|---|
   | `CLERK_ISSUER` | Frontend API URL, e.g. `https://<slug>.clerk.accounts.dev` |
   | `CLERK_JWKS_URL` | `<CLERK_ISSUER>/.well-known/jwks.json` |
   | `CLERK_PUBLISHABLE_KEY` | `pk_test_…` / `pk_live_…` |
   | `CLERK_AUTHORIZED_PARTIES` | `PUBLIC_BASE_URL` (comma-separate if there are several origins) |

   With `AUTH_ENABLED=true` and either of the first two empty, **every non-public route answers 503 `auth_unavailable`** while the marketing site, `/api/public/*` and `/health` stay up. That is the fail-closed rule, not a bug.

## 3. Stripe (test mode first)

1. **Product**: Products → Add product → "MarketMosaic Pro". Two recurring prices: **$29.99 / month** and **$299 / year** (USD). Copy both price ids (`price_…`).
2. **Customer Portal**: Settings → Billing → Customer portal. Allow: cancel subscription (at period end), update payment method, invoice history. Do **not** allow plan switching unless both prices are added to the portal configuration. Copy the configuration id (`bpc_…`) if you created a non-default one; otherwise leave `STRIPE_PORTAL_CONFIGURATION_ID` empty.
3. **Webhook endpoint**: Developers → Webhooks → Add endpoint → URL `${PUBLIC_BASE_URL}/api/billing/webhook`, API version *2024-06-20* (the client pins `Stripe-Version` per request; a newer default is also read — period fields on items, invoice `parent.subscription_details`). Select exactly these events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.paid`
   - `invoice.payment_failed`
   - `customer.deleted`

   Copy the **signing secret** (`whsec_…`). Anything else Stripe sends is answered `200 ignored_unhandled`.
4. **Stripe Tax / refunds** (owner decision, plan §9 Q11): decide whether Stripe Tax is on and what the billing terms say about refunds before `BILLING_ENABLED` flips in live mode.
5. Set on the **web** service only:

   | Variable | Value |
   |---|---|
   | `STRIPE_SECRET_KEY` | `sk_test_…` now; `sk_live_…` at launch |
   | `STRIPE_WEBHOOK_SECRET` | `whsec_…` from step 3 (a live-mode endpoint has its own) |
   | `STRIPE_PRICE_PRO_MONTHLY` | `price_…` |
   | `STRIPE_PRICE_PRO_ANNUAL` | `price_…` |
   | `STRIPE_PORTAL_CONFIGURATION_ID` | `bpc_…` or empty |

   `POST /api/billing/checkout` answers 503 `billing_unavailable` until key, webhook secret, both price ids **and** `PUBLIC_BASE_URL` are all set; the webhook route answers 503 (so Stripe retries) until the webhook secret is set.

## 4. Environment on the web service

| Variable | Purpose |
|---|---|
| `PUBLIC_BASE_URL` | Checkout success/cancel URLs and the Portal return URL (`/app/billing/success`, `/app/billing/canceled`, `/app/account`); also the Clerk allowed origin. `marketmosaic.ai` refuses :443 today (see MEMORY.md) — launch on `https://marketmosaic.onrender.com` or fix DNS first; changing it later means updating Clerk origins and the Stripe webhook URL. |
| `ABUSE_HASH_SALT` | Random 32+ chars. Salts the bootstrap IP / user-agent hashes on `users` that feed `GET /api/admin/abuse-telemetry`. Rotating it invalidates the "same address" grouping, nothing else. |
| `SAMPLE_TICKERS` | Defaults to `NVDA,COST,JPM`. Change only with stored memos for the new tickers; then run the sample rebuild (§5 step 2). |
| `TRIAL_DAYS` / `GRACE_DAYS` | Default 7 / 7. |
| `ENTITLEMENT_OVERRIDES_JSON` / `RATE_LIMIT_OVERRIDES_JSON` | Change an allowance or a rate scope without a deploy, e.g. `{"pm_chat": {"free": 5}}` / `{"research": "5/hour"}`. Bad JSON is logged and ignored. |
| `LEGAL_REVIEWED_AT` | Set to the review date once privacy / terms / billing terms / cookie pages are reviewed; removes the "draft" banner. |
| `TRUSTED_PROXY_HOPS` | Already `"1"` in `render.yaml` (Render's single proxy). Every per-IP ceiling depends on it. |

The **worker** gets none of the above except what it already has. Do not add `CLERK_*` or `STRIPE_*` there.

## 5. Flag order (each step is a dashboard edit + the verification beside it)

1. **`AUTH_ENABLED=true`** with `USAGE_LIMITS_ENABLED=false`, `BILLING_ENABLED=false`, and the Clerk values from §2 set. Internal accounts only (Clerk dev instance, or restrict sign-ups in Clerk).
   - Verify: signed-out, `/` and `/samples/NVDA` render; `GET /api/stocks` → 401 `auth_required`. Sign in → `POST /api/me/bootstrap` returns `trial_started_now: true` once, and `GET /api/me` shows `plan.plan: "pro"`, `plan.source: "trial"`, an exact `trial_ends_at`. A second bootstrap does not move the date.
   - Verify fail-closed once, deliberately: blank `CLERK_JWKS_URL` → protected routes 503, `/api/public/config` still 200. Restore.
2. **Rebuild the public samples**: `POST /api/admin/samples/rebuild` (admin bearer). The worker builds within ~10 minutes; `GET /api/admin/cron-health` shows `sample_build_loop` with a `built=…` note and `billing_loop` reporting hourly (`expired=… gc=…`).
3. **`USAGE_LIMITS_ENABLED=true`**. Verify with a Free account (let a trial lapse, or use `POST /api/admin/billing/overrides` with `kind: "suspend"` revoked afterwards — or simply a second verified account after its trial is over): 4th distinct memo in a month → 402 `quota_exceeded` with `used: 3, limit: 3, resets_at` = first of next month; `POST /analyze` a second time → 402; DCF on a ticker whose memo was not opened → 402 `plan_required` (follows memo). Support lever: `POST /api/admin/billing/overrides` with `kind: "quota", feature: "memo_view", value: "unlimited"` (or an integer) lifts that one meter for the user until `expires_at`; the 402 becomes a 200 on the next request and `/api/me` shows the new `limit`.
4. **`BILLING_ENABLED=true`** with Stripe **test** keys. Test purchase with card `4242 4242 4242 4242`:
   - Mid-trial checkout: the response says `billing_starts: "at_trial_end"` and Stripe shows the subscription `trialing` until the local trial end; `GET /api/me` shows `billing.stripe_status: "trialing"` and `plan.source: "subscription"`.
   - `/app/billing/success` must not claim Pro from the URL — it polls `/api/me` and offers "Refresh from Stripe" (`POST /api/billing/reconcile`).
   - Portal: cancel at period end → `plan.warning` "Pro ends on <date>"; `subscription_canceled` appears in `analytics_events`.
   - Dashboard → Webhooks → *Resend* an already-delivered event → response `{"outcome": "duplicate"}`. Send a `customer.subscription.updated` older than the last applied one (replay from the event log) → `ignored_stale`.
   - Failed payment (test card `4000 0000 0000 0341` on renewal, or *Subscriptions → … → Mark past due* in test mode) → `plan.source: "grace"` for 7 days, warning "Payment failed — update your card".
   - `GET /api/admin/billing/users/<clerk user id>` shows the subscription row and the last usage events; `billing_webhook_events` (via that view or SQL) shows `applied` rows.
5. **Swap to live keys**: new `STRIPE_SECRET_KEY`, a **live-mode** webhook endpoint with its own `STRIPE_WEBHOOK_SECRET`, live price ids. Repeat one real purchase and refund it in the dashboard.
6. **Clerk production instance** (`pk_live`, production Frontend API on your domain): update the four `CLERK_*` values. Invite-only first, then public sign-up.

Rollback at any step is the flag back to `"false"`: every table is inert with the flags off, `resolve_plan` is read-time (no cron ever downgrades anyone), and `GET /api/stocks/{t}/memo` goes back to today's inline behaviour.

## 6. Verification points after launch

- `GET /api/admin/cron-health`: `billing_loop` fresh (< 26h), `sample_build_loop` fresh (< 8d), no `never run`.
- `GET /api/admin/abuse-telemetry`: `share_429` well under 0.001 for real traffic; `trials_per_ip_hash` with no address above a handful; `rate_limit_hits.by_scope` dominated by `ip`/`bootstrap` if anything.
- Render logs contain no `Bearer `, `sk_`, `whsec_` or `eyJ` strings (`assert_no_secrets_in_logs` is the same check the tests run).
- Stripe dashboard → Webhooks: delivery success rate 100%; any 500s mean a database write failed and Stripe is retrying (fine); any 400s mean the signing secret is wrong (not fine — fix before events age past Stripe's retry window).
- `LLMCallLog.user_id` / `feature` populated for customer runs, so `GET /api/admin/llm-metrics` can be read per plan.

## 7. Canary and rollback

- Flip flags one at a time, during a quiet hour, and watch `/api/admin/cron-health`, `/api/admin/abuse-telemetry` and the Stripe webhook log for 30 minutes before the next one.
- If sign-in breaks (503 `auth_unavailable` everywhere): check `CLERK_ISSUER` / `CLERK_JWKS_URL` first, then Clerk status; cached keys keep verifying for 24h during a Clerk outage.
- If Stripe breaks: checkout/portal 503, **entitlements unchanged** (plans are derived from the DB; reconcile never downgrades on a fetch error). No action needed beyond waiting.
- Rollback = the flag back to `"false"`. No data is deleted; trial dates and subscription rows survive for the next attempt.

## 8. Still owner-only after this checklist

Legal copy review (Q11), refund policy and Stripe Tax, the custom domain, Google OAuth client, reminder emails (Q12), and whether the DEVPLAN allowances survive the measured unit costs (`docs/economics/unit-costs-2026-09.md`, Q1).
