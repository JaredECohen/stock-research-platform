import React from "react";
import { Link } from "react-router-dom";
import { Check } from "lucide-react";
import { useConfig } from "@/auth/ConfigProvider";
import { ACCOUNT_PATH, APP_PATH, BTN_GHOST, BTN_PRIMARY, SIGN_UP_PATH } from "./ctas";
import { trialCta } from "./Hero";
import { formatCents, matrixKnown, monthlyEquivalent, rowCells } from "./allowance";
import { useVisitor } from "./hooks";

/**
 * Two plans. Prices come from `config.prices` (cents), the trial length
 * from `config.trial_days`, the bullet numbers from `config.features`.
 * The three honesty rules baked in: no card for the trial; a checkout
 * during the trial keeps the remaining days only when ≥48h remain
 * (Stripe's minimum) and otherwise bills at checkout; periods are UTC
 * calendar months.
 */
function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2 text-sm text-slate-200">
      <Check size={16} className="mt-0.5 text-accent-500 shrink-0" aria-hidden />
      <span>{children}</span>
    </li>
  );
}

export default function PricingTable({ headingLevel = 2 }: { headingLevel?: 2 | 3 }) {
  const { config } = useConfig();
  const { signedIn, authEnabled, loading } = useVisitor();
  const H = `h${headingLevel}` as "h2" | "h3";
  const known = matrixKnown(config.features);
  const cell = (feature: string) => (known ? rowCells(config.features, feature) : null);
  const memo = cell("memo_view");
  const runs = cell("research_run");
  const chat = cell("pm_chat");
  const dcf = cell("dcf");
  const currency = config.prices.currency || "usd";
  const monthly = formatCents(config.prices.monthly_cents, currency);
  const annual = formatCents(config.prices.annual_cents, currency);
  const trialDays = config.trial_days;

  const freeCta = loading ? null : signedIn ? (
    <Link to={APP_PATH} className={`${BTN_GHOST} w-full`}>Continue to app</Link>
  ) : (
    <Link to={SIGN_UP_PATH} className={`${BTN_GHOST} w-full`}>Sign up free</Link>
  );
  const proCta = loading ? null : !authEnabled ? (
    <Link to={APP_PATH} className={`${BTN_PRIMARY} w-full`}>Continue to app</Link>
  ) : signedIn ? (
    <Link to={ACCOUNT_PATH} className={`${BTN_PRIMARY} w-full`}>Manage plan</Link>
  ) : (
    <Link to={SIGN_UP_PATH} className={`${BTN_PRIMARY} w-full`}>{trialCta(trialDays)}</Link>
  );

  return (
    <section aria-labelledby="pricing-heading" className="mt-16">
      <H id="pricing-heading" className="text-2xl font-semibold tracking-tight">Pricing</H>
      <p className="text-sm text-slate-400 mt-1 max-w-2xl">
        {trialDays
          ? `Every new account starts with a ${trialDays}-day Pro trial — no card required. `
          : "Every new account starts with a Pro trial — no card required. "}
        When it ends you move to Free Explorer; nothing is charged unless you subscribe.
      </p>

      <div className="grid gap-4 md:grid-cols-2 mt-6">
        <div className="card flex flex-col" data-plan="free">
          <div className="text-xs uppercase tracking-widest text-slate-400 font-semibold">Free Explorer</div>
          <div className="mt-2 flex items-baseline gap-1">
            <span className="text-4xl font-semibold font-mono">{formatCents(0, currency)}</span>
            <span className="text-sm text-slate-400">forever</span>
          </div>
          <p className="text-sm text-slate-300 mt-2">Explore the committee's stored work.</p>
          <ul className="mt-4 space-y-2 flex-1">
            <Bullet>{memo ? `${memo.free} of stored research memos` : "Stored research memos, on a monthly allowance"}</Bullet>
            <Bullet>{runs ? `${runs.free} research run` : "A research run each month"}</Bullet>
            <Bullet>{chat ? `${chat.free} Ask-the-PM turns` : "Ask-the-PM turns, on a monthly allowance"}</Bullet>
            <Bullet>{dcf ? `DCF and comps: ${dcf.free.toLowerCase()}` : "DCF and comps for companies whose memo you opened this month"}</Bullet>
            <Bullet>Sample research, methodology and the screener</Bullet>
          </ul>
          <div className="mt-5">{freeCta}</div>
        </div>

        <div className="card flex flex-col border-accent-600/40 bg-accent-600/[0.04]" data-plan="pro">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-widest text-accent-500 font-semibold">Pro</div>
            <span className="badge-bull">{trialDays ? `${trialDays}-day trial, no card` : "Trial, no card"}</span>
          </div>
          <div className="mt-2 flex items-baseline gap-1">
            <span className="text-4xl font-semibold font-mono">{monthly}</span>
            <span className="text-sm text-slate-400">per month</span>
          </div>
          <div className="text-sm text-slate-300 mt-1">
            or <span className="font-mono text-slate-100">{annual}</span> per year
            <span className="text-slate-400"> ({monthlyEquivalent(config.prices.annual_cents, currency)} a month equivalent)</span>
          </div>
          <p className="text-sm text-slate-300 mt-2">Underwrite: run research, DCF, comps and portfolios on your own tickers.</p>
          <ul className="mt-4 space-y-2 flex-1">
            <Bullet>{memo ? `${memo.pro} stored research memos` : "Stored research memos"}</Bullet>
            <Bullet>{runs ? `${runs.pro} research runs` : "Research runs"}</Bullet>
            <Bullet>{chat ? `${chat.pro} Ask-the-PM turns` : "Ask-the-PM turns"}</Bullet>
            <Bullet>DCF and comps on any ticker</Bullet>
            <Bullet>Portfolio builder, macro scenarios, track record, memo history</Bullet>
          </ul>
          <div className="mt-5">{proCta}</div>
        </div>
      </div>

      <ul className="mt-5 space-y-1.5 text-xs text-slate-400 max-w-3xl" aria-label="Billing notes">
        <li>All allowances are per UTC calendar month and reset on the first of the month at 00:00 UTC.</li>
        <li>
          Subscribe during the trial and the remaining trial days are kept when at least 48 hours remain (billing starts when the trial
          would have ended); with less than 48 hours left, billing starts at checkout — the checkout page says which applies.
        </li>
        <li>Cancel any time from the billing portal; Pro stays active until the end of the paid period.</li>
        {authEnabled && !config.billing_enabled ? (
          <li data-testid="billing-closed">Subscriptions are not open on this deployment yet; the trial and Free Explorer are available.</li>
        ) : null}
        {!known ? <li>Allowance numbers could not be loaded just now; the app's account page shows the current values.</li> : null}
      </ul>
    </section>
  );
}
