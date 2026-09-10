import React from "react";
import { Link } from "react-router-dom";
import { useConfig } from "@/auth/ConfigProvider";
import { APP_PATH, BTN_GHOST, BTN_PRIMARY, PRICING_PATH, SAMPLES_PATH, SIGN_IN_PATH, SIGN_UP_PATH } from "./ctas";
import { useVisitor } from "./hooks";

/**
 * The value proposition, first. The trial length in the primary CTA comes
 * from `/api/public/config`; when the config fetch fell back the button
 * says "Start your Pro trial" rather than naming a number the backend may
 * not grant.
 */
export function trialCta(trialDays: number | null): string {
  return trialDays ? `Start ${trialDays}-day Pro trial` : "Start your Pro trial";
}

export default function Hero() {
  const { config } = useConfig();
  const { signedIn, loading } = useVisitor();

  return (
    <section aria-labelledby="hero-heading" className="pt-14 sm:pt-20 pb-6">
      <div className="max-w-3xl">
        <div className="text-xs uppercase tracking-widest text-accent-500 font-semibold">AI investment committee</div>
        <h1 id="hero-heading" className="text-3xl sm:text-5xl font-semibold tracking-tight mt-2 leading-tight">
          Research that shows its work.
        </h1>
        <p className="text-base sm:text-lg text-slate-300 mt-4 leading-relaxed">
          MarketMosaic puts a committee of specialist agents on a company — fundamentals, filings, earnings calls, valuation,
          comparable companies, risks, catalysts and scenarios — and has a portfolio-manager agent reconcile them into one
          explainable memo: what the market expects, what we expect, the gap, and the evidence that would prove us wrong.
        </p>
        <div className="flex flex-wrap gap-2 mt-6">
          {loading ? null : signedIn ? (
            <Link to={APP_PATH} className={BTN_PRIMARY}>Continue to app</Link>
          ) : (
            <>
              <Link to={SIGN_UP_PATH} className={BTN_PRIMARY}>{trialCta(config.trial_days)}</Link>
              <Link to={SIGN_IN_PATH} className={BTN_GHOST}>Sign in</Link>
            </>
          )}
          <Link to={SAMPLES_PATH} className={BTN_GHOST}>View sample research</Link>
          <Link to={PRICING_PATH} className={BTN_GHOST}>See pricing</Link>
        </div>
        <p className="text-xs text-slate-400 mt-4">
          {signedIn && !config.auth_enabled
            ? "Research and education only. Not personalised financial advice."
            : "No card required for the trial. Research and education only — not personalised financial advice."}
        </p>
      </div>
    </section>
  );
}
