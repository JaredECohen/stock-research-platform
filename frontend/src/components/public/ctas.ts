// FEAT-002 (S6) — the handful of destinations every marketing CTA points
// at, in one place so a renamed route is one edit. `returnTo` is the app
// root: `auth/returnTo.safeReturnTo` only honours /app paths, so this is
// the one value that survives sanitising on the sign-in page.

export const APP_PATH = "/app";
export const ACCOUNT_PATH = "/app/account";
export const SIGN_UP_PATH = "/sign-up?returnTo=%2Fapp";
export const SIGN_IN_PATH = "/sign-in?returnTo=%2Fapp";
export const PRICING_PATH = "/pricing";
export const SAMPLES_PATH = "/samples";
export const METHODOLOGY_PATH = "/methodology";
export const FAQ_PATH = "/faq";

export const SITE_ORIGIN = "https://marketmosaic.ai";

/** Visible keyboard focus on the dark palette — every interactive element
 *  in the public components carries it. */
export const FOCUS_RING =
  "focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-500";

export const BTN_PRIMARY = `btn-primary motion-safe:transition-colors ${FOCUS_RING}`;
export const BTN_GHOST = `btn-ghost motion-safe:transition-colors ${FOCUS_RING}`;
export const LINK = `text-accent-500 hover:text-accent-600 underline underline-offset-2 rounded-sm ${FOCUS_RING}`;

/** Research-and-education-only wording, shared by the footer, the
 *  disclosure block and the sign-in pages' sibling copy. */
export const RESEARCH_ONLY =
  "MarketMosaic is for investment research and education only. Every memo, model and answer is a model output, not a recommendation, and not personalised financial, investment, legal or tax advice.";
