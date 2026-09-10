// FEAT-002 (S6) — FAQ copy for /faq and the landing page. Plain data so
// the copy test (`content/copy.test.ts`) can scan it for claims we never
// make: no invented social proof, counts, performance or outcome promises.
//
// Allowance numbers are NOT written here. Answers that depend on a
// number say where it lives (the pricing page, which reads the matrix
// the backend enforces) rather than repeating a literal that
// ENTITLEMENT_OVERRIDES_JSON could change without a deploy.

export interface FAQEntry {
  id: string;
  question: string;
  answer: string;
  /** Shown on the landing page too. */
  featured?: boolean;
}

export const FAQ: FAQEntry[] = [
  {
    id: "what-is-it",
    question: "What is MarketMosaic?",
    answer:
      "A research tool that puts an AI investment committee on a company. Specialist agents cover fundamentals, filings, earnings calls, valuation, comparable companies, macro sensitivity and risk; a portfolio-manager agent reconciles them into one memo with a rating, a mispricing thesis and the evidence that would prove it wrong. It is research and education software, not an advisor.",
    featured: true,
  },
  {
    id: "advice",
    question: "Is this investment advice?",
    answer:
      "No. Every memo, DCF, comps table and chat answer is a model output for research and education. Nothing on this site or in the app is personalised financial, investment, legal or tax advice, and MarketMosaic makes no claim about how any security or strategy will perform. Decisions are yours and you should consult a licensed professional where appropriate.",
    featured: true,
  },
  {
    id: "trial",
    question: "How does the Pro trial work?",
    answer:
      "Every new account starts a Pro trial on sign-up — no card is asked for. When it ends you drop to Free Explorer automatically; nothing is charged unless you subscribe. If you subscribe while the trial is still running, the remaining trial days are kept when at least 48 hours remain (billing starts when the trial would have ended); with less than that left, billing starts at checkout and the checkout page says so. The trial length shown on the pricing page is the one the backend grants.",
    featured: true,
  },
  {
    id: "free",
    question: "What does Free Explorer include?",
    answer:
      "Free Explorer lets you explore the committee's stored work: a monthly allowance of stored memos (counted as distinct companies, so re-opening the same memo does not use another), one research run, a monthly allowance of Ask-the-PM turns, and DCF and comps for any company whose memo you opened this month. The exact numbers are on the pricing page, read from the same table the backend enforces.",
    featured: true,
  },
  {
    id: "periods",
    question: "When do allowances reset?",
    answer:
      "All allowances are per UTC calendar month: they reset at 00:00 UTC on the first of each month, on both Free Explorer and Pro. Your account page shows what you have used so far this month and when the period ends.",
  },
  {
    id: "distinct",
    question: "What does “distinct companies” mean for memo views?",
    answer:
      "The Free memo allowance counts companies, not page loads. Opening NVDA's memo three times in a month uses one of the allowance; opening NVDA, COST and JPM uses three. Pro has no monthly cap on stored memos.",
  },
  {
    id: "research-run",
    question: "What is a research run?",
    answer:
      "A research run puts the full agent committee on a ticker and writes a fresh memo. It reads filings, the latest earnings call, fundamentals and prices, runs a DCF and comps, and records the committee's disagreements. Runs are queued and processed by a worker, so a memo can take a few minutes; the app shows progress and the stored memo appears when it is done.",
  },
  {
    id: "data",
    question: "Where does the data come from, and how fresh is it?",
    answer:
      "Filings come from SEC EDGAR; fundamentals, estimates and prices come from market-data providers with a fallback chain. Memos are snapshots: each one records the price and the data it was written from, and the app tells you when a memo is stale. The public samples on this site are rebuilt weekly from stored research and show their build date.",
  },
  {
    id: "samples",
    question: "Are the sample pages live?",
    answer:
      "No. The samples are read-only copies of stored research for three companies, rebuilt on a schedule. Nothing on the marketing site runs a model or calls a data provider; that only happens inside the app, against your allowance.",
  },
  {
    id: "cancel",
    question: "Can I cancel?",
    answer:
      "Yes. Pro is a monthly or yearly subscription managed through the billing portal on your account page. Cancelling stops renewal; you keep Pro until the end of the period you already paid for, then return to Free Explorer. The billing terms page has the details.",
  },
  {
    id: "privacy",
    question: "What do you store about me?",
    answer:
      "An account record keyed to your sign-in provider's user id, a hash of your email (never the address itself), your plan and usage counters, and product events tied to a random anonymous id. There are no third-party analytics scripts on the marketing site; the privacy and cookie pages describe exactly what is kept and for how long.",
  },
  {
    id: "methodology",
    question: "How is the research framed?",
    answer:
      "Around four questions: what will happen, who captures the economic benefit, what is already priced in, and what evidence will reveal the gap. Every memo keeps four expectation columns apart — reported consensus, management guidance, price-implied assumptions and the committee's own forecast — and separates observed data from interpretation. A blank means the evidence was not obtained, not that it is zero. The methodology page goes deeper.",
  },
];

export function featuredFAQ(): FAQEntry[] {
  return FAQ.filter((f) => f.featured);
}
