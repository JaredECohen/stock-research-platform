import React from "react";
import { Link } from "react-router-dom";
import Disclosure from "@/components/public/Disclosure";
import { COLUMN_META } from "@/components/public/ExpectationsLedger";
import PublicShell from "@/components/public/PublicShell";
import { BTN_GHOST, LINK, SAMPLES_PATH } from "@/components/public/ctas";

/**
 * The research process the product is shaped toward, drawn from
 * docs/research/README.md: the four questions, the two mandates, the
 * four-column expectations ledger, observed vs. interpretation, and
 * "blank means not obtained, not zero". Descriptive — it explains how a
 * memo is framed, not a promise about outcomes.
 */
const QUESTIONS = [
  { q: "What will happen?", a: "The operating outcome: revenue, margins, cash flow, capital needs — and over what horizon." },
  { q: "Who captures the economic benefit?", a: "Growth in an industry is not the same as profit for one company. Suppliers, customers, labour and competitors all take a share." },
  { q: "What is already priced in?", a: "The current price implies a set of assumptions. A view that matches them has no edge, however well argued." },
  { q: "What evidence will reveal the gap?", a: "Every view names the observations that would confirm or falsify it, so the memo can be checked against what actually happens." },
];

export default function Methodology() {
  return (
    <PublicShell
      title="Methodology"
      description="How MarketMosaic frames research: four questions, two mandates, a four-column expectations ledger, observed data kept apart from interpretation."
    >
      <article className="pt-12 max-w-3xl">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">Methodology</h1>
        <p className="text-slate-300 mt-3 leading-relaxed">
          The aim is to find mispriced future cash flows — not merely good companies or exciting themes. That framing decides what the
          committee looks for, how a memo is laid out, and what it refuses to pretend to know.
        </p>

        <section aria-labelledby="four-questions" className="mt-10">
          <h2 id="four-questions" className="text-2xl font-semibold tracking-tight">Four questions</h2>
          <p className="text-sm text-slate-400 mt-1">A memo is an attempt to answer these better than the market does.</p>
          <ol className="mt-4 space-y-3">
            {QUESTIONS.map((x, i) => (
              <li key={x.q} className="card-tight">
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-xs text-accent-500">0{i + 1}</span>
                  <h3 className="text-base font-semibold">{x.q}</h3>
                </div>
                <p className="text-sm text-slate-300 mt-1 leading-relaxed">{x.a}</p>
              </li>
            ))}
          </ol>
        </section>

        <section aria-labelledby="two-mandates" className="mt-10">
          <h2 id="two-mandates" className="text-2xl font-semibold tracking-tight">Two mandates, kept apart</h2>
          <div className="grid gap-3 sm:grid-cols-2 mt-4">
            <div className="card-tight">
              <h3 className="text-base font-semibold">Long-term compounders</h3>
              <p className="text-sm text-slate-300 mt-1 leading-relaxed">
                Smaller businesses whose economics can compound for years. Judged on durability of the engine, capital allocation and the
                price paid for it; exited when the engine breaks, not when the quarter disappoints.
              </p>
            </div>
            <div className="card-tight">
              <h3 className="text-base font-semibold">Mispriced inflections</h3>
              <p className="text-sm text-slate-300 mt-1 leading-relaxed">
                Shorter-term situations where expectations and evidence are diverging. Judged on the size of the gap and the timing of the
                evidence; exited when the gap closes or the evidence fails to arrive.
              </p>
            </div>
          </div>
          <p className="text-sm text-slate-400 mt-3">
            Each mandate has its own checklist and exit discipline. Mixing them — holding an inflection trade as if it were a compounder —
            is the classic way a thesis quietly changes underneath a position.
          </p>
        </section>

        <section aria-labelledby="ledger" className="mt-10">
          <h2 id="ledger" className="text-2xl font-semibold tracking-tight">The expectations ledger</h2>
          <p className="text-sm text-slate-300 mt-2 leading-relaxed">
            Every serious view keeps four expectation columns apart. Conflating any two of them is how research talks itself into a
            conclusion.
          </p>
          <dl className="grid gap-3 sm:grid-cols-2 mt-4">
            {(Object.keys(COLUMN_META) as Array<keyof typeof COLUMN_META>).map((k) => (
              <div key={k} className="card-tight">
                <dt className="text-base font-semibold">{COLUMN_META[k].title}</dt>
                <dd className="text-sm text-slate-300 mt-1">{COLUMN_META[k].hint}</dd>
              </div>
            ))}
          </dl>
          <p className="text-sm text-slate-400 mt-3">
            In a memo this is the <span className="font-mono">mispricing_thesis</span> block — consensus view, our view, the gap and the
            falsifiers — with the price-implied leg read from the DCF and management guidance read from the earnings call.
          </p>
        </section>

        <section aria-labelledby="observed" className="mt-10">
          <h2 id="observed" className="text-2xl font-semibold tracking-tight">Observed data versus interpretation</h2>
          <p className="text-sm text-slate-300 mt-2 leading-relaxed">
            A quoted price, a filed number, a sentence from a transcript: observed. What the committee makes of it: interpretation. The
            memo and the sample ledger style the two differently so a reader always knows which one they are looking at, and a
            specialist's confidence score is attached to the interpretation, never to the data.
          </p>
        </section>

        <section aria-labelledby="blank" className="mt-10">
          <h2 id="blank" className="text-2xl font-semibold tracking-tight">Blank means not obtained, not zero</h2>
          <p className="text-sm text-slate-300 mt-2 leading-relaxed">
            Missing evidence stays visibly unknown rather than being scored as neutral. A DCF that cannot compute an implied price prints
            "n/a", not $0.00; a ledger column the memo did not capture says "not captured" and why; a comps cell without data shows a
            dash. Historical datasets are never manufactured to fill a gap, and estimates are only compared on the same basis.
          </p>
        </section>

        <section aria-labelledby="sources" className="mt-10">
          <h2 id="sources" className="text-2xl font-semibold tracking-tight">Where the facts come from</h2>
          <p className="text-sm text-slate-300 mt-2 leading-relaxed">
            SEC filings and investor-relations material are the primary facts. Fundamentals, estimates and prices come from market-data
            providers through a fallback chain, and every memo records the price and the data it was written from. Industry context comes
            from a maintained knowledge base covering each GICS industry's economic engine, indicators, moats and failure modes.
          </p>
        </section>

        <div className="flex flex-wrap gap-2 mt-10">
          <Link to={SAMPLES_PATH} className={BTN_GHOST}>See the ledger on a real sample</Link>
          <Link to="/faq" className={`${BTN_GHOST}`}>Read the FAQ</Link>
        </div>
        <p className="text-xs text-slate-400 mt-4">
          The full process library lives with the research team; this page is the part that shapes the product. Questions about how a
          specific memo was framed belong in <Link to="/faq#methodology" className={LINK}>the FAQ</Link>.
        </p>
        <div className="mt-10">
          <Disclosure />
        </div>
      </article>
    </PublicShell>
  );
}
