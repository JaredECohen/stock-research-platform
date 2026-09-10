import React from "react";
import { Link } from "react-router-dom";
import { FAQ_PATH, FOCUS_RING, METHODOLOGY_PATH, PRICING_PATH, RESEARCH_ONLY, SAMPLES_PATH } from "./ctas";

const PRODUCT = [
  { to: SAMPLES_PATH, label: "Sample research" },
  { to: METHODOLOGY_PATH, label: "Methodology" },
  { to: PRICING_PATH, label: "Pricing" },
  { to: FAQ_PATH, label: "FAQ" },
];

const LEGAL = [
  { to: "/terms", label: "Terms of service" },
  { to: "/billing-terms", label: "Billing terms" },
  { to: "/privacy", label: "Privacy" },
  { to: "/cookies", label: "Cookies and storage" },
];

export default function PublicFooter() {
  const link = `text-sm text-slate-300 hover:text-slate-100 rounded-sm ${FOCUS_RING}`;
  return (
    <footer className="border-t border-ink-800 mt-16">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10 grid gap-8 sm:grid-cols-3">
        <div>
          <div className="text-base font-semibold tracking-tight">MarketMosaic</div>
          <p className="text-xs text-slate-400 mt-2 leading-relaxed">{RESEARCH_ONLY}</p>
        </div>
        <nav aria-label="Product">
          <div className="section-title mb-2">Product</div>
          <ul className="space-y-1.5">
            {PRODUCT.map((l) => (
              <li key={l.to}>
                <Link to={l.to} className={link}>{l.label}</Link>
              </li>
            ))}
          </ul>
        </nav>
        <nav aria-label="Legal">
          <div className="section-title mb-2">Legal</div>
          <ul className="space-y-1.5">
            {LEGAL.map((l) => (
              <li key={l.to}>
                <Link to={l.to} className={link}>{l.label}</Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </footer>
  );
}
