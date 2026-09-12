import React, { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { Activity, Menu, X } from "lucide-react";
import { useVisitor } from "./hooks";
import {
  APP_PATH,
  BTN_GHOST,
  BTN_PRIMARY,
  FAQ_PATH,
  FOCUS_RING,
  METHODOLOGY_PATH,
  PRICING_PATH,
  SAMPLES_PATH,
  SIGN_IN_PATH,
  SIGN_UP_PATH,
} from "./ctas";

const NAV = [
  { to: SAMPLES_PATH, label: "Sample research" },
  { to: METHODOLOGY_PATH, label: "Methodology" },
  { to: PRICING_PATH, label: "Pricing" },
  { to: FAQ_PATH, label: "FAQ" },
];

/**
 * Marketing header. Signed-in visitors (or any visitor while the login
 * wall is off) get one "Continue to app" CTA; everyone else gets Sign in
 * and Sign up. The mobile menu is a plain disclosure button — no
 * transitions to respect, keyboard reachable, `aria-expanded` announced.
 */
export default function PublicNav() {
  const { signedIn, loading } = useVisitor();
  const [open, setOpen] = useState(false);

  const linkCls = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-lg text-sm motion-safe:transition-colors ${FOCUS_RING} ${
      isActive ? "text-accent-500 bg-accent-600/10" : "text-slate-300 hover:text-slate-100 hover:bg-ink-800"
    }`;

  const ctas = loading ? null : signedIn ? (
    <Link to={APP_PATH} className={`${BTN_PRIMARY} text-sm`}>Continue to app</Link>
  ) : (
    <>
      <Link to={SIGN_IN_PATH} className={`${BTN_GHOST} text-sm`}>Sign in</Link>
      <Link to={SIGN_UP_PATH} className={`${BTN_PRIMARY} text-sm`}>Sign up free</Link>
    </>
  );

  return (
    <header className="border-b border-ink-800 bg-ink-950/80 backdrop-blur sticky top-0 z-40">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-3">
        <Link to="/" className={`flex items-center gap-2 rounded-lg ${FOCUS_RING}`} aria-label="MarketMosaic home">
          <div className="h-8 w-8 rounded-lg bg-accent-600/20 border border-accent-600/40 flex items-center justify-center text-accent-500">
            <Activity size={18} aria-hidden />
          </div>
          <div className="leading-tight">
            <div className="text-base font-semibold tracking-tight">MarketMosaic</div>
            <div className="text-[11px] uppercase tracking-widest text-slate-400">AI Investment Committee</div>
          </div>
        </Link>

        <nav aria-label="Primary" className="hidden md:flex items-center gap-1 ml-6">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} className={linkCls}>
              {n.label}
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto hidden md:flex items-center gap-2">{ctas}</div>

        <button
          type="button"
          className={`ml-auto md:hidden btn-ghost !px-2.5 ${FOCUS_RING}`}
          aria-expanded={open}
          aria-controls="public-mobile-menu"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? <X size={18} aria-hidden /> : <Menu size={18} aria-hidden />}
          <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
        </button>
      </div>

      <div id="public-mobile-menu" hidden={!open} className="md:hidden border-t border-ink-800 bg-ink-950">
        <nav aria-label="Primary, mobile" className="max-w-6xl mx-auto px-4 py-3 flex flex-col gap-1">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} className={linkCls} onClick={() => setOpen(false)}>
              {n.label}
            </NavLink>
          ))}
          <div className="flex flex-wrap gap-2 pt-2">{ctas}</div>
        </nav>
      </div>
    </header>
  );
}
