import React, { useEffect, useRef, useState } from "react";
import { Link, NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  Activity,
  Award,
  BarChart3,
  Briefcase,
  ChevronDown,
  GanttChart,
  Gauge,
  LineChart,
  Lock,
  MessageCircle,
  Newspaper,
  Search,
  Settings,
  Sparkles,
  TrendingUp,
  UserCircle,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import ProviderHealthBanner from "@/components/ProviderHealthBanner";
import TrialBanner from "@/components/TrialBanner";
import { useAuth } from "@/auth/AuthContext";
import { useConfig } from "@/auth/ConfigProvider";
import { useAccount } from "@/auth/useAccount";
import { planLabel } from "@/lib/entitlements";

/** `feature` names the entitlement that gates the item; a plan without it
 *  sees a lock and lands on the account page's upgrade prompt instead of a
 *  402 from the page. UX only — the backend still refuses the API call. */
const links: Array<{ to: string; label: string; icon: LucideIcon; end?: boolean; feature?: string }> = [
  { to: "/app", label: "Dashboard", icon: Sparkles, end: true },
  { to: "/app/chat", label: "Ask the PM", icon: MessageCircle },
  { to: "/app/research", label: "Stock Research", icon: Newspaper },
  { to: "/app/dcf", label: "DCF Lab", icon: TrendingUp },
  { to: "/app/comps", label: "Comps", icon: BarChart3 },
  // Available on every plan (the plan shapes the chart, not the page), so
  // no `feature` lock here.
  { to: "/app/fundamentals", label: "Fundamentals", icon: LineChart },
  { to: "/app/screener", label: "Screener", icon: Search },
  // Phase 6: reads are Pro under the wall (`auth/features.py` `scorecard`);
  // the lock only shows once /api/me lists the entitlement.
  { to: "/app/scorecard", label: "Scorecard", icon: Gauge, feature: "scorecard" },
  { to: "/app/portfolio", label: "Portfolio Builder", icon: Briefcase, feature: "portfolio" },
  { to: "/app/macro", label: "Macro", icon: GanttChart, feature: "macro" },
  { to: "/app/track-record", label: "Track Record", icon: Award, feature: "track_record" },
  { to: "/app/settings", label: "Settings", icon: Settings },
];

export default function Layout() {
  const { config } = useConfig();
  const { account } = useAccount();
  const authOn = config.auth_enabled;

  const isLocked = (feature?: string) =>
    !!feature && authOn && !!account && account.entitlements[feature] !== undefined && !account.entitlements[feature].allowed;

  return (
    <div className="flex min-h-screen">
      <aside className="hidden md:flex w-60 flex-col border-r border-ink-800 bg-ink-900/60 backdrop-blur">
        <div className="px-5 py-6 border-b border-ink-800">
          <Link to="/app" className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-accent-600/20 border border-accent-600/40 flex items-center justify-center text-accent-500">
              <Activity size={18} />
            </div>
            <div>
              <div className="text-base font-semibold tracking-tight">MarketMosaic</div>
              <div className="text-[11px] uppercase tracking-widest text-slate-500">AI Investment Committee</div>
            </div>
          </Link>
        </div>
        <nav className="px-3 py-4 space-y-1">
          {links.map(({ to, label, icon: Icon, end, feature }) => {
            const locked = isLocked(feature);
            const target = locked ? `/app/account?upgrade=${encodeURIComponent(feature!)}` : to;
            return (
              <NavLink
                key={to}
                to={target}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                    isActive && !locked
                      ? "bg-accent-600/15 text-accent-500 border border-accent-600/30"
                      : locked
                        ? "text-slate-500 hover:bg-ink-800 hover:text-slate-300 border border-transparent"
                        : "text-slate-300 hover:bg-ink-800 hover:text-slate-100 border border-transparent"
                  }`
                }
                end={end}
                title={locked ? `${label} is part of Pro` : undefined}
                aria-label={locked ? `${label} (Pro feature)` : undefined}
              >
                <Icon size={16} />
                <span className="flex-1">{label}</span>
                {locked && <Lock size={12} aria-hidden data-testid={`lock-${feature}`} />}
              </NavLink>
            );
          })}
        </nav>
        {authOn && <AccountMenu />}
        <div className="mt-auto px-5 py-4 text-[11px] leading-snug text-slate-500 border-t border-ink-800">
          Research & education only.<br />Not personalized financial advice.
        </div>
      </aside>
      <main className="flex-1 px-6 lg:px-10 py-6 max-w-[1400px] mx-auto w-full">
        {/* Lives in the shell, not a page, so a degraded provider is visible
            wherever the user lands; renders null when healthy. */}
        <ProviderHealthBanner />
        {/* Trial end date / billing warning from /api/me; null with auth off. */}
        <TrialBanner />
        <Outlet />
      </main>
    </div>
  );
}

/** Plan badge + email + Account / Sign out. Sidebar only, auth on. */
function AccountMenu() {
  const auth = useAuth();
  const navigate = useNavigate();
  const { account } = useAccount();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const plan = account?.plan;
  const badge = plan ? planLabel(plan.plan, plan.source) : null;
  const badgeCls = plan?.plan === "pro" ? "border-accent-600/40 text-accent-500" : "border-ink-700 text-slate-300";

  return (
    <div className="px-3 pb-3 relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Account menu"
        className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-300 hover:bg-ink-800 border border-ink-800"
      >
        <UserCircle size={16} />
        <span className="flex-1 truncate text-left text-xs">{auth.user?.email || "Account"}</span>
        {badge && (
          <span className={`badge text-[10px] ${badgeCls}`} data-testid="plan-badge">
            {badge}
          </span>
        )}
        <ChevronDown size={12} />
      </button>
      {open && (
        <div role="menu" className="absolute left-3 right-3 mt-1 rounded-lg border border-ink-700 bg-ink-900 shadow-lg z-20 py-1 text-sm">
          <Link role="menuitem" to="/app/account" className="block px-3 py-2 hover:bg-ink-800" onClick={() => setOpen(false)}>
            Account & usage
          </Link>
          {plan && plan.source !== "subscription" && (
            <Link role="menuitem" to="/pricing" className="block px-3 py-2 hover:bg-ink-800" onClick={() => setOpen(false)}>
              Plans & pricing
            </Link>
          )}
          <button
            role="menuitem"
            type="button"
            className="w-full text-left px-3 py-2 hover:bg-ink-800 text-slate-300"
            onClick={() => {
              setOpen(false);
              void auth.signOut().then(() => navigate("/", { replace: true }));
            }}
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
