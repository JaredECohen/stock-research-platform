import React from "react";
import { NavLink, Outlet } from "react-router-dom";
import { Activity, BarChart3, Briefcase, GanttChart, MessageCircle, Newspaper, Search, Settings, Sparkles, TrendingUp } from "lucide-react";

const links = [
  { to: "/", label: "Dashboard", icon: Sparkles },
  { to: "/chat", label: "Ask the PM", icon: MessageCircle },
  { to: "/research", label: "Stock Research", icon: Newspaper },
  { to: "/dcf", label: "DCF Lab", icon: TrendingUp },
  { to: "/comps", label: "Comps", icon: BarChart3 },
  { to: "/screener", label: "Screener", icon: Search },
  { to: "/portfolio", label: "Portfolio Builder", icon: Briefcase },
  { to: "/macro", label: "Macro", icon: GanttChart },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      <aside className="hidden md:flex w-60 flex-col border-r border-ink-800 bg-ink-900/60 backdrop-blur">
        <div className="px-5 py-6 border-b border-ink-800">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-accent-600/20 border border-accent-600/40 flex items-center justify-center text-accent-500">
              <Activity size={18} />
            </div>
            <div>
              <div className="text-base font-semibold tracking-tight">MarketMosaic</div>
              <div className="text-[11px] uppercase tracking-widest text-slate-500">AI Investment Committee</div>
            </div>
          </div>
        </div>
        <nav className="px-3 py-4 space-y-1">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-accent-600/15 text-accent-500 border border-accent-600/30"
                    : "text-slate-300 hover:bg-ink-800 hover:text-slate-100 border border-transparent"
                }`
              }
              end={to === "/"}
            >
              <Icon size={16} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto px-5 py-4 text-[11px] leading-snug text-slate-500 border-t border-ink-800">
          Research & education only.<br />Not personalized financial advice.
        </div>
      </aside>
      <main className="flex-1 px-6 lg:px-10 py-6 max-w-[1400px] mx-auto w-full">
        <Outlet />
      </main>
    </div>
  );
}
