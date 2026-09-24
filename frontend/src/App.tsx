import React from "react";
import { Navigate, Route, Routes, useLocation, useParams, type RouteObject } from "react-router-dom";
import Layout from "@/components/Layout";
import RequireAuth from "@/components/RequireAuth";
import RouteTracker from "@/components/RouteTracker";
import SignInPage from "@/auth/SignInPage";
import Dashboard from "@/pages/Dashboard";
import Chat from "@/pages/Chat";
import Research from "@/pages/Research";
import DCFLab from "@/pages/DCFLab";
import Comps from "@/pages/Comps";
import Fundamentals from "@/pages/Fundamentals";
import IndustryAnalysis from "@/pages/IndustryAnalysis";
import Screener from "@/pages/Screener";
import Scorecard from "@/pages/Scorecard";
import PortfolioBuilder from "@/pages/PortfolioBuilder";
import Macro from "@/pages/Macro";
import Settings from "@/pages/Settings";
import TrackRecord from "@/pages/TrackRecord";
import Account from "@/pages/Account";
import BillingSuccess from "@/pages/BillingSuccess";
import BillingCanceled from "@/pages/BillingCanceled";
import NotFound from "@/pages/NotFound";
import { publicRoutes } from "@/pages/public";

/**
 * Route tree (FEAT-002 §6.1):
 *   /                         marketing (publicRoutes, slice S6)
 *   /sign-in/*, /sign-up/*    auth pages
 *   /app/*                    RequireAuth > Layout > product pages
 *   /chat, /research, …       legacy paths → /app/… keeping ?query and #hash
 *   *                         NotFound
 * With the login wall off, RequireAuth is a pass-through and /app renders
 * exactly what / used to.
 */
export const LEGACY_APP_PATHS = [
  "/chat",
  "/research",
  "/dcf",
  "/comps",
  "/fundamentals",
  "/screener",
  "/scorecard",
  "/industries",
  "/portfolio",
  "/macro",
  "/track-record",
  "/settings",
] as const;

function LegacyRedirect({ to }: { to: string }) {
  const { search, hash } = useLocation();
  return <Navigate replace to={{ pathname: to, search, hash }} />;
}

/** `/industries/:code` → `/app/industries/:code`, keeping `?version=` and
 *  `?tab=` — the parts of the URL that make an edition citable. `:code` is
 *  the group's public slug; an old link's internal code still resolves,
 *  and IndustryAnalysis replaces it with the slug the API answers with. */
function LegacyIndustryRedirect() {
  const { code = "" } = useParams();
  return <LegacyRedirect to={`/app/industries/${encodeURIComponent(code)}`} />;
}

/** RouteObject[] → <Route> elements, recursing into children. */
function renderRouteObjects(routes: RouteObject[]): React.ReactNode {
  return routes.map((r, i) => (
    <Route key={r.path ?? `idx-${i}`} path={r.path} index={r.index as false | undefined} element={r.element}>
      {r.children ? renderRouteObjects(r.children) : null}
    </Route>
  ));
}

export default function App() {
  return (
    <>
      <RouteTracker />
      <Routes>
        {renderRouteObjects(publicRoutes)}

        <Route path="/app" element={<RequireAuth />}>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="chat" element={<Chat />} />
            <Route path="research" element={<Research />} />
            <Route path="dcf" element={<DCFLab />} />
            <Route path="comps" element={<Comps />} />
            <Route path="fundamentals" element={<Fundamentals />} />
            <Route path="screener" element={<Screener />} />
            <Route path="scorecard" element={<Scorecard />} />
            {/* FEAT-003: the index lists the groups, `:code` reads one
                edition. `?version=` and `?tab=` are search params, not
                path segments, so an edition and a section stay citable
                without multiplying routes. */}
            <Route path="industries" element={<IndustryAnalysis />} />
            <Route path="industries/:code" element={<IndustryAnalysis />} />
            <Route path="portfolio" element={<PortfolioBuilder />} />
            <Route path="macro" element={<Macro />} />
            <Route path="track-record" element={<TrackRecord />} />
            <Route path="settings" element={<Settings />} />
            <Route path="account" element={<Account />} />
            <Route path="billing/success" element={<BillingSuccess />} />
            <Route path="billing/canceled" element={<BillingCanceled />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Route>

        {LEGACY_APP_PATHS.map((p) => (
          <Route key={p} path={p} element={<LegacyRedirect to={`/app${p}`} />} />
        ))}
        {/* The only legacy path with a segment under it: a shared
            /industries/<group> link (a slug, or an old internal code) has
            to keep that segment, and the flat map above cannot carry one.
            The page then canonicalises an old code to the public slug. */}
        <Route path="/industries/:code" element={<LegacyIndustryRedirect />} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
