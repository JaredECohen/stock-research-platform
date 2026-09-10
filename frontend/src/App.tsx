import React from "react";
import { Navigate, Route, Routes, useLocation, type RouteObject } from "react-router-dom";
import Layout from "@/components/Layout";
import RequireAuth from "@/components/RequireAuth";
import RouteTracker from "@/components/RouteTracker";
import SignInPage from "@/auth/SignInPage";
import Dashboard from "@/pages/Dashboard";
import Chat from "@/pages/Chat";
import Research from "@/pages/Research";
import DCFLab from "@/pages/DCFLab";
import Comps from "@/pages/Comps";
import Screener from "@/pages/Screener";
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
  "/screener",
  "/portfolio",
  "/macro",
  "/track-record",
  "/settings",
] as const;

function LegacyRedirect({ to }: { to: string }) {
  const { search, hash } = useLocation();
  return <Navigate replace to={{ pathname: to, search, hash }} />;
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
            <Route path="screener" element={<Screener />} />
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

        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
