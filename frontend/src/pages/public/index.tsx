import React from "react";
import type { RouteObject } from "react-router-dom";
import SignInPage from "@/auth/SignInPage";
import FAQPage from "./FAQPage";
import Landing from "./Landing";
import Legal from "./Legal";
import Methodology from "./Methodology";
import Pricing from "./Pricing";
import SampleDetail from "./SampleDetail";
import Samples from "./Samples";

/**
 * Marketing site (FEAT-002 slice S6).
 *
 * Contract with App.tsx: export `publicRoutes: RouteObject[]`, mounted
 * outside the /app shell (no RequireAuth, no Layout). Every page here is
 * reachable signed out and reads only token-free endpoints
 * (`api/publicClient.ts`); the backend authorises everything else.
 *
 * /sign-in and /sign-up render Clerk's components when the login wall is
 * on and the stub page otherwise — `auth/SignInPage` already does both,
 * so it is mounted here as part of the public surface.
 */
export { Landing };

export const publicRoutes: RouteObject[] = [
  { path: "/", element: <Landing /> },
  { path: "/pricing", element: <Pricing /> },
  { path: "/methodology", element: <Methodology /> },
  { path: "/faq", element: <FAQPage /> },
  { path: "/privacy", element: <Legal doc="privacy" /> },
  { path: "/terms", element: <Legal doc="terms" /> },
  { path: "/billing-terms", element: <Legal doc="billing-terms" /> },
  { path: "/cookies", element: <Legal doc="cookies" /> },
  { path: "/samples", element: <Samples /> },
  { path: "/samples/:ticker", element: <SampleDetail /> },
  { path: "/sign-in/*", element: <SignInPage mode="sign-in" /> },
  { path: "/sign-up/*", element: <SignInPage mode="sign-up" /> },
];
