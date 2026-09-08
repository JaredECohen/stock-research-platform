import React from "react";
import { Link } from "react-router-dom";

/** Public 404 — reachable signed out, so it links to both surfaces. */
export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center px-6">
      <div className="card max-w-md text-center">
        <div className="text-xs uppercase tracking-widest text-slate-500">404</div>
        <h1 className="text-2xl font-semibold mt-1">That page does not exist</h1>
        <p className="text-sm text-slate-400 mt-2">
          The link may be old — the research workspace now lives under <span className="font-mono">/app</span>.
        </p>
        <div className="flex flex-wrap justify-center gap-2 mt-4">
          <Link to="/" className="btn-ghost text-sm">Home</Link>
          <Link to="/app" className="btn-primary text-sm">Open the app</Link>
          <Link to="/pricing" className="btn-ghost text-sm">Pricing</Link>
        </div>
      </div>
    </div>
  );
}
