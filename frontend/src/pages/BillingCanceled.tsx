import React from "react";
import { Link } from "react-router-dom";

/** /app/billing/canceled — Stripe Checkout's cancel return. */
export default function BillingCanceled() {
  return (
    <div className="max-w-xl space-y-4">
      <h1 className="text-2xl font-semibold">Checkout cancelled</h1>
      <div className="card">
        <p className="text-sm text-slate-300 leading-relaxed">
          Nothing was charged and your plan is unchanged. You can come back to this whenever you like.
        </p>
        <div className="flex flex-wrap gap-2 mt-3">
          <Link to="/pricing" className="btn-primary text-sm">See plans</Link>
          <Link to="/app/account" className="btn-ghost text-sm">Your account</Link>
          <Link to="/app" className="btn-ghost text-sm">Dashboard</Link>
        </div>
      </div>
    </div>
  );
}
