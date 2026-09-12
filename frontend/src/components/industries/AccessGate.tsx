import React from "react";
import { Link } from "react-router-dom";
import { Lock } from "lucide-react";
import type { IndustryGate } from "./format";

/**
 * What a viewer sees instead of a surface they cannot read yet.
 *
 * It is a *state*, not an error: the deployment's own policy (the
 * `access` block every Industry Analysis response carries) says this
 * surface is Pro and the login wall is on, so the page says so and
 * offers the remedy. Nothing is fetched behind it — an anonymous read of
 * a Pro route is a guaranteed 401, and the client's shared 401 handler
 * would sign the session out and bounce the viewer to sign-in from a page
 * they were only browsing.
 *
 * The gate is what the UI believes; the backend authorises every call
 * regardless. This component never claims the viewer *was* refused —
 * only that the surface costs a tier they are not known to hold.
 */
export default function AccessGate({
  gate,
  what,
  latestAvailable = false,
  className = "",
}: {
  gate: NonNullable<IndustryGate>;
  /** The thing being gated, in the reader's words ("edition history"). */
  what: string;
  /**
   * Whether the published edition is on the page BELOW this card.
   *
   * It is false by default, and the reassurance is printed only when it
   * is true and this card is not the one standing where that edition
   * would be. The sentence was once unconditional, which made the card
   * assert the opposite of the policy it had just read: a deployment
   * with `INDUSTRY_ANALYSIS_ACCESS=pro` renders this card INSTEAD of the
   * report and then told the reader the edition below was public. There
   * was no edition below.
   */
  latestAvailable?: boolean;
  className?: string;
}) {
  const tier = gate.tier === "pro" ? "Pro" : gate.tier;
  const alsoBelow = latestAvailable && gate.surface !== "latest";
  return (
    <div
      className={`card-tight text-sm text-slate-300 ${className}`}
      role="status"
      data-testid={`industry-gate-${gate.surface}`}
      data-gate-reason={gate.reason}
    >
      <div className="flex items-center gap-2 mb-1">
        <Lock size={14} aria-hidden className="text-slate-400" />
        <span className="section-title">{what} is part of {tier}</span>
      </div>
      {gate.reason === "sign_in" ? (
        <p className="text-slate-400">
          This deployment serves {what} to signed-in {tier} accounts.{" "}
          <Link to="/sign-in" className="underline text-accent-500">
            Sign in
          </Link>{" "}
          to read it.{alsoBelow ? " The published edition itself is still shown below." : ""}
        </p>
      ) : (
        <p className="text-slate-400">
          Your plan does not include {what}.{" "}
          <Link to="/app/account?upgrade=industry_analysis" className="underline text-accent-500">
            See plans
          </Link>
          .{alsoBelow ? " The published edition itself is still shown below." : ""}
        </p>
      )}
    </div>
  );
}
