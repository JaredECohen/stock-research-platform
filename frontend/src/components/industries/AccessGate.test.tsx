import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import AccessGate from "@/components/industries/AccessGate";
import { gateFor } from "@/components/industries/format";
import * as fx from "@/test/fixtures/industry";

function mount(gate: NonNullable<ReturnType<typeof gateFor>>, what = "edition history") {
  return render(
    <MemoryRouter>
      <AccessGate gate={gate} what={what} />
    </MemoryRouter>,
  );
}

describe("gateFor", () => {
  const access = fx.taxonomy.access;

  it("gates nothing while the wall is off, whatever the tiers say", () => {
    expect(access.enforced).toBe(false);
    expect(access.surfaces.history).toBe("pro");
    expect(gateFor("history", access, { signedIn: false })).toBeNull();
  });

  it("never gates a public surface", () => {
    const enforced = { ...access, enforced: true };
    expect(enforced.surfaces.latest).toBe("public");
    expect(gateFor("latest", enforced, { signedIn: false })).toBeNull();
  });

  it("asks an anonymous reader to sign in for a Pro surface", () => {
    const gate = gateFor("history", { ...access, enforced: true }, { signedIn: false });
    expect(gate).toEqual({ reason: "sign_in", tier: "pro", surface: "history" });
  });

  it("does NOT claim a refusal for a signed-in reader whose entitlement is unknown", () => {
    // The account may still be loading, or the feature may not appear in
    // the snapshot. Guessing "denied" would hide a surface the reader has
    // paid for; the server authorises the call either way.
    expect(gateFor("history", { ...access, enforced: true }, { signedIn: true })).toBeNull();
    expect(gateFor("history", { ...access, enforced: true }, { signedIn: true, entitlementAllowed: null })).toBeNull();
  });

  it("gates on the plan only when the entitlement is explicitly refused", () => {
    const gate = gateFor("history", { ...access, enforced: true }, { signedIn: true, entitlementAllowed: false });
    expect(gate).toEqual({ reason: "plan", tier: "pro", surface: "history" });
  });

  it("fails closed on a surface the policy has never heard of", () => {
    const gate = gateFor("some_future_surface", { ...access, enforced: true }, { signedIn: false });
    expect(gate?.tier).toBe("pro");
  });

  it("gates nothing at all when there is no access block yet", () => {
    expect(gateFor("history", null, { signedIn: false })).toBeNull();
  });
});

describe("AccessGate", () => {
  it("offers sign-in, and says the public edition is still there", () => {
    mount({ reason: "sign_in", tier: "pro", surface: "history" });
    const card = screen.getByTestId("industry-gate-history");
    expect(card).toHaveAttribute("data-gate-reason", "sign_in");
    expect(card).toHaveTextContent("edition history is part of Pro");
    expect(screen.getByRole("link", { name: "Sign in" })).toHaveAttribute("href", "/sign-in");
  });

  it("offers the plan page when the refusal is the plan, not the session", () => {
    mount({ reason: "plan", tier: "pro", surface: "latest" }, "industry reports");
    const card = screen.getByTestId("industry-gate-latest");
    expect(card).toHaveAttribute("data-gate-reason", "plan");
    expect(screen.getByRole("link", { name: "See plans" })).toHaveAttribute(
      "href",
      "/app/account?upgrade=industry_analysis",
    );
  });

  it("is a status, not an alert — nothing failed", () => {
    mount({ reason: "sign_in", tier: "pro", surface: "changes" });
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
