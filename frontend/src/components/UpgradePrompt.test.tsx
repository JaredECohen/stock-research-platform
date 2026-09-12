import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import UpgradePrompt, { refusalForFeature } from "@/components/UpgradePrompt";
import { featureMatrix, renderWithProviders } from "@/test/providers";
import type { EntitlementRefusal } from "@/types";

const QUOTA: EntitlementRefusal = {
  code: "quota_exceeded",
  feature: "research_run",
  plan: "free",
  used: 1,
  limit: 1,
  resets_at: "2026-10-01T00:00:00",
  upgrade_url: "/pricing",
  message: "Free Explorer includes 1 full research run per month.",
};

describe("UpgradePrompt", () => {
  it("names the feature, the used/limit numbers and the reset date for a quota refusal", () => {
    renderWithProviders(<UpgradePrompt refusal={QUOTA} />);
    const box = screen.getByTestId("upgrade-prompt");
    expect(box).toHaveTextContent("You've used 1 of 1 research runs this month");
    expect(box).toHaveTextContent("A research run puts the full agent committee");
    expect(box).toHaveTextContent("Your Free allowance resets on Oct 1, 2026 (UTC calendar month).");
    expect(screen.getByRole("link", { name: "See Pro plans" })).toHaveAttribute("href", "/pricing");
    expect(screen.getByRole("link", { name: "View your usage" })).toHaveAttribute("href", "/app/account");
    // Never a generic error.
    expect(box).not.toHaveTextContent(/error/i);
  });

  it("explains a plan_required refusal in terms of Pro", () => {
    renderWithProviders(<UpgradePrompt refusal={refusalForFeature("portfolio", "free")} />);
    const box = screen.getByTestId("upgrade-prompt");
    expect(box).toHaveTextContent("Portfolio build is part of Pro");
    expect(box).toHaveTextContent("Portfolio Builder turns a market view into a diversified scenario portfolio");
    expect(screen.getByRole("link", { name: "See Pro plans" })).toBeInTheDocument();
  });

  it("does not sell Pro to someone already on Pro who hit the Pro cap", () => {
    renderWithProviders(<UpgradePrompt refusal={{ ...QUOTA, plan: "pro", used: 20, limit: 20 }} />);
    const box = screen.getByTestId("upgrade-prompt");
    expect(box).toHaveTextContent("You've used 20 of 20 research runs this month");
    expect(box).toHaveTextContent("The Pro allowance for research runs resets on Oct 1, 2026");
    expect(screen.queryByRole("link", { name: "See Pro plans" })).not.toBeInTheDocument();
  });

  it("uses the backend upgrade_url and calls onDismiss", () => {
    const onDismiss = vi.fn();
    renderWithProviders(<UpgradePrompt refusal={{ ...QUOTA, upgrade_url: "/pricing?from=research" }} onDismiss={onDismiss} />);
    expect(screen.getByRole("link", { name: "See Pro plans" })).toHaveAttribute("href", "/pricing?from=research");
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("copes with an unknown feature name", () => {
    renderWithProviders(<UpgradePrompt refusal={refusalForFeature("new_thing")} />);
    expect(screen.getByTestId("upgrade-prompt")).toHaveTextContent("New thing is part of Pro");
  });

  // Allowance numbers are rendered from /api/public/config `features`, the
  // same table the backend enforces (ENTITLEMENT_OVERRIDES_JSON included),
  // so the prompt can never promise a cap the backend does not apply.
  describe("plan allowances come from the config matrix, never from copy", () => {
    it("renders the default Pro allowance from the matrix", () => {
      renderWithProviders(<UpgradePrompt refusal={QUOTA} />, { config: { features: featureMatrix() } });
      const box = screen.getByTestId("upgrade-prompt");
      expect(box).toHaveTextContent("A research run puts the full agent committee");
      expect(box).toHaveTextContent("Pro includes 20 research runs a month.");
    });

    it("follows an operator override (pm_chat pro=500) instead of the shipped default", () => {
      const refusal: EntitlementRefusal = { ...QUOTA, feature: "pm_chat", used: 10, limit: 10 };
      renderWithProviders(<UpgradePrompt refusal={refusal} />, {
        config: { features: featureMatrix({ pm_chat: { pro: 500 } }) },
      });
      const box = screen.getByTestId("upgrade-prompt");
      expect(box).toHaveTextContent("Pro includes 500 Ask-the-PM turns a month.");
      expect(box).not.toHaveTextContent("300");
    });

    it("describes an unlimited metered feature as uncapped and a follows-memo Free rule honestly", () => {
      renderWithProviders(<UpgradePrompt refusal={{ ...QUOTA, feature: "memo_view", used: 3, limit: 3 }} />, {
        config: { features: featureMatrix() },
      });
      expect(screen.getByTestId("upgrade-prompt")).toHaveTextContent(
        "Pro has no monthly cap on stored memo views (distinct tickers).",
      );
      const dcf = renderWithProviders(<UpgradePrompt refusal={refusalForFeature("dcf", "free")} />, {
        config: { features: featureMatrix() },
      });
      expect(dcf.container).toHaveTextContent("Pro runs a DCF on any ticker.");
      expect(dcf.container).toHaveTextContent("On Free, a ticker is available once its memo has been opened this month.");
    });

    it("states no number at all when the matrix is unknown (config fetch fell back)", () => {
      renderWithProviders(<UpgradePrompt refusal={{ ...QUOTA, feature: "pm_chat", used: 10, limit: 10 }} />);
      const body = screen.getByTestId("upgrade-prompt").querySelector("p");
      expect(body?.textContent).toContain("Ask-the-PM answers from the committee");
      expect(body?.textContent).not.toMatch(/\d/);
      expect(body?.textContent).not.toMatch(/Pro includes/);
    });
  });
});
