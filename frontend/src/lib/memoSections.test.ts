import { describe, expect, it } from "vitest";
import {
  REASON_TEXT,
  UNAVAILABLE_TEXT,
  bannerCount,
  hiddenKeys,
  intakeRationale,
  isBlankedFinding,
  isHidden,
  sectionStatus,
} from "@/lib/memoSections";
import { makeMemo } from "@/test/fixtures/memo";
import { PRESENTED_MEMO_NAMES, presentedMemo } from "@/test/fixtures/memoSections";
import type { SectionAvailability } from "@/types";

function av(status: SectionAvailability["status"], reason: SectionAvailability["reason"]): SectionAvailability {
  return { status, reason, hidden_items: 0, headline_hidden: false, basis: [] };
}

describe("memoSections", () => {
  it("uses the presenter's exact placeholder wording", () => {
    // The server writes this text into every hidden prose field; the UI's
    // constant must be the same string or a hidden section reads twice.
    expect(presentedMemo("googl_live_prepflag").final_pm_view).toBe(UNAVAILABLE_TEXT);
    expect(UNAVAILABLE_TEXT).toBe("Unavailable in this version.");
  });

  it("words every reason the captured presenter output uses", () => {
    for (const name of PRESENTED_MEMO_NAMES) {
      for (const entry of Object.values(presentedMemo(name).section_availability ?? {})) {
        if (entry.reason) expect(REASON_TEXT[entry.reason], `${name}: ${entry.reason}`).toBeTruthy();
      }
    }
  });

  it("reads an absent map or key as available (pre-W2a bodies)", () => {
    const legacy = makeMemo({ section_availability: undefined });
    expect(sectionStatus(legacy, "final_pm_view")).toBe("available");
    expect(isHidden(legacy, "final_pm_view")).toBe(false);
    expect(sectionStatus(makeMemo(), "one_sentence_thesis")).toBe("available");
  });

  it("hides only technical, the critic and portfolio fit on the agentic META memo", () => {
    expect(hiddenKeys(presentedMemo("meta_v1")).sort()).toEqual([
      "portfolio_fit",
      "risk_committee_challenge",
      "technical_agent_view",
    ]);
  });

  it("counts neither template_always, intake skips, nor not-produced sections in the banner", () => {
    // META: portfolio fit (template_always), technical (intake skip) and two
    // not-produced optional sections are all unavailable; only the critic counts.
    expect(bannerCount(presentedMemo("meta_v1"))).toBe(1);
    const onlyExcluded = makeMemo({
      section_availability: {
        portfolio_fit: av("unavailable", "template_always"),
        technical_agent_view: av("unavailable", "skipped_by_intake"),
        earnings_qoq_delta: av("unavailable", "not_produced"),
        "comps_agent_view.long_form_report": av("unavailable", "template_fallback"),
        rating_label: av("degraded", "pm_view_unavailable"),
      },
    });
    expect(bannerCount(onlyExcluded)).toBe(0);
    // GOOGL (template PM view, no flag): thesis, PM view, confidence,
    // mispricing, sector view, sector synthesis, critic and final verdict.
    expect(bannerCount(presentedMemo("googl_live_prepflag"))).toBe(8);
  });

  it("reads the intake rationale from the finding, then from the intake decision", () => {
    const meta = presentedMemo("meta_v1");
    expect(intakeRationale(meta, meta.technical_agent_view)).toBe(
      "Synthetic intake rationale 64 for the meta_v1 fixture.",
    );
    // The public-sample size reducer can empty a finding's `data`.
    const reduced = { ...meta.technical_agent_view!, data: {} };
    expect(intakeRationale(meta, reduced)).toBe("Synthetic intake rationale 64 for the meta_v1 fixture.");
  });

  it("recognises a round finding the presenter blanked", () => {
    const msft = presentedMemo("msft_live");
    const round1 = msft.round_findings![1].findings;
    expect(isBlankedFinding(round1.valuation)).toBe(true);
    expect(isBlankedFinding(round1.earnings)).toBe(false);
    expect(isBlankedFinding(undefined)).toBe(false);
  });
});
