import { describe, expect, it } from "vitest";
import {
  FULL_MEMO_SECTIONS,
  MEMO_CARD_SECTIONS,
  REASON_TEXT,
  SAMPLE_SUMMARY_SECTIONS,
  SECTION_KEYS,
  SECTION_REASON_TEXT,
  UNAVAILABLE_TEXT,
  availability,
  bannerCount,
  isCounted,
  reasonText,
  hiddenKeys,
  intakeRationale,
  isBlankedFinding,
  isHidden,
  sectionStatus,
} from "@/lib/memoSections";
import { makeMemo } from "@/test/fixtures/memo";
import { PRESENTED_MEMO_NAMES, PROBES, presentedMemo } from "@/test/fixtures/memoSections";
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

  it("places the debate after the cases and words its states in plain language", () => {
    // D4: `SECTION_KEYS` mirrors the presenter (the backend contract test
    // compares the two); the debate's reasons carry no internal codes.
    expect(SECTION_KEYS.indexOf("debate")).toBe(SECTION_KEYS.indexOf("bear_case") + 1);
    for (const reason of ["debate_unavailable", "not_run", "rebuttals_unavailable"] as const) {
      expect(REASON_TEXT[reason]).toMatch(/^[A-Z][a-z ,.]+\.$/);
    }
    // A partial debate's note is the design's wording.
    expect(reasonText(av("degraded", "rebuttals_unavailable"))).toBe("Rebuttals unavailable in this version.");
    // A debate that was not run hid nothing, so it never counts toward the banner.
    const memo = makeMemo({ section_availability: { debate: av("unavailable", "not_run") } });
    expect(isCounted(memo, "debate")).toBe(false);
    // Every captured legacy memo has the entry, as not produced.
    for (const name of PRESENTED_MEMO_NAMES) {
      expect(availability(presentedMemo(name), "debate")?.reason, name).toBe("not_produced");
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
    expect(bannerCount(presentedMemo("meta_v1"), MEMO_CARD_SECTIONS)).toBe(1);
    const onlyExcluded = makeMemo({
      section_availability: {
        portfolio_fit: av("unavailable", "template_always"),
        technical_agent_view: av("unavailable", "skipped_by_intake"),
        earnings_qoq_delta: av("unavailable", "not_produced"),
        "comps_agent_view.long_form_report": av("unavailable", "template_fallback"),
        rating_label: av("degraded", "pm_view_unavailable"),
      },
    });
    for (const shown of [MEMO_CARD_SECTIONS, FULL_MEMO_SECTIONS, SAMPLE_SUMMARY_SECTIONS]) {
      expect(bannerCount(onlyExcluded, shown)).toBe(0);
    }
    // GOOGL (template PM view, no flag): thesis, PM view, confidence,
    // mispricing, sector view, sector synthesis, critic and final verdict.
    expect(bannerCount(presentedMemo("googl_live_prepflag"), MEMO_CARD_SECTIONS)).toBe(8);
  });

  it("counts only the sections the calling renderer can show", () => {
    // FEAT-003 routing gives every unmapped ticker a hidden industry finding
    // that no memo renderer displays; counting it named a section the reader
    // could not find.
    const industry = PROBES.industry();
    expect(isCounted(industry, "extra_agent_views.industry_group")).toBe(true);
    expect(bannerCount(industry, MEMO_CARD_SECTIONS)).toBe(1); // the critic alone
    expect(bannerCount(industry, FULL_MEMO_SECTIONS)).toBe(1);
    // The public summary shows no critic, so nothing it shows is missing.
    expect(bannerCount(industry, SAMPLE_SUMMARY_SECTIONS)).toBe(0);
    // A missing scorecard row is shown (and counted) by the full memo only.
    const scorecard = PROBES.scorecard();
    expect(bannerCount(scorecard, FULL_MEMO_SECTIONS)).toBe(2);
    expect(bannerCount(scorecard, MEMO_CARD_SECTIONS)).toBe(1);
    // The full memo has no sector-synthesis block, so GOOGL counts 7 there.
    expect(bannerCount(presentedMemo("googl_live_prepflag"), FULL_MEMO_SECTIONS)).toBe(7);
  });

  it("words a degraded thesis as a kept claim beside builder wording, not as removed text", () => {
    const thesis = availability(PROBES.degradedThesis(), "one_sentence_thesis");
    expect(reasonText(thesis, "one_sentence_thesis")).toBe(SECTION_REASON_TEXT.one_sentence_thesis.partial_template);
    expect(reasonText(thesis, "one_sentence_thesis")).not.toBe(REASON_TEXT.partial_template);
    // Other sections keep the generic sentence (the presenter does remove
    // items from a partially templated list).
    expect(reasonText(thesis, "key_risks")).toBe(REASON_TEXT.partial_template);
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
