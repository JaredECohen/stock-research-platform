import { describe, expect, it } from "vitest";
import {
  axisTickFormatter,
  formatCount,
  formatCurrency,
  formatIndexed,
  formatMissing,
  formatMultiple,
  formatPercent,
  formatRatio,
  formatValue,
  humanizeMetric,
  REASON_TEXT,
  reasonText,
  unitLabel,
} from "@/lib/fundamentals/format";
import type { MissingReason } from "@/types/fundamentals";

describe("formatValue by unit", () => {
  const cases: Array<[number, Parameters<typeof formatValue>[1], string, string | null]> = [
    [274.5e9, "currency", "$274.5B", null],
    [1.5e12, "currency", "$1.50T", null],
    [12.34e6, "currency", "$12.3M", null],
    [999, "currency", "$999", null],
    [-1.2e9, "currency", "-$1.2B", null],
    [1.2e9, "currency", "€1.2B", "EUR"],
    [1.2e9, "currency", "SEK 1.2B", "SEK"],
    [1.2e9, "currency", "CHF 1.2B", "CHF"],
    [0.462, "percent", "46.2%", null],
    [-0.05, "percent", "-5.0%", null],
    [1.234, "ratio", "1.23", null],
    [12.34, "multiple", "12.3x", null],
    [15.2e9, "count", "15.2B", null],
    [142.33, "index", "142.3", null],
  ];
  for (const [v, unit, expected, currency] of cases) {
    it(`${unit}${currency ? ` (${currency})` : ""}: ${v} → ${expected}`, () => {
      expect(formatValue(v, unit, { currency })).toBe(expected);
    });
  }

  it("never renders a missing value as zero — always n/a with the reason", () => {
    expect(formatValue(null, "currency", { reason: "no_price" })).toBe("n/a (no price at period end)");
    expect(formatValue(undefined, "percent", { reason: "denominator_nonpositive" })).toBe("n/a (denominator is zero or negative)");
    expect(formatValue(null, "multiple", { reason: "base_nonpositive" })).toBe("n/a (prior-period base is zero or negative)");
    expect(formatValue(null, "count", { reason: "no_shares" })).toBe("n/a (no diluted share count)");
    expect(formatValue(null, "currency", { reason: "missing_line" })).toBe("n/a (line item not reported)");
    expect(formatValue(null, "currency", { reason: "not_backfilled" })).toBe("n/a (history not loaded)");
    expect(formatValue(null, "index", { reason: "before_index_base" })).toBe("n/a (before the index base period)");
    expect(formatValue(null, "currency")).toBe("n/a (not obtained)");
    expect(formatValue(Number.NaN, "currency")).toBe("n/a (not obtained)");
  });

  it("covers every reason code with text", () => {
    const reasons: MissingReason[] = ["base_nonpositive", "denominator_nonpositive", "no_price", "no_shares", "missing_line", "not_backfilled", "before_index_base"];
    for (const r of reasons) expect(REASON_TEXT[r]).toBeTruthy();
    expect(reasonText("something_new" as MissingReason)).toBe("something new");
    expect(formatMissing(null)).toBe("n/a (not obtained)");
  });
});

describe("individual formatters", () => {
  it("return n/a for null", () => {
    expect(formatCurrency(null)).toBe("n/a");
    expect(formatPercent(null)).toBe("n/a");
    expect(formatRatio(null)).toBe("n/a");
    expect(formatMultiple(null)).toBe("n/a");
    expect(formatCount(null)).toBe("n/a");
    expect(formatIndexed(null)).toBe("n/a");
  });

  it("axis ticks are terser than cell values", () => {
    expect(axisTickFormatter("currency", "USD")(250e9)).toBe("$250B");
    expect(axisTickFormatter("currency", "EUR")(-3e9)).toBe("-€3B");
    expect(axisTickFormatter("percent")(0.4)).toBe("40%");
    expect(axisTickFormatter("multiple")(12.6)).toBe("13x");
    expect(axisTickFormatter("count")(16e9)).toBe("16B");
    expect(axisTickFormatter("index")(142.4)).toBe("142");
    expect(axisTickFormatter("ratio")(1.234)).toBe("1.2");
  });

  it("unit labels", () => {
    expect(unitLabel("currency", "usd")).toBe("USD");
    expect(unitLabel("currency")).toBe("currency");
    expect(unitLabel("percent")).toBe("%");
    expect(unitLabel("multiple")).toBe("x");
    expect(unitLabel("index")).toBe("index (base = 100)");
  });

  it("humanizes metric ids as a fallback label", () => {
    expect(humanizeMetric("free_cash_flow")).toBe("Free cash flow");
  });
});
