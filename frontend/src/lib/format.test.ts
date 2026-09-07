import { describe, expect, it } from "vitest";
import { fmtPrice, fmtUpside, numOrNull } from "@/lib/format";

// The DCF engine emits null for numbers it could not compute. These
// helpers are the single rendering of that: "n/a", matching the backend's
// memo prose, never "$0.00" / "+0.0%".
describe("DCF formatters", () => {
  it("fmtPrice renders n/a for null / undefined / NaN and a 2dp price otherwise", () => {
    expect(fmtPrice(null)).toBe("n/a");
    expect(fmtPrice(undefined)).toBe("n/a");
    expect(fmtPrice(Number.NaN)).toBe("n/a");
    expect(fmtPrice(0)).toBe("$0.00");
    expect(fmtPrice(1234.5)).toBe("$1,234.50");
  });

  it("fmtUpside renders n/a for null and a signed percentage otherwise", () => {
    expect(fmtUpside(null)).toBe("n/a");
    expect(fmtUpside(undefined)).toBe("n/a");
    expect(fmtUpside(0.1234)).toBe("+12.3%");
    expect(fmtUpside(-0.05)).toBe("-5.0%");
    expect(fmtUpside(0)).toBe("0.0%");
    expect(fmtUpside(0.1234, 0)).toBe("+12%");
  });

  it("numOrNull never coerces null / missing to 0", () => {
    expect(numOrNull(null)).toBeNull();
    expect(numOrNull(undefined)).toBeNull();
    expect(numOrNull("")).toBeNull();
    expect(numOrNull("abc")).toBeNull();
    expect(numOrNull(Number.POSITIVE_INFINITY)).toBeNull();
    expect(numOrNull(0)).toBe(0);
    expect(numOrNull(12.5)).toBe(12.5);
    expect(numOrNull("12.5")).toBe(12.5);
  });
});
