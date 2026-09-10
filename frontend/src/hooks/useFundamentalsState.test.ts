import { describe, expect, it } from "vitest";
import { encodeState, parseMetrics, parseState, parseTickers, parseView, parseYears } from "@/hooks/useFundamentalsState";

describe("useFundamentalsState parsers", () => {
  it("upper-cases, de-duplicates and clamps tickers to the absolute ceiling", () => {
    expect(parseTickers("aapl, AAPL,msft $bad brk.b")).toEqual(["AAPL", "MSFT", "BRK.B"]);
    expect(parseTickers("a,b,c,d,e,f,g")).toEqual(["A", "B", "C", "D", "E", "F", "G"].slice(0, 5));
    expect(parseTickers(null)).toEqual([]);
  });

  it("keeps plausible metric ids until the catalog is known, then drops unknown ones", () => {
    expect(parseMetrics("Revenue,bogus,revenue", null)).toEqual(["revenue", "bogus"]);
    expect(parseMetrics("Revenue,bogus", new Set(["revenue"]))).toEqual(["revenue"]);
    expect(parseMetrics("a,b,c,d,e", null)).toHaveLength(4);
  });

  it("accepts only the offered ranges and view modes", () => {
    expect(parseYears("5")).toBe(5);
    expect(parseYears("10")).toBe(10);
    expect(parseYears("max")).toBeNull();
    expect(parseYears("7")).toBeNull();
    expect(parseView("dual-axis")).toBe("dual-axis");
    expect(parseView("sideways")).toBe("auto");
  });

  it("encodes the canonical URL with literal commas, omits defaults and carries foreign params", () => {
    expect(encodeState({ tickers: ["AAPL", "MSFT"], metrics: ["revenue"], years: 10, view: "table" })).toBe("?t=AAPL,MSFT&m=revenue&y=10&v=table");
    expect(encodeState({ tickers: [], metrics: [], years: null, view: "auto" })).toBe("");
    expect(encodeState({ tickers: ["AAPL"], metrics: [], years: null, view: "auto" }, new URLSearchParams("utm=x&t=old"))).toBe("?t=AAPL&utm=x");
  });

  it("round-trips: encode(parse(x)) is a fixed point", () => {
    const known = new Set(["revenue", "gross_margin"]);
    const once = encodeState(parseState(new URLSearchParams("t=aapl,AAPL&m=revenue,bogus&y=10&v=nope"), known));
    expect(once).toBe("?t=AAPL&m=revenue&y=10");
    expect(encodeState(parseState(new URLSearchParams(once), known))).toBe(once);
  });
});
