import { describe, expect, it } from "vitest";
import { buildRows, indexSeries, latestValueRatio, observedChange, seriesId, splitSeriesId, statusCounts } from "@/lib/fundamentals/transform";
import { makeSeries, makeSeriesSet } from "@/test/fixtures/fundamentals";

describe("seriesId", () => {
  it("round-trips tickers containing dots", () => {
    const id = seriesId({ ticker: "BRK.B", metric: "free_cash_flow" });
    expect(id).toBe("BRK.B:free_cash_flow");
    expect(splitSeriesId(id)).toEqual({ ticker: "BRK.B", metric: "free_cash_flow" });
  });
});

describe("buildRows", () => {
  it("builds one row per period, chronological, with nulls and reasons for gaps", () => {
    const rows = buildRows(makeSeriesSet());
    expect(rows.map((r) => r.period)).toEqual(["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"]);
    const fy22 = rows[2];
    expect(fy22.values["MSFT:revenue"]).toBeNull();
    expect(fy22.reasons["MSFT:revenue"]).toBe("missing_line");
    expect(fy22.values["AAPL:revenue"]).toBe(394.3e9);
    expect(fy22.estimated["AAPL:gross_margin"]).toBe(true);
    expect(fy22.estimated["MSFT:gross_margin"]).toBe(false);
  });

  it("fills a period one series lacks entirely with null (not obtained), keeping row counts equal", () => {
    const short = makeSeries("NEW", "revenue", [1e9, 2e9, 3e9]);
    const long = makeSeries("OLD", "revenue", [5e9, 6e9, 7e9, 8e9, 9e9]);
    const rows = buildRows([short, long]);
    expect(rows).toHaveLength(5);
    expect(rows[4].values["NEW:revenue"]).toBeNull();
    expect(rows[4].reasons["NEW:revenue"]).toBeNull();
    expect(rows[4].estimated["NEW:revenue"]).toBe(false);
  });
});

describe("statusCounts", () => {
  it("counts observed, missing and estimated points and carries staleness", () => {
    const [aaplRev, msftRev, aaplGm, msftGm] = makeSeriesSet();
    expect(statusCounts(aaplRev)).toEqual({ observed: 5, missing: 0, estimated: 0, stale: false });
    expect(statusCounts(msftRev)).toEqual({ observed: 4, missing: 1, estimated: 0, stale: false });
    expect(statusCounts(aaplGm)).toEqual({ observed: 5, missing: 0, estimated: 1, stale: false });
    expect(statusCounts(msftGm).stale).toBe(true);
  });
});

describe("indexSeries", () => {
  it("rebases to 100 at the first period where every series is observed, blanking earlier observed points", () => {
    const a = makeSeries("A", "revenue", [null, 100, 110, 121, 133.1]);
    const b = makeSeries("B", "revenue", [50, 55, 60.5, 66.55, 73.205]);
    const out = indexSeries([a, b]);
    expect(out.basePeriod).toBe("FY2021");
    expect(out.skipped).toEqual([]);
    const [ia, ib] = out.series;
    // A's leading gap keeps its server-side reason.
    expect(ia.points[0]).toMatchObject({ value: null, reason: "missing_line" });
    expect(ia.points.slice(1).map((p) => p.value!)).toEqual([100, 110, 121, 133.1].map((v) => expect.closeTo(v, 6)));
    // B's FY2020 was observed but precedes the base, so it is blanked with a client reason.
    expect(ib.points[0]).toMatchObject({ value: null, reason: "before_index_base" });
    expect(ib.points.slice(1).map((p) => p.value!)).toEqual([100, 110, 121, 133.1].map((v) => expect.closeTo(v, 6)));
    // The originals are untouched.
    expect(b.points[0].value).toBe(50);
  });

  it("skips an all-missing series and still indexes the rest", () => {
    const empty = makeSeries("EMPTY", "revenue", [null, null, null, null, null]);
    const a = makeSeries("A", "revenue", [10, 20, 30, 40, 50]);
    const out = indexSeries([empty, a]);
    expect(out.skipped).toEqual([{ id: "EMPTY:revenue", reason: "no observed points to index" }]);
    expect(out.basePeriod).toBe("FY2020");
    expect(out.series.map(seriesId)).toEqual(["A:revenue"]);
    expect(out.series[0].points.map((p) => p.value)).toEqual([100, 200, 300, 400, 500]);
  });

  it("keeps an internal gap as a gap in the indexed series", () => {
    const a = makeSeries("A", "revenue", [10, null, 30, 40, 50]);
    const out = indexSeries([a]);
    expect(out.series[0].points[1]).toMatchObject({ value: null, reason: "missing_line" });
    expect(out.series[0].points[2].value).toBe(300);
  });

  it("refuses when no period has every series positive, listing each series", () => {
    const neg = makeSeries("NEG", "net_income", [-1, -2, -3, -4, -5]);
    const pos = makeSeries("POS", "net_income", [1, 2, 3, 4, 5]);
    const out = indexSeries([neg, pos]);
    expect(out.basePeriod).toBeNull();
    expect(out.series).toEqual([]);
    expect(out.skipped.map((s) => s.reason)).toEqual(["no period where every series is observed and positive", "no period where every series is observed and positive"]);
  });

  it("returns nothing for an empty input", () => {
    expect(indexSeries([])).toEqual({ basePeriod: null, series: [], skipped: [] });
  });
});

describe("latestValueRatio", () => {
  it("uses the latest observed positive value of each series", () => {
    const big = makeSeries("BIG", "revenue", [1, 2, 3, 4, 400]);
    const small = makeSeries("SMALL", "revenue", [1, 2, 3, 4, null]); // latest observed = 4
    expect(latestValueRatio([big, small])).toBe(100);
  });

  it("is null with fewer than two positive latest values", () => {
    const big = makeSeries("BIG", "revenue", [1, 2, 3, 4, 400]);
    const neg = makeSeries("NEG", "revenue", [1, 2, 3, 4, -1]);
    expect(latestValueRatio([big, neg])).toBeNull();
    expect(latestValueRatio([big])).toBeNull();
  });
});

describe("observedChange", () => {
  it("returns a null pct when the first observed value is not positive", () => {
    const s = makeSeries("X", "net_income", [-1e9, null, 2e9, 3e9, 4e9]);
    const c = observedChange(s.points)!;
    expect(c.first.period).toBe("FY2020");
    expect(c.last.period).toBe("FY2024");
    expect(c.pct).toBeNull();
  });

  it("returns null with no observed points", () => {
    expect(observedChange(makeSeries("X", "revenue", [null, null]).points)).toBeNull();
  });
});
