// FEAT-002 (S6) — shapes served by the token-free public API.
// Mirrors backend/app/services/public_samples.py (`list_samples`,
// `assemble`, `build_expectations_ledger`). The backend is the authority;
// every field a sample may lack is nullable here so the marketing pages
// render what exists and say what does not, never crash.

import type { CompsResult, DCFResult, ScreenerRow, StockMemoOut } from "@/types";

/** One row of `GET /api/public/samples` — listed whether built or not. */
export interface SampleSummary {
  ticker: string;
  company_name: string | null;
  sector: string | null;
  /** ISO timestamp of the newest built kind; null when nothing is built. */
  built_at: string | null;
  /** Kinds with a stored row (`memo`, `dcf`, `comps`, `fundamentals`, `prices`, `screener_row`, `commentary`). */
  kinds: string[];
}

export interface FundamentalsPoint {
  period: string;
  period_end: string | null;
  value: number | null;
}

export interface FundamentalsSeries {
  metric: string;
  statement?: string;
  cadence?: "annual" | "quarterly" | string;
  points: FundamentalsPoint[];
}

export interface PricePoint {
  date: string;
  close: number;
}

export interface SampleCommentary {
  text: string;
  generated_at: string;
  model: string;
}

/** The screener row as the worker stores it: the app's `ScreenerRow` plus
 *  the universe size it was ranked against and when. */
export interface SampleScreenerRow extends ScreenerRow {
  universe_size?: number;
  scored_at?: string | null;
}

// --- expectations ledger ------------------------------------------------------

export type LedgerBasis = "observed" | "interpretation";
export type LedgerStatus = "available" | "not_captured" | "n/a";
export type LedgerColumnKey = "reported_consensus" | "management_guidance" | "price_implied" | "our_forecast";

export interface GuidanceCellValue {
  prior: string | null;
  current: string | null;
  direction: string;
  rationale: string;
}

/** A cell's value depends on the leg: prose, a number (price / upside),
 *  a list of falsifiers, or a guidance change. */
export type LedgerCellValue = string | number | string[] | GuidanceCellValue | null;

export interface LedgerCell {
  label: string;
  value: LedgerCellValue;
  basis: LedgerBasis;
  /** Memo field the cell was read from, e.g. `mispricing_thesis.our_view`. */
  source: string;
  as_of?: string;
}

export interface LedgerColumn {
  status: LedgerStatus;
  items: LedgerCell[];
  /** Always set when status is not `available`: why the leg is blank. */
  reason: string | null;
}

export interface ExpectationsLedger {
  columns: LedgerColumnKey[];
  note: string;
  reported_consensus: LedgerColumn;
  management_guidance: LedgerColumn;
  price_implied: LedgerColumn;
  our_forecast: LedgerColumn;
}

/** `GET /api/public/samples/{ticker}` — a 200 even when unbuilt. */
export interface SamplePayload {
  ticker: string;
  company_name: string | null;
  sector: string | null;
  built_at: string | null;
  /** Public copy of the stored memo: long-form reports and the diligence dialog stripped. */
  memo: StockMemoOut | null;
  dcf: DCFResult | null;
  comps: CompsResult | null;
  fundamentals: { series: FundamentalsSeries[] } | null;
  prices: PricePoint[] | null;
  screener_row: SampleScreenerRow | null;
  commentary: SampleCommentary | null;
  expectations_ledger: ExpectationsLedger;
  kinds_built: string[];
  /** Human-readable notes on what is missing or kept from an earlier build. */
  degraded: string[];
  disclosures: string[];
}
