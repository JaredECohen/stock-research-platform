// FEAT-002 (S6) — data hooks over `api/publicClient`. Both settle into an
// explicit status instead of throwing so the page renders a degraded
// state; both ignore results that arrive after unmount or after the
// ticker changed (a slow NVDA response must not overwrite JPM).
import { useEffect, useState } from "react";
import { getSample, listSamples, type SampleResult } from "@/api/publicClient";
import type { SamplePayload, SampleSummary } from "@/types/public";

export interface SampleListState {
  status: "loading" | "ok" | "error";
  samples: SampleSummary[];
}

export function useSampleList(): SampleListState {
  const [state, setState] = useState<SampleListState>({ status: "loading", samples: [] });
  useEffect(() => {
    let cancelled = false;
    void listSamples().then((res) => {
      if (cancelled) return;
      setState(res === null ? { status: "error", samples: [] } : { status: "ok", samples: res });
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return state;
}

export interface SampleState {
  status: "idle" | "loading" | "ok" | "not_found" | "unavailable";
  ticker: string | null;
  sample: SamplePayload | null;
  /** On 404: the tickers that ARE public, from the backend's error body. */
  sampleTickers: string[];
}

const IDLE: SampleState = { status: "idle", ticker: null, sample: null, sampleTickers: [] };

export function useSample(ticker: string | null): SampleState {
  const [state, setState] = useState<SampleState>(IDLE);
  useEffect(() => {
    if (!ticker) {
      setState(IDLE);
      return undefined;
    }
    let cancelled = false;
    setState({ status: "loading", ticker, sample: null, sampleTickers: [] });
    void getSample(ticker).then((res: SampleResult) => {
      if (cancelled) return;
      if (res.status === "ok") setState({ status: "ok", ticker, sample: res.sample, sampleTickers: [] });
      else if (res.status === "not_found") setState({ status: "not_found", ticker, sample: null, sampleTickers: res.sampleTickers });
      else setState({ status: "unavailable", ticker, sample: null, sampleTickers: [] });
    });
    return () => {
      cancelled = true;
    };
  }, [ticker]);
  return state;
}
