import { useCallback, useEffect, useState } from "react";
import { api } from "@/api/client";
import type { Account } from "@/types";
import { useAuth } from "./AuthContext";
import { useConfig } from "./ConfigProvider";

/**
 * SWR-style cache of `GET /api/me`, shared by every subscriber (Layout
 * badge, TrialBanner, Account page) so the shell makes one call, not
 * three. Refreshes on window focus (throttled) and whenever the client
 * sees a 402 (`mm:account-refresh`) — the meters just moved.
 *
 * Null with the wall off: there is no account to read.
 */
export const ACCOUNT_REFRESH_EVENT = "mm:account-refresh";
const FOCUS_THROTTLE_MS = 30_000;

interface CacheState {
  account: Account | null;
  error: string | null;
  fetchedAt: number;
  inflight: Promise<void> | null;
}

const cache: CacheState = { account: null, error: null, fetchedAt: 0, inflight: null };
const listeners = new Set<() => void>();

function notify() {
  listeners.forEach((fn) => fn());
}

function load(): Promise<void> {
  if (cache.inflight) return cache.inflight;
  cache.inflight = api
    .me()
    .then((acct) => {
      cache.account = acct;
      cache.error = null;
    })
    .catch((e: Error) => {
      // Keep the last good snapshot on a transient failure; the page can
      // still render meters from it and the next refresh will replace it.
      cache.error = e.message || "Failed to load account";
    })
    .finally(() => {
      cache.fetchedAt = Date.now();
      cache.inflight = null;
      notify();
    });
  return cache.inflight;
}

/** Drop the cached account (sign-out, tests). */
export function resetAccountCache(): void {
  cache.account = null;
  cache.error = null;
  cache.fetchedAt = 0;
  cache.inflight = null;
  notify();
}

/** Ask every subscriber to refetch; used after checkout/reconcile. */
export function invalidateAccount(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent(ACCOUNT_REFRESH_EVENT));
}

export interface UseAccountResult {
  account: Account | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  /** True when there is an account to talk about (wall on, signed in). */
  enabled: boolean;
}

export function useAccount(): UseAccountResult {
  const { config } = useConfig();
  const { status } = useAuth();
  const enabled = config.auth_enabled && status === "signed_in";
  const [, force] = useState(0);

  useEffect(() => {
    const bump = () => force((n) => n + 1);
    listeners.add(bump);
    return () => {
      listeners.delete(bump);
    };
  }, []);

  useEffect(() => {
    if (!enabled) return;
    if (cache.account === null && cache.inflight === null) void load();
    const onRefresh = () => void load();
    const onFocus = () => {
      if (Date.now() - cache.fetchedAt > FOCUS_THROTTLE_MS) void load();
    };
    window.addEventListener(ACCOUNT_REFRESH_EVENT, onRefresh);
    window.addEventListener("focus", onFocus);
    return () => {
      window.removeEventListener(ACCOUNT_REFRESH_EVENT, onRefresh);
      window.removeEventListener("focus", onFocus);
    };
  }, [enabled]);

  useEffect(() => {
    if (status === "signed_out") resetAccountCache();
  }, [status]);

  const refresh = useCallback(() => (enabled ? load() : Promise.resolve()), [enabled]);

  return {
    account: enabled ? cache.account : null,
    loading: enabled && cache.account === null && cache.error === null,
    error: enabled ? cache.error : null,
    refresh,
    enabled,
  };
}
