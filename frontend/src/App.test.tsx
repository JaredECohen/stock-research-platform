import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "@/App";
import { AuthProvider } from "@/auth/AuthProvider";
import { ConfigProvider } from "@/auth/ConfigProvider";
import { resetAccountCache } from "@/auth/useAccount";
import { LocationSpy, freeAccount, makeAccount, okJson, stubFetch } from "@/test/providers";
import type { PublicConfig } from "@/types";

// Full app: real ConfigProvider (seeded, no fetch) + real AuthProvider
// (stub via VITE_AUTH_STUB) + the route tree. fetch is stubbed by URL.

function mountApp(route: string, config: Partial<PublicConfig>) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <ConfigProvider initial={config}>
        <AuthProvider>
          <LocationSpy />
          <App />
        </AuthProvider>
      </ConfigProvider>
    </MemoryRouter>,
  );
}

async function settle() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

const location = () => screen.getByTestId("location").textContent;

describe("App routing", () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    resetAccountCache();
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  describe("with the login wall off", () => {
    it("renders the marketing landing at / and the dashboard at /app", async () => {
      stubFetch();
      const view = mountApp("/", { auth_enabled: false });
      await settle();
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Research that shows its work.");
      // The CTA appears in both the header and the hero.
      expect(screen.getAllByRole("link", { name: "Continue to app" })[0]).toHaveAttribute("href", "/app");
      view.unmount();

      stubFetch();
      mountApp("/app", { auth_enabled: false });
      await settle();
      expect(screen.getByText("Your AI Investment Committee")).toBeInTheDocument();
      expect(location()).toBe("/app");
      // No auth chrome when the wall is off.
      expect(screen.queryByRole("button", { name: "Account menu" })).not.toBeInTheDocument();
    });

    it.each([
      ["/research?ticker=NVDA", "/app/research?ticker=NVDA"],
      ["/chat?q=Analyze%20NVDA#x", "/app/chat?q=Analyze%20NVDA#x"],
      ["/dcf", "/app/dcf"],
      ["/comps", "/app/comps"],
      ["/screener", "/app/screener"],
      ["/portfolio", "/app/portfolio"],
      ["/macro", "/app/macro"],
      ["/track-record?horizon=90", "/app/track-record?horizon=90"],
      ["/settings", "/app/settings"],
    ])("redirects legacy %s to %s keeping query and hash", async (from, to) => {
      stubFetch();
      mountApp(from, { auth_enabled: false });
      await settle();
      expect(location()).toBe(to);
    });

    it("sends /sign-in straight to the app since there is nothing to sign into", async () => {
      stubFetch();
      mountApp("/sign-in?returnTo=%2Fapp%2Fresearch", { auth_enabled: false });
      await settle();
      expect(location()).toBe("/app/research");
    });

    it("renders NotFound for unknown paths, inside and outside the shell", async () => {
      stubFetch();
      const v = mountApp("/nope", { auth_enabled: false });
      await settle();
      expect(screen.getByText("That page does not exist")).toBeInTheDocument();
      v.unmount();
      stubFetch();
      mountApp("/app/nope", { auth_enabled: false });
      await settle();
      expect(screen.getByText("That page does not exist")).toBeInTheDocument();
    });
  });

  describe("with the login wall on and the stub provider", () => {
    it("redirects a signed-out visitor from /app/* to /sign-in with a returnTo", async () => {
      vi.stubEnv("VITE_AUTH_STUB", "signed_out");
      stubFetch();
      mountApp("/app/research?ticker=NVDA", { auth_enabled: true });
      await settle();
      expect(location()).toBe("/sign-in?returnTo=%2Fapp%2Fresearch%3Fticker%3DNVDA");
      expect(screen.getByTestId("stub-sign-in")).toBeInTheDocument();
      expect(screen.queryByText("Stock Research")).not.toBeInTheDocument();
    });

    it("returns to the deep link after the stub sign-in and bootstraps once", async () => {
      vi.stubEnv("VITE_AUTH_STUB", "signed_out");
      const mock = stubFetch([
        ["/api/me/bootstrap", () => okJson({ ...makeAccount(), trial_started_now: true })],
        ["/api/me", () => okJson(makeAccount())],
      ]);
      mountApp("/app/research?ticker=NVDA", { auth_enabled: true });
      await settle();
      fireEvent.click(screen.getByRole("button", { name: "Continue as test user" }));
      await settle();
      await settle();
      expect(location()).toBe("/app/research?ticker=NVDA");
      expect(screen.getByRole("heading", { name: "Stock Research" })).toBeInTheDocument();
      const boots = mock.mock.calls.filter(([u]) => String(u).includes("/api/me/bootstrap"));
      expect(boots).toHaveLength(1);
      expect(new Headers(boots[0][1]?.headers).get("authorization")).toBe("Bearer stub-token");
    });

    it("rejects an unsafe returnTo and lands on /app", async () => {
      vi.stubEnv("VITE_AUTH_STUB", "signed_out");
      stubFetch([["/api/me", () => okJson(makeAccount())]]);
      mountApp("/sign-in?returnTo=%2F%2Fevil.example%2Fapp", { auth_enabled: true });
      await settle();
      fireEvent.click(screen.getByRole("button", { name: "Continue as test user" }));
      await settle();
      await settle();
      expect(location()).toBe("/app");
    });

    it.each(["https%3A%2F%2Fevil.example", "%2Fapi%2Fme", "javascript%3Aalert(1)", "%2Fpricing"])(
      "rejects returnTo=%s",
      async (bad) => {
        vi.stubEnv("VITE_AUTH_STUB", "signed_in");
        stubFetch([["/api/me", () => okJson(makeAccount())]]);
        mountApp(`/sign-in?returnTo=${bad}`, { auth_enabled: true });
        await settle();
        expect(location()).toBe("/app");
      },
    );

    it("renders the shell with plan badge and locks Pro items for a Free account", async () => {
      vi.stubEnv("VITE_AUTH_STUB", "signed_in");
      stubFetch([
        ["/api/me/bootstrap", () => okJson({ ...freeAccount(), trial_started_now: false })],
        ["/api/me", () => okJson(freeAccount())],
      ]);
      mountApp("/app", { auth_enabled: true, billing_enabled: true });
      await settle();
      await settle();
      expect(screen.getByText("Your AI Investment Committee")).toBeInTheDocument();
      expect(screen.getByTestId("plan-badge")).toHaveTextContent("Free");
      expect(screen.getByTestId("lock-portfolio")).toBeInTheDocument();
      expect(screen.getByRole("link", { name: "Portfolio Builder (Pro feature)" })).toHaveAttribute(
        "href",
        "/app/account?upgrade=portfolio",
      );
      // Free items are plain links to their pages.
      expect(screen.getByRole("link", { name: "Stock Research" })).toHaveAttribute("href", "/app/research");
      expect(screen.queryByTestId("lock-chat")).not.toBeInTheDocument();
    });

    it("shows the unavailable notice when the wall is on but no provider is configured", async () => {
      stubFetch();
      mountApp("/app", { auth_enabled: true, clerk_publishable_key: null });
      await settle();
      expect(screen.getByRole("alert")).toHaveTextContent("Sign-in is temporarily unavailable");
      expect(screen.queryByText("Your AI Investment Committee")).not.toBeInTheDocument();
    });

    it("keeps the landing page reachable signed out", async () => {
      vi.stubEnv("VITE_AUTH_STUB", "signed_out");
      stubFetch();
      mountApp("/", { auth_enabled: true });
      await settle();
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Research that shows its work.");
      // Header and hero both link to sign-in.
      expect(screen.getAllByRole("link", { name: "Sign in" })[0]).toHaveAttribute("href", "/sign-in?returnTo=%2Fapp");
    });
  });
});
