import { describe, expect, it } from "vitest";
import { safeReturnTo, signInPath } from "@/auth/returnTo";

describe("safeReturnTo", () => {
  it("keeps same-origin paths inside the app shell, query and hash included", () => {
    expect(safeReturnTo("/app")).toBe("/app");
    expect(safeReturnTo("/app/research?ticker=NVDA")).toBe("/app/research?ticker=NVDA");
    expect(safeReturnTo("/app/chat?q=hi#x")).toBe("/app/chat?q=hi#x");
    expect(safeReturnTo("/app?tab=1")).toBe("/app?tab=1");
    // Encoded once by signInPath, decoded once here.
    expect(safeReturnTo(encodeURIComponent("/app/research?ticker=NVDA"))).toBe("/app/research?ticker=NVDA");
  });

  it("rejects protocol-relative, scheme, API and non-app destinations", () => {
    for (const bad of [
      "//evil.example",
      "https://evil.example/app",
      "javascript:alert(1)",
      "JavaScript:alert(1)",
      "/api/me",
      "/api/admin/ui-log",
      "/apple",
      "/",
      "/pricing",
      "app/research",
      "/app/\\evil",
      "/\\evil.example",
      "/app/x y",
      "%2F%2Fevil.example",
      "",
      null,
      undefined,
    ]) {
      expect(safeReturnTo(bad as string)).toBe("/app");
    }
  });

  it("falls back on undecodable input", () => {
    expect(safeReturnTo("/app/%E0%A4%A")).toBe("/app");
  });
});

describe("signInPath", () => {
  it("encodes the destination into returnTo", () => {
    expect(signInPath("/app/research?ticker=NVDA")).toBe("/sign-in?returnTo=%2Fapp%2Fresearch%3Fticker%3DNVDA");
    expect(signInPath("/app", "sign-up")).toBe("/sign-up?returnTo=%2Fapp");
  });
});
