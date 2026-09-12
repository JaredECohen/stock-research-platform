import { describe, expect, it } from "vitest";
import { safeReturnTo, signInPath } from "@/auth/returnTo";
// Vite `?raw`: the file as text, so the test can inspect its bytes.
import returnToSource from "./returnTo.ts?raw";

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

  it("rejects control characters, DEL and whitespace anywhere in the path", () => {
    for (const bad of [
      "/app/x\u0000y",
      "/app/x\u007fy",
      "/app/x\ty",
      "/app/x\ny",
      "/app/x\ry",
      "/app/x\u001by",
      "/app/research?ticker=NV\u0000DA",
      // Percent-encoded controls decode to the same bytes and are refused too.
      "/app/x%00y",
      "/app/x%7Fy",
      "/app/x%0Ay",
    ]) {
      expect(safeReturnTo(bad)).toBe("/app");
    }
  });

  it("keeps the sanitiser reviewable: the source contains no raw control bytes", () => {
    // A literal NUL/DEL in the regex once made git classify this file as
    // binary, so a change to the redirect allowlist shipped with no diff.
    const src = returnToSource;
    expect(src).not.toMatch(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/);
    expect(src).toContain("/[\\x00-\\x20\\x7f]/");
  });
});

describe("signInPath", () => {
  it("encodes the destination into returnTo", () => {
    expect(signInPath("/app/research?ticker=NVDA")).toBe("/sign-in?returnTo=%2Fapp%2Fresearch%3Fticker%3DNVDA");
    expect(signInPath("/app", "sign-up")).toBe("/sign-up?returnTo=%2Fapp");
  });
});
