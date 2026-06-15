// Vitest setup — registers jest-dom matchers on Vitest's `expect` and
// unmounts rendered trees between tests (RTL's auto-cleanup only fires
// when test globals are enabled, which we keep off).
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
