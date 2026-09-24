import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import Settings from "@/pages/Settings";
import { okJson, renderWithProviders, stubFetch } from "@/test/providers";

// FIX-008b: `enable_vector_search=false` was read as "semantic retrieval is
// off", but no retrieval code reads the flag. The backend now sends a note
// beside the boolean; the page must show it, and still render older payloads.
const NOTE =
  "Not consulted by retrieval: the filing and earnings analysts always search the vector index first; " +
  "the filing analyst falls back to BM25 keyword search only when that search returns nothing.";

function status(extra: Record<string, unknown> = {}) {
  return {
    mode: "live",
    providers: {},
    missing_api_keys: [],
    llm_configured: true,
    feature_flags: { enable_agent_critic: true, enable_vector_search: false },
    ...extra,
  };
}

describe("Settings feature flags", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows the backend's note under a flag that no code consults, and tolerates older payloads", async () => {
    stubFetch([["/api/providers/status", () => okJson(status({ feature_flag_notes: { enable_vector_search: NOTE } }))]]);
    const view = renderWithProviders(<Settings />);
    expect(await screen.findByTestId("flag-note-enable_vector_search")).toHaveTextContent(NOTE);
    expect(screen.getByText("enable_vector_search")).toBeInTheDocument();
    expect(screen.queryByTestId("flag-note-enable_agent_critic")).toBeNull();
    view.unmount();

    // A backend that predates the notes still renders every flag, note-free.
    stubFetch([["/api/providers/status", () => okJson(status())]]);
    renderWithProviders(<Settings />);
    expect(await screen.findByText("enable_vector_search")).toBeInTheDocument();
    expect(screen.queryByTestId("flag-note-enable_vector_search")).toBeNull();
  });
});
