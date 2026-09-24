import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import HistoryPicker from "@/components/industries/HistoryPicker";
import * as fx from "@/test/fixtures/industry";

describe("HistoryPicker", () => {
  it("lists every edition on file plus the 'latest published' entry", () => {
    render(<HistoryPicker history={fx.history} value="latest" onSelect={() => {}} />);
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(fx.history.items.length + 1);
    expect(options[0]).toHaveTextContent("Latest published");
  });

  it("labels an edition that is no longer the published one with its status", () => {
    render(<HistoryPicker history={fx.history} value="latest" onSelect={() => {}} />);
    const superseded = fx.history.items.find((i) => !i.is_latest_good)!;
    expect(screen.getByRole("option", { name: new RegExp(`v${superseded.version}.*${superseded.status}`) })).toBeInTheDocument();
  });

  it("hands the chosen version back as a string the URL can carry", () => {
    const onSelect = vi.fn();
    render(<HistoryPicker history={fx.history} value="latest" onSelect={onSelect} />);
    const target = fx.history.items[fx.history.items.length - 1];
    fireEvent.change(screen.getByTestId("history-select"), { target: { value: String(target.version) } });
    expect(onSelect).toHaveBeenCalledWith(String(target.version));
  });

  it("counts the editions a capped response dropped", () => {
    const h = fx.clone(fx.history);
    h.count = 26;
    h.limit = 26;
    h.truncated = 14;
    render(<HistoryPicker history={h} value="latest" onSelect={() => {}} />);
    expect(screen.getByTestId("history-count")).toHaveTextContent("14 older not shown (limit 26)");
  });

  it("explains the version gap audit-only editions leave, and stays quiet when there is none", () => {
    // The captured history holds no withheld edition; the count line says nothing about one.
    expect(fx.history.withheld).toBe(0);
    render(<HistoryPicker history={fx.history} value="latest" onSelect={() => {}} />);
    expect(screen.getByTestId("history-count")).not.toHaveTextContent("audit only");

    const h = fx.clone(fx.history);
    h.withheld = 2;
    render(<HistoryPicker history={h} value="latest" onSelect={() => {}} />);
    expect(screen.getAllByTestId("history-count")[1]).toHaveTextContent(
      "2 editions were kept for audit only (no validated analyst edition) and are not published, so version numbers skip.",
    );
  });

  it("reports the last refresh attempt, or says none is on file", () => {
    render(<HistoryPicker history={fx.history} value="latest" onSelect={() => {}} />);
    expect(screen.getByTestId("history-count")).toHaveTextContent(`Last refresh attempt: ${fx.history.last_attempt!.status}`);

    const none = fx.clone(fx.history);
    none.last_attempt = null;
    render(<HistoryPicker history={none} value="latest" onSelect={() => {}} />);
    expect(screen.getAllByTestId("history-count")[1]).toHaveTextContent("No refresh attempt is recorded");
  });

  it("says the group has no editions rather than rendering an empty select", () => {
    const empty = fx.clone(fx.history);
    empty.items = [];
    render(<HistoryPicker history={empty} value="latest" onSelect={() => {}} />);
    expect(screen.getByTestId("history-empty")).toHaveTextContent("no editions on file");
  });
});
