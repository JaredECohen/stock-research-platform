import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import RateLimitNotice from "@/components/RateLimitNotice";
import type { RateLimitRefusal } from "@/types";

const LIMITED: RateLimitRefusal = { code: "rate_limited", scope: "user:llm_light", retry_after: 3, window_seconds: 60, message: "" };

describe("RateLimitNotice", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("keeps the retry disabled until retry_after elapses, then enables it and calls onRetry", () => {
    const onRetry = vi.fn();
    render(<RateLimitNotice refusal={LIMITED} onRetry={onRetry} preservedNote="Your message is kept in the box below." />);
    const box = screen.getByTestId("rate-limit-notice");
    expect(box).toHaveTextContent("Too many requests for AI requests.");
    expect(box).toHaveTextContent("Your message is kept in the box below.");
    const button = screen.getByRole("button", { name: "Retry in 3s" });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onRetry).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByRole("button", { name: "Retry in 1s" })).toBeDisabled();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    const ready = screen.getByRole("button", { name: "Retry" });
    expect(ready).toBeEnabled();
    expect(box).toHaveTextContent("You can retry now.");
    fireEvent.click(ready);
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("is immediately retryable when retry_after is 0 and describes concurrency limits", () => {
    const onRetry = vi.fn();
    render(<RateLimitNotice refusal={{ ...LIMITED, code: "concurrency_limited", retry_after: 0 }} onRetry={onRetry} />);
    expect(screen.getByTestId("rate-limit-notice")).toHaveTextContent("Another request of this kind is still running");
    const button = screen.getByRole("button", { name: "Retry" });
    expect(button).toBeEnabled();
    fireEvent.click(button);
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("restarts the countdown when a new refusal arrives", () => {
    const view = render(<RateLimitNotice refusal={LIMITED} onRetry={() => {}} />);
    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(screen.getByRole("button", { name: "Retry" })).toBeEnabled();
    view.rerender(<RateLimitNotice refusal={{ ...LIMITED, retry_after: 5 }} onRetry={() => {}} />);
    expect(screen.getByRole("button", { name: "Retry in 5s" })).toBeDisabled();
  });
});
