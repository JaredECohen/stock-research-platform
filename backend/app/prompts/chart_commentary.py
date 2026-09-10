"""FEAT-001 — the chart-commentary prompt (slice S3).

One bounded, cheap-route call. The model is handed EXACTLY what the
chart displays — the recomputed series, every missing point with its
reason — plus short excerpts of the stored memo for each selected
company, and is asked for one thing only: how each memo's stated view
relates to what the chart shows. The observed section of the response
is computed in Python (`services/chart_commentary.observed_items`) and
is given to the model as context, never authored by it, so a number in
"Observed in the data" is always one the chart can be checked against.

Why a Python module rather than a `.md` in this package: the prompt is
mostly generated structure (tables, per-ticker blocks) with a small
fixed preamble, and its size has to be provable — `MAX_PROMPT_CHARS`
bounds the user prompt (chars/4 ≈ tokens, so 24k chars ≈ 6k tokens) and
`build_prompt` shrinks the memo excerpts until it fits. `PROMPT_VERSION`
is part of the commentary cache key: a wording change must not serve a
row generated under the old instructions.

Research-process fidelity (docs/research/README.md): observed data and
interpretation stay separate on the wire and in the prompt; memo views
are labelled as the memo's scenario, not a recommendation; a missing
value is shown as `n/a(<reason>)`, never as zero.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

PROMPT_VERSION: Final = "2026.09.1"

# ~6k tokens at the chars/4 rule of thumb. Hard ceiling on the user
# prompt; the system prompt is fixed and small.
MAX_PROMPT_CHARS: Final = 24_000

# Bounds on what one memo contributes. `MEMO_FIELD_CHARS` clips every
# excerpt field; `MEMO_BLOCK_CHARS` caps the whole block for one ticker.
# `build_prompt` halves both (down to `MIN_MEMO_FIELD_CHARS`) when the
# prompt would otherwise exceed the ceiling.
MEMO_FIELD_CHARS: Final = 280
MEMO_BLOCK_CHARS: Final = 1_400
MIN_MEMO_FIELD_CHARS: Final = 80

# What the model may return: a few sentences per memo, a few caveats.
MAX_MEMO_VIEW_PER_TICKER: Final = 3
MAX_MEMO_VIEW_TOTAL: Final = 12
MAX_MEMO_VIEW_CHARS: Final = 600
MAX_MODEL_CAVEATS: Final = 6
MAX_CAVEAT_CHARS: Final = 300

SYSTEM_PROMPT: Final = (
    "You are a buy-side research associate writing a short note for a research and "
    "education tool. You are given (1) the exact data a fundamentals chart displays, "
    "(2) observations already computed from that data, and (3) short excerpts from a "
    "stored research memo for some of the companies.\n"
    "Your only task: for each company that has a memo excerpt, say in one to three "
    "sentences how the memo's stated view relates to what the displayed data shows — "
    "where the data supports it, where it cuts against it, and what the memo could not "
    "have seen if it predates the latest period.\n"
    "Rules:\n"
    "- Use only the tickers, metrics and periods provided. Never mention any other company.\n"
    "- Do not compute or invent numbers. If you cite a figure, it must appear verbatim in "
    "the data or the observations.\n"
    "- Treat a memo's view as that memo's scenario, not as a recommendation. Never tell "
    "the reader to buy, sell or hold.\n"
    "- A value shown as n/a is unknown, not zero and not neutral. Say so if it matters.\n"
    "- A memo marked stale may be out of date; say what it could not have seen.\n"
    "Return ONLY a JSON object of the form "
    '{"memo_view": [{"ticker": "<TICKER>", "text": "<sentences>"}], "caveats": ["<short caveat>"]}. '
    "Omit tickers that have no memo excerpt. Keep caveats for things the reader should "
    "know about the comparison itself."
)


@dataclass(frozen=True)
class MetricLine:
    """One metric as the prompt describes it (from the catalog)."""
    id: str
    label: str
    unit_type: str
    kind: str
    formula_text: str


@dataclass(frozen=True)
class SeriesLine:
    """One displayed series, values already formatted by the caller
    (`value` is the display string or `n/a(<reason>)`)."""
    ticker: str
    metric: str
    unit_type: str
    currency: str | None
    indexed: bool
    points: tuple[tuple[str, str], ...]   # (period, rendered value)
    stale: bool
    stale_reason: str | None


@dataclass(frozen=True)
class MemoBlock:
    """The bounded memo excerpt for one ticker. `fields` is ordered:
    the first entries survive longest when the block is shrunk."""
    ticker: str
    version: int
    generated_at: str          # ISO date
    stale: bool
    stale_reason: str | None
    fields: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    return text[: max(0, n - 1)].rstrip() + "…"


def render_memo_block(block: MemoBlock, *, field_chars: int, block_chars: int) -> tuple[str, bool]:
    """One ticker's memo section. Returns `(text, truncated)` — truncated
    is True when any field was clipped or dropped, so the service can
    tell the reader the model saw an abridged memo."""
    head = (
        f"[{block.ticker}] stored memo v{block.version}, generated {block.generated_at}"
        + (f"; STALE: {block.stale_reason}" if block.stale and block.stale_reason else
           ("; STALE" if block.stale else ""))
    )
    lines = [head]
    truncated = False
    used = len(head)
    for name, value in block.fields:
        if not value:
            continue
        clipped = _clip(value, field_chars)
        truncated = truncated or clipped != " ".join(value.split())
        line = f"  {name}: {clipped}"
        if used + len(line) + 1 > block_chars:
            truncated = True
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines), truncated


def _render_series(lines: Iterable[SeriesLine]) -> str:
    out: list[str] = []
    for s in lines:
        unit = s.unit_type + (f" {s.currency}" if s.currency else "") + (", indexed to 100" if s.indexed else "")
        stale = f" [stale: {s.stale_reason}]" if s.stale and s.stale_reason else (" [stale]" if s.stale else "")
        values = ", ".join(f"{period}={value}" for period, value in s.points)
        out.append(f"{s.ticker} {s.metric} ({unit}){stale}: {values}")
    return "\n".join(out)


def _render_metrics(metrics: Iterable[MetricLine]) -> str:
    return "\n".join(
        f"- {m.id} ({m.label}; {m.unit_type}; {m.kind}): {m.formula_text}" for m in metrics
    )


@dataclass(frozen=True)
class BuiltPrompt:
    text: str
    field_chars: int
    truncated_tickers: tuple[str, ...]
    # True when the ceiling could not be met even at the minimum memo
    # size and the memo section was dropped entirely — the caller must
    # say so rather than let the model answer without the memos.
    memos_dropped: bool

    @property
    def estimated_tokens(self) -> int:
        return len(self.text) // 4


def build_prompt(
    *,
    tickers: list[str],
    metrics: list[MetricLine],
    periods: list[str],
    normalize: str,
    series: list[SeriesLine],
    observed: list[str],
    memos: Mapping[str, MemoBlock | None],
    unavailable: Mapping[str, str],
    max_chars: int = MAX_PROMPT_CHARS,
) -> BuiltPrompt:
    """Assemble the user prompt within `max_chars`.

    The data section is never cut — the model must see exactly what the
    chart shows — so the budget is met by shrinking the memo excerpts:
    field and block caps are halved until the prompt fits or the caps
    reach `MIN_MEMO_FIELD_CHARS`; beyond that the memo section is dropped
    and `memos_dropped` is reported. At the largest selection the route
    allows (5 companies × 4 metrics × the stored history) the data
    section alone stays well under the ceiling; the test suite pins that.
    """
    header = (
        "DISPLAYED SELECTION\n"
        f"tickers: {', '.join(tickers)}\n"
        f"periods (fiscal years, oldest first): {', '.join(periods) if periods else '(none)'}\n"
        f"normalize: {normalize}\n"
        "metrics:\n" + _render_metrics(metrics)
    )
    unavailable_text = ""
    if unavailable:
        unavailable_text = "\nNO STORED HISTORY (nothing displayed for these): " + "; ".join(
            f"{t}: {reason}" for t, reason in unavailable.items()
        )
    data = "DISPLAYED DATA (n/a(<reason>) = no value; reasons are closed-set codes)\n" + (
        _render_series(series) or "(no series)"
    )
    obs = "OBSERVED IN THE DATA (computed, not to be restated as your own)\n" + (
        "\n".join(f"- {o}" for o in observed) or "- (no observations)"
    )
    task = (
        "TASK\nFor each ticker with a memo excerpt above, write memo_view sentences relating "
        "that memo's view to the displayed data. Return only the JSON object described in the "
        "system message."
    )

    fixed = "\n\n".join([header + unavailable_text, data, obs])
    field_chars = MEMO_FIELD_CHARS
    block_chars = MEMO_BLOCK_CHARS
    while True:
        blocks: list[str] = []
        truncated: list[str] = []
        for t in tickers:
            block = memos.get(t)
            if block is None:
                blocks.append(f"[{t}] no stored memo — nothing to relate; omit from memo_view")
                continue
            text, was_truncated = render_memo_block(block, field_chars=field_chars, block_chars=block_chars)
            blocks.append(text)
            if was_truncated:
                truncated.append(t)
        memo_section = "STORED MEMO EXCERPTS (each memo's own scenario, not advice)\n" + "\n".join(blocks)
        prompt = "\n\n".join([fixed, memo_section, task])
        if len(prompt) <= max_chars:
            return BuiltPrompt(prompt, field_chars, tuple(truncated), memos_dropped=False)
        if field_chars <= MIN_MEMO_FIELD_CHARS:
            break
        field_chars = max(MIN_MEMO_FIELD_CHARS, field_chars // 2)
        block_chars = max(MIN_MEMO_FIELD_CHARS * 3, block_chars // 2)

    # Could not fit even the smallest excerpts: the data must win.
    prompt = "\n\n".join([fixed, "STORED MEMO EXCERPTS\n(omitted: prompt budget exhausted by the data)", task])
    return BuiltPrompt(prompt[:max_chars], field_chars, tuple(memos), memos_dropped=True)


def parse_output(raw: Any) -> tuple[list[tuple[str, str]], list[str]] | None:
    """Shape-check the model's JSON: `(memo_view pairs, caveats)` or None
    when the object is not usable at all. Selection membership and the
    existence of a memo are checked by the service, not here."""
    if not isinstance(raw, dict):
        return None
    items = raw.get("memo_view")
    if not isinstance(items, list):
        return None
    pairs: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = item.get("ticker")
        text = item.get("text")
        if not isinstance(ticker, str) or not isinstance(text, str):
            continue
        ticker = ticker.strip().upper()
        text = _clip(text, MAX_MEMO_VIEW_CHARS)
        if ticker and text:
            pairs.append((ticker, text))
    caveats_raw = raw.get("caveats")
    caveats: list[str] = []
    if isinstance(caveats_raw, list):
        for c in caveats_raw:
            if isinstance(c, str) and c.strip():
                caveats.append(_clip(c, MAX_CAVEAT_CHARS))
            if len(caveats) >= MAX_MODEL_CAVEATS:
                break
    return pairs, caveats
