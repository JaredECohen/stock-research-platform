"""A deterministic stand-in for the Industry Analysis analyst model.

Owner decision 1 (2026-09-24): only analyst-written editions are ever
published. The test environment has no LLM (blank keys, demo-only mode),
so without a stand-in every edition a test drains is a template — stored
audit-only, never served — and nothing downstream of publication (routes,
the PM block, the UI fixture) can be exercised against a real published
shape.

This stub is the seam. It makes the writer believe a model is configured
(``settings.has_llm``, ``llm._demo_only``) and replaces ONE function,
``industry_report_writer._llm_call``, with a "model" that returns the
writer's own deterministic interpretation of each requested section,
relabelled. Everything else runs for real: fact building, the per-section
bookkeeping (``narrative_by_section``, the ``analyst_narrative:*``
markers), the validator, the store's display rule. Because the returned
prose IS the template prose, it passes the same grounding gate the
template does — no number or causal link is invented to make a test green.

The relabel is visible on purpose: every string that said
"Deterministic edition (...)" says ``LABEL`` instead, so a fixture
captured with the stub can never be mistaken for a real analyst's words.

One addition to the template: the outlook registers ONE forecast
assumption (``registered_assumption``), anchored on the edition's first
rate anchor and holding it at its own observed value, and the base
scenario lists it — so a published edition exercises the registered-
assumption contract end to end without the stub inventing a number.

``sections`` restricts which sections the stub "writes" (the rest come
back empty, exactly as a model that skipped them) — how a test builds an
analyst edition too thin to publish.

Two entry points: ``install(monkeypatch)`` for pytest, and the
``stubbed_analyst()`` context manager for scripts (the UI fixture capture),
which must not depend on pytest.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import Any
from unittest import mock

LABEL = "Fixture analyst (stubbed model)"
MODEL = "fixture-analyst-stub"
REASON = (
    "the test environment has no LLM; the stub returns the writer's own deterministic "
    "interpretation relabelled as '" + LABEL + "', so published editions exist without "
    "inventing any analyst prose or numbers"
)

_TEMPLATE_LABELS = (
    "Deterministic edition (no analyst narrative)",
    "Deterministic edition (deterministic mode)",
)


def _relabel(obj: Any) -> Any:
    if isinstance(obj, str):
        for label in _TEMPLATE_LABELS:
            obj = obj.replace(label, LABEL)
        return obj
    if isinstance(obj, dict):
        return {k: _relabel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relabel(v) for v in obj]
    return obj


ASSUMPTION_ID = "FA1"
ASSUMPTION_HORIZON = "next 4 quarters"
ASSUMPTION_FALSIFIER = (
    "A later weekly statistics row shows this observed measure moving away from the level "
    "the assumption holds it at."
)


def registered_assumption(outlook_facts: dict[str, Any]) -> dict[str, Any] | None:
    """The one forecast assumption the stub registers in an outlook.

    The registered-assumption contract (owner decision 1) needs a published
    edition that carries one, or the assumptions table has nothing real to
    render. The stub still invents no number: it anchors on the FIRST rate
    anchor in the edition's own catalogue and assumes that observation
    simply persists — the value IS the anchor's value, rendered as a
    percentage. ``None`` when the catalogue has no rate anchor (the
    no-prices week), exactly as a model with nothing to anchor to must
    declare nothing.
    """
    anchor = next((a for a in outlook_facts.get("anchors") or [] if a.get("family") == "rate"), None)
    if anchor is None:
        return None
    value = f"{float(anchor['value']) * 100:.1f}%"
    return {
        "type": "forecast_assumption",
        "id": ASSUMPTION_ID,
        "text": (f"{LABEL} assumption: the first observed rate anchor holds at {value} over the "
                 f"{ASSUMPTION_HORIZON}, against {value} observed."),
        "value": value,
        "horizon": ASSUMPTION_HORIZON,
        "anchor": anchor["path"],
        "basis": [anchor["path"]],
        "falsifier": ASSUMPTION_FALSIFIER,
    }


def make_llm_call(sections: Iterable[str] | None = None):
    """A replacement for ``industry_report_writer._llm_call``. ``sections``
    (default: all) are the ones the stub answers for."""
    from app.agents import industry_report_writer as writer

    allowed = None if sections is None else set(sections)

    def _stub(analyst: Any, facts: dict[str, dict[str, Any]], requested: tuple[str, ...], *,
              run_id: str, generation: dict[str, Any], **_ignored: Any) -> dict[str, Any] | None:
        generation["llm_calls"] += 1
        generation["model"] = MODEL
        interp = writer._deterministic_interpretation(facts, analyst, writer.NARRATIVE_LLM)
        out = {name: _relabel(interp[name]) for name in requested
               if name in interp and (allowed is None or name in allowed)}
        if "outlook" in out:
            fa = registered_assumption(facts.get("outlook") or {})
            if fa is not None:
                out["outlook"]["claims"].append(fa)
                out["outlook"]["scenarios"]["base"]["assumption_ids"] = [ASSUMPTION_ID]
        return out or None

    return _stub


def _targets():
    from app.agents import industry_report_writer as writer
    from app.agents import llm
    from app.config import Settings
    return writer, llm, Settings


def install(monkeypatch, *, sections: Iterable[str] | None = None) -> None:
    """Install the stub for one pytest test (``monkeypatch`` undoes it)."""
    writer, llm, settings_cls = _targets()
    monkeypatch.setattr(settings_cls, "has_llm", property(lambda self: True))
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    monkeypatch.setattr(writer, "_llm_call", make_llm_call(sections))


@contextlib.contextmanager
def stubbed_analyst(*, sections: Iterable[str] | None = None) -> Iterator[None]:
    """The same stub for code that runs outside pytest."""
    writer, llm, settings_cls = _targets()
    with mock.patch.object(settings_cls, "has_llm", property(lambda self: True)), \
            mock.patch.object(llm, "_demo_only", lambda: False), \
            mock.patch.object(writer, "_llm_call", make_llm_call(sections)):
        yield
