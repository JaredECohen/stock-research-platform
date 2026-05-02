"""Validate which configured models actually respond against the real APIs.

Run from `backend/`:

    python -m scripts.validate_model_access

Reads keys + model names from `app.config.settings` (which loads
`config.env` then `.env`), fires one tiny chat call per model, prints a
PASS/FAIL table, and exits non-zero if any configured model failed.

Costs roughly $0.001-0.005 total. Safe to run anytime — the calls are
3-tokens-out maximum.

Use this whenever:
- You add API keys for a new provider.
- You change a model env var (e.g. `OPENAI_PM_MODEL`) in config.env.
- A memo run starts returning empty findings and you suspect a model
  ID is no longer valid on your account.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Allow live behavior for this validation regardless of conftest's defaults.
os.environ.pop("ENABLE_LIVE_DATA", None)
os.environ.pop("USE_DEMO_DATA", None)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str: return f"\033[31m{s}\033[0m"
def _dim(s: str) -> str: return f"\033[2m{s}\033[0m"


def _result(label: str, model: str, ok: bool, detail: str = "") -> Tuple[bool, str]:
    badge = _green("OK  ") if ok else _red("FAIL")
    return ok, f"  {badge}  {label:13s} {model:25s} {detail}"


def _check_openai(api_key: str, models: Iterable[Tuple[str, str]]) -> list:
    """Each `models` entry: (label, model_id)."""
    rows: list = []
    if not api_key:
        return [(False, _dim("  (skipped — OPENAI_API_KEY not set)"))]
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        return [(False, _red(f"  openai sdk import failed: {exc}"))]

    # Import after sys.path setup so we can use the same kwarg helper that
    # the production code uses — keeps the validator honest.
    from app.agents.llm import _openai_token_kwarg

    c = OpenAI(api_key=api_key)
    for label, model in models:
        if not model:
            rows.append((False, _dim(f"  (skipped — {label} model not set)")))
            continue
        try:
            r = c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                **_openai_token_kwarg(model, 5),
            )
            tot = r.usage.total_tokens if r.usage else "?"
            rows.append(_result(label, model, True, f"({tot} tok)"))
        except Exception as exc:
            err = str(exc).splitlines()[0][:140]
            rows.append(_result(label, model, False, _red(f"— {err}")))
    return rows


def _check_anthropic(api_key: str, models: Iterable[Tuple[str, str]]) -> list:
    rows: list = []
    if not api_key:
        return [(False, _dim("  (skipped — ANTHROPIC_API_KEY not set)"))]
    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as exc:
        return [(False, _red(f"  anthropic sdk import failed: {exc}"))]
    c = Anthropic(api_key=api_key)
    for label, model in models:
        if not model:
            rows.append((False, _dim(f"  (skipped — {label} model not set)")))
            continue
        try:
            r = c.messages.create(
                model=model, max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            tot = (r.usage.input_tokens + r.usage.output_tokens) if r.usage else "?"
            rows.append(_result(label, model, True, f"({tot} tok)"))
        except Exception as exc:
            err = str(exc).splitlines()[0][:140]
            rows.append(_result(label, model, False, _red(f"— {err}")))
    return rows


def _check_gemini(api_key: str, models: Iterable[Tuple[str, str]]) -> list:
    rows: list = []
    if not api_key:
        return [(False, _dim("  (skipped — GEMINI_API_KEY not set)"))]
    try:
        from google import genai  # type: ignore
    except Exception as exc:
        return [(False, _red(f"  google-genai sdk import failed: {exc}"))]
    c = genai.Client(api_key=api_key)
    for label, model in models:
        if not model:
            rows.append((False, _dim(f"  (skipped — {label} model not set)")))
            continue
        try:
            r = c.models.generate_content(
                model=model, contents="hi", config={"max_output_tokens": 5},
            )
            meta = getattr(r, "usage_metadata", None)
            tot = (
                (meta.prompt_token_count or 0) + (meta.candidates_token_count or 0)
                if meta else "?"
            )
            rows.append(_result(label, model, True, f"({tot} tok)"))
        except Exception as exc:
            err = str(exc).splitlines()[0][:140]
            rows.append(_result(label, model, False, _red(f"— {err}")))
    return rows


def main() -> int:
    from app.config import settings
    print()
    print("Key presence:")
    print(f"  OPENAI_API_KEY:    {'SET' if settings.openai_api_key else _dim('empty')}")
    print(f"  ANTHROPIC_API_KEY: {'SET' if settings.anthropic_api_key else _dim('empty')}")
    print(f"  GEMINI_API_KEY:    {'SET' if settings.gemini_api_key else _dim('empty')}")
    print()

    all_rows: list = []

    print("OpenAI:")
    rows = _check_openai(settings.openai_api_key, [
        ("PM",     settings.openai_pm_model),
        ("Sector", settings.openai_sector_model),
        ("Tool",   settings.openai_tool_model),
        ("Macro",  settings.openai_macro_model),
        ("Strong", settings.openai_strong_model),
        ("Cheap",  settings.openai_cheap_model),
    ])
    for r in rows:
        print(r[1])
        all_rows.append(r)
    print()

    print("Anthropic:")
    rows = _check_anthropic(settings.anthropic_api_key, [
        ("Critic", settings.anthropic_critic_model),
        ("Strong", settings.anthropic_strong_model),
        ("Cheap",  settings.anthropic_cheap_model),
    ])
    for r in rows:
        print(r[1])
        all_rows.append(r)
    print()

    print("Gemini:")
    rows = _check_gemini(settings.gemini_api_key, [
        ("News",   settings.gemini_news_model),
        ("Social", settings.gemini_social_model),
        ("Long-doc", settings.gemini_longdoc_model),
    ])
    for r in rows:
        print(r[1])
        all_rows.append(r)
    print()

    fails = [r for r in all_rows if r[0] is False and "skipped" not in r[1]]
    total = sum(1 for r in all_rows if "skipped" not in r[1])
    if fails:
        print(_red(f"FAILED: {len(fails)} of {total} configured models did not respond."))
        return 1
    print(_green(f"OK: all {total} configured models responded."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
