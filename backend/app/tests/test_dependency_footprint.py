"""Guards on what the app drags into memory at import time.

Every module imported at startup is resident for the life of the
process. `pandas` alone is ~48MB, and it was in `requirements.txt` with
zero call sites anywhere in the repo. Dropping it only stays dropped if
something notices when an import sneaks back in.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys


def _app_import_probe(expr: str) -> str:
    """Import `app.main` in a clean subprocess and evaluate `expr`.

    `cwd` must be the backend package root: pytest may be invoked from
    there or from the repo root, and `import app.main` only resolves from
    the former. Deriving it from __file__ keeps this independent of how
    the suite was launched.
    """
    backend = pathlib.Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c", f"import app.main, sys; print({expr})"],
        capture_output=True, text=True, cwd=str(backend),
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


def test_pandas_is_not_imported_by_the_app():
    """pandas was removed from requirements — nothing may import it."""
    assert _app_import_probe("'pandas' in sys.modules") == "False", (
        "app.main now imports pandas; either restore the dependency in "
        "requirements.txt or keep the import lazy"
    )


def test_numpy_is_not_imported_at_startup():
    """numpy (~28MB) is used by factor analytics and the vector-search
    scoring path, but both import it lazily. Hoisting it to module scope
    would add that cost to every process, including ones that never run
    a search."""
    assert _app_import_probe("'numpy' in sys.modules") == "False", (
        "numpy is now imported at startup; keep it lazy (see "
        "services/vector_store._numpy)"
    )
