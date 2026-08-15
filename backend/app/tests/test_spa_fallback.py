"""SPA deep links must serve the app shell, not a JSON 404.

Found 2026-08-15 by driving the deployed site: every client-side route
404'd on direct navigation.

    GET /              200
    GET /track-record  404   {"detail":"Not Found"}
    GET /screener      404
    ... every route except "/"

`StaticFiles(html=True)` serves index.html for a DIRECTORY request, but
"/track-record" looks for a FILE of that name, doesn't find one, and
falls through to FastAPI's 404. In-app navigation always worked (React
Router never touches the server), so the break was invisible to anyone
clicking through the UI — it only hit typing a URL, refreshing,
bookmarking, or following a shared link.

The wiring lives in the Dockerfile's generated `/app/server.py`, which no
test could reach. This rebuilds the same composition against the real
`app.main:app` so the behaviour is pinned even though the file is
generated at image-build time. If you change the fallback in the
Dockerfile, change `_build_spa_app` to match.
"""
from __future__ import annotations

import pytest
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from app.main import app as api_app

# Mirrors the Dockerfile heredoc. Kept as a literal copy rather than
# imported, because the real one only exists inside the built image.
_API_PREFIXES = ("api/", "health", "docs", "redoc", "openapi.json")


@pytest.fixture(scope="module")
def spa_client(tmp_path_factory):
    dist = tmp_path_factory.mktemp("dist")
    (dist / "index.html").write_text("<!doctype html><div id=root>APP_SHELL</div>")
    (dist / "favicon.ico").write_text("FAVICON")
    (dist / "assets").mkdir()
    (dist / "assets" / "main-abc123.js").write_text("console.log('bundle')")
    # A file OUTSIDE dist, to prove traversal can't reach it.
    (dist.parent / "outside_secret.txt").write_text("SECRET_TOKEN_XYZ")

    index = dist / "index.html"

    @api_app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str):
        if full_path.startswith(_API_PREFIXES):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)

    api_app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui_test")
    yield TestClient(api_app)
    # Leave the app as we found it — other test modules share this import.
    api_app.router.routes = [
        r for r in api_app.router.routes
        if getattr(r, "name", None) not in ("_spa_fallback", "ui_test")
    ]


@pytest.mark.parametrize("route", [
    "/track-record", "/screener", "/dcf-lab", "/comps", "/macro", "/settings",
])
def test_deep_link_serves_the_app_shell(spa_client, route):
    """The actual bug: these all returned {"detail":"Not Found"}."""
    resp = spa_client.get(route)
    assert resp.status_code == 200
    assert "APP_SHELL" in resp.text, f"{route} did not serve index.html"


def test_real_assets_are_still_served(spa_client):
    """The fallback must not shadow the bundle — serving index.html for a
    .js request would break the app far worse than the 404 did."""
    resp = spa_client.get("/assets/main-abc123.js")
    assert resp.status_code == 200
    assert "console.log('bundle')" in resp.text
    assert spa_client.get("/favicon.ico").text == "FAVICON"


def test_root_still_serves_index(spa_client):
    assert "APP_SHELL" in spa_client.get("/").text


def test_unknown_api_path_still_returns_json_404(spa_client):
    """A client-side `fetch('/api/typo')` must get a parseable JSON 404, not
    200 with an HTML body — otherwise every API typo surfaces as a JSON
    parse error somewhere unrelated."""
    resp = spa_client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Not Found"


def test_real_api_routes_are_unaffected(spa_client):
    """The catch-all is registered last, so genuine routes must still win."""
    assert spa_client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("attack", [
    "/../outside_secret.txt",
    "/..%2f..%2foutside_secret.txt",
    "/%2e%2e/outside_secret.txt",
    "/assets/../../outside_secret.txt",
])
def test_path_traversal_cannot_escape_the_dist_directory(spa_client, attack):
    """Serving files by path is only safe with the containment check —
    without `is_relative_to`, these read arbitrary files off the container."""
    resp = spa_client.get(attack)
    assert "SECRET_TOKEN_XYZ" not in resp.text, f"{attack} escaped dist/"
