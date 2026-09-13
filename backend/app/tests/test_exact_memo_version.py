from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import memo_store
from app.tests.test_memo_store import _stub_memo


@pytest.mark.parametrize("wall", [False, True])
def test_exact_version_never_serves_latest_or_runs_research(monkeypatch, wall):
    from app.api import routes_stocks
    from app.config import settings
    monkeypatch.setattr(settings, "auth_enabled", wall)
    # This isolates routing from the separately tested account middleware.
    monkeypatch.setattr(routes_stocks, "customer_wall_on", lambda: wall)
    class Grant:
        def commit(self, db): pass
        def release(self, db): pass
    monkeypatch.setattr(routes_stocks, "authorize", lambda *a, **k: Grant())
    def forbidden(*a, **k):
        raise AssertionError("Historical GET must neither regenerate nor choose latest")
    ticker = "ZZVERSION" + str(int(wall))
    first = memo_store.save_memo(_stub_memo(ticker))
    memo_store.save_memo(_stub_memo(ticker).model_copy(update={"final_pm_view": "new version"}))
    monkeypatch.setattr(memo_store, "latest_memo", forbidden)
    monkeypatch.setattr(routes_stocks, "run_stock_memo", forbidden)
    # Keep HTTP middleware in developer mode; the handler branch still tests wall=true.
    monkeypatch.setattr(settings, "auth_enabled", False)
    with TestClient(app) as client:
        r = client.get(f"/api/stocks/{ticker}/memo?version={first.version}")
        assert r.status_code == 200, r.text
        assert r.headers["X-Memo-Version"] == str(first.version)
        assert r.json()["final_pm_view"] == "pm view"
        assert client.get(f"/api/stocks/{ticker}/memo?version=99999").status_code == 404
        assert client.get(f"/api/stocks/{ticker}/memo?version=0").status_code == 422
        assert client.get(f"/api/stocks/{ticker}/memo?version=1&ondemand=true").status_code == 422
        assert client.get(f"/api/stocks/{ticker}/memo?version=1&as_of=2020-01-01").status_code == 422


def test_version_reader_excludes_backtest_snapshots():
    snap = memo_store.save_memo(_stub_memo("ZZPAST"), as_of_date=date(2020, 1, 1))
    assert memo_store.memo_version("ZZPAST", snap.version) is None
