"""W7 §8.3 — the corpus repair is targeted, identity-preserving, capped and
resumable.

It re-embeds unusable chunks in place (ids and text unchanged, provenance
stamped), indexes zero-chunk in-scope sources without ever running a
post-pass, and stops *before* any cap — dollars from the provider's reported
usage, rows, sources, added bytes, and an absolute database-size ceiling.
Orphan, demo and good rows are never touched.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from app.config import settings
from app.models import DocChunk
from app.services import corpus_inventory as ci
from app.services import corpus_repair as cr
from app.services import embeddings as emb_svc
from app.services import filing_memory
from app.tests.corpus_fixtures import NOW, corpus_db, embedding_value_refs, live_openai  # noqa: F401
from app.tests.test_corpus_inventory import seed

GENEROUS = {"max_usd": 5.0, "max_rows": 100_000, "max_added_mb": 500.0}
TARGETS = ("hash", "legacy", "other", "no_emb")  # the 7 re-embed targets in `seed`
UNTOUCHABLE = ("ok", "orphan", "demo_ticker", "demo_accession")


def _target_ids(ids):
    return sorted(i for k in TARGETS for i in ids[k])


def _snapshot(db):
    return {i: (r.text, r.embedding_dim, r.embedding_model, dict(r.meta or {}), r.source_id)
            for i, r in db.rows().items()}


def test_plan_writes_nothing(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    before = _snapshot(corpus_db)
    corpus_db.statements.clear()
    planned = cr.plan(now=NOW)
    receipt = cr.run(mode="plan", now=NOW)
    writes = [s for s in corpus_db.statements if s.lstrip().split()[0].upper() in ("INSERT", "UPDATE", "DELETE")]
    assert writes == [] and _snapshot(corpus_db) == before
    assert live_openai.calls == []  # a plan never spends
    assert planned["targets"]["reembed"]["rows"] == 7 and planned["targets"]["index-missing"]["sources"] == 3
    assert receipt["exit_code"] == 0 and receipt["targets"] == planned["targets"]


def test_reembed_in_place_preserves_ids_text_and_records_provenance(corpus_db, live_openai):  # noqa: F811
    ids, _ = seed(corpus_db)
    before = _snapshot(corpus_db)
    corpus_db.statements.clear()
    res = cr.reembed(**GENEROUS, receipt="cr-test-1", now=NOW)
    # The stream reads text and sizes, never the stored vectors.
    selects = [s for s in corpus_db.statements if s.lstrip().upper().startswith("SELECT")]
    assert selects and not [s for s in selects if embedding_value_refs(s)]
    assert res["stopped_reason"] is None and res["rows_repaired"] == 7 and res["skipped_changed"] == 0
    after = _snapshot(corpus_db)
    assert set(after) == set(before)  # no row added or removed: ids preserved
    for i in _target_ids(ids):
        text, dim, model, meta, source_id = after[i]
        assert (text, source_id) == (before[i][0], before[i][4])
        assert (dim, model) == (emb_svc.EMBEDDING_DIM, emb_svc.EMBEDDING_MODEL)
        assert meta["k"] == "v"  # existing meta kept
        assert meta["embedding_repair"] == {
            "from_model": before[i][2], "from_dim": before[i][1], "at": NOW.isoformat(), "receipt": "cr-test-1",
        }
    with corpus_db.Session() as db:
        assert all(len(r.embedding) == emb_svc.EMBEDDING_DIM for r in db.query(DocChunk).filter(
            DocChunk.id.in_(_target_ids(ids))))
    assert res["chunk_id_ranges"] == [[min(_target_ids(ids)), max(_target_ids(ids))]]
    assert res["tokens"] == 7 * live_openai.tokens_per_text
    # Resumable and idempotent: nothing is left to repair.
    again = cr.reembed(**GENEROUS, now=NOW)
    assert again["rows_repaired"] == 0 and again["stopped_reason"] is None


def test_ok_orphan_demo_rows_untouched(corpus_db, live_openai):  # noqa: F811
    ids, _ = seed(corpus_db)
    before = _snapshot(corpus_db)
    cr.reembed(**GENEROUS, now=NOW)
    after = _snapshot(corpus_db)
    for key in UNTOUCHABLE:
        for i in ids[key]:
            assert after[i] == before[i], key


def test_concurrent_change_guard_skips(corpus_db, live_openai, monkeypatch):  # noqa: F811
    ids, _ = seed(corpus_db)
    raced = ids["hash"][1]  # mid-range, so the receipt's id ranges must show the gap
    real_embed = emb_svc.embed

    def embed_while_reindexed(texts, **kwargs):
        # A poller re-indexes this row between our read and our write.
        with corpus_db.Session() as db:
            row = db.get(DocChunk, raced)
            row.embedding, row.embedding_dim, row.embedding_model = [0.9] * 1536, 1536, "fresh"
            row.meta = {"fresh": True}
            db.commit()
        return real_embed(texts, **kwargs)

    monkeypatch.setattr(emb_svc, "embed", embed_while_reindexed)
    res = cr.reembed(**GENEROUS, now=NOW)
    assert res["skipped_changed"] == 1 and res["rows_repaired"] == 6
    row = corpus_db.rows()[raced]
    assert (row.embedding_model, row.meta) == ("fresh", {"fresh": True})
    # The receipt claims only the rows it rewrote.
    targets = _target_ids(ids)
    assert res["chunk_id_ranges"] == [[targets[0], raced - 1], [raced + 1, targets[-1]]]


def _repair_with(corpus_db, **caps):  # noqa: F811
    return cr.reembed(**{**GENEROUS, **caps}, now=NOW, batch_size=2)


def test_caps_stop_before_exceeding_rows(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    res = _repair_with(corpus_db, max_rows=3)
    assert res["stopped_reason"] == cr.STOP_MAX_ROWS and res["rows_repaired"] == 3


def test_caps_stop_before_exceeding_usd(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    cap = emb_svc.tokens_to_usd(45)
    res = _repair_with(corpus_db, max_usd=cap)
    assert res["stopped_reason"] == cr.STOP_MAX_USD
    assert 0 < res["rows_repaired"] < 7 and emb_svc.tokens_to_usd(res["tokens"]) <= cap


def test_caps_stop_before_exceeding_bytes(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    # Every re-embedded row is charged gross: 31 000 + 6 160 = 37 160 bytes.
    # Batch 1 = two hash rows: 74 320. Batch 2 = a hash and a 3072-dim legacy
    # row. Netting out the old values used to price the legacy row at
    # −24 280 (it "shrinks"), so batch 2 looked like +7 760 and ran; on
    # Postgres the old value stays on disk until VACUUM, so it really adds
    # 74 320 more, crossing 100 000.
    res = _repair_with(corpus_db, max_added_mb=100_000 / ci.MB)
    assert res["stopped_reason"] == cr.STOP_MAX_ADDED_MB
    assert res["rows_repaired"] == 2 and res["added_bytes_est"] == 74_320


def test_db_ceiling_refuses(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    before = _snapshot(corpus_db)
    with corpus_db.Session() as db:
        size = ci.database_bytes(db)
    assert size and size > 0  # measured, not assumed
    ceiling_mb = size / ci.MB  # already full: any growth would cross it
    res = cr.reembed(**GENEROUS, max_db_mb=ceiling_mb, now=NOW)
    assert res["stopped_reason"] == cr.STOP_MAX_DB_MB and res["rows_repaired"] == 0
    out = cr.index_missing(max_sources=10, max_usd=1.0, max_added_mb=100, max_db_mb=ceiling_mb, now=NOW)
    assert out["stopped_reason"] == cr.STOP_MAX_DB_MB and out["sources_indexed"] == 0
    assert live_openai.calls == [] and _snapshot(corpus_db) == before
    # The CLI path: --apply without a ceiling is refused, with one it stops (exit 1).
    assert cr.run(mode="apply", klass="reembed", now=NOW)["exit_code"] == 2
    receipt = cr.run(mode="apply", klass="index-missing", max_db_mb=ceiling_mb, now=NOW)
    assert receipt["exit_code"] == 1 and receipt["stopped_reason"] == cr.STOP_MAX_DB_MB


def test_usage_tokens_drive_the_cap(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    # The provider bills 1 000 tokens a text, far over the rows' token_count
    # (5-19). Estimates alone would repair all 7 rows for 95 tokens.
    live_openai.tokens_per_text = 1_000
    res = _repair_with(corpus_db, max_usd=emb_svc.tokens_to_usd(1_500))
    # The documented bound: only the first batch billed above its estimate
    # can overshoot (here 2 000 against 1 500); billing is known only after
    # the call. The run stops there instead of repairing all 7.
    assert res["tokens"] == 2_000 and res["rows_repaired"] == 2
    assert res["stopped_reason"] == cr.STOP_MAX_USD
    # A response without `usage` is still charged, at our own count.
    live_openai.tokens_per_text = None
    res = cr.reembed(**GENEROUS, now=NOW)
    assert res["tokens_estimated"] > 0 and res["tokens"] == res["tokens_estimated"]


def test_observed_billing_ratio_prevents_a_second_overshoot(corpus_db, live_openai):  # noqa: F811
    """Batch 1 (rows 5+5 tokens) bills 20: twice our estimate. Batch 2 is
    estimated at 5+11=16, which at face value fits a 39-token cap (20+16)
    but bills 20 more, to 40. Scaled by the observed ratio it projects to
    52 and the run stops within the cap."""
    seed(corpus_db)
    cap = emb_svc.tokens_to_usd(39)
    res = _repair_with(corpus_db, max_usd=cap)
    assert res["stopped_reason"] == cr.STOP_MAX_USD
    assert res["tokens"] == 20 and res["rows_repaired"] == 2
    assert emb_svc.tokens_to_usd(res["tokens"]) <= cap


@pytest.mark.parametrize("cap", ["max_usd", "max_added_mb", "max_rows", "max_sources", "max_db_mb"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_non_finite_or_negative_caps_are_refused(corpus_db, live_openai, monkeypatch, capsys, cap, bad):  # noqa: F811
    """`nan < 0` is False and nothing is ever `> inf`: such a cap passed the
    sign check and then never bound, so `--max-db-mb inf` satisfied the
    mandatory ceiling. A negative one escaped as a traceback (exit 1, which
    means "stopped by a cap, resumable") after the census had been built."""
    seed(corpus_db)
    before = _snapshot(corpus_db)
    klass = "index-missing" if cap == "max_sources" else "reembed"
    kwargs = {"max_usd": 1.0, "max_rows": 100, "max_sources": 10, "max_added_mb": 100.0,
              "max_db_mb": 10_000.0, cap: bad}
    monkeypatch.setattr(cr.inv_svc, "build_inventory", lambda **kw: pytest.fail("census built before refusing"))
    receipt = cr.run(mode="apply", klass=klass, now=NOW, **kwargs)
    assert receipt["exit_code"] == 2 and cap in receipt["refused"]
    json.dumps(receipt, allow_nan=False)  # a strict-JSON receipt, even echoing a NaN cap
    with pytest.raises(ValueError):
        if klass == "reembed":
            cr.reembed(max_usd=kwargs["max_usd"], max_rows=kwargs["max_rows"],
                       max_added_mb=kwargs["max_added_mb"], max_db_mb=kwargs["max_db_mb"], now=NOW)
        else:
            cr.index_missing(max_sources=kwargs["max_sources"], max_usd=kwargs["max_usd"],
                             max_added_mb=kwargs["max_added_mb"], max_db_mb=kwargs["max_db_mb"], now=NOW)
    assert live_openai.calls == [] and _snapshot(corpus_db) == before
    if cap in ("max_rows", "max_sources") or math.isfinite(bad):
        return  # argparse's int type already rejects nan/inf; -1 is covered above
    # Through the CLI: argparse's float accepts "nan" and "inf"; the run
    # refuses them and the printed line stays strict JSON.
    flag = "--" + cap.replace("_", "-")
    assert _cli().main(["--apply", "--class", klass, "--max-db-mb", "10000", flag, str(bad)]) == 2
    line = capsys.readouterr().out.strip().splitlines()[-1]
    json.loads(line.split(" ", 1)[1], parse_constant=lambda c: pytest.fail(f"bare {c} in the receipt"))


def test_db_ceiling_zero_is_refused(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    receipt = cr.run(mode="apply", klass="reembed", max_db_mb=0, now=NOW)
    assert receipt["exit_code"] == 2 and "max_db_mb" in receipt["refused"]


def _growing_db(monkeypatch, step=200_000, base=10 * ci.MB):
    """`_measure_db` reporting a database that grows `step` bytes per read."""
    reads = []

    def measure():
        reads.append(base + step * len(reads))
        return reads[-1]

    monkeypatch.setattr(cr, "_measure_db", measure)
    return reads


def test_db_ceiling_is_remeasured_before_every_batch(corpus_db, live_openai, monkeypatch):  # noqa: F811
    """Estimates miss real growth (dead tuples, TOAST, index pages); only the
    measured size sees it, so it is read before every batch — a size read
    once at the start would let a long run grow far past the ceiling."""
    seed(corpus_db)
    reads = _growing_db(monkeypatch)
    ceiling_mb = (10 * ci.MB + 220_000) / ci.MB
    # Batch 1 projects 10 MB + 74 320: fits. Batch 2 reads 10 MB + 200 000,
    # projects 274 320 over the base: past the ceiling.
    res = _repair_with(corpus_db, max_db_mb=ceiling_mb)
    assert res["stopped_reason"] == cr.STOP_MAX_DB_MB and res["rows_repaired"] == 2 and len(reads) == 2


def test_db_ceiling_is_remeasured_before_every_source(corpus_db, live_openai, monkeypatch):  # noqa: F811
    _, sources = seed(corpus_db)
    reads = _growing_db(monkeypatch)
    ceiling_mb = (10 * ci.MB + 220_000) / ci.MB
    # Source 1 (2 est. chunks, 79 320) fits; source 2 reads +200 000 and
    # adds 39 660: past the ceiling.
    res = cr.index_missing(max_sources=10, max_usd=1.0, max_added_mb=100, max_db_mb=ceiling_mb, now=NOW)
    assert res["stopped_reason"] == cr.STOP_MAX_DB_MB and len(reads) == 2
    assert res["indexed"] == [f"AAA:filing:{sources['f_indexable']}"]
    assert res["deferred"] == [f"AAA:transcript:{sources['t_indexable']}", f"CCC:filing:{sources['f_memo_scope']}"]


def test_unmeasurable_db_with_a_ceiling_stops_before_spending(corpus_db, live_openai, monkeypatch):  # noqa: F811
    seed(corpus_db)
    before = _snapshot(corpus_db)
    monkeypatch.setattr(cr, "_measure_db", lambda: None)
    res = cr.reembed(**GENEROUS, max_db_mb=10_000, now=NOW)
    out = cr.index_missing(max_sources=10, max_usd=1.0, max_added_mb=100, max_db_mb=10_000, now=NOW)
    assert res["stopped_reason"] == out["stopped_reason"] == cr.STOP_DB_SIZE_UNKNOWN
    assert live_openai.calls == [] and _snapshot(corpus_db) == before


def test_reembed_outage_stops_and_is_resumable(corpus_db, live_openai, monkeypatch):  # noqa: F811
    """An outage mid-run keeps every committed batch, stops at once (no call
    per remaining batch), exits 1 with a receipt, and the next run finishes."""
    ids, _ = seed(corpus_db)
    real_create = live_openai.create

    def create_then_fail(**kw):
        if live_openai.calls:
            live_openai.calls.append(list(kw["input"]))
            raise ConnectionError("provider down")
        return real_create(**kw)

    monkeypatch.setattr(live_openai, "create", create_then_fail)
    res = _repair_with(corpus_db)
    assert res["stopped_reason"] == cr.STOP_EMBEDDING_UNAVAILABLE
    assert res["rows_repaired"] == 2 and len(live_openai.calls) == 2
    unusable = [i for i in _target_ids(ids) if corpus_db.rows()[i].embedding_dim != emb_svc.EMBEDDING_DIM
                or corpus_db.rows()[i].embedding is None]
    assert len(unusable) == 5
    receipt = cr.run(mode="apply", klass="reembed", max_db_mb=10_000, now=NOW)
    assert receipt["exit_code"] == 1 and receipt["stopped_reason"] == cr.STOP_EMBEDDING_UNAVAILABLE
    assert receipt["results"]["rows_repaired"] == 0 and "provider down" not in json.dumps(receipt)
    monkeypatch.setattr(live_openai, "create", real_create)
    again = cr.reembed(**GENEROUS, now=NOW)
    assert again["rows_repaired"] == 5 and again["stopped_reason"] is None


def test_index_missing_caps_stop_before_exceeding_usd(corpus_db, live_openai):  # noqa: F811
    _, sources = seed(corpus_db)
    everything = [f"AAA:filing:{sources['f_indexable']}", f"AAA:transcript:{sources['t_indexable']}",
                  f"CCC:filing:{sources['f_memo_scope']}"]
    # The newest source is estimated at ceil(400 × 1.3) = 520 tokens.
    res = cr.index_missing(max_sources=10, max_usd=emb_svc.tokens_to_usd(519), max_added_mb=100, now=NOW)
    assert res["stopped_reason"] == cr.STOP_MAX_USD and res["sources_indexed"] == 0
    assert res["deferred"] == everything and live_openai.calls == []
    # Room for the first two (520, then 65 on top of what was billed), not
    # the 1 170-token third.
    res = cr.index_missing(max_sources=10, max_usd=emb_svc.tokens_to_usd(600), max_added_mb=100, now=NOW)
    assert res["stopped_reason"] == cr.STOP_MAX_USD and res["indexed"] == everything[:2]
    assert res["deferred"] == everything[2:]


def test_index_missing_caps_stop_before_exceeding_bytes(corpus_db, live_openai):  # noqa: F811
    _, sources = seed(corpus_db)
    # The newest source: 2 est. chunks × (31 000 + 6 160 + 2 500) = 79 320.
    res = cr.index_missing(max_sources=10, max_usd=1.0, max_added_mb=79_319 / ci.MB, now=NOW)
    assert res["stopped_reason"] == cr.STOP_MAX_ADDED_MB and res["sources_indexed"] == 0
    assert len(res["deferred"]) == 3 and live_openai.calls == []


def test_index_missing_rechecks_chunks_and_isolates_failures(corpus_db, live_openai, monkeypatch):  # noqa: F811
    ids, sources = seed(corpus_db)
    with corpus_db.Session() as db:
        stale = next(s for s in ci.missing_sources(db, now=NOW) if s.id == sources["f_indexable"])
    # A census that is out of date: filing `ok[0]`'s source already has chunks.
    with corpus_db.Session() as db:
        has_chunks = db.get(DocChunk, ids["ok"][0]).source_id
    real_missing = ci.missing_sources
    monkeypatch.setattr(ci, "missing_sources", lambda db, **kw: [
        ci.MissingSource("filing", has_chunks, "AAA", stale.date, 400, "indexable"), *real_missing(db, **kw)])
    before = _snapshot(corpus_db)
    real_index = filing_memory.index_filing

    def index_or_fail(row):
        if row.id == sources["f_indexable"]:
            raise ValueError("unparseable section")
        return real_index(row)

    monkeypatch.setattr(filing_memory, "index_filing", index_or_fail)
    res = cr.index_missing(max_sources=2, max_usd=1.0, max_added_mb=100, now=NOW)
    # The stale candidate is skipped without a slot; the failing one is named
    # by type and does not stop the rest.
    assert res["skipped_changed"] == [f"AAA:filing:{has_chunks}"]
    assert res["failures"] == [f"AAA:filing:{sources['f_indexable']}:ValueError"]
    assert res["indexed"] == [f"AAA:transcript:{sources['t_indexable']}"]
    after = _snapshot(corpus_db)
    assert {i: v for i, v in after.items() if i in before} == before  # good chunks untouched


def test_apply_without_a_class_is_refused(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    before = _snapshot(corpus_db)
    assert _cli().main(["--apply", "--max-db-mb", "10000"]) == 2  # --class defaults to "all"
    assert live_openai.calls == [] and _snapshot(corpus_db) == before


def test_apply_refuses_without_key(corpus_db, monkeypatch):  # noqa: F811
    seed(corpus_db)
    before = _snapshot(corpus_db)
    monkeypatch.setattr(settings, "openai_api_key", "")
    receipt = cr.run(mode="apply", klass="reembed", max_db_mb=10_000, now=NOW)
    assert receipt["exit_code"] == 2 and "no semantic embedding provider" in receipt["refused"]
    # Demo-only mode with a key would write hash vectors: refused too.
    monkeypatch.setattr(settings, "openai_api_key", "test-not-a-secret")
    monkeypatch.setattr(settings, "use_demo_data", True)
    monkeypatch.setattr(settings, "enable_live_data", False)
    assert cr.run(mode="apply", klass="index-missing", max_db_mb=10_000, now=NOW)["exit_code"] == 2
    with pytest.raises(cr.RepairRefused):
        cr.reembed(**GENEROUS, now=NOW)
    assert _snapshot(corpus_db) == before


def test_index_missing_targets_indexable_newest_first_and_never_calls_post_pass(
    corpus_db, live_openai, monkeypatch,  # noqa: F811
):
    _, sources = seed(corpus_db)
    monkeypatch.setattr(filing_memory, "post_pass", lambda *a, **k: pytest.fail("index_missing ran a post-pass"))
    res = cr.index_missing(max_sources=2, max_usd=1.0, max_added_mb=100, now=NOW)
    assert res["indexed"] == [f"AAA:filing:{sources['f_indexable']}", f"AAA:transcript:{sources['t_indexable']}"]
    assert res["deferred"] == [f"CCC:filing:{sources['f_memo_scope']}"]
    assert res["stopped_reason"] == cr.STOP_MAX_SOURCES
    with corpus_db.Session() as db:
        written = (db.query(DocChunk).filter_by(source_type="filing", source_id=sources["f_indexable"]).all()
                   + db.query(DocChunk).filter_by(source_type="transcript", source_id=sources["t_indexable"]).all())
        assert written and {r.embedding_dim for r in written} == {emb_svc.EMBEDDING_DIM}
        for key in ("f_old", "f_fetch_error", "f_empty", "f_tier", "f_not_company", "f_demo", "t_old"):
            kind = "transcript" if key.startswith("t_") else "filing"
            assert not db.query(DocChunk).filter_by(source_type=kind, source_id=sources[key]).count(), key
    assert res["tokens"] == sum(len(c) for c in live_openai.calls) * live_openai.tokens_per_text
    # The remainder is picked up by the next run; a source with chunks is not re-indexed.
    rest = cr.index_missing(max_sources=10, max_usd=1.0, max_added_mb=100, now=NOW)
    assert rest["indexed"] == [f"CCC:filing:{sources['f_memo_scope']}"] and rest["stopped_reason"] is None


def test_index_missing_outage_stops_and_names_the_remainder(corpus_db, live_openai):  # noqa: F811
    seed(corpus_db)
    live_openai.fail = ConnectionError("provider down")
    res = cr.index_missing(max_sources=10, max_usd=1.0, max_added_mb=100, now=NOW)
    assert res["stopped_reason"] == cr.STOP_EMBEDDING_UNAVAILABLE
    assert res["sources_indexed"] == 0 and len(res["deferred"]) == 3
    assert "provider down" not in json.dumps(res)


def _cli():
    path = Path(__file__).resolve().parents[2] / "scripts" / "corpus_repair.py"
    spec = importlib.util.spec_from_file_location("corpus_repair_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_receipt_shape(corpus_db, live_openai, capsys):  # noqa: F811
    seed(corpus_db)
    receipt = cr.run(mode="apply", klass="reembed", max_db_mb=10_000, now=NOW)
    assert receipt["exit_code"] == 0 and receipt["stopped_reason"] is None
    assert {"receipt_id", "mode", "class", "openai_configured", "semantic_embeddings", "caps",
            "before", "after", "targets", "results", "stopped_reason", "exit_code"} <= set(receipt)
    assert receipt["before"]["unusable"] == 10 and receipt["after"]["unusable"] == 3  # orphan + 2 demo
    assert receipt["results"]["chunk_id_ranges"] and receipt["results"]["receipt"] == receipt["receipt_id"]

    cli = _cli()
    assert cli.main(["--plan", "--class", "reembed"]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("CORPUS_REPAIR_RECEIPT ")
    printed = json.loads(line.split(" ", 1)[1])
    assert printed["mode"] == "plan" and set(printed["targets"]) == {"reembed"}
    assert "test-not-a-secret" not in line
    assert cli.main(["--apply", "--class", "reembed"]) == 2  # no --max-db-mb
    assert cli.main(["--inventory"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("CORPUS_INVENTORY ")
