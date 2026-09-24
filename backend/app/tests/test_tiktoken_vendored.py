"""FIX-012: the `cl100k_base` BPE ranks are vendored, and load with no network.

Token budgets in the chunker are only exact when tiktoken's encoder loads.
It used to download its ranks on first use, which the netguard (rightly)
refuses, so a fresh CI runner measured every chunk budget on a character
heuristic and skipped the encoder-specific tests. The ranks now live in
`app/data/tiktoken/`, and these tests keep that arrangement from rotting.

The failure mode they guard is quiet and destructive: tiktoken checks the
cached bytes against a SHA-256 pinned in `tiktoken_ext` and, on a mismatch,
DELETES the file and re-fetches. So a changed byte (an EOL conversion) or a
tiktoken upgrade that moves the pin would not error — it would delete a
tracked file and put CI straight back on the heuristic. Each check below
fails with the specific cause instead.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services import embeddings as emb
from app.tests import netguard

RANKS_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
# Verified against tiktoken 0.14.0's pin when the file was vendored.
PINNED_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
PINNED_SIZE = 1_681_126

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED = REPO_ROOT / "backend" / "app" / "data" / "tiktoken" / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"


def _installed_pin() -> tuple[str, str]:
    """(url, expected_hash) that the installed tiktoken uses for cl100k_base.

    Read from the constructor's source rather than by calling it: calling it
    loads the ranks, which is the thing under test.
    """
    import tiktoken_ext.openai_public as public

    src = inspect.getsource(public.cl100k_base)
    url = re.search(r'"(https://[^"]+/cl100k_base\.tiktoken)"', src)
    pin = re.search(r'expected_hash="([0-9a-f]{64})"', src)
    assert url and pin, f"cannot find cl100k_base's URL and pin in tiktoken's source:\n{src}"
    return url.group(1), pin.group(1)


def test_vendored_file_matches_pinned_hash():
    assert VENDORED.is_file(), (
        f"{VENDORED} is missing. tiktoken deletes a cache file whose hash does not "
        "match its pin, so a deletion in `git status` means the bytes were altered."
    )
    data = VENDORED.read_bytes()
    assert len(data) == PINNED_SIZE, f"size {len(data)} != {PINNED_SIZE}: EOL conversion?"
    assert hashlib.sha256(data).hexdigest() == PINNED_SHA256


def test_installed_tiktoken_pins_the_same_hash():
    """A tiktoken upgrade that moves the pin must fail here, not in production.

    Otherwise the new tiktoken would treat the vendored bytes as corrupt,
    delete them, and silently go back to fetching over the network.
    """
    url, pin = _installed_pin()
    assert url == RANKS_URL == emb.TIKTOKEN_RANKS_URL
    assert pin == PINNED_SHA256, (
        "the installed tiktoken pins a different cl100k_base hash than the vendored "
        "file; re-vendor the ranks from the new pin (and update PINNED_* here)"
    )
    assert hashlib.sha256(emb.VENDORED_TIKTOKEN_FILE.read_bytes()).hexdigest() == pin


def test_file_named_by_tiktoken_cache_key():
    """tiktoken only finds `<TIKTOKEN_CACHE_DIR>/<sha1(url)>`; any other name is inert."""
    import tiktoken.load

    # Pin the key derivation itself, so a tiktoken that renames its cache
    # entries fails here rather than quietly missing the file.
    assert "hashlib.sha1(blobpath.encode()).hexdigest()" in inspect.getsource(
        tiktoken.load.read_file_cached
    )
    key = hashlib.sha1(RANKS_URL.encode()).hexdigest()
    assert emb.VENDORED_TIKTOKEN_FILE.name == key == VENDORED.name
    assert emb.VENDORED_TIKTOKEN_FILE.resolve() == VENDORED.resolve()
    assert emb.VENDORED_TIKTOKEN_FILE.is_file()


def test_gitattributes_keeps_the_ranks_byte_exact():
    """`-text` is what stops a checkout from rewriting line endings in the file.

    The rule's presence is not enough: the last matching line in
    .gitattributes wins, so a `* text=auto` appended below it would silently
    re-enable conversion. Ask git for the attribute it actually applies.
    With `text` unset git ignores `eol` too, so `unset` is the whole answer.
    """
    lines = (REPO_ROOT / ".gitattributes").read_text().splitlines()
    rule = [ln.split() for ln in lines if ln.strip().startswith("backend/app/data/tiktoken/")]
    assert rule and "-text" in rule[0], f".gitattributes has no -text rule for the ranks: {rule}"

    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("no git checkout to ask for the effective attribute")
    out = subprocess.run(
        [git, "check-attr", "text", "--", str(VENDORED.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out.endswith(": text: unset"), (
        f"git applies {out!r} to the vendored ranks; a later .gitattributes line "
        "overrides the -text rule, so a checkout may rewrite the bytes"
    )


@pytest.fixture
def cold_tiktoken(monkeypatch, request, tmp_path):
    """No loaded encodings, no cache dir configured, and no way to fetch.

    `DATA_GYM_CACHE_DIR` points tiktoken's fallback cache at an empty dir, so
    a warm `<tmp>/data-gym-cache` on a dev machine cannot satisfy the load:
    only the vendored file can. `read_file` (tiktoken's only network path) is
    replaced so a fetch fails even under MM_ALLOW_NETWORK=1.
    """
    import tiktoken.load
    import tiktoken.registry

    # setenv first so monkeypatch records the original value and restores it,
    # including removing the variable `_encoding()` is about to set.
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "placeholder")
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tmp_path / "cold-cache"))
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})

    def _no_fetch(blobpath):
        raise AssertionError(f"tiktoken tried to fetch {blobpath}")

    monkeypatch.setattr(tiktoken.load, "read_file", _no_fetch)
    emb._encoding.cache_clear()
    request.addfinalizer(emb._encoding.cache_clear)


def test_encoder_loads_offline_under_netguard(cold_tiktoken, request):
    """From a cold process state, with no cache dir configured, and no network.

    The fetch is blocked twice over: the netguard refuses sockets, and the
    `cold_tiktoken` fixture replaces `read_file`.
    """
    before = netguard.hits().get(request.node.nodeid, [])

    enc = emb._encoding()

    assert enc is not None, "vendored cl100k_base failed to load"
    assert enc.name == "cl100k_base"
    assert enc.encode("hello world") == [15339, 1917]
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(emb.VENDORED_TIKTOKEN_DIR)
    assert netguard.hits().get(request.node.nodeid, []) == before == []
    assert VENDORED.is_file(), "tiktoken deleted the vendored file"


def test_an_operator_set_cache_dir_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    emb._use_vendored_tiktoken_cache()
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(tmp_path)


def test_a_missing_vendored_file_leaves_tiktoken_defaults_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "placeholder")
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    monkeypatch.setattr(emb, "VENDORED_TIKTOKEN_FILE", tmp_path / "absent")
    emb._use_vendored_tiktoken_cache()
    assert "TIKTOKEN_CACHE_DIR" not in os.environ


@pytest.mark.parametrize("text", ["Revenue rose 12.4% to $394,328 million.", "營業收入", "🚀"])
def test_count_tokens_is_the_real_encoder(cold_tiktoken, text):
    """With the ranks vendored, count_tokens is exact, never the heuristic."""
    enc = emb._encoding()
    assert enc is not None, "vendored cl100k_base failed to load"
    assert emb.count_tokens(text) == len(enc.encode(text, disallowed_special=()))


def test_an_unreadable_vendored_dir_does_not_escape_encoding(cold_tiktoken, monkeypatch):
    """`_encoding()` never raises, and neither may the cache redirect before it.

    `Path.is_file` swallows only "not there" errors; EACCES on the directory
    escapes it, and `lru_cache` does not cache exceptions, so every
    `count_tokens` call inside a filing index would raise again. Cold, so
    the fallback cannot fetch: the expected outcome is the heuristic.
    """

    class _Unreadable:
        def is_file(self) -> bool:
            raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(emb, "VENDORED_TIKTOKEN_FILE", _Unreadable())

    emb._use_vendored_tiktoken_cache()
    assert "TIKTOKEN_CACHE_DIR" not in os.environ
    assert emb._encoding() is None
    assert emb.count_tokens("hello world") > 0
