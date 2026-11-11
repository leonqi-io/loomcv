"""Suite-wide fixtures shared by every test module."""

import gc

import pytest


@pytest.fixture(autouse=True)
def _clear_chroma_system_cache():
    """Release every Chroma client's file handles after each test.

    `chromadb.PersistentClient` caches one `System` per path in a
    process-wide registry, and each keeps its SQLite/HNSW files open. Tests
    use a fresh `tmp_path`, so without this the registry grows by one open
    store per test until macOS's default `ulimit -n 256` is hit and an
    unrelated later test fails with "unable to open database file"
    (`ulimit -n 4096` was the workaround).
    Lives here, not in one module, because `test_index.py`, `test_cli.py`
    and `test_search.py` all open Chroma stores.

    Clearing the registry alone is not enough: the stopped systems sit in
    reference cycles and keep their files open until the cycle collector
    runs, so the collection is forced here (measured: +10 fds per test
    without it, flat with it).
    """
    yield
    from chromadb.api.shared_system_client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    gc.collect()
