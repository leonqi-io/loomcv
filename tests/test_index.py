"""Tests for the SQLite + Chroma incremental index.

Fixtures are synthetic (`tests/fixtures/index_vault.md`); the
embedding adapter is always the in-process `FakeEmbeddingAdapter` below —
zero network, zero model downloads. All stores live under `tmp_path`.

Behaviour exercised here:
- SQLite is the source of record, Chroma a derived index; write order
  is embed -> Chroma -> SQLite-committed-last; a store-consistency guard
  (`StoreInconsistentError`) catches a hand-deleted `vector_cache/`.
- Chroma per-bullet metadata is
  exactly `{bullet_id, content_hash}`; a metadata-only update (status,
  updated, order, project label, keywords, tags) writes SQLite only and
  makes no Chroma call at all.
- Four sync outcomes (added/updated/deleted/metadata_only) plus
  unchanged; a status-block-only change or a pure reorder is
  `metadata_only` with zero re-embeds; draft bullets are excluded by
  default and a draft<->confirmed flip is a plain add/delete.
- v3.1: a job with zero visible bullets has its `jobs` row deleted
  (SQLite mirrors the vault, not a history).
- Schema v3 (`jobs` replaces `projects`), the
  stepwise migration loop, and the 2 -> 3 step being a full rebuild.

All fixture content is fictional — no real career data (M1 protocol).
"""

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from loomcv.sync.index_assets import (
    COLLECTION_NAME,
    SchemaVersionError,
    StoreInconsistentError,
    VaultParseError,
    VectorSpaceMismatchError,
    index_vault,
)
from loomcv.sync.parse_assets import parse_star_vault

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture()
def vault_text() -> str:
    return _read("index_vault.md")


@pytest.fixture()
def metadata_vault_text() -> str:
    return _read("metadata_vault.md")


@pytest.fixture()
def paths(tmp_path):
    return tmp_path / "pipeline_storage.db", tmp_path / "vector_cache"


@pytest.fixture(autouse=True)
def _no_default_embedding_function(monkeypatch):
    """No code path may ever call Chroma's default (ONNX) embedding fn.

    Chroma caches that model under ~/.cache/chroma/, not under our
    chroma_path, so asserting "no cache under chroma_path" would be
    vacuously true. A missing chromadb must fail this suite, never skip it,
    so the import below is a hard import.

    Patched on `__call__`, not the module-level class name: Chroma evaluates
    `embedding_function=ef.DefaultEmbeddingFunction()` at import time, and
    that class's ONNX download is lazy — it happens on first `__call__`, not
    on construction. Replacing the class name after import would leave an
    already-built default instance fully functional.
    """
    import chromadb.utils.embedding_functions as ef

    def _raise(*_args, **_kwargs):
        raise RuntimeError("default Chroma embedding function must never be called")

    for cls_name in ("DefaultEmbeddingFunction", "ONNXMiniLM_L6_V2"):
        cls = getattr(ef, cls_name, None)
        if cls is not None:
            monkeypatch.setattr(cls, "__call__", _raise)


# ------------------------------------------------------------ fake adapter


class FakeEmbeddingAdapter:
    """Deterministic, network-free stand-in for `EmbeddingAdapter`."""

    def __init__(
        self,
        *,
        model_id: str = "fake-model",
        vector_space_version: int = 1,
        raise_on_call: bool = False,
    ) -> None:
        self._model_id = model_id
        self._vector_space_version = vector_space_version
        self.raise_on_call = raise_on_call
        self.calls = 0
        self.texts_embedded = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vector_space_version(self) -> int:
        return self._vector_space_version

    def embed(self, texts):
        if self.raise_on_call:
            raise RuntimeError("fake embedding adapter forced failure")
        self.calls += 1
        self.texts_embedded += len(texts)
        return [_deterministic_vector(t) for t in texts]


def _deterministic_vector(text: str) -> list[float]:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:8]]


# -------------------------------------------------------------- store helpers


def _sqlite_bullet_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[0] for row in conn.execute("SELECT bullet_id FROM bullets")}
    finally:
        conn.close()


def _sqlite_job_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[0] for row in conn.execute("SELECT job_id FROM jobs")}
    finally:
        conn.close()


def _sqlite_job_row(db_path: Path, job_id: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def _sqlite_row(db_path: Path, bullet_id: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM bullets WHERE bullet_id = ?", (bullet_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def _chroma_client(chroma_path: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_path))


def _chroma_collection(chroma_path: Path):
    return _chroma_client(chroma_path).get_collection(COLLECTION_NAME, embedding_function=None)


def _chroma_ids(chroma_path: Path) -> set[str]:
    return set(_chroma_collection(chroma_path).get(include=[])["ids"])


def _chroma_embeddings(chroma_path: Path) -> dict[str, list[float]]:
    got = _chroma_collection(chroma_path).get(include=["embeddings"])
    return dict(zip(got["ids"], (list(v) for v in got["embeddings"])))


def _chroma_snapshot(chroma_path: Path) -> dict:
    """Everything Chroma holds per bullet (ids, vectors, metadata), order-normalised."""
    got = _chroma_collection(chroma_path).get(include=["embeddings", "metadatas"])
    return {
        bid: (list(vec), dict(meta))
        for bid, vec, meta in zip(got["ids"], got["embeddings"], got["metadatas"])
    }


def _all_bullet_ids(parse_result) -> set[str]:
    return {b.bullet_id for j in parse_result.jobs for b in j.bullets}


def _snapshot_dir(path: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}


# ---------------------------------------------------------------- fresh sync


def test_01_fresh_sync_sqlite_counts(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    assert _sqlite_job_ids(db_path) == {j.job_id for j in parse_result.jobs}
    assert _sqlite_bullet_ids(db_path) == _all_bullet_ids(parse_result)


def test_02_fresh_sync_chroma_count(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    expected_ids = _all_bullet_ids(parse_result)
    collection = _chroma_collection(chroma_path)
    assert collection.count() == len(expected_ids)
    assert _chroma_ids(chroma_path) == expected_ids


def test_03_fresh_sync_report_counts(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    fake = FakeEmbeddingAdapter()
    report = index_vault(parse_result, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    expected_ids = _all_bullet_ids(parse_result)
    assert set(report.added) == expected_ids
    assert report.updated == []
    assert report.deleted == []
    assert report.metadata_only == []
    assert report.unchanged == []
    assert report.embedded == len(expected_ids) == fake.texts_embedded


def test_04_collection_has_no_default_embedding_function(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    collection = _chroma_collection(chroma_path)
    # Attribute name is chromadb-version-specific; the real guarantee is the
    # autouse fixture above, which makes the default EF unconstructible.
    assert getattr(collection, "_embedding_function", None) is None


def test_05_chunk_metadata_payload_shape(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    # Nothing in loomcv/ reads per-bullet Chroma
    # metadata, so it is exactly the two keys that change only with the text.
    collection = _chroma_collection(chroma_path)
    got = collection.get(include=["metadatas"])
    hashes = {b.bullet_id: b.content_hash for j in parse_result.jobs for b in j.bullets}
    assert set(got["ids"]) == set(hashes)
    for bid, meta in zip(got["ids"], got["metadatas"]):
        assert set(meta) == {"bullet_id", "content_hash"}
        assert meta["bullet_id"] == bid
        assert meta["content_hash"] == hashes[bid]


# --------------------------------------------------------------- incremental


def test_06_idempotent_resync(vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    parse_result = parse_star_vault(vault_text)
    index_vault(parse_result, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    before = _chroma_embeddings(chroma_path)
    calls_before = fake.calls

    report = index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    assert fake.calls == calls_before
    assert report.added == []
    assert report.updated == []
    assert report.deleted == []
    assert report.metadata_only == []
    assert report.jobs_written == []
    assert report.jobs_deleted == []
    assert set(report.unchanged) == _all_bullet_ids(parse_result)
    assert _chroma_embeddings(chroma_path) == before


def test_07_single_text_edit_one_reembed(vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )
    before = _chroma_embeddings(chroma_path)
    embedded_before = fake.texts_embedded

    edited_text = vault_text.replace("from 800ms to 200ms", "from 800ms to 150ms")
    after_result = parse_star_vault(edited_text)
    edited_bullet = next(b for j in after_result.jobs for b in j.bullets if "150ms" in b.text)

    report = index_vault(after_result, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert report.updated == [edited_bullet.bullet_id]
    assert report.added == report.deleted == report.metadata_only == []
    assert fake.texts_embedded - embedded_before == 1

    after = _chroma_embeddings(chroma_path)
    for bid in before:
        if bid == edited_bullet.bullet_id:
            assert after[bid] != before[bid]
        else:
            assert after[bid] == before[bid]


def test_08_bullet_removed_row_and_vector_deleted(vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    embedded_before = fake.texts_embedded
    removed_id = "lumen-retail-search-engineer::aur002"
    removed_line = (
        "  - [Reliability] Reduced imaginary search timeouts by 30% across 2 fake"
        " regions. {id: aur002, status: confirmed, updated: 2026-08-10}\n"
        "    * **Project**: Aurora Search Revamp\n"
    )
    edited_text = vault_text.replace(removed_line, "")
    after_result = parse_star_vault(edited_text)
    assert removed_id not in _all_bullet_ids(after_result)

    report = index_vault(after_result, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert report.deleted == [removed_id]
    assert fake.texts_embedded == embedded_before
    assert removed_id not in _sqlite_bullet_ids(db_path)
    assert removed_id not in _chroma_ids(chroma_path)


def test_09_job_emptied_jobs_row_kept(vault_text, paths):
    # A job whose confirmed bullets all went away is a
    # bullet-less job in the vault, so its `jobs` row stays; its bullets and vectors still go.
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )
    assert "cobalt-analytics-data-engineer" in _sqlite_job_ids(db_path)

    edited_text = vault_text.replace("{id: cas001}", "{id: cas001, status: draft}").replace(
        "{id: cas002}", "{id: cas002, status: draft}"
    )
    after_result = parse_star_vault(edited_text)
    cobalt = next(j for j in after_result.jobs if j.job_id == "cobalt-analytics-data-engineer")
    assert cobalt.bullets == []

    report = index_vault(after_result, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert "cobalt-analytics-data-engineer" not in report.jobs_deleted
    assert "cobalt-analytics-data-engineer" in _sqlite_job_ids(db_path)
    for tail in ("cas001", "cas002"):
        gone = f"cobalt-analytics-data-engineer::{tail}"
        assert gone not in _sqlite_bullet_ids(db_path)
        assert gone not in _chroma_ids(chroma_path)
    # untouched
    assert "lumen-retail-search-engineer" in _sqlite_job_ids(db_path)
    assert "placeholder-lab-research-intern" in _sqlite_job_ids(db_path)


def test_10_tag_edit_is_metadata_only_id_edit_is_delete_plus_add(vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )
    bid = "lumen-retail-search-engineer::bea002"
    old_line = (
        "  - [Observability] Instrumented pretend delivery tracing across 6 imaginary services."
        " {id: bea002}\n"
    )
    assert old_line in vault_text

    # tags feed neither bullet_id nor content_hash: metadata-only, SQLite only
    snapshot = _chroma_snapshot(chroma_path)
    calls_before, embedded_before = fake.calls, fake.texts_embedded
    tagged_text = vault_text.replace(
        old_line, old_line.replace("[Observability]", "[Observability][Tooling]")
    )
    report = index_vault(
        parse_star_vault(tagged_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )
    assert report.metadata_only == [bid]
    assert report.added == report.deleted == report.updated == []
    assert report.embedded == 0
    assert fake.calls == calls_before and fake.texts_embedded == embedded_before
    assert json.loads(_sqlite_row(db_path, bid)["tags_json"]) == ["Observability", "Tooling"]
    assert _chroma_snapshot(chroma_path) == snapshot  # no Chroma write of any kind

    # an id edit is a different bullet: delete + add, one embed
    new_id = "lumen-retail-search-engineer::bea009"
    renamed_text = tagged_text.replace("{id: bea002}", "{id: bea009}")
    report = index_vault(
        parse_star_vault(renamed_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )
    assert report.deleted == [bid]
    assert report.added == [new_id]
    assert report.updated == []
    assert fake.texts_embedded - embedded_before == 1
    assert bid not in _sqlite_bullet_ids(db_path)
    assert new_id in _sqlite_bullet_ids(db_path)
    assert bid not in _chroma_ids(chroma_path)
    assert new_id in _chroma_ids(chroma_path)


# ------------------------------------------------------------- metadata-only


LUMEN = "lumen-retail-search-engineer::"
_BEACON_A = (
    "  - [Messaging][Backend] Fanned out 1M fictional notifications per minute"
    " across 4 fake queues. {id: bea001}\n"
    "    * **Project**: Beacon Notification Service\n"
)
_BEACON_B = (
    "  - [Observability] Instrumented pretend delivery tracing across 6 imaginary services."
    " {id: bea002}\n"
    "    * **Project**: Beacon Notification Service\n"
)


def _bump_updated_date(text: str) -> str:
    return text.replace(
        "{id: aur002, status: confirmed, updated: 2026-08-10}",
        "{id: aur002, status: confirmed, updated: 2026-08-15}",
    )


def _swap_beacon_bullets(text: str) -> str:
    assert _BEACON_A + _BEACON_B in text
    return text.replace(_BEACON_A + _BEACON_B, _BEACON_B + _BEACON_A)


def _edit_keywords(text: str) -> str:
    old = "    * **Keywords**: Caching; Search Relevance\n"
    assert old in text
    return text.replace(old, "    * **Keywords**: Caching; Query Latency\n")


def _add_project_label(text: str) -> str:
    return text.replace("{id: lab001}\n", "{id: lab001}\n    * **Project**: Lab Spike\n")


def _check_status_updated(db_path):
    assert _sqlite_row(db_path, LUMEN + "aur002")["updated"] == "2026-08-15"


def _check_reorder(db_path):
    assert _sqlite_row(db_path, LUMEN + "bea002")["bullet_order"] == 2
    assert _sqlite_row(db_path, LUMEN + "bea001")["bullet_order"] == 3


def _check_keywords(db_path):
    row = _sqlite_row(db_path, LUMEN + "aur001")
    assert json.loads(row["keywords_json"]) == ["Caching", "Query Latency"]


def _check_label(db_path):
    row = _sqlite_row(db_path, "placeholder-lab-research-intern::lab001")
    assert row["project"] == "Lab Spike"


@pytest.mark.parametrize(
    "mutate,changed_ids,check",
    [
        (_bump_updated_date, [LUMEN + "aur002"], _check_status_updated),
        (_swap_beacon_bullets, [LUMEN + "bea001", LUMEN + "bea002"], _check_reorder),
        (_edit_keywords, [LUMEN + "aur001"], _check_keywords),
        (_add_project_label, ["placeholder-lab-research-intern::lab001"], _check_label),
    ],
    ids=["status_updated", "reorder", "keywords_edit", "project_label_added"],
)
def test_11_metadata_only_change(vault_text, paths, mutate, changed_ids, check):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )
    snapshot = _chroma_snapshot(chroma_path)
    calls_before, embedded_before = fake.calls, fake.texts_embedded

    after_result = parse_star_vault(mutate(vault_text))
    report = index_vault(after_result, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert set(report.metadata_only) == set(changed_ids)
    assert report.updated == []
    assert report.added == []
    assert report.deleted == []
    assert report.embedded == 0
    assert fake.texts_embedded == embedded_before
    assert fake.calls == calls_before
    # A metadata-only update is SQLite-only — Chroma is byte-equal
    assert _chroma_snapshot(chroma_path) == snapshot
    check(db_path)


# ------------------------------------------------------------------- drafts


def test_12_draft_excluded_then_confirmed_is_added(vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    draft_id = "cobalt-analytics-data-engineer::cas003"
    assert draft_id not in _sqlite_bullet_ids(db_path)
    assert draft_id not in _chroma_ids(chroma_path)
    embedded_before = fake.texts_embedded

    confirmed_text = vault_text.replace("{id: cas003, status: draft}", "{id: cas003}")
    assert confirmed_text != vault_text
    report = index_vault(
        parse_star_vault(confirmed_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    assert report.added == [draft_id]
    assert fake.texts_embedded - embedded_before == 1
    assert draft_id in _sqlite_bullet_ids(db_path)
    assert draft_id in _chroma_ids(chroma_path)


def test_13_confirmed_to_draft_is_deleted(vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    target_id = "lumen-retail-search-engineer::aur002"
    assert target_id in _sqlite_bullet_ids(db_path)
    embedded_before = fake.texts_embedded

    drafted_text = vault_text.replace(
        "status: confirmed, updated: 2026-08-10", "status: draft, updated: 2026-08-10"
    )
    report = index_vault(
        parse_star_vault(drafted_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    assert report.deleted == [target_id]
    assert fake.texts_embedded == embedded_before
    assert target_id not in _sqlite_bullet_ids(db_path)
    assert target_id not in _chroma_ids(chroma_path)


# ------------------------------------------------------------------ aborts


def test_14_parse_error_aborts_fresh_no_files_created(paths):
    db_path, chroma_path = paths
    malformed = parse_star_vault(_read("malformed_vault.md"))
    assert malformed.errors()

    with pytest.raises(VaultParseError):
        index_vault(
            malformed, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
        )

    assert not db_path.exists()
    assert not chroma_path.exists()


def test_14b_parse_error_aborts_after_good_sync_untouched(vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    db_before = db_path.read_bytes()
    chroma_before = _snapshot_dir(chroma_path)

    malformed = parse_star_vault(_read("malformed_vault.md"))
    with pytest.raises(VaultParseError):
        index_vault(malformed, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert db_path.read_bytes() == db_before
    assert _snapshot_dir(chroma_path) == chroma_before


# (test_15 — "#reserve project excluded by default" — was deleted: the
# reserve concept is gone; `lab001` below is an ordinary, indexed bullet.)


# ------------------------------------------------------- vector-space guard


@pytest.mark.parametrize(
    "make_second_adapter",
    [
        lambda: FakeEmbeddingAdapter(vector_space_version=2),
        lambda: FakeEmbeddingAdapter(model_id="different-model"),
    ],
    ids=["bumped_version", "changed_model_id"],
)
def test_16_vector_space_mismatch_hard_error(vault_text, paths, make_second_adapter):
    db_path, chroma_path = paths
    index_vault(
        parse_star_vault(vault_text),
        db_path=db_path,
        chroma_path=chroma_path,
        adapter=FakeEmbeddingAdapter(),
    )

    db_before = db_path.read_bytes()
    ids_before = _chroma_ids(chroma_path)

    second_adapter = make_second_adapter()
    with pytest.raises(VectorSpaceMismatchError):
        index_vault(
            parse_star_vault(vault_text),
            db_path=db_path,
            chroma_path=chroma_path,
            adapter=second_adapter,
        )
    assert second_adapter.calls == 0
    assert db_path.read_bytes() == db_before
    assert _chroma_ids(chroma_path) == ids_before


# ------------------------------------------------------------------ rebuild


def test_17_rebuild_path_fresh_fully_reembedded(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    new_adapter = FakeEmbeddingAdapter(model_id="new-model", vector_space_version=2)
    report = index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=new_adapter, rebuild=True
    )

    expected_ids = _all_bullet_ids(parse_result)
    assert set(report.added) == expected_ids
    assert report.embedded == len(expected_ids) == new_adapter.texts_embedded

    collection = _chroma_collection(chroma_path)
    assert _chroma_ids(chroma_path) == expected_ids
    meta = collection.metadata or {}
    assert meta.get("model_id") == "new-model"
    assert meta.get("vector_space_version") == 2


def test_17b_rebuild_mid_embed_failure_leaves_old_stores_intact(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    db_before = db_path.read_bytes()
    before_ids = _chroma_ids(chroma_path)

    failing_adapter = FakeEmbeddingAdapter(
        model_id="new-model", vector_space_version=2, raise_on_call=True
    )
    with pytest.raises(RuntimeError):
        index_vault(
            parse_result,
            db_path=db_path,
            chroma_path=chroma_path,
            adapter=failing_adapter,
            rebuild=True,
        )

    assert db_path.read_bytes() == db_before
    assert _chroma_ids(chroma_path) == before_ids


# ---------------------------------------------------------- store guard


def test_18_stores_inconsistent_hard_error_chroma_dir_deleted(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    db_before = db_path.read_bytes()
    shutil.rmtree(chroma_path)

    fake2 = FakeEmbeddingAdapter()
    with pytest.raises(StoreInconsistentError):
        index_vault(parse_result, db_path=db_path, chroma_path=chroma_path, adapter=fake2)

    assert db_path.read_bytes() == db_before
    assert fake2.calls == 0

    recovered_adapter = FakeEmbeddingAdapter()
    recovered = index_vault(
        parse_result,
        db_path=db_path,
        chroma_path=chroma_path,
        adapter=recovered_adapter,
        rebuild=True,
    )
    expected_ids = _all_bullet_ids(parse_result)
    assert set(recovered.added) == expected_ids
    assert _chroma_ids(chroma_path) == expected_ids


def test_18b_stores_inconsistent_hard_error_collection_dropped(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    db_before = db_path.read_bytes()
    _chroma_client(chroma_path).delete_collection(COLLECTION_NAME)

    fake2 = FakeEmbeddingAdapter()
    with pytest.raises(StoreInconsistentError):
        index_vault(parse_result, db_path=db_path, chroma_path=chroma_path, adapter=fake2)

    assert db_path.read_bytes() == db_before
    assert fake2.calls == 0


def test_18c_crash_between_chroma_delete_and_sqlite_commit_recovers(vault_text, paths):
    """A crash after collection.delete() but before the SQLite commit (D-3) must

    self-heal on the next sync, not demand a rebuild: the id was already on its
    way out, so SQLite catching up is just the delete finishing late.
    """
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    removed_id = "lumen-retail-search-engineer::aur002"
    _chroma_collection(chroma_path).delete(ids=[removed_id])  # simulates the crash window

    calls_before = fake.calls
    removed_line = (
        "  - [Reliability] Reduced imaginary search timeouts by 30% across 2 fake"
        " regions. {id: aur002, status: confirmed, updated: 2026-08-10}\n"
        "    * **Project**: Aurora Search Revamp\n"
    )
    edited_text = vault_text.replace(removed_line, "")
    report = index_vault(
        parse_star_vault(edited_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    assert report.deleted == [removed_id]
    assert fake.calls == calls_before
    assert removed_id not in _sqlite_bullet_ids(db_path)
    assert removed_id not in _chroma_ids(chroma_path)

    resync_report = index_vault(
        parse_star_vault(edited_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )
    assert resync_report.added == []
    assert resync_report.updated == []
    assert resync_report.deleted == []
    assert resync_report.metadata_only == []
    assert set(resync_report.unchanged) == _all_bullet_ids(parse_star_vault(edited_text))


def test_18d_id_missing_from_chroma_but_still_in_vault_hard_error(vault_text, paths):
    """The mirror case: an id the vault still wants, missing from Chroma, must

    still raise — that is the real inconsistency the D-3 guard exists to catch.
    """
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    still_wanted_id = "lumen-retail-search-engineer::aur002"
    _chroma_collection(chroma_path).delete(ids=[still_wanted_id])

    db_before = db_path.read_bytes()
    fake2 = FakeEmbeddingAdapter()
    with pytest.raises(StoreInconsistentError):
        index_vault(
            parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake2
        )
    assert db_path.read_bytes() == db_before
    assert fake2.calls == 0


# ------------------------------------------------------------ schema guard


def test_19_schema_version_mismatch_hard_error(vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    fake2 = FakeEmbeddingAdapter()
    with pytest.raises(SchemaVersionError):
        index_vault(parse_result, db_path=db_path, chroma_path=chroma_path, adapter=fake2)
    assert fake2.calls == 0


# --------------------------------------------- jobs, keywords, schema v3


def _sqlite_table_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        conn.close()


def _sqlite_schema_version(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    finally:
        conn.close()


def test_20_jobs_and_keywords_land_in_sqlite_and_chroma(metadata_vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    parse_result = parse_star_vault(metadata_vault_text)
    report = index_vault(parse_result, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    # jobs rows: fixture is written Zephyr-then-Helios; job_order is by date
    helios = _sqlite_job_row(db_path, "helios-streaming-senior-player-engineer")
    assert helios["name"] == "Helios Streaming — Senior Player Engineer"
    assert helios["employer"] == "Helios Streaming"
    assert helios["title"] == "Senior Player Engineer"
    assert (helios["date_from"], helios["date_to"]) == ("2023-07", "present")
    assert json.loads(helios["defaults_json"]) == ["TypeScript", "Shaka Player"]
    assert helios["job_order"] == 0

    zephyr = _sqlite_job_row(db_path, "zephyr-analytics-data-engineer")
    assert (zephyr["date_from"], zephyr["date_to"]) == ("2019-09", "2023-06")
    assert json.loads(zephyr["defaults_json"]) == []
    assert zephyr["job_order"] == 1

    # bullets rows carry the bullet's OWN keyword list and label; defaults not merged
    hel = _sqlite_row(db_path, "helios-streaming-senior-player-engineer::hel001")
    assert hel["job_id"] == "helios-streaming-senior-player-engineer"
    assert hel["project"] == "Helios Player Rewrite"
    assert json.loads(hel["keywords_json"]) == ["Shaka Player", "Buffer Tuning"]
    zep = _sqlite_row(db_path, "zephyr-analytics-data-engineer::zep001")
    assert zep["project"] == "Zephyr Batch Loader"
    assert json.loads(zep["keywords_json"]) == []
    tun = _sqlite_row(db_path, "zephyr-analytics-data-engineer::tun001")
    assert json.loads(tun["keywords_json"]) == ["Alerting", "Dedupe Rules", "Observability"]

    # Chroma carries exactly {bullet_id, content_hash}; collection metadata unchanged
    collection = _chroma_collection(chroma_path)
    got = collection.get(include=["metadatas"])
    assert got["ids"]
    assert all(set(m) == {"bullet_id", "content_hash"} for m in got["metadatas"])
    assert collection.metadata["model_id"] == fake.model_id
    assert collection.metadata["vector_space_version"] == fake.vector_space_version

    assert _sqlite_schema_version(db_path) == "3"
    assert "projects" not in _sqlite_table_names(db_path)
    assert report.schema_rebuild_from is None


def _edit_helios_defaults(text: str) -> str:
    return text.replace("* **Defaults**: TypeScript; Shaka Player", "* **Defaults**: TypeScript")


def _edit_zephyr_title(text: str) -> str:
    return text.replace("* **Title**: Data Engineer", "* **Title**: Senior Data Engineer")


def _edit_zephyr_to_no_reorder(text: str) -> str:
    return text.replace("* **To**: 2023-06", "* **To**: 2023-08")


def _edit_name_same_slug(text: str) -> str:
    return text.replace("## [Job] Helios Streaming — Senior", "## [Job] Helios Streaming - Senior")


@pytest.mark.parametrize(
    "mutate,expected_jobs",
    [
        (_edit_helios_defaults, ["helios-streaming-senior-player-engineer"]),
        (_edit_zephyr_title, ["zephyr-analytics-data-engineer"]),
        (_edit_zephyr_to_no_reorder, ["zephyr-analytics-data-engineer"]),
        (_edit_name_same_slug, ["helios-streaming-senior-player-engineer"]),
    ],
    ids=["defaults_edit", "title_edit", "to_edit_no_reorder", "name_same_slug"],
)
def test_21_job_level_edit_touches_job_row_only(metadata_vault_text, paths, mutate, expected_jobs):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(metadata_vault_text),
        db_path=db_path,
        chroma_path=chroma_path,
        adapter=fake,
    )
    snapshot = _chroma_snapshot(chroma_path)
    calls_before = fake.calls

    edited_text = mutate(metadata_vault_text)
    assert edited_text != metadata_vault_text
    edited = parse_star_vault(edited_text)
    assert edited.errors() == []
    report = index_vault(edited, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert sorted(report.jobs_written) == sorted(expected_jobs)
    # a job-level edit never touches a bullet
    assert set(report.unchanged) == _all_bullet_ids(edited)
    assert report.added == report.updated == report.deleted == report.metadata_only == []
    assert report.embedded == 0
    assert fake.calls == calls_before
    assert _chroma_snapshot(chroma_path) == snapshot


def test_21_to_edit_that_reorders_flips_job_order(metadata_vault_text, paths):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    index_vault(
        parse_star_vault(metadata_vault_text),
        db_path=db_path,
        chroma_path=chroma_path,
        adapter=fake,
    )
    helios_id = "helios-streaming-senior-player-engineer"
    zephyr_id = "zephyr-analytics-data-engineer"
    assert _sqlite_job_row(db_path, helios_id)["job_order"] == 0

    edited = parse_star_vault(metadata_vault_text.replace("* **To**: present", "* **To**: 2018-12"))
    report = index_vault(edited, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert sorted(report.jobs_written) == sorted([helios_id, zephyr_id])
    assert _sqlite_job_row(db_path, zephyr_id)["job_order"] == 0
    assert _sqlite_job_row(db_path, helios_id)["job_order"] == 1
    assert _sqlite_job_row(db_path, helios_id)["date_to"] == "2018-12"
    assert set(report.unchanged) == _all_bullet_ids(edited)
    assert report.embedded == 0


_V1_PROJECTS_DDL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    context    TEXT NOT NULL DEFAULT '',
    tags_json  TEXT NOT NULL DEFAULT '[]'
);
"""
_V2_PROJECT_COLUMNS = """
ALTER TABLE projects ADD COLUMN domain TEXT;
ALTER TABLE projects ADD COLUMN year INTEGER;
ALTER TABLE projects ADD COLUMN tech_json TEXT NOT NULL DEFAULT '[]';
"""
_OLD_BULLETS_DDL = """
CREATE TABLE bullets (
    bullet_id    TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    text         TEXT NOT NULL,
    tags_json    TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'confirmed',
    updated      TEXT,
    bullet_order INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_bullets_project ON bullets(project_id);
"""
_OLD_ID = "old-project::old-tag"


def _build_old_store(db_path: Path, chroma_path: Path, adapter, *, version: int) -> None:
    """Hand-build a pre-v3 store: old SQLite layout + a Chroma collection holding one old id."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_V1_PROJECTS_DDL)
        if version == 2:
            conn.executescript(_V2_PROJECT_COLUMNS)
        conn.executescript(_OLD_BULLETS_DDL)
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(version),))
        conn.execute("INSERT INTO projects (project_id, name) VALUES ('old-project', 'Old')")
        conn.execute(
            "INSERT INTO bullets (bullet_id, project_id, text, content_hash)"
            " VALUES (?, 'old-project', 'Old text.', 'sha1:old')",
            (_OLD_ID,),
        )
        conn.commit()
    finally:
        conn.close()
    collection = _chroma_client(chroma_path).create_collection(
        COLLECTION_NAME,
        embedding_function=None,
        metadata={
            "model_id": adapter.model_id,
            "vector_space_version": adapter.vector_space_version,
        },
    )
    collection.upsert(
        ids=[_OLD_ID], embeddings=[_deterministic_vector("Old text.")], metadatas=[{"x": 1}]
    )


@pytest.mark.parametrize("version", [1, 2])
def test_22_old_schema_chain_rebuilds_to_v3(vault_text, paths, version):
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    _build_old_store(db_path, chroma_path, fake, version=version)

    parse_result = parse_star_vault(vault_text)
    report = index_vault(parse_result, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    expected = _all_bullet_ids(parse_result)
    assert _sqlite_schema_version(db_path) == "3"
    tables = _sqlite_table_names(db_path)
    assert {"jobs", "bullets"} <= tables
    assert "projects" not in tables
    assert _sqlite_bullet_ids(db_path) == expected
    assert _OLD_ID not in _sqlite_bullet_ids(db_path)
    # the 2 -> 3 step forces a full rebuild: everything added and embedded, no orphan vectors
    assert report.schema_rebuild_from == version
    assert set(report.added) == expected
    assert report.embedded == len(expected)
    assert _chroma_ids(chroma_path) == expected

    # a second sync is an ordinary no-op
    again = index_vault(
        parse_star_vault(vault_text), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )
    assert again.schema_rebuild_from is None
    assert set(again.unchanged) == expected
    assert again.embedded == 0


def test_23_unknown_higher_schema_version_still_raises(metadata_vault_text, paths):
    db_path, chroma_path = paths
    parse_result = parse_star_vault(metadata_vault_text)
    index_vault(
        parse_result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    fake2 = FakeEmbeddingAdapter()
    with pytest.raises(SchemaVersionError):
        index_vault(parse_result, db_path=db_path, chroma_path=chroma_path, adapter=fake2)
    assert fake2.calls == 0


HARROWGATE = "harrowgate-mutual-insurance-claims-operations-manager"
PELLWORTH = "pellworth-general-insurance-senior-claims-adjuster"
TALLIS = "tallis-reed-loss-adjusting-claims-trainee"
_TALLIS_TO = "* **To**: 2013-12\n"


def _claims_vault(*, tallis_defaults: bool = True) -> str:
    """The claims persona confirmed in-test (the fixture file stays draft); Tallis & Reed has no
    bullets, and gets a `Defaults` line when asked (X-1)."""
    text = _read("claims_expected_vault.md").replace("status: draft", "status: confirmed")
    if tallis_defaults:
        assert _TALLIS_TO in text
        text = text.replace(_TALLIS_TO, _TALLIS_TO + "* **Defaults**: Xactimate; Subrogation\n")
    return text


def _meta_value(db_path: Path, key: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    return row[0] if row is not None else None


def _meta_skills(db_path: Path) -> list[str] | None:
    raw = _meta_value(db_path, "skills_json")
    return None if raw is None else json.loads(raw)


def test_x1_bulletless_job_written_with_defaults(paths):
    # X-1: a job with no bullets is written, `Defaults` and all,
    # and `index_vault` still raises nothing.
    db_path, chroma_path = paths
    result = parse_star_vault(_claims_vault())
    assert result.errors() == []
    assert any(j.job_id == TALLIS for j in result.jobs)

    report = index_vault(
        result, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter()
    )
    assert TALLIS in report.jobs_written
    assert len(report.jobs_written) == 3
    assert _sqlite_job_ids(db_path) == {HARROWGATE, PELLWORTH, TALLIS}
    row = _sqlite_job_row(db_path, TALLIS)
    assert json.loads(row["defaults_json"]) == ["Xactimate", "Subrogation"]
    assert not any(b.startswith(TALLIS) for b in _sqlite_bullet_ids(db_path))


def test_x2_bulletless_job_removed_is_deleted(paths):
    # X-2: a bullet-less job leaves `jobs` only when it leaves the vault.
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    text = _claims_vault()
    index_vault(parse_star_vault(text), db_path=db_path, chroma_path=chroma_path, adapter=fake)
    assert TALLIS in _sqlite_job_ids(db_path)

    kept = text.split("## [Job] Tallis & Reed")[0]
    report = index_vault(
        parse_star_vault(kept), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    assert report.jobs_deleted == [TALLIS]
    assert _sqlite_job_ids(db_path) == {HARROWGATE, PELLWORTH}
    assert report.embedded == 0


@pytest.mark.parametrize("rebuild", [False, True])
def test_x3_skills_line_in_meta(paths, rebuild):
    # X-3: the vault-level `Skills` line is one `meta` row, replaced on every
    # sync (both paths); an edit to it is a SQLite-only write.
    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    text = _claims_vault()
    parsed = parse_star_vault(text)
    assert len(parsed.skills) == 14

    first = index_vault(
        parsed, db_path=db_path, chroma_path=chroma_path, adapter=fake, rebuild=rebuild
    )
    assert _meta_skills(db_path) == parsed.skills
    assert first.skills == 14
    assert first.counts()["skills"] == 14

    edited = text.replace("; Regulatory compliance", "")
    second = index_vault(
        parse_star_vault(edited),
        db_path=db_path,
        chroma_path=chroma_path,
        adapter=fake,
        rebuild=rebuild,
    )
    assert _meta_skills(db_path) == parsed.skills[:-1]
    assert second.skills == 13
    if not rebuild:
        assert second.embedded == 0
        assert second.updated == []

    no_line = (
        "\n".join(ln for ln in edited.splitlines() if not ln.startswith("* **Skills**:")) + "\n"
    )
    third = index_vault(
        parse_star_vault(no_line),
        db_path=db_path,
        chroma_path=chroma_path,
        adapter=fake,
        rebuild=rebuild,
    )
    assert _meta_skills(db_path) == []
    assert third.skills == 0


def test_x4_head_db_upgraded_without_rebuild(paths):
    # X-4: an older DB (no `skills_json` row, no bullet-less job rows)
    # reads back as no skills and is upgraded by the next sync — no rebuild, no re-embed.
    from loomcv.retrieve.search import load_claims_inputs

    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    parsed = parse_star_vault(_claims_vault())
    index_vault(parsed, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM meta WHERE key = 'skills_json'")
    conn.execute("DELETE FROM jobs WHERE job_id = ?", (TALLIS,))
    conn.commit()
    conn.close()
    jobs, skills = load_claims_inputs(db_path)
    assert skills == []
    assert [job_id for job_id, _defaults in jobs] == [HARROWGATE, PELLWORTH]

    report = index_vault(parsed, db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert report.schema_rebuild_from is None
    assert report.embedded == 0
    assert TALLIS in report.jobs_written
    assert _meta_skills(db_path) == parsed.skills
    jobs, skills = load_claims_inputs(db_path)
    assert [job_id for job_id, _defaults in jobs] == [HARROWGATE, PELLWORTH, TALLIS]
    assert dict(jobs)[TALLIS] == ["Xactimate", "Subrogation"]
    assert skills == parsed.skills


# -------------------------------------- vault header + Skills groups in meta (I-1 … I-4)


def test_l3a_header_and_groups_meta_rows(paths):
    # I-1 — two more `meta` rows beside `skills_json`, which keeps its flat shape.
    from loomcv.sync.parse_assets import SkillGroup, VaultHeader

    db_path, chroma_path = paths
    parsed = parse_star_vault(_read("header_vault.md"))
    assert parsed.errors() == []

    index_vault(parsed, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter())

    assert VaultHeader.model_validate_json(_meta_value(db_path, "header_json")) == parsed.header
    groups = [
        SkillGroup.model_validate(g) for g in json.loads(_meta_value(db_path, "skill_groups_json"))
    ]
    assert groups == parsed.skill_groups
    assert _meta_skills(db_path) == parsed.skills
    assert len(parsed.skills) == 7


def test_l3a_header_edit_is_a_meta_only_write(paths):
    # I-2 — the header is not a bullet: a Headline edit replaces the row and re-embeds nothing.
    from loomcv.sync.parse_assets import VaultHeader

    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    text = _read("header_vault.md")
    index_vault(parse_star_vault(text), db_path=db_path, chroma_path=chroma_path, adapter=fake)
    calls = fake.calls

    edited = text.replace("Senior Web Platform Engineer", "Principal Platform Engineer")
    report = index_vault(
        parse_star_vault(edited), db_path=db_path, chroma_path=chroma_path, adapter=fake
    )

    header = VaultHeader.model_validate_json(_meta_value(db_path, "header_json"))
    assert header.headline == "Principal Platform Engineer"
    assert (report.added, report.updated, report.metadata_only) == ([], [], [])
    assert report.embedded == 0
    assert fake.calls == calls


def test_l3a_pre_l3a_db_reads_empty_header_and_one_unnamed_group(paths):
    # I-3 — a DB synced before this ticket has neither row.
    from loomcv.retrieve.search import load_skill_groups, load_vault_header
    from loomcv.sync.parse_assets import SkillGroup, VaultHeader

    db_path, chroma_path = paths
    parsed = parse_star_vault(_read("header_vault.md"))
    index_vault(parsed, db_path=db_path, chroma_path=chroma_path, adapter=FakeEmbeddingAdapter())
    assert load_vault_header(db_path) == parsed.header
    assert load_skill_groups(db_path) == parsed.skill_groups

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM meta WHERE key IN ('header_json', 'skill_groups_json')")
    conn.commit()
    assert load_vault_header(db_path) == VaultHeader()
    assert load_skill_groups(db_path) == [SkillGroup(name="", items=parsed.skills)]

    conn.execute("UPDATE meta SET value = '[]' WHERE key = 'skills_json'")
    conn.commit()
    conn.close()
    assert load_skill_groups(db_path) == []


def test_l3a_regrouping_leaves_claims_inputs_unchanged(paths):
    # I-4 — the same items in other groups: `load_claims_inputs` (what `claims_of` reads) is equal.
    from loomcv.retrieve.search import load_claims_inputs, load_skill_groups

    db_path, chroma_path = paths
    fake = FakeEmbeddingAdapter()
    text = _read("header_vault.md")
    index_vault(parse_star_vault(text), db_path=db_path, chroma_path=chroma_path, adapter=fake)
    before = load_claims_inputs(db_path)

    regrouped = text.replace(
        "  * **Languages, Frameworks, and Data**: Python; TypeScript; React; PostgreSQL\n"
        "  * **Platforms and Delivery**: Kubernetes; Terraform; GitHub Actions; Python\n",
        "  * **Everything**: Python; TypeScript; React; PostgreSQL; Kubernetes\n"
        "  * **Delivery**: Terraform; GitHub Actions\n",
    )
    assert regrouped != text
    index_vault(parse_star_vault(regrouped), db_path=db_path, chroma_path=chroma_path, adapter=fake)

    assert [g.name for g in load_skill_groups(db_path)] == ["Everything", "Delivery"]
    assert load_claims_inputs(db_path) == before


# ------------------------------------------------- the vault fingerprint
#
# `sync_vault` writes the `meta` row `vault_sha256` = SHA-256 of the vault text as
# read, after every other write of the sync, and `SyncReport` is unchanged.

_SYNC_REPORT_FIELDS_AT_845A353 = {
    "added",
    "updated",
    "deleted",
    "metadata_only",
    "unchanged",
    "jobs_written",
    "jobs_deleted",
    "skills",
    "embedded",
    "schema_rebuild_from",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sync_file(vault: Path, paths, adapter) -> object:
    from loomcv.sync.index_assets import sync_vault

    db_path, chroma_path = paths
    return sync_vault(vault, db_path=db_path, chroma_path=chroma_path, adapter=adapter)


def test_f1_sync_writes_the_vault_fingerprint(vault_text, paths, tmp_path):
    from loomcv.sync.index_assets import SyncReport

    vault = tmp_path / "vault.md"
    vault.write_text(vault_text, encoding="utf-8")
    _sync_file(vault, paths, FakeEmbeddingAdapter())
    db_path, _ = paths
    assert _meta_value(db_path, "vault_sha256") == _sha256(vault.read_text(encoding="utf-8"))
    assert set(SyncReport.model_fields) == _SYNC_REPORT_FIELDS_AT_845A353


def test_f2_same_file_synced_twice_leaves_the_row(vault_text, paths, tmp_path):
    vault = tmp_path / "vault.md"
    vault.write_text(vault_text, encoding="utf-8")
    fake = FakeEmbeddingAdapter()
    _sync_file(vault, paths, fake)
    db_path, _ = paths
    first = _meta_value(db_path, "vault_sha256")
    second_report = _sync_file(vault, paths, fake)
    assert second_report.embedded == 0
    assert _meta_value(db_path, "vault_sha256") == first == _sha256(vault_text)


def test_f3_status_block_only_change_rewrites_the_row(vault_text, paths, tmp_path):
    vault = tmp_path / "vault.md"
    vault.write_text(vault_text, encoding="utf-8")
    fake = FakeEmbeddingAdapter()
    _sync_file(vault, paths, fake)
    changed = _bump_updated_date(vault_text)
    assert changed != vault_text
    vault.write_text(changed, encoding="utf-8")
    report = _sync_file(vault, paths, fake)
    db_path, _ = paths
    assert report.metadata_only
    assert report.embedded == 0
    assert _meta_value(db_path, "vault_sha256") == _sha256(changed)


def test_f4_failed_sync_leaves_no_row_or_the_old_one(vault_text, paths, tmp_path):
    db_path, chroma_path = paths
    vault = tmp_path / "vault.md"
    vault.write_text(vault_text, encoding="utf-8")

    # Fresh database: the embedding step raises → no fingerprint row.
    with pytest.raises(RuntimeError):
        _sync_file(vault, paths, FakeEmbeddingAdapter(raise_on_call=True))
    assert not db_path.exists() or _meta_value(db_path, "vault_sha256") is None

    # Existing database: synced text A, then text B (one bullet's text changed) fails → row = A.
    shutil.rmtree(chroma_path, ignore_errors=True)
    if db_path.exists():
        db_path.unlink()
    _sync_file(vault, paths, FakeEmbeddingAdapter())
    old = "Reduced imaginary search timeouts by 30%"
    assert old in vault_text
    text_b = vault_text.replace(old, "Reduced imaginary search timeouts by 35%")
    vault.write_text(text_b, encoding="utf-8")
    with pytest.raises(RuntimeError):
        _sync_file(vault, paths, FakeEmbeddingAdapter(raise_on_call=True))
    assert _meta_value(db_path, "vault_sha256") == _sha256(vault_text)
