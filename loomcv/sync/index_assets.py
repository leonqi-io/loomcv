"""SQLite + Chroma incremental index.

Turns a `VaultParseResult` (from `parse_assets.parse_star_vault`, strict
mode) into persisted state: SQLite is the source of record
(`02_database/pipeline_storage.db`), Chroma a derived, persistent vector
index (`02_database/vector_cache/`) reconciled from it — keyed on stable
`bullet_id` + `content_hash`.

Write ordering: embed once (batched) -> Chroma writes -> SQLite
committed last. A crash between the two only duplicates work on the next
sync (both are idempotent by ID), never diverges the stores permanently.

Schema v3: `jobs` replaces `projects`, a bullet carries its own
`project` label and `keywords_json`. Chroma holds the vector plus exactly
`{bullet_id, content_hash}` per bullet (nothing in `loomcv/` reads more):
a keyword, label, tag, status or job-level edit is a SQLite-only update
with zero re-embeds and no Chroma call.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.api.client import SharedSystemClient
from pydantic import BaseModel

from loomcv.sync.embedding import EmbeddingAdapter
from loomcv.sync.embedding import build_embedding_adapter as _build_embedding_adapter
from loomcv.sync.embedding import load_embedding_config as _load_embedding_config
from loomcv.sync.parse_assets import (
    ParsedBullet,
    ParsedJob,
    SkillGroup,
    VaultHeader,
    VaultParseResult,
    parse_star_vault,
)

SCHEMA_VERSION = 3
COLLECTION_NAME = "star_bullets"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    employer      TEXT,
    title         TEXT,
    date_from     TEXT NOT NULL,
    date_to       TEXT NOT NULL,
    defaults_json TEXT NOT NULL DEFAULT '[]',
    job_order     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bullets (
    bullet_id     TEXT PRIMARY KEY,
    job_id        TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    project       TEXT,
    text          TEXT NOT NULL,
    tags_json     TEXT NOT NULL DEFAULT '[]',
    keywords_json TEXT NOT NULL DEFAULT '[]',
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'confirmed',
    updated       TEXT,
    bullet_order  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bullets_job ON bullets(job_id);
"""


class VaultParseError(RuntimeError):
    """Strict-mode parse abort: `parse_result.errors()` is non-empty."""


class VectorSpaceMismatchError(RuntimeError):
    """Running embedding config disagrees with the existing collection (§3.4)."""


class StoreInconsistentError(RuntimeError):
    """SQLite knows bullet_ids Chroma does not — the derived index is stale/gone (D-3)."""


class SchemaVersionError(RuntimeError):
    """`meta.schema_version` in an existing DB does not match `SCHEMA_VERSION` (D-2)."""


class SyncReport(BaseModel):
    added: list[str] = []
    updated: list[str] = []
    deleted: list[str] = []
    metadata_only: list[str] = []
    unchanged: list[str] = []
    jobs_written: list[str] = []
    jobs_deleted: list[str] = []
    # the number of items on the vault-level `Skills` line, as stored
    skills: int = 0
    embedded: int = 0
    # The stored schema version when a migration step forced a full rebuild
    # (feeds the CLI's stderr line); deliberately not part of `counts()`.
    schema_rebuild_from: int | None = None

    def counts(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "updated": len(self.updated),
            "deleted": len(self.deleted),
            "metadata_only": len(self.metadata_only),
            "unchanged": len(self.unchanged),
            "jobs_written": len(self.jobs_written),
            "jobs_deleted": len(self.jobs_deleted),
            "skills": self.skills,
        }


# --------------------------------------------------------------- SQLite


def _ensure_schema(conn: sqlite3.Connection) -> int | None:
    """Create or migrate the SQLite schema to ``SCHEMA_VERSION``.

    Returns the stored version when a migration step forced a full rebuild
    (the 2 -> 3 step does: every ``bullet_id`` changed prefix), else ``None``.
    A version above ``SCHEMA_VERSION``, or a gap in ``_MIGRATIONS``, raises
    ``SchemaVersionError`` before any step runs.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )
        conn.commit()
        return None

    stored = int(row["value"])
    if stored > SCHEMA_VERSION or any(
        version not in _MIGRATIONS for version in range(stored, SCHEMA_VERSION)
    ):
        raise SchemaVersionError(
            f"SQLite schema_version {row['value']} != expected {SCHEMA_VERSION};"
            " delete 02_database/ and re-sync"
        )

    rebuild_required = False
    version = stored
    while version < SCHEMA_VERSION:
        rebuild_required |= _MIGRATIONS[version](conn)
        version += 1
    # Idempotent; runs after the steps (the v3 index on bullets(job_id) would
    # fail against a v2 bullets table).
    conn.executescript(_SCHEMA_SQL)
    return stored if rebuild_required else None


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> bool:
    """In-place v1 -> v2 upgrade: add the three `projects` columns and
    bump meta.schema_version to 2. Idempotent — existing columns are skipped via
    PRAGMA table_info, so a crash between the ALTERs and the meta bump recovers.
    Returns False: no rebuild is needed by this step alone."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(projects)")}
    if "domain" not in existing:
        conn.execute("ALTER TABLE projects ADD COLUMN domain TEXT")
    if "year" not in existing:
        conn.execute("ALTER TABLE projects ADD COLUMN year INTEGER")
    if "tech_json" not in existing:
        conn.execute("ALTER TABLE projects ADD COLUMN tech_json TEXT NOT NULL DEFAULT '[]'")
    conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
    conn.commit()
    return False


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> bool:
    """v2 -> v3 is a rebuild, not a row migration: every
    `bullet_id` changes prefix under v3, so no v2 row survives. Drops `bullets` and
    `projects`, creates the v3 tables and bumps meta. Returns True: `index_vault`
    takes the `_sync_rebuild` path (which also drops the old Chroma collection)."""
    conn.execute("DROP TABLE IF EXISTS bullets")
    conn.execute("DROP TABLE IF EXISTS projects")
    conn.executescript(_SCHEMA_SQL)
    conn.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
    conn.commit()
    return True


# Keyed by the version each step upgrades *from*.
_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], bool]] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


def _read_current_bullets(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {row["bullet_id"]: row for row in conn.execute("SELECT * FROM bullets")}


def _read_current_jobs(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {row["job_id"]: row for row in conn.execute("SELECT * FROM jobs")}


def _write_bullet(conn: sqlite3.Connection, job_id: str, bullet: ParsedBullet) -> None:
    updated_iso = bullet.updated.isoformat() if bullet.updated else None
    conn.execute(
        "INSERT OR REPLACE INTO bullets"
        " (bullet_id, job_id, project, text, tags_json, keywords_json, content_hash,"
        " status, updated, bullet_order)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            bullet.bullet_id,
            job_id,
            bullet.project,
            bullet.text,
            json.dumps(bullet.tags),
            json.dumps(bullet.keywords),
            bullet.content_hash,
            bullet.status,
            updated_iso,
            bullet.order,
        ),
    )


def _job_row_tuple(job: ParsedJob) -> tuple:
    """The stored job columns, in the order compared to decide `jobs_written`."""
    return (
        job.name,
        job.employer,
        job.title,
        job.date_from,
        job.date_to,
        json.dumps(job.defaults),
        job.order,
    )


def _write_job(conn: sqlite3.Connection, job: ParsedJob) -> None:
    conn.execute(
        "INSERT INTO jobs"
        " (job_id, name, employer, title, date_from, date_to, defaults_json, job_order)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(job_id) DO UPDATE SET name = excluded.name, employer = excluded.employer,"
        " title = excluded.title, date_from = excluded.date_from, date_to = excluded.date_to,"
        " defaults_json = excluded.defaults_json, job_order = excluded.job_order",
        (job.job_id, *_job_row_tuple(job)),
    )


def _write_skills(
    conn: sqlite3.Connection,
    skills: list[str],
    header: VaultHeader,
    skill_groups: list[SkillGroup],
) -> None:
    """The vault-level lines as `meta` rows, replaced on every sync: the flat
    `Skills` list (`skills_json`, unchanged), and the header
    (`header_json`) and the groups (`skill_groups_json`). SQLite-only: no vector depends on
    them."""
    rows = (
        ("skills_json", json.dumps(skills)),
        ("header_json", header.model_dump_json()),
        ("skill_groups_json", json.dumps([g.model_dump() for g in skill_groups])),
    )
    conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", rows)


# --------------------------------------------------------------- Chroma


def open_chroma_client(chroma_path: Path) -> chromadb.ClientAPI:
    if not chroma_path.exists():
        # A hand-deleted chroma_path can otherwise be masked by chromadb's
        # per-process SharedSystemClient cache, which still remembers the old
        # (now-gone) collection for this path — see D-3 store-consistency
        # guard and the plan record's "Known pitfall" note.
        SharedSystemClient.clear_system_cache()
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def get_collection(client: chromadb.ClientAPI):
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=None), True
    except chromadb.errors.NotFoundError:
        return None, False


def _create_collection(client: chromadb.ClientAPI, adapter: EmbeddingAdapter):
    return client.create_collection(
        COLLECTION_NAME,
        embedding_function=None,
        metadata={
            "model_id": adapter.model_id,
            "vector_space_version": adapter.vector_space_version,
        },
    )


def check_vector_space(collection, adapter: EmbeddingAdapter) -> None:
    meta = collection.metadata or {}
    if (
        meta.get("model_id") != adapter.model_id
        or meta.get("vector_space_version") != adapter.vector_space_version
    ):
        raise VectorSpaceMismatchError(
            f"running embedding config (model={adapter.model_id!r},"
            f" vector_space_version={adapter.vector_space_version}) does not match the"
            f" existing collection (model={meta.get('model_id')!r},"
            f" vector_space_version={meta.get('vector_space_version')!r});"
            " rebuild=True is required"
        )


def _check_store_consistency(collection, current_bullet_ids) -> None:
    current_bullet_ids = list(current_bullet_ids)
    if not current_bullet_ids:
        return
    present = set(collection.get(ids=current_bullet_ids, include=[])["ids"])
    missing = sorted(set(current_bullet_ids) - present)
    if missing:
        raise StoreInconsistentError(
            "SQLite has bullet_ids missing from the Chroma collection"
            f" (the vector index is stale or was deleted out-of-band): {missing};"
            " rebuild=True is required"
        )


def _bullet_metadata(bullet: ParsedBullet) -> dict:
    """Per-bullet Chroma metadata: exactly the two keys that change only with the
    text. Everything else lives in SQLite, the source of record."""
    return {"bullet_id": bullet.bullet_id, "content_hash": bullet.content_hash}


# ----------------------------------------------------------------- diff


def _diff(
    current_bullets: dict[str, sqlite3.Row],
    jobs: list[ParsedJob],
) -> tuple[dict[str, tuple[ParsedJob, ParsedBullet]], dict[str, list[str]]]:
    incoming: dict[str, tuple[ParsedJob, ParsedBullet]] = {}
    for job in jobs:
        for bullet in job.bullets:
            incoming[bullet.bullet_id] = (job, bullet)

    current_ids = set(current_bullets)
    incoming_ids = set(incoming)

    added = sorted(incoming_ids - current_ids)
    deleted = sorted(current_ids - incoming_ids)
    updated: list[str] = []
    metadata_only: list[str] = []
    unchanged: list[str] = []

    for bullet_id in sorted(current_ids & incoming_ids):
        current_row = current_bullets[bullet_id]
        _job, bullet = incoming[bullet_id]
        if current_row["content_hash"] != bullet.content_hash:
            updated.append(bullet_id)
            continue
        stored = (
            current_row["status"],
            current_row["updated"] or "",
            current_row["bullet_order"],
            current_row["project"],
            current_row["keywords_json"],
            current_row["tags_json"],
        )
        incoming_state = (
            bullet.status,
            bullet.updated.isoformat() if bullet.updated else "",
            bullet.order,
            bullet.project,
            json.dumps(bullet.keywords),
            json.dumps(bullet.tags),
        )
        if stored != incoming_state:
            metadata_only.append(bullet_id)
        else:
            unchanged.append(bullet_id)

    return incoming, {
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "metadata_only": metadata_only,
        "unchanged": unchanged,
    }


# ------------------------------------------------------------ sync paths


def _sync_incremental(
    conn: sqlite3.Connection,
    client: chromadb.ClientAPI,
    collection,
    current_bullets: dict[str, sqlite3.Row],
    current_jobs: dict[str, sqlite3.Row],
    jobs: list[ParsedJob],
    *,
    skills: list[str],
    header: VaultHeader,
    skill_groups: list[SkillGroup],
    adapter: EmbeddingAdapter,
) -> SyncReport:
    incoming, diff = _diff(current_bullets, jobs)
    to_embed_ids = diff["added"] + diff["updated"]

    if to_embed_ids:
        vectors = adapter.embed([incoming[bid][1].text for bid in to_embed_ids])
        embedding_map = dict(zip(to_embed_ids, vectors))
    else:
        embedding_map = {}

    if collection is None:
        collection = _create_collection(client, adapter)

    if diff["deleted"]:
        collection.delete(ids=diff["deleted"])
    if to_embed_ids:
        collection.upsert(
            ids=to_embed_ids,
            embeddings=[embedding_map[bid] for bid in to_embed_ids],
            metadatas=[_bullet_metadata(incoming[bid][1]) for bid in to_embed_ids],
        )
    # A metadata-only change (status, updated, order, label, keywords, tags) writes
    # SQLite only — Chroma carries nothing it could change.

    # Every parsed job gets its row, bullet-less ones included: no bullet row
    # joins to it, so the funnel and the merge never see it.
    incoming_jobs = {j.job_id: j for j in jobs}
    jobs_deleted = sorted(set(current_jobs) - set(incoming_jobs))
    jobs_written = []
    for job_id, job in incoming_jobs.items():
        existing = current_jobs.get(job_id)
        if existing is None or (
            existing["name"],
            existing["employer"],
            existing["title"],
            existing["date_from"],
            existing["date_to"],
            existing["defaults_json"],
            existing["job_order"],
        ) != _job_row_tuple(job):
            jobs_written.append(job_id)

    for job_id in jobs_written:
        _write_job(conn, incoming_jobs[job_id])
    for bullet_id in diff["deleted"]:
        conn.execute("DELETE FROM bullets WHERE bullet_id = ?", (bullet_id,))
    for bullet_id in to_embed_ids + diff["metadata_only"]:
        job, bullet = incoming[bullet_id]
        _write_bullet(conn, job.job_id, bullet)
    for job_id in jobs_deleted:
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
    _write_skills(conn, skills, header, skill_groups)

    conn.commit()

    return SyncReport(
        added=diff["added"],
        updated=diff["updated"],
        deleted=diff["deleted"],
        metadata_only=diff["metadata_only"],
        unchanged=diff["unchanged"],
        jobs_written=sorted(jobs_written),
        jobs_deleted=jobs_deleted,
        skills=len(skills),
        embedded=len(to_embed_ids),
    )


def _sync_rebuild(
    conn: sqlite3.Connection,
    client: chromadb.ClientAPI,
    jobs: list[ParsedJob],
    *,
    skills: list[str],
    header: VaultHeader,
    skill_groups: list[SkillGroup],
    adapter: EmbeddingAdapter,
) -> SyncReport:
    incoming = sorted(
        ((job, bullet) for job in jobs for bullet in job.bullets),
        key=lambda jb: jb[1].bullet_id,
    )

    # Embed everything first, while both stores are still fully intact: a
    # mid-embed failure here must leave the old index untouched.
    vectors = adapter.embed([bullet.text for _, bullet in incoming]) if incoming else []

    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    collection = _create_collection(client, adapter)
    if incoming:
        collection.upsert(
            ids=[bullet.bullet_id for _, bullet in incoming],
            embeddings=vectors,
            metadatas=[_bullet_metadata(bullet) for _, bullet in incoming],
        )

    conn.execute("DELETE FROM bullets")
    conn.execute("DELETE FROM jobs")
    incoming_jobs = {j.job_id: j for j in jobs}
    for job in incoming_jobs.values():
        _write_job(conn, job)
    for job, bullet in incoming:
        _write_bullet(conn, job.job_id, bullet)
    _write_skills(conn, skills, header, skill_groups)

    conn.commit()

    return SyncReport(
        added=[bullet.bullet_id for _, bullet in incoming],
        jobs_written=sorted(incoming_jobs),
        skills=len(skills),
        embedded=len(incoming),
    )


# ------------------------------------------------------------- entry points


def index_vault(
    parse_result: VaultParseResult,
    *,
    db_path: Path,
    chroma_path: Path,
    adapter: EmbeddingAdapter,
    rebuild: bool = False,
) -> SyncReport:
    if parse_result.errors():
        details = "; ".join(f"line {i.line_no}: {i.message}" for i in parse_result.errors())
        raise VaultParseError(
            f"vault has parse errors; aborting before any store is touched: {details}"
        )

    db_path = Path(db_path)
    chroma_path = Path(chroma_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        schema_rebuild_from = _ensure_schema(conn)
        # A migration step that renamed every bullet_id (2 -> 3) forces the rebuild
        # path whatever the caller passed: the old collection is dropped there.
        rebuild = rebuild or schema_rebuild_from is not None

        client = open_chroma_client(chroma_path)
        collection, collection_exists = get_collection(client)

        current_bullets = _read_current_bullets(conn)
        current_jobs = _read_current_jobs(conn)
        incoming_ids = {b.bullet_id for j in parse_result.jobs for b in j.bullets}

        if not rebuild:
            if collection_exists:
                check_vector_space(collection, adapter)
                # Only guard ids the vault still wants: a crash between a
                # Chroma delete() and the SQLite commit leaves SQLite ahead
                # of Chroma on ids that were being deleted anyway (D-3 says
                # a crash there only duplicates work, never diverges) — the
                # next sync's delete + no-op re-delete recovers those on its
                # own. Ids SQLite still wants but Chroma lacks are the real
                # inconsistency this guard exists to catch.
                _check_store_consistency(collection, current_bullets.keys() & incoming_ids)
            elif current_bullets:
                raise StoreInconsistentError(
                    "SQLite has bullets but no Chroma collection exists at all;"
                    " rebuild=True is required to recover"
                )

        if rebuild:
            report = _sync_rebuild(
                conn,
                client,
                parse_result.jobs,
                skills=parse_result.skills,
                header=parse_result.header,
                skill_groups=parse_result.skill_groups,
                adapter=adapter,
            )
            report.schema_rebuild_from = schema_rebuild_from
            return report
        return _sync_incremental(
            conn,
            client,
            collection,
            current_bullets,
            current_jobs,
            parse_result.jobs,
            skills=parse_result.skills,
            header=parse_result.header,
            skill_groups=parse_result.skill_groups,
            adapter=adapter,
        )
    finally:
        conn.close()


def sync_vault(
    vault_path: Path,
    *,
    db_path: Path,
    chroma_path: Path,
    adapter: EmbeddingAdapter | None = None,
    config_path: Path | None = None,
    include_draft: bool = False,
    rebuild: bool = False,
) -> SyncReport:
    text = Path(vault_path).read_text(encoding="utf-8")
    parse_result = parse_star_vault(text, include_draft=include_draft)
    if adapter is None:
        adapter = _build_embedding_adapter(_load_embedding_config(config_path))
    report = index_vault(
        parse_result,
        db_path=db_path,
        chroma_path=chroma_path,
        adapter=adapter,
        rebuild=rebuild,
    )
    # Written last, in its own commit: any failure inside
    # `index_vault` leaves no row on a fresh database and the previous row on an existing one.
    _write_fingerprint(Path(db_path), vault_fingerprint(text))
    return report


def vault_fingerprint(text: str) -> str:
    """SHA-256 of the vault text as read (UTF-8), hex. The one definition: the sync writes it,
    `loomcv.app_state` compares against it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_fingerprint(db_path: Path, fingerprint: str) -> None:
    """INSERT OR REPLACE the fingerprint's `meta` row; own connection, own commit. No schema
    bump, no `SyncReport` field, no counter."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('vault_sha256', ?)", (fingerprint,)
        )
        conn.commit()
    finally:
        conn.close()
