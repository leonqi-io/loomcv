"""Index-backed search: SQLite + Chroma -> per-line funnel -> rerank -> merge.

The I/O half of the retrieval library layer (`rank.py` / `rerank.py` hold the
pure halves).

One `match` = one `adapter.embed` call over all requirement strings (must +
responsibilities + nice) + one exact `collection.get` of every loaded
candidate's stored embedding (the approximate
`collection.query` could silently drop ids) + one `reranker.score` call over the
sum of every line's own top `top_k` pairs + zero or one
coverage-mapping call: a profile that already carries its `coverage` is
replayed after `check_coverage`, otherwise the `coverage` adapter is called once (twice
after a retry) and the answer is stored on the returned profile. A second call of
the same kind runs on the lines graded `none`, over the vault's claims (`claims_of`; stored and
replayed like the coverage; never moves a grade, a gap or a slot). The merge across lines
(`merge.caps_for` + `merge.walk`) is pure and runs last, on the in-process results.

Similarity: raw cosine `dot / (|a| |b|)`, computed in process in
pure Python and returned in [-1, 1]; it does not depend on the collection's
distance space, so it is `vector_space_version`-neutral. A candidate with no
stored vector is reported on stderr and counted in `diagnostics.vector_missing`
by `funnel()`; it is never scored as 0.

Candidates come from SQLite only, never from Chroma metadata (Chroma holds
`bullet_id` + `content_hash` and nothing else); no Chroma `where=` clause is ever
built. A bullet's `keywords` are its own complete list, read by the BM25 and reranker
documents and by the coverage keyword list.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

from pydantic import BaseModel

from loomcv.coverage import check_claims, check_coverage, claims_of, map_coverage, vault_keywords
from loomcv.llm import GenerationAdapter
from loomcv.retrieve.merge import caps_for, walk
from loomcv.retrieve.rank import (
    Candidate,
    ClaimLine,
    LineResult,
    MatchDiagnostics,
    MatchResult,
    funnel,
    requirements_of,
)
from loomcv.retrieve.rerank import RerankAdapter, rerank_lines
from loomcv.state import JDProfile, RetrievalConfig
from loomcv.sync.embedding import EmbeddingAdapter
from loomcv.sync.index_assets import (
    SCHEMA_VERSION,
    SchemaVersionError,
    check_vector_space,
    get_collection,
    open_chroma_client,
)
from loomcv.sync.parse_assets import SkillGroup, VaultHeader

_CANDIDATES_SQL = """
SELECT b.bullet_id, b.text, b.tags_json, b.keywords_json, b.project, b.status,
       j.job_id, j.name, j.job_order
FROM bullets b JOIN jobs j USING (job_id)
ORDER BY j.job_order, b.bullet_order
"""


class IndexNotReadyError(RuntimeError):
    """No usable index: DB file / tables / Chroma dir / collection missing, or 0 bullets.

    The message always says to run `pipeline sync` first; the CLI maps it to
    exit 3.
    """


def load_candidates(db_path: Path) -> list[Candidate]:
    db_path = Path(db_path)
    if not db_path.exists():
        raise IndexNotReadyError(f"no index database at {db_path}; run `pipeline sync` first")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        except sqlite3.OperationalError as exc:
            raise IndexNotReadyError(
                f"{db_path} is not an initialised index ({exc}); run `pipeline sync` first"
            ) from exc
        if row is None:
            raise IndexNotReadyError(f"{db_path} has no schema_version; run `pipeline sync` first")
        if int(row["value"]) != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"SQLite schema_version {row['value']} != expected {SCHEMA_VERSION};"
                " run `pipeline sync` to migrate"
            )
        rows = conn.execute(_CANDIDATES_SQL).fetchall()
    finally:
        conn.close()

    candidates: list[Candidate] = []
    for r in rows:
        if r["status"] == "draft":
            continue
        candidates.append(
            Candidate(
                bullet_id=r["bullet_id"],
                text=r["text"],
                job_id=r["job_id"],
                job=r["name"],
                project=r["project"],
                tags=json.loads(r["tags_json"]),
                keywords=json.loads(r["keywords_json"]),
                job_order=r["job_order"],
            )
        )
    return candidates


def load_claims_inputs(db_path: Path) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """What the vault states beyond its bullets: every job's `Defaults` — bullet-less
    jobs included — as `(job_id, defaults)` in job order, and the vault-level `Skills` list
    (`[]` when the row is absent, i.e. an older DB). Call after `load_candidates`,
    which has checked the schema version."""
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        job_rows = conn.execute(
            "SELECT job_id, defaults_json FROM jobs ORDER BY job_order, job_id"
        ).fetchall()
        skills_row = conn.execute("SELECT value FROM meta WHERE key = 'skills_json'").fetchone()
    finally:
        conn.close()
    jobs = [(job_id, json.loads(defaults)) for job_id, defaults in job_rows]
    return jobs, json.loads(skills_row[0]) if skills_row is not None else []


def _meta_row(db_path: Path, key: str) -> str | None:
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    return row[0] if row is not None else None


def load_vault_header(db_path: Path) -> VaultHeader:
    """The vault's header lines; `VaultHeader()` for a DB synced before
    the `header_json` row existed."""
    raw = _meta_row(db_path, "header_json")
    return VaultHeader() if raw is None else VaultHeader.model_validate_json(raw)


def load_skill_groups(db_path: Path) -> list[SkillGroup]:
    """The vault's Skills groups in order. A DB synced before the
    `skill_groups_json` row existed reads as one unnamed group of its flat `skills_json` list
    (`[]` when that is empty)."""
    raw = _meta_row(db_path, "skill_groups_json")
    if raw is not None:
        return [SkillGroup.model_validate(g) for g in json.loads(raw)]
    flat = load_claims_inputs(db_path)[1]
    return [SkillGroup(name="", items=flat)] if flat else []


class JobRow(BaseModel):
    """One `jobs` row as the résumé header needs it."""

    job_id: str
    name: str
    employer: str | None
    title: str | None
    date_from: str
    date_to: str
    job_order: int


def load_job_rows(db_path: Path) -> list[JobRow]:
    """Every job of the index — bullet-less and all-draft jobs included (sync writes every
    parsed job) — ordered `job_order, job_id`. Read-only; call after `load_candidates`, which
    has checked the schema version (the `load_claims_inputs` pattern)."""
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT job_id, name, employer, title, date_from, date_to, job_order"
            " FROM jobs ORDER BY job_order, job_id"
        ).fetchall()
    finally:
        conn.close()
    return [
        JobRow(
            job_id=r[0],
            name=r[1],
            employer=r[2],
            title=r[3],
            date_from=r[4],
            date_to=r[5],
            job_order=r[6],
        )
        for r in rows
    ]


def vector_scores_for(
    requirements: list[str],
    routed_ids: list[str],
    *,
    chroma_path: Path,
    adapter: EmbeddingAdapter,
) -> dict[str, dict[str, float]]:
    chroma_path = Path(chroma_path)
    if not chroma_path.exists():
        raise IndexNotReadyError(f"no vector index at {chroma_path}; run `pipeline sync` first")
    client = open_chroma_client(chroma_path)
    collection, exists = get_collection(client)
    if not exists:
        raise IndexNotReadyError(
            f"no Chroma collection under {chroma_path}; run `pipeline sync` first"
        )
    check_vector_space(collection, adapter)

    unique_reqs = list(dict.fromkeys(requirements))
    if not unique_reqs or not routed_ids:
        return {r: {} for r in unique_reqs}

    res = collection.get(ids=list(routed_ids), include=["embeddings"])
    got_embeddings = res["embeddings"] if res["embeddings"] is not None else []
    stored: dict[str, list[float]] = {}
    for bid, emb in zip(res["ids"], got_embeddings, strict=True):
        # `len(emb)`, never `if not emb`: chromadb returns numpy rows, whose truth value raises.
        if emb is not None and len(emb) != 0:
            stored[bid] = [float(x) for x in emb]

    missing = [bid for bid in routed_ids if bid not in stored]
    if missing:
        print(
            f"warning: {len(missing)} bullets have no vector in the index: {', '.join(missing)}",
            file=sys.stderr,
        )
    if not stored:
        return {r: {} for r in unique_reqs}

    embeddings = adapter.embed(unique_reqs)
    return {
        req: {bid: _cosine(q, stored[bid]) for bid in routed_ids if bid in stored}
        for req, q in zip(unique_reqs, embeddings, strict=True)
    }


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return min(max(dot / norm, -1.0), 1.0)


def match(
    profile: JDProfile,
    *,
    db_path: Path,
    chroma_path: Path,
    adapter: EmbeddingAdapter,
    reranker: RerankAdapter,
    config: RetrievalConfig,
    coverage: GenerationAdapter | None = None,
) -> MatchResult:
    """Per-line funnel → one reranker call → the coverage grades → `MatchResult`.

    A profile with a non-empty `coverage` is replayed (zero adapter calls, checked by
    `check_coverage`, which raises `StoredCoverageError`); otherwise the `coverage` adapter
    is required and called once. Neither → `ValueError` before any store is opened.
    """
    if not profile.coverage and coverage is None:
        raise ValueError(
            "match() needs a profile that carries its coverage, or a coverage= adapter to map it"
        )
    candidates = load_candidates(db_path)
    if not candidates:
        raise IndexNotReadyError(f"index at {db_path} holds 0 bullets; run `pipeline sync` first")
    lines = requirements_of(profile)
    keywords = vault_keywords(candidates)

    # The coverage step runs before the embedding / rerank work, so a failed call wastes none.
    coverage_calls = 0
    if profile.coverage:
        check_coverage(profile.coverage, lines, keywords)
    else:
        mapping, coverage_calls = map_coverage(lines, keywords, adapter=coverage)
        profile = profile.model_copy(
            update={
                "coverage": mapping,
                "coverage_model": coverage.model_id if coverage_calls else None,
            }
        )

    # The claims step: the lines the evidence keywords left at `none`, asked once more
    # over the keywords the vault states but no bullet carries. Grades and `gaps` do not move.
    claims = claims_of(*load_claims_inputs(db_path), candidates)
    claim_keywords = [c.keyword for c in claims]
    asked = [r for r in lines if profile.coverage[r.text].match == "none"] if claims else []
    claims_calls = 0
    if profile.claims is not None:
        check_claims(profile.claims, asked, claim_keywords)
    else:
        if asked and coverage is None:
            raise ValueError(
                "match() needs a coverage= adapter to ask the claims of a profile that has none"
            )
        mapping, claims_calls = (
            map_coverage(asked, claim_keywords, adapter=coverage) if asked else ({}, 0)
        )
        profile = profile.model_copy(
            update={
                "claims": mapping,
                "claims_model": coverage.model_id if claims_calls else None,
            }
        )

    vector_scores = vector_scores_for(
        [r.text for r in lines],
        [c.bullet_id for c in candidates],
        chroma_path=chroma_path,
        adapter=adapter,
    )
    funnel_result = funnel(candidates, profile, vector_scores, config)
    ranked_lists, rerank_pairs = rerank_lines(funnel_result, candidates, reranker)

    results: list[LineResult] = []
    for line, ranked in zip(funnel_result.lines, ranked_lists, strict=True):
        answer = profile.coverage[line.text]
        results.append(
            LineResult(
                text=line.text,
                kind=line.kind,
                grade=answer.match,
                mapped_keywords=list(answer.keywords),
                ranked=ranked,
                max_cosine=line.max_cosine,
                max_bm25=line.max_bm25,
                top_rrf_score=line.top_rrf_score,
                parts=list(answer.parts),
            )
        )
    # Stage 3: the merge across lines. Jobs are the ones among the loaded candidates;
    # each line's whole RRF order is read in process, never from the payload.
    jobs = sorted({(c.job_id, c.job_order) for c in candidates}, key=lambda j: (j[1], j[0]))
    caps = caps_for(jobs, config.total_slots, config.recency_shares)
    selection = walk(
        results,
        candidates,
        {line.text: line.order for line in funnel_result.lines},
        config.total_slots,
        caps,
    )
    by_keyword = {c.keyword.casefold(): c for c in claims}
    claim_lines = [
        ClaimLine(
            line=r.text,
            kind=r.kind,
            match=answer.match,
            keywords=[by_keyword[k.casefold()] for k in answer.keywords],
        )
        for r in asked
        if (answer := profile.claims[r.text]).match != "none"
    ]
    return MatchResult(
        profile=profile,
        requirements=results,
        gaps=[r.text for r in results if r.kind == "must" and r.grade == "none"],
        claims=claim_lines,
        selection=selection,
        diagnostics=MatchDiagnostics(
            vector_missing=funnel_result.vector_missing,
            rerank_pairs=rerank_pairs,
            rerank_model=reranker.model_id,
            coverage_model=profile.coverage_model,
            coverage_calls=coverage_calls,
            claims_model=profile.claims_model,
            claims_calls=claims_calls,
        ),
    )
