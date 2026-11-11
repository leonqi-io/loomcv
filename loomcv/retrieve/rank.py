"""Pure per-line funnel functions for the retrieval layer.

No I/O: candidates and pre-computed vector similarities are passed in by the
caller (`search.match()` fills `vector_scores` from Chroma). Implements stage 1
of the funnel, with rank fusion (RRF):

- Requirements: must-haves, responsibilities, nice-to-haves, in that order, first
  occurrence wins; constraints and traits are never read. A line has no weight.
- One funnel per line, nothing shared across lines: every candidate is ranked once
  by raw cosine over its text embedding and once by BM25 over
  `" ".join(keywords) + " " + text`, fused with unweighted Reciprocal Rank Fusion
  (`rrf_k`); the line's top `top_k` go on to the reranker. The RRF value is a rank
  artefact, never compared to a threshold.

Nothing here decides evidence or gaps: the coverage grade (`loomcv/coverage.py`) is
a model's answer, and `rerank.rerank_lines()` only orders each line's own cut.
The result shapes of the whole match path live here so every stage shares them, the
merge's (`Selection` and its parts, filled by `merge.py`) among them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from rank_bm25 import BM25Okapi

from loomcv import resources
from loomcv.state import CoverageMatch, CoveragePart, JDProfile, RetrievalConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS_PATH = resources.path("stopwords_en.txt")


def _load_stopwords(path: Path) -> frozenset[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImportError(f"cannot load stopword list {path}: {exc}") from exc
    words = (line.split("|", 1)[0].strip().lower() for line in text.splitlines())
    return frozenset(w for w in words if w)


# The Snowball English stop list, vendored verbatim; dropped from
# requirement and bullet text alike.
STOPWORDS: frozenset[str] = _load_stopwords(_STOPWORDS_PATH)

RequirementKind = Literal["must", "responsibility", "nice"]


class Candidate(BaseModel):
    bullet_id: str
    text: str
    job_id: str
    job: str
    project: str | None = None
    tags: list[str] = []
    # The bullet's own complete keyword list; read by the BM25 and reranker documents.
    keywords: list[str] = []
    # `jobs.job_order` (0 = most recent); read by the slot step only, never by an order.
    # The default keeps hand-built candidates valid; `load_candidates` always sets the real value.
    job_order: int = 0


class Requirement(BaseModel):
    text: str
    kind: RequirementKind


class LineRanking(BaseModel):
    """One line's funnel: its RRF order, the top `top_k` of it, and the raw signals' maxima.

    `order` is the line's whole `(-rrf, bullet_id)` order (every bullet with RRF > 0); the
    merge reads it in process for rule D's third key. `top == order[:top_k]`.
    """

    text: str
    kind: RequirementKind
    order: list[str]
    top: list[str]
    top_rrf_score: float | None
    max_cosine: float | None
    max_bm25: float | None


class FunnelResult(BaseModel):
    lines: list[LineRanking]
    vector_missing: int


class RankedBullet(BaseModel):
    bullet_id: str
    job: str
    project: str | None
    text: str
    # 1-based position in the line's RRF order / in the reranker's order over that cut
    rrf_rank: int
    rerank_score: float
    rerank_rank: int


class LineResult(BaseModel):
    text: str
    kind: RequirementKind
    grade: CoverageMatch
    mapped_keywords: list[str]
    ranked: list[RankedBullet]
    max_cosine: float | None
    max_bm25: float | None
    top_rrf_score: float | None
    # The named parts of an all-required compound line, from the coverage answer; a
    # payload written before this field loads with [].
    parts: list[CoveragePart] = []


class MatchDiagnostics(BaseModel):
    vector_missing: int
    rerank_pairs: int
    rerank_model: str
    coverage_model: str | None
    # adapter calls actually made: 0 on a replay, 1 after a call, 2 after a retry
    coverage_calls: int
    # the claims call, counted the same way
    claims_model: str | None
    claims_calls: int


class Claim(BaseModel):
    """A stated keyword no loaded bullet carries, and where it is stated."""

    keyword: str
    # job_ids in job order, then "skills" for the vault-level line
    at: list[str]


class ClaimLine(BaseModel):
    """A line graded `none` on the bullets' keywords that the vault's claims cover."""

    line: str
    kind: RequirementKind
    match: Literal["exact", "adjacent"]
    keywords: list[Claim]


class SelectedBullet(BaseModel):
    """A placed bullet with the rule D inputs that put it first on its line at that pass."""

    # `pass` is a Python keyword: the attribute is `pass_number`, the JSON key is `pass`.
    model_config = ConfigDict(serialize_by_alias=True)

    bullet_id: str
    project: str | None
    text: str
    line: str
    kind: RequirementKind
    pass_number: int = Field(
        serialization_alias="pass", validation_alias=AliasChoices("pass_number", "pass")
    )
    walk_position: int
    overlap: int
    rerank_rank: int | None
    rrf_position: int | None
    covers: list[str]


class JobSelection(BaseModel):
    job_id: str
    job: str
    job_order: int
    cap: int
    placed: int
    bullets: list[SelectedBullet]


class WalkStep(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    bullet_id: str
    job_id: str
    line: str
    kind: RequirementKind
    pass_number: int = Field(
        serialization_alias="pass", validation_alias=AliasChoices("pass_number", "pass")
    )


class UnplacedLine(BaseModel):
    line: str
    kind: RequirementKind
    reason: Literal["no_candidates", "all_taken", "budget"]


class UncoveredLine(BaseModel):
    line: str
    kind: RequirementKind


class Selection(BaseModel):
    """The merge across lines: placed bullets by job, the walk, and the coverage lookups.

    No field is a score; `covers` / `covered_by` / `uncovered` are set lookups over the lines'
    mapped keywords.
    """

    total_slots: int
    jobs: list[JobSelection]
    walk: list[WalkStep]
    unplaced: list[UnplacedLine]
    covered_by: dict[str, list[str]]
    uncovered: list[UncoveredLine]
    passes: int


class MatchResult(BaseModel):
    profile: JDProfile
    requirements: list[LineResult]
    gaps: list[str]
    claims: list[ClaimLine]
    selection: Selection
    diagnostics: MatchDiagnostics


class MatchPayload(BaseModel):
    """The JSON `pipeline match` prints, read back by
    `pipeline draft`. `config` is the `RetrievalConfig` the run used; a
    missing key fails validation, unknown keys are ignored."""

    profile: JDProfile
    parse_warnings: list[str]
    requirements: list[LineResult]
    gaps: list[str]
    claims: list[ClaimLine]
    selection: Selection
    config: RetrievalConfig
    diagnostics: MatchDiagnostics


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in STOPWORDS]


def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]]) -> list[float]:
    if not query_tokens or not docs_tokens:
        return [0.0] * len(docs_tokens)
    bm25 = BM25Okapi(docs_tokens)
    return [float(x) for x in bm25.get_scores(query_tokens)]


def ranks(values: list[float], *, skip_zero: bool) -> list[int | None]:
    """Dense 1-based ranks, highest value first; equal values share a rank.

    With `skip_zero`, every value <= 0 is unranked (`None`) — `rank_bm25` can
    return negative scores on small corpora.
    """

    def ranked(v: float) -> bool:
        return not (skip_zero and v <= 0)

    distinct = sorted({v for v in values if ranked(v)}, reverse=True)
    position = {v: i + 1 for i, v in enumerate(distinct)}
    return [position[v] if ranked(v) else None for v in values]


def requirements_of(profile: JDProfile) -> list[Requirement]:
    """must + responsibilities + nice, in that order.

    A text already seen (byte-equal, in any list) is skipped — first occurrence wins.
    `traits` and `constraints` are never read.
    """
    ordered: list[tuple[str, RequirementKind]] = [
        *((r, "must") for r in profile.must_have),
        *((r, "responsibility") for r in profile.responsibilities),
        *((r, "nice") for r in profile.nice_to_have),
    ]
    seen: set[str] = set()
    result: list[Requirement] = []
    for text, kind in ordered:
        if text in seen:
            continue
        seen.add(text)
        result.append(Requirement(text=text, kind=kind))
    return result


def bm25_document(candidate: Candidate) -> str:
    """The BM25 document: the keyword list, then the text (`run_t1.py` 294, literal)."""
    return " ".join(candidate.keywords) + " " + candidate.text


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def funnel(
    candidates: list[Candidate],
    profile: JDProfile,
    vector_scores: dict[str, dict[str, float]],
    config: RetrievalConfig,
) -> FunnelResult:
    """Fuse per line with unweighted RRF over all candidates; the line's cut = top `top_k`.

    RRF 0 never counts and ties break by `bullet_id`. `vector_missing` counts
    candidates absent from every line's `vector_scores` entry; `search.vector_scores_for`
    returns the same id set for every line, so in `match()` this is the number of
    bullets with no vector in the index.
    """
    docs_tokens = [tokenize(bm25_document(c)) for c in candidates]
    k = config.rrf_k
    requirements = requirements_of(profile)
    lines: list[LineRanking] = []

    for req in requirements:
        bm = bm25_scores(tokenize(req.text), docs_tokens)
        r_bm = ranks(bm, skip_zero=True)

        req_vec_scores = vector_scores.get(req.text, {})
        cos: list[float | None] = [
            min(max(req_vec_scores[c.bullet_id], -1.0), 1.0)
            if c.bullet_id in req_vec_scores
            else None
            for c in candidates
        ]
        present = [v for v in cos if v is not None]
        present_ranks = iter(ranks(present, skip_zero=False))
        r_vec = [next(present_ranks) if v is not None else None for v in cos]

        rrf: dict[str, float] = {}
        for i, c in enumerate(candidates):
            vec_term = 1 / (k + r_vec[i]) if r_vec[i] is not None else 0.0
            bm_term = 1 / (k + r_bm[i]) if r_bm[i] is not None else 0.0
            rrf[c.bullet_id] = vec_term + bm_term

        order = sorted((bid for bid, v in rrf.items() if v > 0), key=lambda b: (-rrf[b], b))
        lines.append(
            LineRanking(
                text=req.text,
                kind=req.kind,
                order=order,
                top=order[: config.top_k],
                top_rrf_score=_round(rrf[order[0]]) if order else None,
                max_cosine=_round(max(present)) if present else None,
                max_bm25=_round(max(bm)) if bm else None,
            )
        )

    vector_missing = (
        sum(
            1
            for c in candidates
            if all(c.bullet_id not in vector_scores.get(r.text, {}) for r in requirements)
        )
        if requirements
        else 0
    )
    return FunnelResult(lines=lines, vector_missing=vector_missing)
