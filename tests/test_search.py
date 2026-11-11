"""Tests for index-backed search (`loomcv/retrieve/search.py`).

Fixtures are synthetic and read-only (`tests/fixtures/metadata_vault.md`,
`tests/fixtures/index_vault.md`); every store is synced into `tmp_path` with the
in-process `FakeEmbeddingAdapter` below — zero network, zero model downloads.

`loomcv.retrieve.search` is imported **inside** each test (via `_search()`), not at
module scope, so a missing module fails each test at red instead of aborting the
whole run at collection.

The test `RetrievalConfig` (`_cfg()`) is explicit, independent of `retrieval.yaml`
tuning (S3); a factory, so a config-shape change fails each test at red instead of
the whole module at import. `match()` runs one funnel per
requirement line (BM25 over keywords + text, vector, RRF, the line's top `top_k`) and
reranks only those pairs, in one call: `FakeRerankAdapter` scores a pair as
`(1 + shared lower-case tokens) / 10`, so every pair is > 0 and P4's `build queue`
scores highest on P4 — tests hold by arithmetic, not by luck of the sha1 hash.
The sha1 fake vectors are non-negative, so every bullet has a cosine and an RRF
value > 0 on every line. Grades and `gaps` come from the coverage mapping:
`FakeCoverageAdapter` answers the rendered prompt from a `{line: (match, keywords)}`
rule, by default `none` / `[]` for every line.

All fixture content is fictional — no real career data.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from loomcv.state import JDProfile, RetrievalConfig
from loomcv.sync.index_assets import index_vault
from loomcv.sync.parse_assets import parse_star_vault

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The four metadata-fixture bullets are told apart by their
# `* **Project**:` label (the project level is gone).
P1 = "Helios Player Rewrite"
P2 = "Zephyr Batch Loader"
P3 = "Tundra Alert Console"
P4 = "Quartz Build Fleet"

P4_PHRASE = "build queue"
NONSENSE = "quantum zebra choreography"


def _cfg(**overrides) -> RetrievalConfig:
    data = {
        "top_k": 2,
        "rrf_k": 60,
        "total_slots": 42,
        "recency_shares": [1],
    }
    data.update(overrides)
    return RetrievalConfig(**data)


def _search():
    import loomcv.retrieve.search as search

    return search


@pytest.fixture(autouse=True)
def _no_default_embedding_function(monkeypatch):
    """No code path may ever call Chroma's default (ONNX) embedding fn.

    Copied from `tests/test_index.py` (re-declared locally, not imported).
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


class FakeRerankAdapter:
    """Deterministic, network-free stand-in for `RerankAdapter`; records every call."""

    def __init__(self, *, model_id: str = "fake-reranker") -> None:
        self._model_id = model_id
        self.calls: list[list[tuple[str, str]]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def score(self, pairs):
        self.calls.append(list(pairs))
        return [(1 + len(_words(q) & _words(d))) / 10 for q, d in pairs]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _none_rule(_line: str) -> tuple[str, list[str]]:
    return "none", []


def _requested_lines(user: str) -> list[str]:
    """The `- ` lines between `Requirements` and `Keywords (n)` of a rendered coverage prompt."""
    block = user.split("Requirements\n", 1)[1].split("\n\nKeywords (", 1)[0]
    return [ln[2:] for ln in block.splitlines()]


class FakeCoverageAdapter:
    """Network-free stand-in for the coverage node's `GenerationAdapter`.

    `rule(line_text) -> (match, keywords)`; default: every line `none` with no keywords.
    """

    def __init__(self, rule=None, *, model_id: str = "fake-coverage") -> None:
        self._rule = rule or _none_rule
        self._model_id = model_id
        self.calls: list[tuple[str, str, float]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def temperature(self) -> float:
        return 0.0

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        import json

        self.calls.append((system, user, temperature))
        reply = {}
        for line in _requested_lines(user):
            match, keywords = self._rule(line)
            reply[line] = {"match": match, "keywords": keywords}
        return json.dumps(reply)


class RaisingCoverageAdapter:
    """A coverage adapter that must never be called (stored-coverage replay)."""

    model_id = "never-called"
    temperature = 0.0

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        raise AssertionError("the coverage adapter must not run with a stored coverage")


# ------------------------------------------------------------------ helpers


def _parse(fixture: str, *, include_draft=False):
    text = (FIXTURES / fixture).read_text(encoding="utf-8")
    return parse_star_vault(text, include_draft=include_draft)


def _sync(tmp_path: Path, fixture: str, *, include_draft: bool = False):
    """Parse + index a fixture into `tmp_path`; return (db, chroma, parse_result)."""
    db, chroma = tmp_path / "pipeline_storage.db", tmp_path / "vector_cache"
    parse_result = _parse(fixture, include_draft=include_draft)
    index_vault(parse_result, db_path=db, chroma_path=chroma, adapter=FakeEmbeddingAdapter())
    return db, chroma, parse_result


def _sync_metadata(tmp_path: Path):
    return _sync(tmp_path, "metadata_vault.md")


def _profile(**overrides) -> JDProfile:
    data = {
        "title": "Synthetic Engineer",
        "seniority": "senior",
        "domain": "fictional",
        "must_have": ["fictional"],
    }
    data.update(overrides)
    return JDProfile(**data)


def _match(profile, db, chroma, config, adapter=None, reranker=None, coverage=None):
    """Run `match()`; a profile without a stored coverage gets a default fake coverage adapter."""
    if coverage is None and not profile.coverage:
        coverage = FakeCoverageAdapter()
    return _search().match(
        profile,
        db_path=db,
        chroma_path=chroma,
        adapter=adapter or FakeEmbeddingAdapter(),
        reranker=reranker or FakeRerankAdapter(),
        config=config,
        coverage=coverage,
    )


def _ranked_ids(result) -> set[str]:
    return {b.bullet_id for line in result.requirements for b in line.ranked}


def _project_of(parse_result) -> dict[str, str | None]:
    """bullet_id -> the bullet's `* **Project**:` label (None when unlabelled)."""
    return {b.bullet_id: b.project for j in parse_result.jobs for b in j.bullets}


def _bullets_of(parse_result, project: str) -> list[str]:
    return [b.bullet_id for j in parse_result.jobs for b in j.bullets if b.project == project]


# ---------------------------------------------------------------- 1. load_candidates


def test_load_candidates_fields_and_draft(tmp_path):
    search = _search()

    # Metadata vault: order = job_order, bullet_order; keywords are the bullet's own.
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    db, _chroma, pr = _sync_metadata(meta_dir)
    cands = search.load_candidates(db)
    assert [c.bullet_id for c in cands] == [b.bullet_id for j in pr.jobs for b in j.bullets]
    by_project = {c.project: c for c in cands}

    parsed = {b.bullet_id: (j, b) for j in pr.jobs for b in j.bullets}
    for cand in cands:
        job, bullet = parsed[cand.bullet_id]
        assert cand.keywords == bullet.keywords
        assert cand.job == job.name
        assert cand.job_id == job.job_id
        assert cand.job_order == job.order
        assert cand.project == bullet.project

    # the job's `Defaults` (TypeScript) are not merged into the bullet's own list
    assert by_project[P1].keywords == ["Shaka Player", "Buffer Tuning"]
    assert by_project[P2].keywords == []
    assert by_project[P3].keywords == ["Alerting", "Dedupe Rules", "Observability"]

    # index_vault synced with a draft present in SQLite; the former reserve
    # bullet (`lab001`) is an ordinary candidate now.
    idx_dir = tmp_path / "idx"
    idx_dir.mkdir()
    db2, _chroma2, pr_all = _sync(idx_dir, "index_vault.md", include_draft=True)
    draft_ids = {b.bullet_id for j in pr_all.jobs for b in j.bullets if b.status == "draft"}
    assert draft_ids

    default_ids = {c.bullet_id for c in search.load_candidates(db2)}
    assert not default_ids & draft_ids
    expected_default = set(_project_of(_parse("index_vault.md")))
    assert default_ids == expected_default
    assert "placeholder-lab-research-intern::lab001" in default_ids


# ------------------------------------------------------------- 2. vector_scores_for


def test_vector_scores_for_routed_ids_range_determinism(tmp_path):
    search = _search()
    db, chroma, pr = _sync_metadata(tmp_path)
    all_ids = sorted(_project_of(pr))
    routed_ids = all_ids[:2]
    reqs = [P4_PHRASE, P4_PHRASE, "x"]

    adapter = FakeEmbeddingAdapter()
    first = search.vector_scores_for(reqs, routed_ids, chroma_path=chroma, adapter=adapter)
    assert adapter.calls == 1

    assert set(first) == {P4_PHRASE, "x"}
    for req in first:
        assert set(first[req]) == set(routed_ids)
        for sim in first[req].values():
            assert 0.0 < sim <= 1.0

    adapter2 = FakeEmbeddingAdapter()
    second = search.vector_scores_for(reqs, routed_ids, chroma_path=chroma, adapter=adapter2)
    assert adapter2.calls == 1
    assert second == first


# ------------------------------------------------------------ 4. BM25 phrase


def test_match_bm25_phrase_ranks_p4_first(tmp_path):
    db, chroma, pr = _sync_metadata(tmp_path)
    (p4_bullet,) = _bullets_of(pr, P4)

    result = _match(_profile(must_have=[P4_PHRASE]), db, chroma, _cfg())

    (line,) = result.requirements
    assert line.text == P4_PHRASE
    assert line.ranked
    assert line.ranked[0].bullet_id == p4_bullet


# ------------------------- TR7. one rerank call, pairs = the sum of the lines' own top_k


def test_match_one_rerank_call_per_line_lists(tmp_path, monkeypatch):
    search = _search()
    db, chroma, pr = _sync_metadata(tmp_path)
    must, resp, nice = "fictional", "ship the fake player", "pretend"
    profile = _profile(must_have=[must], responsibilities=[resp], nice_to_have=[nice])

    asked: list[tuple[list[str], list[str]]] = []
    real_vector_scores_for = search.vector_scores_for

    def spy(requirements, routed_ids, **kwargs):
        asked.append((list(requirements), list(routed_ids)))
        return real_vector_scores_for(requirements, routed_ids, **kwargs)

    monkeypatch.setattr(search, "vector_scores_for", spy)
    reranker = FakeRerankAdapter()
    coverage = FakeCoverageAdapter()

    result = _match(profile, db, chroma, _cfg(), reranker=reranker, coverage=coverage)

    # Call 1 = the coverage call; call 2 = the claims call on the `none` lines
    # (Helios declares `TypeScript` in `Defaults` and no bullet carries it).
    assert len(coverage.calls) == 2

    ((requirements, ids),) = asked
    assert requirements == [must, resp, nice]
    assert sorted(ids) == sorted(_project_of(pr))

    assert [(ln.text, ln.kind) for ln in result.requirements] == [
        (must, "must"),
        (resp, "responsibility"),
        (nice, "nice"),
    ]
    assert len(reranker.calls) == 1
    (pairs,) = reranker.calls
    expected = sum(min(_cfg().top_k, len(ids)) for _ in result.requirements)
    assert expected == 6
    assert len(pairs) == expected == result.diagnostics.rerank_pairs
    assert sum(len(ln.ranked) for ln in result.requirements) == len(pairs)
    # Each line's pairs are its own: queries in line order, top_k each.
    assert [q for q, _d in pairs] == [r for r in (must, resp, nice) for _ in range(2)]
    assert result.diagnostics.rerank_model == reranker.model_id


# ------------------------------------------------- TR11. one coverage call


def test_match_runs_coverage_once(tmp_path):
    db, chroma, _pr = _sync_metadata(tmp_path)
    resp, nice = "ship the fake player", "pretend"
    profile = _profile(
        must_have=[NONSENSE, P4_PHRASE], responsibilities=[resp], nice_to_have=[nice]
    )

    def rule(line: str) -> tuple[str, list[str]]:
        if line == P4_PHRASE:
            return "exact", ["Alerting"]
        return "none", []

    coverage = FakeCoverageAdapter(rule)

    result = _match(profile, db, chroma, _cfg(), coverage=coverage)

    # The coverage call is still one and first; the second is the claims call.
    assert len(coverage.calls) == 2
    assert result.diagnostics.claims_calls == 1
    _system, user, _temperature = coverage.calls[0]
    assert "Keywords (" in user
    assert "- Alerting" in user
    assert profile.coverage == {}  # the caller's profile is not mutated
    assert list(result.profile.coverage) == [NONSENSE, P4_PHRASE, resp, nice]
    assert result.profile.coverage_model == "fake-coverage"
    assert result.profile.coverage[P4_PHRASE].keywords == ["Alerting"]
    assert [(ln.text, ln.grade) for ln in result.requirements] == [
        (NONSENSE, "none"),
        (P4_PHRASE, "exact"),
        (resp, "none"),
        (nice, "none"),
    ]
    assert result.requirements[1].mapped_keywords == ["Alerting"]
    # Only the must line graded none is a gap; a nice or responsibility `none` is not.
    assert result.gaps == [NONSENSE]
    assert result.diagnostics.coverage_calls == 1
    assert result.diagnostics.coverage_model == "fake-coverage"


# ------------------------------------------------------------ TR13. stored coverage


def test_match_replays_stored_coverage_zero_calls(tmp_path):
    db, chroma, _pr = _sync_metadata(tmp_path)
    profile = _profile(must_have=[P4_PHRASE, NONSENSE])
    first = _match(profile, db, chroma, _cfg(), coverage=FakeCoverageAdapter())

    replay = _match(first.profile, db, chroma, _cfg(), coverage=RaisingCoverageAdapter())

    assert replay.diagnostics.coverage_calls == 0
    assert replay.requirements == first.requirements
    assert replay.gaps == first.gaps == [P4_PHRASE, NONSENSE]
    assert replay.profile == first.profile


def test_match_replays_profile_without_parts_zero_calls(tmp_path):
    # C-6: a stored profile written before `parts` existed has no `parts` key anywhere
    import json

    db, chroma, _pr = _sync_metadata(tmp_path)
    profile = _profile(must_have=[P4_PHRASE, NONSENSE])
    first = _match(profile, db, chroma, _cfg(), coverage=FakeCoverageAdapter())
    stored = json.loads(first.profile.model_dump_json())
    for answer in stored["coverage"].values():
        answer.pop("parts", None)
    old_profile = JDProfile.model_validate(stored)

    replay = _match(old_profile, db, chroma, _cfg(), coverage=RaisingCoverageAdapter())

    assert replay.diagnostics.coverage_calls == 0
    assert [line.parts for line in replay.requirements] == [[], []]
    assert all(answer.parts == [] for answer in replay.profile.coverage.values())


def test_match_line_parts_reach_payload(tmp_path):
    # C-8: the parts ride from the coverage answer to the payload JSON;
    # nothing else on the match path moves (N-1 at unit level).
    import json

    db, chroma, _pr = _sync_metadata(tmp_path)
    profile = _profile(must_have=[P4_PHRASE, NONSENSE])

    class PartsAdapter(FakeCoverageAdapter):
        def complete(self, system: str, user: str, *, temperature: float) -> str:
            reply = json.loads(super().complete(system, user, temperature=temperature))
            if P4_PHRASE in reply:  # the claims call asks only for the lines graded none
                reply[P4_PHRASE]["parts"] = [{"term": P4_PHRASE, "match": "exact"}]
            return json.dumps(reply)

    def rule(line: str) -> tuple[str, list[str]]:
        return ("exact", ["Alerting"]) if line == P4_PHRASE else ("none", [])

    plain = _match(profile, db, chroma, _cfg(), coverage=FakeCoverageAdapter(rule))
    parted = _match(profile, db, chroma, _cfg(), coverage=PartsAdapter(rule))

    by_text = {line.text: line for line in parted.requirements}
    assert [(p.term, p.match) for p in by_text[P4_PHRASE].parts] == [(P4_PHRASE, "exact")]
    assert by_text[NONSENSE].parts == []
    dumped = json.loads(parted.model_dump_json())
    assert dumped["requirements"][0]["parts"] == [{"term": P4_PHRASE, "match": "exact"}]
    assert [line.grade for line in parted.requirements] == [
        line.grade for line in plain.requirements
    ]
    assert parted.selection == plain.selection
    assert parted.gaps == plain.gaps


def test_match_requires_coverage_or_adapter(tmp_path):
    search = _search()
    db, chroma = tmp_path / "no.sqlite", tmp_path / "no_chroma"

    # `match="coverage"`: a pydantic ValidationError is also a ValueError; only the real
    # guard's message names the coverage source.
    with pytest.raises(ValueError, match="coverage"):
        search.match(
            _profile(),
            db_path=db,
            chroma_path=chroma,
            adapter=FakeEmbeddingAdapter(),
            reranker=FakeRerankAdapter(),
            config=_cfg(),
            coverage=None,
        )

    assert not db.exists()  # no store was opened


def test_match_stored_coverage_other_profile_raises(tmp_path):
    from loomcv.coverage import StoredCoverageError

    db, chroma, _pr = _sync_metadata(tmp_path)
    first = _match(_profile(must_have=[P4_PHRASE, NONSENSE]), db, chroma, _cfg())
    other = first.profile.model_copy(update={"must_have": [P4_PHRASE]})

    with pytest.raises(StoredCoverageError, match="not a line of this profile"):
        _match(other, db, chroma, _cfg(), coverage=RaisingCoverageAdapter())


# ------------------------------------------------ S-1. exact get, never query


def test_vector_scores_for_exact_get_never_query(tmp_path, monkeypatch):
    import math

    from chromadb.api.models.Collection import Collection

    from loomcv.sync.index_assets import get_collection, open_chroma_client

    search = _search()
    db, chroma, pr = _sync_metadata(tmp_path)
    routed_ids = sorted(_project_of(pr))
    reqs = [P4_PHRASE, "fictional", P4_PHRASE]

    def _no_query(*_args, **_kwargs):
        raise AssertionError("collection.query must never be called (F-S3-1)")

    monkeypatch.setattr(Collection, "query", _no_query)

    scores = search.vector_scores_for(
        reqs, routed_ids, chroma_path=chroma, adapter=FakeEmbeddingAdapter()
    )

    assert set(scores) == {P4_PHRASE, "fictional"}
    collection, _exists = get_collection(open_chroma_client(chroma))
    got = collection.get(ids=routed_ids, include=["embeddings"])
    stored = {bid: [float(x) for x in emb] for bid, emb in zip(got["ids"], got["embeddings"])}

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    for req, per_bullet in scores.items():
        assert set(per_bullet) == set(routed_ids)
        q = _deterministic_vector(req)
        for bid, sim in per_bullet.items():
            assert -1.0 <= sim <= 1.0
            assert sim == pytest.approx(cosine(q, stored[bid]), abs=1e-9)


# ------------------------------------------------------------ S-2. determinism


def test_match_three_runs_identical(tmp_path):
    db, chroma, _pr = _sync_metadata(tmp_path)
    first = _match(_profile(must_have=[P4_PHRASE, "fictional", NONSENSE]), db, chroma, _cfg())
    replayed = first.profile  # carries the coverage: the acceptance path replays, never calls

    runs = [
        _match(replayed, db, chroma, _cfg(), coverage=RaisingCoverageAdapter()).model_dump_json()
        for _ in range(3)
    ]

    assert runs[0] == runs[1] == runs[2]
    # The comparison covers the merge's result, which is part of the dump.
    assert '"selection"' in runs[0]


# ---------------------------------------------------------- S-3. missing vector


def test_match_missing_vector_warned_and_counted(tmp_path, capsys):
    from loomcv.sync.index_assets import get_collection, open_chroma_client

    db, chroma, pr = _sync_metadata(tmp_path)
    (p4_bullet,) = _bullets_of(pr, P4)
    collection, _exists = get_collection(open_chroma_client(chroma))
    collection.delete(ids=[p4_bullet])
    capsys.readouterr()

    result = _match(_profile(must_have=[P4_PHRASE]), db, chroma, _cfg())

    err = capsys.readouterr().err
    assert f"warning: 1 bullets have no vector in the index: {p4_bullet}" in err
    assert result.diagnostics.vector_missing == 1
    assert p4_bullet in _ranked_ids(result)


# ------------------------------------------------------------ 7. empty store


def test_match_on_empty_store_raises_index_not_ready(tmp_path):
    search = _search()
    db, chroma = tmp_path / "pipeline_storage.db", tmp_path / "vector_cache"

    with pytest.raises(search.IndexNotReadyError, match="sync"):
        _match(_profile(), db, chroma, _cfg())

    assert not db.exists()


# ------------------------------------------------------ the merge on a real index

SLOT_MUST = "reduce alert noise"
SLOT_RESP = "keep builds and playback fast"


def _slot_rule(line: str) -> tuple[str, list[str]]:
    """The must line maps to one Zephyr keyword; the responsibility line to one keyword on
    each of the two Helios bullets (no single keyword is shared by both)."""
    if line == SLOT_MUST:
        return "exact", ["Alerting"]
    if line == SLOT_RESP:
        return "adjacent", ["Buffer Tuning", "CI/CD"]
    return "none", []


def _full_id(pr, local_id: str) -> str:
    return next(
        b.bullet_id for j in pr.jobs for b in j.bullets if b.bullet_id.endswith(f"::{local_id}")
    )


def _slot_match(tmp_path, *, total_slots: int):
    """Metadata vault, two jobs with `recency_shares=[1, 1]`; returns (result, pr)."""
    db, chroma, pr = _sync_metadata(tmp_path)
    profile = _profile(must_have=[SLOT_MUST], responsibilities=[SLOT_RESP])
    config = _cfg(top_k=4, total_slots=total_slots, recency_shares=[1, 1])
    result = _match(profile, db, chroma, config, coverage=FakeCoverageAdapter(_slot_rule))
    return result, pr


def _helios_carriers_by_rerank(result, pr) -> list[str]:
    """The two Helios carriers of the responsibility line, best reranker rank first."""
    carriers = {_full_id(pr, "hel001"), _full_id(pr, "qua001")}
    line = next(ln for ln in result.requirements if ln.text == SLOT_RESP)
    ranked = sorted(line.ranked, key=lambda b: b.rerank_rank)
    return [b.bullet_id for b in ranked if b.bullet_id in carriers]


def test_match_selection_two_jobs_total_2(tmp_path):
    # Test 9: caps 1 / 1. The first phase puts the must pick in Zephyr (cap spent); pass 2 gives the
    # responsibility line its first carrier in the job with room, Helios.
    result, pr = _slot_match(tmp_path, total_slots=2)
    sel = result.selection
    helios, zephyr = [j.job_id for j in pr.jobs]
    tun001 = _full_id(pr, "tun001")
    first, _second = _helios_carriers_by_rerank(result, pr)

    assert sel.total_slots == 2
    assert [(j.job_id, j.job_order, j.cap, j.placed) for j in sel.jobs] == [
        (helios, 0, 1, 1),
        (zephyr, 1, 1, 1),
    ]
    helios_bullet, zephyr_bullet = sel.jobs[0].bullets[0], sel.jobs[1].bullets[0]
    assert (zephyr_bullet.bullet_id, zephyr_bullet.line, zephyr_bullet.kind) == (
        tun001,
        SLOT_MUST,
        "must",
    )
    assert zephyr_bullet.pass_number == 1
    assert (helios_bullet.bullet_id, helios_bullet.line, helios_bullet.kind) == (
        first,
        SLOT_RESP,
        "responsibility",
    )
    assert helios_bullet.pass_number == 2
    assert [(w.bullet_id, w.job_id) for w in sel.walk] == [(tun001, zephyr), (first, helios)]
    assert sel.passes == 2
    assert sel.covered_by == {SLOT_MUST: [tun001], SLOT_RESP: [first]}
    assert sel.uncovered == []
    assert sel.unplaced == []
    assert result.gaps == []


def test_match_selection_total_3_second_carrier_pass_3(tmp_path):
    # Test 9, second half: caps 2 / 1. The must line has no second carrier, so the second Helios
    # carrier goes to the responsibility line on pass 3.
    result, pr = _slot_match(tmp_path, total_slots=3)
    sel = result.selection
    first, second = _helios_carriers_by_rerank(result, pr)

    assert [(j.cap, j.placed) for j in sel.jobs] == [(2, 2), (1, 1)]
    assert [(w.bullet_id, w.line, w.pass_number) for w in sel.walk] == [
        (_full_id(pr, "tun001"), SLOT_MUST, 1),
        (first, SLOT_RESP, 2),
        (second, SLOT_RESP, 3),
    ]
    assert [b.bullet_id for b in sel.jobs[0].bullets] == [first, second]
    assert sel.passes == 3
    assert sel.covered_by[SLOT_RESP] == [first, second]
    assert sel.uncovered == []


def test_match_none_lines_in_gaps_not_in_selection(tmp_path):
    # Test 10: a `none` line takes no slot; a must one is a gap, a responsibility one is not.
    db, chroma, _pr = _sync_metadata(tmp_path)
    profile = _profile(must_have=[NONSENSE], responsibilities=["nothing here either"])

    result = _match(profile, db, chroma, _cfg(), coverage=FakeCoverageAdapter())

    assert result.gaps == [NONSENSE]
    sel = result.selection
    assert sel.walk == []
    assert sel.unplaced == []
    assert sel.covered_by == {}
    assert sel.uncovered == []
    assert all(j.placed == 0 for j in sel.jobs)


def test_funnel_order_full_and_top_is_prefix(tmp_path):
    # Test 11: `LineRanking.order` is the line's whole RRF order, `top` its first `top_k`.
    from loomcv.retrieve.rank import funnel

    search = _search()
    db, chroma, _pr = _sync(tmp_path, "index_vault.md")
    candidates = search.load_candidates(db)
    ids = [c.bullet_id for c in candidates]
    profile = _profile(must_have=[P4_PHRASE, "fictional"])
    config = _cfg(top_k=2)
    scores = search.vector_scores_for(
        [P4_PHRASE, "fictional"], ids, chroma_path=chroma, adapter=FakeEmbeddingAdapter()
    )

    result = funnel(candidates, profile, scores, config)

    assert len(ids) == 7  # eight bullets, one draft
    for line in result.lines:
        assert sorted(line.order) == sorted(ids)  # sha1 vectors > 0: every bullet has RRF > 0
        assert len(line.top) == 2
        assert line.top == line.order[:2]


# ---------------------------------------- the claims call
#
# The claims vault = `metadata_vault.md` + a vault-level `Skills` line (inline, the fixture file
# stays read-only). Candidate keywords: Shaka Player, Buffer Tuning, Foo Bar, Baz.js, CI/CD,
# Alerting, Dedupe Rules, Observability. Helios declares `TypeScript; Shaka Player` in `Defaults`.
# So the claims set is [TypeScript @ (helios, skills), Kubernetes @ (skills)]; `Alerting` is on a
# bullet, hence evidence, not a claim.

CLAIMS_SKILLS_LINE = "* **Skills**: Kubernetes; TypeScript; Alerting\n"
RESP_LINE = "ship the fake player"


def _claims_vault_text(skills_line: str = CLAIMS_SKILLS_LINE) -> str:
    text = (FIXTURES / "metadata_vault.md").read_text(encoding="utf-8")
    head, rest = text.split("\n", 1)
    return head + "\n" + skills_line + rest


def _sync_text(tmp_path: Path, text: str):
    """Parse + index inline vault text into `tmp_path`; return (db, chroma, parse_result)."""
    db, chroma = tmp_path / "pipeline_storage.db", tmp_path / "vector_cache"
    parse_result = parse_star_vault(text)
    assert parse_result.errors() == []
    index_vault(parse_result, db_path=db, chroma_path=chroma, adapter=FakeEmbeddingAdapter())
    return db, chroma, parse_result


def _helios_id(pr) -> str:
    return next(j.job_id for j in pr.jobs if j.name.startswith("Helios"))


def _keyword_block(user: str) -> list[str]:
    """The `- ` lines under `Keywords (n)` of a rendered coverage prompt."""
    block = user.split("\nKeywords (", 1)[1].split("\n", 1)[1]
    return [ln[2:] for ln in block.splitlines() if ln.startswith("- ")]


class SequencedCoverageAdapter:
    """Call n answers by `rules[n]` (`rule(line) -> (match, keywords)`); records every call."""

    def __init__(self, *rules, model_id: str = "fake-coverage") -> None:
        self._rules = list(rules)
        self._model_id = model_id
        self.calls: list[tuple[str, str, float]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def temperature(self) -> float:
        return 0.0

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        import json

        rule = self._rules[len(self.calls)]
        self.calls.append((system, user, temperature))
        reply = {}
        for line in _requested_lines(user):
            match, keywords = rule(line)
            reply[line] = {"match": match, "keywords": keywords}
        return json.dumps(reply)


def _claims_exact(keyword: str, *, only: str | None = None):
    def rule(line: str) -> tuple[str, list[str]]:
        if only is None or line == only:
            return "exact", [keyword]
        return "none", []

    return rule


def _claims_run(tmp_path: Path, profile: JDProfile, claims_rule):
    """Coverage call grades everything `none`; the claims call answers by `claims_rule`."""
    db, chroma, pr = _sync_text(tmp_path, _claims_vault_text())
    coverage = SequencedCoverageAdapter(_none_rule, claims_rule)
    return _match(profile, db, chroma, _cfg(), coverage=coverage), coverage, (db, chroma, pr)


def test_l1_bulletless_job_changes_nothing(tmp_path):
    # L-1: a persisted bullet-less job is invisible to the funnel and the merge;
    # every line is graded exact, so there is nothing to claim.
    bulletless = (
        "\n## [Job] Old Basement Works — Apprentice\n"
        "* **Employer**: Old Basement Works\n"
        "* **Title**: Apprentice\n"
        "* **From**: 2001-01\n"
        "* **To**: 2002-01\n"
        "* **Bullets**:\n"
    )
    text = (FIXTURES / "metadata_vault.md").read_text(encoding="utf-8")
    plain_dir, with_dir = tmp_path / "plain", tmp_path / "with"
    plain_dir.mkdir()
    with_dir.mkdir()
    plain = _sync_text(plain_dir, text)
    extra = _sync_text(with_dir, text + bulletless)
    assert len(extra[2].jobs) == len(plain[2].jobs) + 1
    profile = _profile(must_have=[P4_PHRASE, "fictional"], responsibilities=[RESP_LINE])

    def run(synced):
        coverage = SequencedCoverageAdapter(_claims_exact("Alerting"))
        db, chroma, _pr = synced
        return _match(profile, db, chroma, _cfg(), coverage=coverage), coverage

    base, base_cov = run(plain)
    other, other_cov = run(extra)

    assert other.claims == [] and base.claims == []
    assert len(other_cov.calls) == len(base_cov.calls) == 1
    assert other.diagnostics.claims_calls == 0
    assert other.requirements == base.requirements
    assert other.gaps == base.gaps
    assert other.selection == base.selection


def test_l2_must_none_line_claim_exact(tmp_path):
    from loomcv.retrieve.rank import Claim, ClaimLine

    profile = _profile(must_have=[NONSENSE])
    result, coverage, synced = _claims_run(tmp_path, profile, _claims_exact("Kubernetes"))

    assert result.claims == [
        ClaimLine(
            line=NONSENSE,
            kind="must",
            match="exact",
            keywords=[Claim(keyword="Kubernetes", at=["skills"])],
        )
    ]
    assert len(coverage.calls) == 2
    assert result.diagnostics.claims_calls == 1
    assert result.diagnostics.claims_model == "fake-coverage"
    assert result.profile.claims_model == "fake-coverage"
    # the grades do not move: the line is still a gap and still takes no slot
    assert result.gaps == [NONSENSE]
    assert result.requirements[0].grade == "none"
    db, chroma, _pr = synced
    plain = _match(
        profile, db, chroma, _cfg(), coverage=SequencedCoverageAdapter(_none_rule, _none_rule)
    )
    assert plain.claims == []
    assert result.gaps == plain.gaps
    assert result.selection == plain.selection
    assert result.requirements == plain.requirements


def test_l3_responsibility_line_adjacent(tmp_path):
    from loomcv.retrieve.rank import Claim

    profile = _profile(must_have=[NONSENSE], responsibilities=[RESP_LINE])

    def rule(line: str) -> tuple[str, list[str]]:
        return ("adjacent", ["TypeScript"]) if line == RESP_LINE else ("none", [])

    result, _cov, synced = _claims_run(tmp_path, profile, rule)

    (claim_line,) = result.claims
    assert (claim_line.line, claim_line.kind, claim_line.match) == (
        RESP_LINE,
        "responsibility",
        "adjacent",
    )
    assert claim_line.keywords == [
        Claim(keyword="TypeScript", at=[_helios_id(synced[2]), "skills"])
    ]
    assert result.gaps == [NONSENSE]  # only the must line is a gap


def test_l4_claims_none_stays_gap(tmp_path):
    from loomcv.state import Coverage

    profile = _profile(must_have=[NONSENSE])
    result, _cov, _synced = _claims_run(tmp_path, profile, _none_rule)

    assert result.claims == []
    assert result.gaps == [NONSENSE]
    # "nothing found" is the mapping as returned, `none` entries included — not `{}`
    assert result.profile.claims == {NONSENSE: Coverage(match="none", keywords=[])}
    assert result.diagnostics.claims_calls == 1


def test_l5_no_none_lines_no_second_call(tmp_path):
    db, chroma, _pr = _sync_text(tmp_path, _claims_vault_text())
    coverage = SequencedCoverageAdapter(_claims_exact("Alerting"))
    profile = _profile(must_have=[P4_PHRASE], responsibilities=[RESP_LINE])

    result = _match(profile, db, chroma, _cfg(), coverage=coverage)

    assert len(coverage.calls) == 1
    assert result.diagnostics.claims_calls == 0
    assert result.diagnostics.claims_model is None
    assert result.profile.claims == {}
    assert result.claims == []


def test_l6_replay_coverage_and_claims_zero_calls(tmp_path):
    profile = _profile(must_have=[NONSENSE])
    first, _cov, synced = _claims_run(tmp_path, profile, _claims_exact("Kubernetes"))
    db, chroma, _pr = synced

    replay = _match(first.profile, db, chroma, _cfg(), coverage=RaisingCoverageAdapter())

    assert replay.claims == first.claims != []
    assert replay.diagnostics.coverage_calls == 0
    assert replay.diagnostics.claims_calls == 0
    assert replay.profile == first.profile


def test_l7_stale_claims_and_pre_r5_profile(tmp_path):
    from loomcv.coverage import StoredCoverageError

    profile = _profile(must_have=[NONSENSE])
    first, _cov, synced = _claims_run(tmp_path, profile, _claims_exact("Kubernetes"))
    db, chroma, _pr = synced

    # (a) the vault changed: `Kubernetes` left the Skills line → the stored claims no longer fit
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    odb, ochroma, _opr = _sync_text(
        other_dir, _claims_vault_text("* **Skills**: TypeScript; Alerting\n")
    )
    with pytest.raises(StoredCoverageError, match="Kubernetes"):
        _match(first.profile, odb, ochroma, _cfg(), coverage=RaisingCoverageAdapter())

    # (b) a pre-R5 profile (coverage, no claims) → exactly one call, the claims call
    pre_r5 = first.profile.model_copy(update={"claims": None, "claims_model": None})
    coverage = SequencedCoverageAdapter(_claims_exact("Kubernetes"))
    result = _match(pre_r5, db, chroma, _cfg(), coverage=coverage)

    assert len(coverage.calls) == 1
    assert result.diagnostics.coverage_calls == 0
    assert result.diagnostics.claims_calls == 1
    assert result.profile.claims is not None
    assert [c.line for c in result.claims] == [NONSENSE]


def test_r1_coverage_then_claims_order(tmp_path):
    profile = _profile(must_have=[NONSENSE, P4_PHRASE], responsibilities=[RESP_LINE])

    def coverage_rule(line: str) -> tuple[str, list[str]]:
        return ("exact", ["Alerting"]) if line == P4_PHRASE else ("none", [])

    db, chroma, _pr = _sync_text(tmp_path, _claims_vault_text())
    coverage = SequencedCoverageAdapter(coverage_rule, _none_rule)

    _match(profile, db, chroma, _cfg(), coverage=coverage)

    first_user, second_user = coverage.calls[0][1], coverage.calls[1][1]
    assert _requested_lines(first_user) == [NONSENSE, P4_PHRASE, RESP_LINE]
    assert "Alerting" in _keyword_block(first_user)
    assert "Kubernetes" not in _keyword_block(first_user)
    assert _requested_lines(second_user) == [NONSENSE, RESP_LINE]  # only the `none` lines
    assert sorted(_keyword_block(second_user)) == ["Kubernetes", "TypeScript"]


def test_r2_claims_call_error_exit_4(tmp_path, monkeypatch, capsys):
    import json

    from loomcv.cli import main

    db, chroma, _pr = _sync_text(tmp_path, _claims_vault_text())

    class ClaimsBreaks:
        model_id = "fake-coverage"
        temperature = 0.0

        def __init__(self) -> None:
            self.calls: list = []

        def complete(self, system: str, user: str, *, temperature: float) -> str:
            self.calls.append((system, user, temperature))
            if len(self.calls) == 1:
                return json.dumps(
                    {ln: {"match": "none", "keywords": []} for ln in _requested_lines(user)}
                )
            return "not json at all"

    adapter = ClaimsBreaks()
    monkeypatch.setattr("loomcv.cli.build_embedding_adapter", lambda cfg: FakeEmbeddingAdapter())
    monkeypatch.setattr("loomcv.cli.build_rerank_adapter", lambda cfg: FakeRerankAdapter())
    monkeypatch.setattr("loomcv.cli.build_generation_adapter", lambda node, provider: adapter)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(_profile(must_have=[NONSENSE]).model_dump()), encoding="utf-8")

    rc = main(["match", "--profile", str(profile), "--db", str(db), "--chroma", str(chroma)])

    assert rc == 4
    captured = capsys.readouterr()
    assert captured.err.count("coverage:") >= 1
    assert captured.out == ""
    assert len(adapter.calls) == 3  # the coverage call, then the claims call and its one retry


def test_r3_coverage_calls_unchanged(tmp_path):
    profile = _profile(must_have=[NONSENSE])
    first, _cov, synced = _claims_run(tmp_path, profile, _none_rule)
    db, chroma, _pr = synced

    assert first.diagnostics.coverage_calls == 1
    assert first.diagnostics.claims_calls == 1

    pre_r5 = first.profile.model_copy(update={"claims": None, "claims_model": None})
    again = _match(
        pre_r5, db, chroma, _cfg(), coverage=SequencedCoverageAdapter(_claims_exact("Kubernetes"))
    )
    assert again.diagnostics.coverage_calls == 0
    assert again.diagnostics.claims_calls == 1


def test_r4_empty_claims_payload_key_diff(tmp_path):
    from loomcv.retrieve.rank import MatchDiagnostics

    text = (FIXTURES / "metadata_vault.md").read_text(encoding="utf-8")
    text = text.replace("* **Defaults**: TypeScript; Shaka Player\n", "")
    assert "Defaults" not in text.split("## [Job]", 1)[1]
    db, chroma, _pr = _sync_text(tmp_path, text)
    coverage = SequencedCoverageAdapter(_none_rule)

    result = _match(_profile(must_have=[NONSENSE]), db, chroma, _cfg(), coverage=coverage)

    assert result.claims == []
    assert len(coverage.calls) == 1
    assert result.diagnostics.claims_calls == 0
    assert result.diagnostics.claims_model is None
    assert result.profile.claims == {}
    assert result.profile.claims_model is None
    head_result = {"profile", "requirements", "gaps", "selection", "diagnostics"}
    head_diagnostics = {
        "vector_missing",
        "rerank_pairs",
        "rerank_model",
        "coverage_model",
        "coverage_calls",
    }
    head_profile = {
        "title",
        "seniority",
        "domain",
        "must_have",
        "nice_to_have",
        "responsibilities",
        "constraints",
        "traits",
        "ats_keywords",
        "skills",
        "coverage",
        "coverage_model",
    }
    assert set(result.model_dump()) - head_result == {"claims"}
    assert set(MatchDiagnostics.model_fields) - head_diagnostics == {"claims_model", "claims_calls"}
    assert set(result.profile.model_dump()) - head_profile == {"claims", "claims_model"}
