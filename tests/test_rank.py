"""Tests for loomcv.state schemas and loomcv.retrieve.rank.

In-test literal synthetic candidates only; no fixture file, no I/O beyond the
real `config/retrieval.yaml` (read-only) for the config-loading test.

`funnel()` runs one independent funnel per requirement line —
BM25 over `" ".join(keywords) + " " + text`, raw cosine over the stored text
vectors, unweighted Reciprocal Rank Fusion (`rrf_k`), the line's top `top_k` by RRF
(RRF 0 never counts, ties by `bullet_id`). Nothing is shared across lines and
nothing decides evidence here. BM25 orders on tiny corpora are built from terms
that each occur in exactly one bullet — a term shared by >= half the documents
gets a non-positive `rank_bm25` score and is unranked. New symbols are imported
inside the tests, so the tokenizer / BM25 tests do not depend on the funnel code.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from loomcv.retrieve.rank import Candidate, bm25_scores, tokenize
from loomcv.state import JDProfile, RetrievalConfig, load_retrieval_config

K = 60


def make_config(**overrides):
    defaults = dict(
        top_k=12,
        rrf_k=K,
        total_slots=8,
        recency_shares=[2, 1],
    )
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


def make_profile(**overrides):
    defaults = dict(
        title="Backend Engineer",
        seniority="senior",
        domain="ad-tech",
        must_have=[],
        nice_to_have=[],
    )
    defaults.update(overrides)
    return JDProfile(**defaults)


def make_candidate(bullet_id, **overrides):
    defaults = dict(
        bullet_id=bullet_id,
        text="",
        job_id="job-1",
        job="Job One",
        project=None,
        keywords=[],
        job_order=0,
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def test_tokenize_lowercases_splits_and_drops_short_tokens():
    assert tokenize("Node.js v20, 3x faster!") == ["node", "js", "v20", "3x", "faster"]


def test_bm25_scores_empty_query_and_term_presence():
    docs = [
        ["python", "backend", "service"],
        ["frontend", "react", "ui"],
        ["database", "sql", "queries"],
    ]
    assert bm25_scores([], docs) == [0.0, 0.0, 0.0]

    scores = bm25_scores(["python"], docs)
    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_bm25_scores_idf_ordering():
    docs = [
        ["common", "rare", "backend"],
        ["common", "frontend", "service"],
        ["common", "database", "service"],
    ]
    scores_common = bm25_scores(["common"], docs)
    scores_rare = bm25_scores(["rare"], docs)
    assert scores_common[0] < scores_rare[0]


def test_tokenize_drops_standard_stopwords_only():
    tokens = set(
        tokenize(
            "Experience with prompting and fine-tuning for LLMs/ML systems, hands-on, via APIs"
        )
    )

    assert {"llms", "ml", "prompting", "fine", "tuning", "apis"} <= tokens
    assert {"hands", "experience", "systems"} <= tokens
    assert not tokens & {"on", "with"}


# ----------------------------------------------- vendored stopword list


def test_load_stopwords_standard_list():
    import hashlib

    from loomcv import resources
    from loomcv.retrieve.rank import _load_stopwords

    path = resources.path("stopwords_en.txt")
    words = _load_stopwords(path)

    assert isinstance(words, frozenset)
    assert len(words) >= 150
    assert all(w == w.lower() for w in words)
    assert {"the", "of", "it", "is"} <= words
    assert not words & {"experience", "years", "hands", "us"}

    lines = path.read_bytes().splitlines(keepends=True)
    header = [ln for ln in lines if ln.startswith(b"|#")]
    body = b"".join(ln for ln in lines if not ln.startswith(b"|#"))
    (sha_line,) = [ln for ln in header if b"sha256" in ln]
    assert sha_line.decode().split(":")[-1].strip() == hashlib.sha256(body).hexdigest()


def test_load_stopwords_missing_file_raises_import_error(tmp_path):
    from loomcv.retrieve.rank import _load_stopwords

    missing = tmp_path / "nope.txt"

    with pytest.raises(ImportError) as excinfo:
        _load_stopwords(missing)

    assert str(missing) in str(excinfo.value)


def test_tokenize_hands_on_experience_with_the_llms_in_it():
    assert tokenize("Hands-on experience with the LLMs in IT") == ["hands", "experience", "llms"]


def test_rank_module_has_no_generic_skill_words():
    import loomcv.retrieve.rank as rank_module

    assert not hasattr(rank_module, "GENERIC_SKILL_WORDS")


def test_ranks_dense_ties_and_nonpositive_unranked():
    from loomcv.retrieve.rank import ranks

    assert ranks([0.9, 0.9, 0.5, 0.0, -0.3], skip_zero=True) == [1, 1, 2, None, None]
    assert ranks([0.2, 0.7], skip_zero=False) == [2, 1]


def _funnel(candidates, profile, vector_scores, config):
    from loomcv.retrieve.rank import funnel

    return funnel(candidates, profile, vector_scores, config)


# ---------------------------------------------------- the per-line funnel


def test_bm25_document_keywords_then_text():
    # TR1: run_t1.py 294, literal — keywords space-joined, then a space, then the text.
    from loomcv.retrieve.rank import bm25_document

    with_kw = make_candidate("A", text="Wrote suites.", keywords=["Cypress", "Test Automation"])
    assert bm25_document(with_kw) == "Cypress Test Automation Wrote suites."

    no_kw = make_candidate("B", text="Wrote suites.")
    assert bm25_document(no_kw) == " Wrote suites."
    assert tokenize(bm25_document(no_kw)) == tokenize("Wrote suites.")


def test_funnel_bm25_reads_keywords():
    # TR1: A's text never says "cypress"; its keyword list does.
    profile = make_profile(must_have=["cypress"])
    candidates = [
        make_candidate("A", text="wrote suites", keywords=["Cypress", "Test Automation"]),
        make_candidate("B", text="zulu yankee"),
        make_candidate("C", text="xray whiskey"),
    ]

    result = _funnel(candidates, profile, {}, make_config())

    assert [ln.text for ln in result.lines] == ["cypress"]
    assert result.lines[0].top == ["A"]


def test_funnel_rrf_arithmetic_unweighted():
    # TR2. BM25 order C (3 unique hits) > B (2) > A (1); cosine order A > B, C has no vector.
    req = "u1 u2 u3 u4 u5 u6"
    profile = make_profile(must_have=[req])
    candidates = [
        make_candidate("A", text="u1"),
        make_candidate("B", text="u2 u3"),
        make_candidate("C", text="u4 u5 u6"),
    ]
    vector_scores = {req: {"A": 0.9, "B": 0.8}}

    result = _funnel(candidates, profile, vector_scores, make_config())

    line = result.lines[0]
    assert line.top == ["A", "B", "C"]
    assert line.order == line.top
    assert line.top_rrf_score == pytest.approx(1 / (K + 1) + 1 / (K + 3), abs=1e-6)

    tuned = _funnel(candidates, profile, vector_scores, make_config(rrf_k=10))
    assert tuned.lines[0].top_rrf_score == pytest.approx(1 / 11 + 1 / 13, abs=1e-6)


def test_funnel_missing_side_contributes_one_term_and_ties_by_bullet_id():
    # TR2: B has a vector only, Q a BM25 hit only (both rank 1 on their side), Z vector rank 2,
    # W neither (rrf 0, excluded).
    profile = make_profile(must_have=["u1"])
    candidates = [
        make_candidate("Q", text="u1"),
        make_candidate("B", text="zz"),
        make_candidate("Z", text="yy"),
        make_candidate("W", text="ww"),
    ]
    vector_scores = {"u1": {"B": 0.9, "Z": 0.8}}

    result = _funnel(candidates, profile, vector_scores, make_config())

    line = result.lines[0]
    assert line.top == ["B", "Q", "Z"]
    assert line.order == line.top
    assert "W" not in line.top
    assert line.top_rrf_score == pytest.approx(1 / (K + 1), abs=1e-6)


def test_funnel_nice_and_must_same_order():
    # TR2: no weight — the same text as a must line and as a nice line gives the same result.
    candidates = [
        make_candidate("b1", text="sample bullet text"),
        make_candidate("b2", text="sample bullet text"),
        make_candidate("b3", text="sample bullet text"),
    ]
    vector_scores = {"req": {"b1": 0.5, "b2": 0.9, "b3": 0.7}}

    as_must = _funnel(candidates, make_profile(must_have=["req"]), vector_scores, make_config())
    as_nice = _funnel(candidates, make_profile(nice_to_have=["req"]), vector_scores, make_config())

    assert as_must.lines[0].kind == "must"
    assert as_nice.lines[0].kind == "nice"
    assert as_must.lines[0].top == as_nice.lines[0].top == ["b2", "b3", "b1"]
    assert as_must.lines[0].top_rrf_score == as_nice.lines[0].top_rrf_score


def test_funnel_cut_top_k_per_line():
    # TR3: five ranked candidates, top_k=2, two lines with different top 2s.
    profile = make_profile(must_have=["alpha"], nice_to_have=["bravo"])
    candidates = [make_candidate(f"c{i}", text=f"t{i}") for i in range(1, 6)]
    vector_scores = {
        "alpha": {"c1": 0.9, "c2": 0.8, "c3": 0.7, "c4": 0.6, "c5": 0.5},
        "bravo": {"c5": 0.9, "c4": 0.8, "c3": 0.7, "c2": 0.6, "c1": 0.5},
    }

    result = _funnel(candidates, profile, vector_scores, make_config(top_k=2))

    assert [ln.top for ln in result.lines] == [["c1", "c2"], ["c5", "c4"]]
    # `order` keeps the line's full RRF order, `top` is its prefix.
    assert [ln.order for ln in result.lines] == [
        ["c1", "c2", "c3", "c4", "c5"],
        ["c5", "c4", "c3", "c2", "c1"],
    ]
    for ln in result.lines:
        assert ln.top == ln.order[:2]
    assert result.vector_missing == 0


def test_funnel_line_rows_kinds_and_vector_missing():
    profile = make_profile(
        must_have=["kafka", "req two"], responsibilities=["run it"], nice_to_have=["nice req"]
    )
    candidates = [
        make_candidate("b1", text="alpha bravo"),
        make_candidate("b2", text="kafka streams"),
        make_candidate("b3", text="charlie delta"),
    ]
    vector_scores = {
        "kafka": {"b1": 0.9, "b2": 0.6, "b3": 0.4},
        "req two": {"b1": 0.3, "b2": 0.2},
    }

    result = _funnel(candidates, profile, vector_scores, make_config())

    assert result.vector_missing == 0
    assert [(ln.text, ln.kind) for ln in result.lines] == [
        ("kafka", "must"),
        ("req two", "must"),
        ("run it", "responsibility"),
        ("nice req", "nice"),
    ]
    kafka, two, resp, nice = result.lines
    # Top RRF bullet is b2 (cosine rank 2 + BM25 rank 1), but the max cosine is b1's.
    assert kafka.top[0] == "b2"
    assert kafka.top_rrf_score == pytest.approx(1 / (K + 2) + 1 / (K + 1), abs=1e-6)
    assert kafka.max_cosine == pytest.approx(0.9, abs=1e-6)
    assert kafka.max_bm25 > 0

    assert two.top_rrf_score == pytest.approx(1 / (K + 1), abs=1e-6)
    assert two.max_cosine == pytest.approx(0.3, abs=1e-6)
    assert two.max_bm25 == 0.0

    for empty in (resp, nice):
        assert empty.top == []
        assert empty.top_rrf_score is None
        assert empty.max_cosine is None
        assert empty.max_bm25 == 0.0

    missing_b3 = {
        "kafka": {"b1": 0.9, "b2": 0.6},
        "req two": {"b1": 0.3, "b2": 0.2},
    }
    assert _funnel(candidates, profile, missing_b3, make_config()).vector_missing == 1


def test_funnel_deterministic():
    # TR4
    profile = make_profile(must_have=["req"], nice_to_have=["other"])
    vector_scores = {"req": {"b2": 0.5, "b1": 0.5}, "other": {"b1": 0.4, "b2": 0.6}}
    candidates = [
        make_candidate("b2", text="sample bullet text", keywords=["K2"]),
        make_candidate("b1", text="sample bullet text", keywords=["K1"]),
    ]

    first = _funnel(candidates, profile, vector_scores, make_config())
    second = _funnel(candidates, profile, vector_scores, make_config())

    assert first.model_dump_json() == second.model_dump_json()
    assert first.lines[0].top == ["b1", "b2"]
    assert first.lines[0].order == ["b1", "b2"]


def test_requirements_of_order_dedup_and_no_weight():
    from loomcv.retrieve.rank import Requirement, requirements_of

    profile = make_profile(
        must_have=["a"], responsibilities=["b", "a"], nice_to_have=["c", "b"], traits=["t"]
    )

    reqs = requirements_of(profile)

    assert [(r.text, r.kind) for r in reqs] == [
        ("a", "must"),
        ("b", "responsibility"),
        ("c", "nice"),
    ]
    assert "weight" not in Requirement.model_fields


def test_routing_pool_and_metadata_are_gone():
    import loomcv.retrieve.rank as rank_module

    for name in ("build_pool", "metadata_hit", "PoolResult", "RankResult", "ScoredBullet"):
        assert not hasattr(rank_module, name)
    for retired in ("domain", "year", "tech"):
        assert retired not in Candidate.model_fields
    for retired in ("domains", "tech"):
        assert retired not in JDProfile.model_fields


def test_load_retrieval_config_reads_real_file():
    # The inverse of the old "ignores extra keys" contract (extra="forbid")
    # is covered by tests/test_config.py::test_retrieval_config_rejects_retired_key.
    config = load_retrieval_config()

    assert isinstance(config.top_k, int)
    assert isinstance(config.rrf_k, int)
    # The résumé length and the recency shares; structure, not values.
    assert isinstance(config.total_slots, int) and config.total_slots >= 1
    assert isinstance(config.recency_shares, list) and config.recency_shares
    assert all(isinstance(s, int) and s >= 1 for s in config.recency_shares)
    assert not hasattr(config, "final_bullets")
    assert not hasattr(config, "nice_to_have_weight")


def test_jdprofile_rejects_invalid_seniority():
    with pytest.raises(ValidationError):
        JDProfile(
            title="Backend Engineer",
            seniority="principal",
            domain="ad-tech",
            must_have=[],
        )
