"""Tests for `loomcv/retrieve/rerank.py` (the cross-encoder adapter; per-line reranking).

In-test synthetic candidates and hand-set pair scores only: the reranker is never a real model —
no `sentence_transformers` import, no network, no model download.

`rerank_lines()` scores only each line's own `top_k` pairs
(`("<line text>", "Keywords: k1; k2. <text>")`) in ONE `reranker.score` call and orders each line's
bullets by `(-score, bullet_id)`. The tests build `LineRanking` inputs by hand, so they do not
depend on the funnel code.

`loomcv.retrieve.rerank` and the `rank` symbols are imported inside each test (`_rr()`, `_rank()`),
so a missing symbol fails each test at red on its own.
"""

from __future__ import annotations

import pytest


def _rr():
    import loomcv.retrieve.rerank as rerank_module

    return rerank_module


def _rank():
    import loomcv.retrieve.rank as rank_module

    return rank_module


def make_candidate(bullet_id, **overrides):
    defaults = dict(
        bullet_id=bullet_id,
        text="sample bullet text",
        job_id=f"job-{bullet_id}",
        job=f"Job {bullet_id}",
        project=None,
        keywords=[],
        job_order=0,
    )
    defaults.update(overrides)
    return _rank().Candidate(**defaults)


def make_line(text, top, *, kind="must"):
    return _rank().LineRanking(
        text=text,
        kind=kind,
        order=list(top),
        top=list(top),
        top_rrf_score=None,
        max_cosine=None,
        max_bm25=None,
    )


class FakeRerankAdapter:
    """Network-free stand-in for `RerankAdapter`; `scorer(query, document) -> float`."""

    def __init__(self, scorer=None, *, model_id: str = "fake-reranker") -> None:
        self._model_id = model_id
        self._scorer = scorer or (lambda query, document: 0.5)
        self.calls: list[list[tuple[str, str]]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def score(self, pairs):
        self.calls.append(list(pairs))
        return [self._scorer(q, d) for q, d in pairs]


def _funnel_result(*lines, vector_missing=0):
    return _rank().FunnelResult(lines=list(lines), vector_missing=vector_missing)


def test_rerank_config_and_adapter_build(tmp_path, monkeypatch):
    import sys

    import yaml

    rr = _rr()
    path = tmp_path / "models.yaml"

    def write(block: dict):
        data = {
            "embedding": {"adapter": "fake", "model": "fake-embed", "vector_space_version": 1},
            "reranker": block,
        }
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    cfg = rr.load_rerank_config(
        write(
            {
                "adapter": "sentence_transformers",
                "model": "fake/cross-encoder",
                "base_url_env": None,
            }
        )
    )
    assert cfg.adapter == "sentence_transformers"
    assert cfg.model == "fake/cross-encoder"
    assert cfg.base_url_env is None

    with pytest.raises(ValueError, match="bogus"):
        rr.build_rerank_adapter(rr.load_rerank_config(write({"adapter": "bogus", "model": "m"})))

    http_cfg = rr.load_rerank_config(
        write({"adapter": "http_rerank", "model": "m", "base_url_env": "FAKE_RERANK_URL"})
    )
    assert http_cfg.adapter == "http_rerank"
    with pytest.raises(NotImplementedError):
        rr.build_rerank_adapter(http_cfg)

    # `None` in sys.modules makes `from sentence_transformers import …` raise ImportError, so no
    # model is ever loaded or downloaded here even when the embed group is installed.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="uv sync --group embed"):
        rr.build_rerank_adapter(cfg)


# ------------------------------------------------------------------ TR5: the document


def test_rerank_document_and_query():
    rr = _rr()
    with_kw = make_candidate("A", text="Wrote suites.", keywords=["Cypress", "End-to-End Testing"])
    no_kw = make_candidate("B", text="Plain text.")
    assert rr.rerank_document(with_kw) == "Keywords: Cypress; End-to-End Testing. Wrote suites."
    assert rr.rerank_document(no_kw) == "Keywords: . Plain text."

    reranker = FakeRerankAdapter()
    line = make_line("browser automation", ["A", "B"])

    rr.rerank_lines(_funnel_result(line), [with_kw, no_kw], reranker)

    assert reranker.calls == [
        [
            ("browser automation", "Keywords: Cypress; End-to-End Testing. Wrote suites."),
            ("browser automation", "Keywords: . Plain text."),
        ]
    ]


# ------------------------------------------------------------------ TR6: order, ranks, rounding


def test_rerank_line_order_ranks_rounding():
    rr = _rr()
    candidates = [make_candidate(f"c{i}", text=f"txt-c{i}") for i in range(1, 6)]
    raw = {
        "txt-c5": 0.12345678,
        "txt-c3": 0.05,
        "txt-c4": 0.05,
        "txt-c2": 0.02,
        "txt-c1": 0.01,
    }

    def scorer(query, document):
        return next(v for k, v in raw.items() if k in document)

    reranker = FakeRerankAdapter(scorer)
    line = make_line("req", ["c1", "c2", "c3", "c4", "c5"])

    lists, pairs = rr.rerank_lines(_funnel_result(line), candidates, reranker)

    (ranked,) = lists
    assert pairs == 5
    # c3 / c4 tie on the raw score: bullet_id order. c5 is RRF #5 and reranks #1.
    assert [b.bullet_id for b in ranked] == ["c5", "c3", "c4", "c2", "c1"]
    assert [b.rrf_rank for b in ranked] == [5, 3, 4, 2, 1]
    assert [b.rerank_rank for b in ranked] == [1, 2, 3, 4, 5]
    assert ranked[0].rerank_score == 0.123457
    assert (ranked[0].job, ranked[0].project, ranked[0].text) == ("Job c5", None, "txt-c5")


# ------------------------------------------------------------------ TR7: one call, own pairs


def test_rerank_lines_one_call_pair_count():
    rr = _rr()
    candidates = [make_candidate(b) for b in ("a", "b", "c", "d", "e")]
    long_line = make_line("line one", ["a", "b", "c"])
    short_line = make_line("line two", ["d", "e"], kind="nice")
    reranker = FakeRerankAdapter()

    lists, pairs = rr.rerank_lines(_funnel_result(long_line, short_line), candidates, reranker)

    assert len(reranker.calls) == 1
    assert pairs == 5 == len(reranker.calls[0])
    assert [len(x) for x in lists] == [3, 2]

    # M1: with nothing to score the adapter is still called exactly once.
    empty_reranker = FakeRerankAdapter()
    lists, pairs = rr.rerank_lines(
        _funnel_result(make_line("x", []), make_line("y", [])), candidates, empty_reranker
    )
    assert lists == [[], []]
    assert pairs == 0
    assert empty_reranker.calls == [[]]


def test_rerank_pairs_only_own_top_k():
    rr = _rr()
    candidates = [make_candidate(b, text=f"text {b}") for b in ("c1", "c2", "c3")]
    line_a = make_line("line A", ["c1", "c2"])
    line_b = make_line("line B", ["c2", "c3"])
    reranker = FakeRerankAdapter()

    rr.rerank_lines(_funnel_result(line_a, line_b), candidates, reranker)

    (pairs,) = reranker.calls
    queries_for = {}
    for query, document in pairs:
        queries_for.setdefault(document, []).append(query)
    assert queries_for == {
        "Keywords: . text c1": ["line A"],
        "Keywords: . text c2": ["line A", "line B"],
        "Keywords: . text c3": ["line B"],
    }
