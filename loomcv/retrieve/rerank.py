"""Cross-encoder reranking, one cut per requirement line.

Two halves, like `loomcv/sync/embedding.py`:

- The adapter contract. A `RerankAdapter` is resolved from `config/models.yaml`'s
  `reranker:` block, or injected directly (tests always inject). No model,
  device or endpoint is hardcoded here. `sentence_transformers` (CrossEncoder,
  optional `embed` dependency group) is imported lazily inside its adapter, so
  importing this module never pulls torch; `http_rerank` is reserved.
- The pure `rerank_lines()`: given each line's RRF cut (`FunnelResult`), it scores
  only the pairs `(line text, rerank_document(bullet))` of that line's own cut in
  ONE adapter call and orders each line's bullets by `(-score, bullet_id)`. A
  reranker score is only ever compared with scores of the same line, and no
  threshold reads it; the order is the output.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel

from loomcv.retrieve.rank import Candidate, FunnelResult, RankedBullet

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class RerankConfig(BaseModel):
    adapter: str
    model: str
    base_url_env: str | None = None


class RerankAdapter(Protocol):
    @property
    def model_id(self) -> str: ...

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


def load_rerank_config(config_path: Path | str | None = None) -> RerankConfig:
    path = config_path or _DEFAULT_CONFIG_PATH
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RerankConfig.model_validate(raw["reranker"])


def build_rerank_adapter(config: RerankConfig) -> RerankAdapter:
    if config.adapter == "sentence_transformers":
        return SentenceTransformersRerankAdapter(config)
    if config.adapter == "http_rerank":
        raise NotImplementedError(
            "reranker adapter 'http_rerank' is not implemented yet (this version ships"
            " sentence_transformers only); set reranker.adapter to sentence_transformers"
        )
    raise ValueError(
        f"unknown reranker adapter {config.adapter!r};"
        " expected 'sentence_transformers' or 'http_rerank'"
    )


class SentenceTransformersRerankAdapter:
    """Local cross-encoder via `sentence-transformers` (optional `embed` dependency group)."""

    def __init__(self, config: RerankConfig) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence_transformers is not installed;"
                " run `uv sync --group embed` to use the sentence_transformers reranker"
            ) from exc
        self._config = config
        self._model = CrossEncoder(config.model)

    @property
    def model_id(self) -> str:
        return self._config.model

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        # The model's default activation (sigmoid for single-label cross-encoders): the
        # adapter's native monotonic scale, returned as plain floats.
        if not pairs:
            return []
        return self._model.predict(list(pairs)).tolist()


def rerank_document(candidate: Candidate) -> str:
    """The reranker's view of a bullet: its keywords in stored order, then its text.

    Byte-equal to `run_t1.py` 303; a bullet without keywords gets `"Keywords: . " + text`.
    """
    return "Keywords: " + "; ".join(candidate.keywords) + ". " + candidate.text


def rerank_lines(
    funnel_result: FunnelResult,
    candidates: list[Candidate],
    reranker: RerankAdapter,
) -> tuple[list[list[RankedBullet]], int]:
    """One `reranker.score` call over every line's own cut → one ordered list per line.

    Returns `(per-line lists in line order, number of pairs scored)`. The adapter is called
    exactly once, also when there is nothing to score.
    """
    by_id = {c.bullet_id: c for c in candidates}
    pairs = [
        (line.text, rerank_document(by_id[bid])) for line in funnel_result.lines for bid in line.top
    ]
    raw = reranker.score(pairs)
    if len(raw) != len(pairs):
        raise ValueError(f"reranker returned {len(raw)} scores for {len(pairs)} pairs")
    scores = iter(float(v) for v in raw)

    lists: list[list[RankedBullet]] = []
    for line in funnel_result.lines:
        line_scores = {bid: next(scores) for bid in line.top}
        order = sorted(line.top, key=lambda bid: (-line_scores[bid], bid))
        ranked: list[RankedBullet] = []
        for rerank_rank, bid in enumerate(order, start=1):
            c = by_id[bid]
            ranked.append(
                RankedBullet(
                    bullet_id=bid,
                    job=c.job,
                    project=c.project,
                    text=c.text,
                    rrf_rank=line.top.index(bid) + 1,
                    rerank_score=round(line_scores[bid], 6),
                    rerank_rank=rerank_rank,
                )
            )
        lists.append(ranked)
    return lists, len(pairs)
