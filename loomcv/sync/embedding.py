"""Embedding adapter interface.

Resolves an `EmbeddingAdapter` from `config/models.yaml`'s `embedding:` block,
or lets a caller inject one directly (tests always inject). No provider,
model, endpoint, or temperature is hardcoded here — everything comes from
config or from the caller.

Heavy imports (`sentence_transformers`, `torch`) are lazy, inside the
adapter that needs them, so importing this module never pulls torch.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class EmbeddingConfig(BaseModel):
    adapter: str
    model: str
    base_url_env: str | None = None
    api_key_env: str | None = None
    vector_space_version: int


class EmbeddingAdapter(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def vector_space_version(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def load_embedding_config(config_path: Path | None = None) -> EmbeddingConfig:
    path = config_path or _DEFAULT_CONFIG_PATH
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EmbeddingConfig.model_validate(raw["embedding"])


def build_embedding_adapter(config: EmbeddingConfig) -> EmbeddingAdapter:
    if config.adapter == "sentence_transformers":
        return SentenceTransformersAdapter(config)
    if config.adapter == "openai_compatible":
        return OpenAICompatibleAdapter(config)
    raise ValueError(
        f"unknown embedding adapter {config.adapter!r};"
        " expected 'sentence_transformers' or 'openai_compatible'"
    )


class SentenceTransformersAdapter:
    """Local embedding via `sentence-transformers` (optional `embed` dependency group)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence_transformers is not installed;"
                " run `uv sync --group embed` to use the sentence_transformers adapter"
            ) from exc
        self._config = config
        self._model = SentenceTransformer(config.model)

    @property
    def model_id(self) -> str:
        return self._config.model

    @property
    def vector_space_version(self) -> int:
        return self._config.vector_space_version

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.encode(list(texts)).tolist()


class OpenAICompatibleAdapter:
    """Local/hosted embedding via any OpenAI-compatible HTTP endpoint (stdlib only)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        if not config.base_url_env:
            raise ValueError(
                "openai_compatible embedding adapter requires config.base_url_env to be set"
            )
        base_url = os.environ.get(config.base_url_env)
        if not base_url:
            raise RuntimeError(
                f"environment variable {config.base_url_env} is not set"
                " (required by the openai_compatible embedding adapter)"
            )
        self._config = config
        self._base_url = base_url.rstrip("/")
        self._api_key = os.environ.get(config.api_key_env) if config.api_key_env else None

    @property
    def model_id(self) -> str:
        return self._config.model

    @property
    def vector_space_version(self) -> int:
        return self._config.vector_space_version

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._config.model, "input": list(texts)}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self._base_url}/embeddings", data=payload, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310 - operator-configured base_url
            body = json.loads(response.read().decode("utf-8"))
        return [item["embedding"] for item in body["data"]]
