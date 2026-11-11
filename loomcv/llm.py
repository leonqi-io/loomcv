"""Generation-side adapter contract.

Resolves a `GenerationAdapter` for one pipeline node from `config/models.yaml`'s
`providers:` + `nodes:` sections (design doc §三), or lets a caller inject one
directly (tests always inject). No provider, model, temperature, endpoint, or key
is hardcoded here — everything comes from the one config file the caller names.

Mirrors `loomcv/sync/embedding.py`: Protocol + `build_*` dispatch + a stdlib
`urllib` OpenAI-compatible client. Only `openai_compatible` is implemented; the
`anthropic` / `openai` adapter names stay valid config but raise until a hosted
provider is actually configured.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import URLError

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
_HOSTED_ADAPTERS = ("anthropic", "openai")


class ProviderConfig(BaseModel):
    name: str
    adapter: str
    base_url_env: str | None = None
    api_key_env: str | None = None


class NodeConfig(BaseModel):
    name: str
    provider: str
    model: str
    temperature: float
    timeout_s: float | None = Field(default=None, gt=0)
    seed: int | None = None


@dataclass(frozen=True)
class CallRecord:
    """What one completed generation call cost; never holds reply or reasoning text."""

    elapsed_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_chars: int | None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reasoning_chars(message: object) -> int | None:
    if not isinstance(message, dict):
        return None
    for key in ("reasoning", "reasoning_content"):
        if isinstance(message.get(key), str):
            return len(message[key])
    return None


class GenerationAdapter(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def temperature(self) -> float: ...

    def complete(self, system: str, user: str, *, temperature: float) -> str: ...


def load_node_config(
    node: str, config_path: Path | None = None
) -> tuple[NodeConfig, ProviderConfig]:
    """Read one node's routing and its provider from exactly one config file."""
    path = Path(config_path or _DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for section in ("providers", "nodes"):
        if not isinstance(raw.get(section), dict):
            raise KeyError(f"section '{section}' missing from {path}")
    if node not in raw["nodes"]:
        raise KeyError(f"node '{node}' not in nodes of {path}")
    node_cfg = NodeConfig.model_validate({**raw["nodes"][node], "name": node})
    if node_cfg.provider not in raw["providers"]:
        raise KeyError(f"provider '{node_cfg.provider}' (node '{node}') not in providers of {path}")
    provider_cfg = ProviderConfig.model_validate(
        {**raw["providers"][node_cfg.provider], "name": node_cfg.provider}
    )
    return node_cfg, provider_cfg


def build_generation_adapter(
    node_cfg: NodeConfig, provider_cfg: ProviderConfig
) -> GenerationAdapter:
    if provider_cfg.adapter == "openai_compatible":
        return OpenAICompatibleGenerationAdapter(node_cfg, provider_cfg)
    if provider_cfg.adapter in _HOSTED_ADAPTERS:
        raise NotImplementedError(
            f"adapter '{provider_cfg.adapter}' is not implemented yet (this version ships"
            " openai_compatible only); route the node to an openai_compatible provider"
        )
    raise ValueError(
        f"unknown generation adapter {provider_cfg.adapter!r} for provider"
        f" '{provider_cfg.name}'; expected 'openai_compatible'"
    )


class OpenAICompatibleGenerationAdapter:
    """Chat completion via any OpenAI-compatible HTTP endpoint (stdlib only)."""

    def __init__(self, node_cfg: NodeConfig, provider_cfg: ProviderConfig) -> None:
        if not provider_cfg.base_url_env:
            raise ValueError(
                f"provider '{provider_cfg.name}' (openai_compatible) requires base_url_env"
            )
        base_url = os.environ.get(provider_cfg.base_url_env)
        if not base_url:
            raise RuntimeError(
                f"environment variable {provider_cfg.base_url_env} is not set"
                f" (required by provider '{provider_cfg.name}')"
            )
        self._node = node_cfg
        self._provider = provider_cfg
        self._base_url = base_url.rstrip("/")
        self._api_key = (
            os.environ.get(provider_cfg.api_key_env) if provider_cfg.api_key_env else None
        )
        self.calls: list[CallRecord] = []

    @property
    def model_id(self) -> str:
        return self._node.model

    @property
    def temperature(self) -> float:
        return self._node.temperature

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        body: dict = {
            "model": self._node.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if self._node.seed is not None:
            body["seed"] = self._node.seed
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions", data=payload, headers=headers, method="POST"
        )
        urlopen_kwargs = {} if self._node.timeout_s is None else {"timeout": self._node.timeout_s}
        started = time.monotonic()
        try:
            with urllib.request.urlopen(  # noqa: S310 - operator-configured base_url
                request, **urlopen_kwargs
            ) as response:
                raw = response.read().decode("utf-8")
        except (TimeoutError, URLError) as exc:
            timed_out = isinstance(exc, TimeoutError) or isinstance(
                getattr(exc, "reason", None), TimeoutError
            )
            if not timed_out:
                raise
            raise RuntimeError(
                f"provider '{self._provider.name}' timed out after {self._node.timeout_s} s"
                f" (node '{self._node.name}')"
            ) from exc
        elapsed = time.monotonic() - started
        try:
            parsed = json.loads(raw)
            message = parsed["choices"][0]["message"]
            content = message["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            parsed, message, content = None, None, None
        if not isinstance(content, str):
            raise RuntimeError(
                f"provider '{self._provider.name}' returned no choices[0].message.content"
            )
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        self.calls.append(
            CallRecord(
                elapsed_s=elapsed,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                reasoning_chars=_reasoning_chars(message),
            )
        )
        return content
