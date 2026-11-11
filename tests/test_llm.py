"""Tests for the generation-side adapter `loomcv/llm.py`.

Config files are written by each test into `tmp_path` — the repo
`config/models.yaml` carries placeholder model ids and is not used. No test reaches
the network: an autouse fixture replaces `urllib.request.urlopen` (the object
`loomcv.llm` calls through `urllib.request`) with one that raises; test 2 overrides
it with a recording fake. `loomcv.llm` is imported inside each test so every case
fails on its own at red.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest
import yaml

_PROVIDERS = {
    "local_eval": {"adapter": "openai_compatible", "base_url_env": "FAKE_BASE_URL"},
}
_NODES = {"parse_jd": {"provider": "local_eval", "model": "fake-model", "temperature": 0}}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("network call in test")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


def _write_config(tmp_path: Path, data: dict, name: str = "models.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _full_config(tmp_path: Path, providers: dict | None = None) -> Path:
    return _write_config(tmp_path, {"providers": providers or _PROVIDERS, "nodes": _NODES})


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._raw = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _recording_urlopen(monkeypatch, body: dict) -> list:
    seen: list = []

    def _fake(request, *args, **kwargs):
        seen.append(request)
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    return seen


# ------------------------------------------------------------------ test 1


def test_load_node_config_and_errors(tmp_path):
    from loomcv.llm import build_generation_adapter, load_node_config

    node, provider = load_node_config("parse_jd", _full_config(tmp_path))
    assert node.model == "fake-model"
    assert node.temperature == 0
    assert node.provider == "local_eval"
    assert provider.name == "local_eval"
    assert provider.adapter == "openai_compatible"
    assert provider.base_url_env == "FAKE_BASE_URL"

    with pytest.raises(KeyError, match="draft_resume"):
        load_node_config("draft_resume", _full_config(tmp_path))

    embedding_only = _write_config(
        tmp_path,
        {"embedding": {"adapter": "openai_compatible", "model": "x", "vector_space_version": 1}},
        name="models.local.yaml",
    )
    with pytest.raises(KeyError, match="providers"):
        load_node_config("parse_jd", embedding_only)

    bogus = _full_config(tmp_path, {"local_eval": {"adapter": "bogus"}})
    node, provider = load_node_config("parse_jd", bogus)
    with pytest.raises(ValueError, match="bogus"):
        build_generation_adapter(node, provider)


# ------------------------------------------------------------------ test 2


def test_openai_compatible_request_shape(tmp_path, monkeypatch):
    from loomcv.llm import build_generation_adapter, load_node_config

    node, provider = load_node_config("parse_jd", _full_config(tmp_path))

    monkeypatch.delenv("FAKE_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="FAKE_BASE_URL"):
        build_generation_adapter(node, provider)

    monkeypatch.setenv("FAKE_BASE_URL", "http://fake.invalid/v1/")
    seen = _recording_urlopen(monkeypatch, {"choices": [{"message": {"content": "hello"}}]})
    adapter = build_generation_adapter(node, provider)

    reply = adapter.complete("sys text", "user text", temperature=0.3)

    assert reply == "hello"
    assert len(seen) == 1
    request = seen[0]
    assert request.full_url == "http://fake.invalid/v1/chat/completions"
    assert request.get_header("Authorization") is None
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "fake-model"
    assert body["temperature"] == 0.3
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert [m["content"] for m in body["messages"]] == ["sys text", "user text"]

    keyed = _full_config(
        tmp_path,
        {
            "local_eval": {
                "adapter": "openai_compatible",
                "base_url_env": "FAKE_BASE_URL",
                "api_key_env": "FAKE_KEY",
            }
        },
    )
    monkeypatch.setenv("FAKE_KEY", "abc")
    seen = _recording_urlopen(monkeypatch, {"choices": [{"message": {"content": "hello"}}]})
    adapter = build_generation_adapter(*load_node_config("parse_jd", keyed))
    adapter.complete("s", "u", temperature=0)
    assert seen[0].get_header("Authorization") == "Bearer abc"

    _recording_urlopen(monkeypatch, {"foo": 1})
    adapter = build_generation_adapter(node, provider)
    with pytest.raises(RuntimeError, match="local_eval"):
        adapter.complete("s", "u", temperature=0)


# ------------------------------------------------------------------ test 3


def test_hosted_adapters_not_implemented(tmp_path):
    from loomcv.llm import build_generation_adapter, load_node_config

    for adapter_name in ("anthropic", "openai"):
        config = _full_config(
            tmp_path, {"local_eval": {"adapter": adapter_name, "api_key_env": "SOME_KEY"}}
        )
        node, provider = load_node_config("parse_jd", config)
        with pytest.raises(NotImplementedError, match="openai_compatible"):
            build_generation_adapter(node, provider)


# ------------------------------------------------ timeout, seed, call record
#
# L-1 … L-5. The existing `_recording_urlopen` is left as it is; the
# new tests use a sibling that also records the keyword arguments `urlopen` received.

_OK_BODY = {"choices": [{"message": {"content": "hello"}}]}


def _node_config(tmp_path: Path, **extra) -> Path:
    nodes = {"parse_jd": {**_NODES["parse_jd"], **extra}}
    return _write_config(tmp_path, {"providers": _PROVIDERS, "nodes": nodes})


def _adapter(tmp_path: Path, monkeypatch, **extra):
    from loomcv.llm import build_generation_adapter, load_node_config

    monkeypatch.setenv("FAKE_BASE_URL", "http://fake.invalid/v1")
    return build_generation_adapter(*load_node_config("parse_jd", _node_config(tmp_path, **extra)))


def _recording_urlopen_kw(monkeypatch, body: dict) -> list:
    seen: list = []

    def _fake(request, *args, **kwargs):
        seen.append((request, kwargs))
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    return seen


def test_node_config_timeout_and_seed(tmp_path):
    # L-1
    from pydantic import ValidationError

    from loomcv.llm import load_node_config

    node, _provider = load_node_config("parse_jd", _node_config(tmp_path, timeout_s=30, seed=7))
    assert node.timeout_s == 30
    assert node.seed == 7

    node, _provider = load_node_config("parse_jd", _full_config(tmp_path))
    assert node.timeout_s is None
    assert node.seed is None

    for bad in (0, -1):
        with pytest.raises(ValidationError):
            load_node_config("parse_jd", _node_config(tmp_path, timeout_s=bad))


def test_timeout_passed_to_urlopen(tmp_path, monkeypatch):
    # L-2
    seen = _recording_urlopen_kw(monkeypatch, _OK_BODY)
    _adapter(tmp_path, monkeypatch, timeout_s=30).complete("s", "u", temperature=0)
    assert seen[0][1]["timeout"] == 30

    seen = _recording_urlopen_kw(monkeypatch, _OK_BODY)
    _adapter(tmp_path, monkeypatch).complete("s", "u", temperature=0)
    assert "timeout" not in seen[0][1]


def test_timeout_becomes_runtime_error(tmp_path, monkeypatch):
    # L-3
    from urllib.error import URLError

    adapter = _adapter(tmp_path, monkeypatch, timeout_s=30)

    def _raising(exc):
        def _fake(request, *args, **kwargs):
            raise exc

        monkeypatch.setattr(urllib.request, "urlopen", _fake)

    for exc in (TimeoutError(), URLError(TimeoutError())):
        _raising(exc)
        with pytest.raises(RuntimeError, match="timed out after 30") as info:
            adapter.complete("s", "u", temperature=0)
        assert "local_eval" in str(info.value)
        assert "parse_jd" in str(info.value)

    class _SlowResponse(_FakeResponse):
        def read(self) -> bytes:
            raise TimeoutError()

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, *a, **k: _SlowResponse({}))
    with pytest.raises(RuntimeError, match="timed out after 30"):
        adapter.complete("s", "u", temperature=0)

    _raising(URLError("refused"))
    with pytest.raises(URLError):
        adapter.complete("s", "u", temperature=0)


def test_seed_in_body_only_when_set(tmp_path, monkeypatch):
    # L-4
    seen = _recording_urlopen_kw(monkeypatch, _OK_BODY)
    _adapter(tmp_path, monkeypatch, seed=7).complete("s", "u", temperature=0)
    assert json.loads(seen[0][0].data.decode("utf-8"))["seed"] == 7

    seen = _recording_urlopen_kw(monkeypatch, _OK_BODY)
    _adapter(tmp_path, monkeypatch).complete("s", "u", temperature=0)
    assert "seed" not in json.loads(seen[0][0].data.decode("utf-8"))


def test_call_records(tmp_path, monkeypatch):
    # L-5
    adapter = _adapter(tmp_path, monkeypatch)
    assert adapter.calls == []

    _recording_urlopen_kw(
        monkeypatch,
        {
            "choices": [{"message": {"content": "hello", "reasoning": "abcd"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5},
        },
    )
    adapter.complete("s", "u", temperature=0)
    assert len(adapter.calls) == 1
    record = adapter.calls[0]
    assert record.elapsed_s >= 0
    assert (record.prompt_tokens, record.completion_tokens, record.reasoning_chars) == (11, 5, 4)

    _recording_urlopen_kw(monkeypatch, _OK_BODY)
    adapter.complete("s", "u", temperature=0)
    second = adapter.calls[1]
    assert (second.prompt_tokens, second.completion_tokens, second.reasoning_chars) == (
        None,
        None,
        None,
    )

    _recording_urlopen_kw(
        monkeypatch, {"choices": [{"message": {"content": "hello", "reasoning_content": "xyz"}}]}
    )
    adapter.complete("s", "u", temperature=0)
    assert [r.reasoning_chars for r in adapter.calls] == [4, None, 3]

    fresh = _adapter(tmp_path, monkeypatch, timeout_s=30)

    def _timeout(request, *args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(urllib.request, "urlopen", _timeout)
    with pytest.raises(RuntimeError):
        fresh.complete("s", "u", temperature=0)
    assert fresh.calls == []
