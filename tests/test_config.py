"""Structural validation for config/models.yaml and config/retrieval.yaml.

This is a structural test only: it checks that both files parse via
yaml.safe_load and that the documented top-level keys are present. It
intentionally does not assert on provider names, model IDs, temperatures,
endpoints, or retrieval values themselves — those are config-driven and must
stay free to change without touching this test.
"""

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_models_yaml_parses_and_has_documented_top_level_keys():
    config = _load("models.yaml")
    assert isinstance(config, dict)
    for key in ("providers", "nodes", "embedding"):
        assert key in config, f"config/models.yaml is missing top-level key: {key}"


def test_models_yaml_embedding_has_vector_space_version():
    config = _load("models.yaml")
    assert "vector_space_version" in config["embedding"], (
        "config/models.yaml embedding block is missing vector_space_version "
        "(required by the embedding index compatibility rule)"
    )


def test_retrieval_yaml_parses_and_has_documented_keys():
    config = _load("retrieval.yaml")
    assert isinstance(config, dict)
    for key in (
        "top_k",
        "rrf_k",
        "total_slots",
        "recency_shares",
    ):
        assert key in config, f"config/retrieval.yaml is missing documented key: {key}"
    # Structure only — the numbers are provisional and stay config values.
    assert isinstance(config["total_slots"], int) and config["total_slots"] >= 1
    shares = config["recency_shares"]
    assert isinstance(shares, list) and shares
    assert all(isinstance(s, int) and s >= 1 for s in shares)
    # The two retired keys are gone.
    assert "final_bullets" not in config
    assert "nice_to_have_weight" not in config
    # v1 of the R4 ticket named a fixed per-job list; it was dropped before it shipped.
    assert "slots_per_job" not in config


_SLOT_BASE = {"top_k": 12, "rrf_k": 60, "total_slots": 14, "recency_shares": [3, 2, 1]}


@pytest.mark.parametrize(
    "data",
    [
        {k: v for k, v in _SLOT_BASE.items() if k != "total_slots"},
        {k: v for k, v in _SLOT_BASE.items() if k != "recency_shares"},
        {**_SLOT_BASE, "total_slots": 0},
        {**_SLOT_BASE, "recency_shares": []},
        {**_SLOT_BASE, "recency_shares": [0]},
    ],
    ids=["no-total-slots", "no-recency-shares", "total-slots-0", "shares-empty", "shares-0"],
)
def test_retrieval_config_slot_keys_required_and_positive(tmp_path, data):
    # Missing, empty and non-positive slot keys fail loudly (no defaults).
    from pydantic import ValidationError

    from loomcv.state import load_retrieval_config

    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(_SLOT_BASE), encoding="utf-8")
    assert load_retrieval_config(good).recency_shares == [3, 2, 1]

    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_retrieval_config(path)


def test_retrieval_config_rejects_retired_key(tmp_path):
    from pydantic import ValidationError

    from loomcv.state import load_retrieval_config

    content = (CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")
    stale = tmp_path / "retrieval.yaml"

    # `per_project_cap`, `final_bullets` and `nice_to_have_weight` are retired keys (a stale
    # yaml fails loudly).
    for retired in (
        "hybrid_alpha",
        "gap_cosine_floor",
        "judge_candidates",
        "per_project_cap",
        "final_bullets",
        "nice_to_have_weight",
    ):
        # an integer value: `per_project_cap: 0.6` would fail as a non-integer while the key
        # was still a field, which would let this test pass for the wrong reason
        stale.write_text(content + f"\n{retired}: 1\n", encoding="utf-8")

        with pytest.raises(ValidationError) as excinfo:
            load_retrieval_config(stale)

        assert retired in str(excinfo.value)


def test_models_yaml_has_reranker_block():
    config = _load("models.yaml")
    assert isinstance(config.get("reranker"), dict), "config/models.yaml has no reranker block"
    for key in ("adapter", "model"):
        assert key in config["reranker"], f"config/models.yaml reranker block is missing {key}"


def test_models_yaml_has_coverage_node():
    # Structure only — provider, model and temperature stay config values.
    config = _load("models.yaml")
    coverage = config["nodes"].get("coverage")
    assert isinstance(coverage, dict), "config/models.yaml nodes has no coverage entry"
    for key in ("provider", "model", "temperature"):
        assert key in coverage, f"config/models.yaml nodes.coverage is missing {key}"
    assert "judge" not in config["nodes"], "the judge node is retired"
    for retired in ("draft_resume", "draft_cl", "verify"):
        assert retired not in config["nodes"], f"the {retired} node is retired (no reader)"
    assert set(config["nodes"]) == {"parse_jd", "coverage", "ingest", "rewrite"}


def test_rewrite_node_is_configured():
    # The profile page's drafting call has a node of its own (its temperature
    # is a config value, not a literal); structure only.
    config = _load("models.yaml")
    rewrite = config["nodes"].get("rewrite")
    assert isinstance(rewrite, dict), "config/models.yaml nodes has no rewrite entry"
    for key in ("provider", "model", "temperature", "timeout_s"):
        assert key in rewrite, f"config/models.yaml nodes.rewrite is missing {key}"
    assert rewrite["provider"] in config["providers"]


def test_vocab_yaml_is_retired():
    # The domain / tech vocabulary has no reader; do not restore it.
    assert not (CONFIG_DIR / "vocab.yaml").exists()


def test_generation_yaml_has_five_keys_with_readers():
    # A key exists only with a reader — `output_dir` and
    # `resume_template` are read by `pipeline draft`, `pandoc_defaults` and `page_limit` by
    # `pipeline finalize`, `bullet_max_chars` by the profile page's drafting call.
    config = _load("generation.yaml")
    assert isinstance(config, dict)
    assert set(config) == {
        "output_dir",
        "resume_template",
        "pandoc_defaults",
        "page_limit",
        "bullet_max_chars",
    }
    assert isinstance(config["output_dir"], str) and config["output_dir"].strip()
    assert isinstance(config["page_limit"], int) and not isinstance(config["page_limit"], bool)
    assert config["page_limit"] > 0
    assert isinstance(config["bullet_max_chars"], int)
    assert not isinstance(config["bullet_max_chars"], bool)
    assert config["bullet_max_chars"] > 0
    from loomcv.state import load_generation_config, resolve_config_file

    for key in ("resume_template", "pandoc_defaults"):
        assert isinstance(config[key], str)
        assert resolve_config_file(config[key]).is_file(), f"{key}: {config[key]} is not a file"

    loaded = load_generation_config()
    assert loaded.model_dump() == config


def test_generation_config_bullet_max_chars_default(tmp_path):
    # A four-key file still loads, with the default budget; zero is refused.
    import yaml
    from pydantic import ValidationError

    from loomcv.state import GenerationConfig, load_generation_config

    four = {k: v for k, v in _load("generation.yaml").items() if k != "bullet_max_chars"}
    path = tmp_path / "generation.yaml"
    path.write_text(yaml.safe_dump(four), encoding="utf-8")
    # The default budget is 250.
    assert load_generation_config(path).bullet_max_chars == 250

    with pytest.raises(ValidationError):
        GenerationConfig.model_validate({**four, "bullet_max_chars": 0})


def test_load_generation_config_default_path():
    from loomcv.state import load_generation_config

    config = load_generation_config()
    assert config.output_dir == _load("generation.yaml")["output_dir"]


# R-1 .. R-4, M-1: the shipped resources live in
# `loomcv/resources/` and are found through `resources.path`; `generation.yaml`'s two pointer
# keys resolve resources first, then the working directory.

_RESOURCE_NAMES = (
    "prompts/parse_jd.md",
    "prompts/coverage.md",
    "prompts/ingest_structure.md",
    "prompts/select_keywords.md",
    "templates/resume_default.md",
    "stopwords_en.txt",
    "pandoc_resume.yaml",
)


def test_resources_path_is_absolute_existing_file():
    # R-1 — the first line is the one `git mv` carried over from `config/prompts/parse_jd.md`.
    from loomcv import resources

    path = resources.path("prompts", "parse_jd.md")
    assert path.is_absolute()
    assert path.is_file()
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first == (
        "You are a structured-extraction assistant for a resume-tailoring system. You read one job"
    )


def test_resources_path_missing_raises():
    # R-2 — never a silent fallback.
    from loomcv import resources

    with pytest.raises(FileNotFoundError, match="prompts/nope.md"):
        resources.path("prompts", "nope.md")


def test_resolve_config_file_finds_packaged_resources():
    # R-3
    from loomcv import resources
    from loomcv.state import resolve_config_file

    template = resolve_config_file("templates/resume_default.md")
    assert template == resources.path("templates", "resume_default.md")
    defaults = resolve_config_file("pandoc_resume.yaml")
    assert defaults == resources.path("pandoc_resume.yaml")
    for path in (template, defaults):
        assert path.is_absolute() and path.is_file()


def test_resolve_config_file_falls_back_to_cwd_and_keeps_absolute(monkeypatch, tmp_path):
    # R-4 — a relative value with no such resource is the user's file beside where they run
    # `pipeline`; a `..` part never enters the package lookup; absolute as is.
    from loomcv.state import resolve_config_file

    monkeypatch.chdir(tmp_path)
    assert resolve_config_file("my_template.md") == Path.cwd() / "my_template.md"
    assert resolve_config_file("../x.md") == Path.cwd() / "../x.md"
    assert resolve_config_file(str(tmp_path / "x.md")) == tmp_path / "x.md"


def test_packaged_resources_and_old_config_paths():
    # M-1 — read through `importlib.resources`, not through a repo path.
    from importlib.resources import files

    root = files("loomcv.resources")
    for name in _RESOURCE_NAMES:
        assert root.joinpath(*name.split("/")).is_file(), name
    repo_root = CONFIG_DIR.parent
    for old in ("prompts", "templates", "stopwords_en.txt", "pandoc_resume.yaml"):
        assert not (repo_root / "config" / old).exists(), f"config/{old} still exists"
