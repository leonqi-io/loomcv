"""CLI tests for the `pipeline sync` entry point.

Fixtures are synthetic (`tests/fixtures/index_vault.md`, read-only);
the embedding adapter is always an in-process fake injected by monkeypatching
`loomcv.cli.build_embedding_adapter`, so no test loads sentence-transformers or
hits the network (same rule as S2). All stores live under `tmp_path`.

The CLI is invoked in-process as `main([...])` with `capsys`, except the
`python -m loomcv.cli` check (test 9), which uses a subprocess. `main` is
imported **inside** each test that needs it (not at module scope) so the
metadata-only test 10 still passes at red when `loomcv/cli.py` is absent.

All fixture content is fictional — no real career data (M1 protocol).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from loomcv import resources
from loomcv.sync.parse_assets import parse_star_vault

FIXTURES = Path(__file__).resolve().parent / "fixtures"
VAULT_NAME = "index_vault.md"

# `--include-reserve` is gone (was SEVEN_FLAGS).
SIX_FLAGS = (
    "--vault",
    "--db",
    "--chroma",
    "--config",
    "--rebuild",
    "--include-draft",
)


def _vault_text() -> str:
    return (FIXTURES / VAULT_NAME).read_text(encoding="utf-8")


def _visible_count(vault_text: str, *, include_draft: bool = False) -> int:
    """Bullet count the default `sync` indexes, derived from the parser.

    `sync_vault` threads the flag into `parse_star_vault`, so this is exactly
    the set it will add — counts are derived, never hardcoded.
    """
    pr = parse_star_vault(vault_text, include_draft=include_draft)
    return sum(len(j.bullets) for j in pr.jobs)


@pytest.fixture(autouse=True)
def _no_default_embedding_function(monkeypatch):
    """No code path may ever call Chroma's default (ONNX) embedding fn.

    Copied from `tests/test_index.py` (re-declared locally, not imported) so the
    CLI suite stays independent of the index suite.
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


# ------------------------------------------------------------------ helpers


def _copy_vault(tmp_path: Path, text: str | None = None, name: str = "vault.md") -> Path:
    dst = tmp_path / name
    if text is None:
        shutil.copy(FIXTURES / VAULT_NAME, dst)
    else:
        dst.write_text(text, encoding="utf-8")
    return dst


def _run(vault: Path, db: Path, chroma: Path, extra: list[str] | None = None) -> int:
    from loomcv.cli import main

    argv = ["sync", "--vault", str(vault), "--db", str(db), "--chroma", str(chroma)]
    if extra:
        argv += extra
    return main(argv)


class FakeRerankAdapter:
    """Network-free stand-in for `RerankAdapter`: every pair scores 0.5."""

    @property
    def model_id(self) -> str:
        return "fake-reranker"

    def score(self, pairs):
        return [0.5 for _ in pairs]


def _requested_lines(user: str) -> list[str]:
    """The `- ` lines between `Requirements` and `Keywords (n)` of a rendered coverage prompt."""
    block = user.split("Requirements\n", 1)[1].split("\n\nKeywords (", 1)[0]
    return [ln[2:] for ln in block.splitlines()]


class FakeCoverageAdapter:
    """Network-free coverage node: answers any rendered prompt, every line `none`."""

    def __init__(self, *, model_id: str = "fake-coverage") -> None:
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
        return json.dumps(
            {line: {"match": "none", "keywords": []} for line in _requested_lines(user)}
        )


def _patch_adapter(monkeypatch, *, vector_space_version: int = 1) -> list:
    """Patch every adapter builder; returns the `(node name, adapter)` generation builds."""
    monkeypatch.setattr(
        "loomcv.cli.build_embedding_adapter",
        lambda cfg: FakeEmbeddingAdapter(vector_space_version=vector_space_version),
    )
    # raising=False: the sync tests also pass through here, before (red) and after the
    # reranker exists in cli.py; no test ever builds a real CrossEncoder.
    monkeypatch.setattr(
        "loomcv.cli.build_rerank_adapter", lambda cfg: FakeRerankAdapter(), raising=False
    )
    built: list = []

    def build(node, provider):
        if node.name != "coverage":
            raise AssertionError(f"unexpected generation node {node.name!r}")
        adapter = FakeCoverageAdapter(model_id=node.model)
        built.append((node.name, adapter))
        return adapter

    monkeypatch.setattr("loomcv.cli.build_generation_adapter", build)
    return built


# ---------------------------------------------------------------- fresh sync


def test_sync_happy_path_exit_0_and_counts(tmp_path, monkeypatch, capsys):
    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    rc = _run(vault, db, chroma)

    assert rc == 0
    out = capsys.readouterr().out
    n = _visible_count(_vault_text())
    assert f"added={n}" in out
    assert f"embedded={n}" in out
    assert "jobs_written=3" in out
    assert "skills=0" in out
    assert "projects_written" not in out
    assert db.exists()
    assert chroma.exists()


def test_sync_second_run_is_unchanged(tmp_path, monkeypatch, capsys):
    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    assert _run(vault, db, chroma) == 0
    capsys.readouterr()
    n = _visible_count(_vault_text())

    assert _run(vault, db, chroma) == 0
    out = capsys.readouterr().out
    assert "added=0" in out
    assert "updated=0" in out
    assert f"unchanged={n}" in out
    assert "embedded=0" in out


def test_sync_single_bullet_edit_reembeds_one(tmp_path, monkeypatch, capsys):
    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    assert _run(vault, db, chroma) == 0
    capsys.readouterr()

    edited = vault.read_text(encoding="utf-8").replace("from 800ms to 200ms", "from 800ms to 150ms")
    vault.write_text(edited, encoding="utf-8")

    assert _run(vault, db, chroma) == 0
    out = capsys.readouterr().out
    assert "updated=1" in out
    assert "embedded=1" in out


def test_sync_parse_error_exit_1_no_store_created(tmp_path, monkeypatch, capsys):
    # The injected defect is a job without its required `From` line
    text = _vault_text().replace("* **From**: 2023-01\n", "", 1)
    assert text != _vault_text()
    vault = _copy_vault(tmp_path, text)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    rc = _run(vault, db, chroma)

    assert rc == 1
    assert "line" in capsys.readouterr().err
    assert not db.exists()
    assert not chroma.exists()


# ---------------------------------------------------------------- rebuild


def test_sync_rebuild_flag_passes_through(tmp_path, monkeypatch, capsys):
    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    assert _run(vault, db, chroma) == 0
    capsys.readouterr()
    n = _visible_count(_vault_text())

    assert _run(vault, db, chroma, extra=["--rebuild"]) == 0
    out = capsys.readouterr().out
    assert f"embedded={n}" in out


# ---------------------------------------------------------------- include flags


def test_sync_include_flags_pass_through(tmp_path, monkeypatch, capsys):
    base = _vault_text()
    n_default = _visible_count(base)
    n_draft = _visible_count(base, include_draft=True)
    assert n_draft == n_default + 1
    _patch_adapter(monkeypatch)

    vault_a = _copy_vault(tmp_path, base, name="vault_default.md")
    db_a, chroma_a = tmp_path / "db_default.sqlite", tmp_path / "chroma_default"
    assert _run(vault_a, db_a, chroma_a) == 0
    assert f"added={n_default}" in capsys.readouterr().out

    vault_b = _copy_vault(tmp_path, base, name="vault_draft.md")
    db_b, chroma_b = tmp_path / "db_draft.sqlite", tmp_path / "chroma_draft"
    assert _run(vault_b, db_b, chroma_b, extra=["--include-draft"]) == 0
    assert f"added={n_draft}" in capsys.readouterr().out


def test_sync_include_reserve_flag_rejected(tmp_path, monkeypatch):
    """`--include-reserve` no longer exists on either subcommand."""
    from loomcv.cli import main

    _patch_adapter(monkeypatch)
    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"

    with pytest.raises(SystemExit) as sync_exc:
        main(
            ["sync", "--vault", str(vault), "--db", str(db), "--chroma", str(chroma)]
            + ["--include-reserve"]
        )
    assert sync_exc.value.code == 2

    with pytest.raises(SystemExit) as match_exc:
        main(["match", "--profile", str(tmp_path / "p.json"), "--include-reserve"])
    assert match_exc.value.code == 2


def test_sync_without_vault_and_no_default_file_exit_1(tmp_path, monkeypatch, capsys):
    """C-6 (was `test_sync_vault_required`): `--vault` is optional with
    the default `02_database/vault.md`; with the file missing the command refuses, exit 1."""
    from loomcv.cli import main

    _patch_adapter(monkeypatch)
    monkeypatch.chdir(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"

    assert main(["sync", "--db", str(db), "--chroma", str(chroma)]) == 1
    err = capsys.readouterr().err
    assert err.strip() == (
        "sync: no vault at 02_database/vault.md — run `pipeline complete` first or pass --vault"
    )
    assert not db.exists()
    assert not chroma.exists()


def test_sync_without_vault_indexes_default_vault(tmp_path, monkeypatch, capsys):
    """C-7: no `--vault` and `02_database/vault.md` present → indexed."""
    from loomcv.cli import main

    _patch_adapter(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "02_database").mkdir()
    shutil.copy(FIXTURES / VAULT_NAME, tmp_path / "02_database" / "vault.md")
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"

    assert main(["sync", "--db", str(db), "--chroma", str(chroma)]) == 0
    out = capsys.readouterr().out
    assert f"added={_visible_count(_vault_text())}" in out
    assert db.exists()


def test_sync_old_grammar_vault_exit_1(tmp_path, monkeypatch, capsys):
    text = "# Old vault\n\n## [Project] Foo\n* **Tags**: #a\n* **Bullets**:\n  - [A] Old bullet.\n"
    vault = _copy_vault(tmp_path, text)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    rc = _run(vault, db, chroma)

    assert rc == 1
    err = capsys.readouterr().err
    assert "line 3" in err
    assert "'## [Project]' is the pre-R1 vault grammar" in err
    assert not db.exists()
    assert not chroma.exists()


def _build_v2_store(db: Path, chroma: Path) -> None:
    """A pre-v3 store: schema_version 2, `projects` table, one old-id bullet in Chroma."""
    import sqlite3

    import chromadb

    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE projects (project_id TEXT PRIMARY KEY, name TEXT NOT NULL,"
        " context TEXT NOT NULL DEFAULT '', tags_json TEXT NOT NULL DEFAULT '[]',"
        " domain TEXT, year INTEGER, tech_json TEXT NOT NULL DEFAULT '[]');"
        "CREATE TABLE bullets (bullet_id TEXT PRIMARY KEY, project_id TEXT NOT NULL"
        " REFERENCES projects(project_id) ON DELETE CASCADE, text TEXT NOT NULL,"
        " tags_json TEXT NOT NULL DEFAULT '[]', content_hash TEXT NOT NULL,"
        " status TEXT NOT NULL DEFAULT 'confirmed', updated TEXT,"
        " bullet_order INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO meta VALUES ('schema_version', '2');"
        "INSERT INTO projects (project_id, name) VALUES ('old-project', 'Old');"
        "INSERT INTO bullets (bullet_id, project_id, text, content_hash)"
        " VALUES ('old-project::old-tag', 'old-project', 'Old text.', 'sha1:old');"
    )
    conn.commit()
    conn.close()
    collection = chromadb.PersistentClient(path=str(chroma)).create_collection(
        "star_bullets",
        embedding_function=None,
        metadata={"model_id": "fake-model", "vector_space_version": 1},
    )
    collection.upsert(ids=["old-project::old-tag"], embeddings=[_deterministic_vector("Old text.")])


def test_sync_v2_db_full_rebuild_line_once(tmp_path, monkeypatch, capsys):
    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _build_v2_store(db, chroma)
    _patch_adapter(monkeypatch)
    n = _visible_count(_vault_text())

    rc = _run(vault, db, chroma)

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err.count("schema 2 → 3: full rebuild") == 1
    assert "every bullet_id changed" in captured.err
    assert f"added={n}" in captured.out

    assert _run(vault, db, chroma) == 0
    assert "schema 2 → 3" not in capsys.readouterr().err


# ---------------------------------------------------------- vector-space guard


def test_sync_vector_space_mismatch_exit_3(tmp_path, monkeypatch, capsys):
    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"

    _patch_adapter(monkeypatch, vector_space_version=1)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()

    _patch_adapter(monkeypatch, vector_space_version=2)
    rc = _run(vault, db, chroma)

    assert rc == 3
    assert "rebuild" in capsys.readouterr().err


# -------------------------------------------------------------- usage errors


def test_no_subcommand_is_usage_error():
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


# ----------------------------------------------------------- module entry point


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "loomcv.cli", "sync", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for flag in SIX_FLAGS:
        assert flag in result.stdout
    assert "--include-reserve" not in result.stdout


# ------------------------------------------------------ console script (metadata)


def test_console_script_registered():
    eps = {ep.name: ep for ep in importlib.metadata.entry_points(group="console_scripts")}
    if "pipeline" not in eps:
        pytest.skip("pipeline console script not installed (run `uv sync`)")
    assert eps["pipeline"].value == "loomcv.cli:main"


# ------------------------------------------------------ pipeline match

_MATCH_KEYS = [
    "profile",
    "parse_warnings",
    "requirements",
    "gaps",
    "claims",
    "selection",
    "config",
    "diagnostics",
]
# The merge's result, one level at a time.
_SELECTION_KEYS = [
    "total_slots",
    "jobs",
    "walk",
    "unplaced",
    "covered_by",
    "uncovered",
    "passes",
]
_JOB_KEYS = ["job_id", "job", "job_order", "cap", "placed", "bullets"]
_SELECTED_BULLET_KEYS = [
    "bullet_id",
    "project",
    "text",
    "line",
    "kind",
    "pass",
    "walk_position",
    "overlap",
    "rerank_rank",
    "rrf_position",
    "covers",
]
_WALK_KEYS = ["bullet_id", "job_id", "line", "kind", "pass"]
_LINE_KEYS = [
    "text",
    "kind",
    "grade",
    "mapped_keywords",
    "ranked",
    "max_cosine",
    "max_bm25",
    "top_rrf_score",
    "parts",
]
_RANKED_KEYS = [
    "bullet_id",
    "job",
    "project",
    "text",
    "rrf_rank",
    "rerank_score",
    "rerank_rank",
]
_DIAGNOSTICS_KEYS = [
    "vector_missing",
    "rerank_pairs",
    "rerank_model",
    "coverage_model",
    "coverage_calls",
    "claims_model",
    "claims_calls",
]


def _write_profile(tmp_path: Path, **overrides) -> Path:
    import json

    data = {
        "title": "Synthetic Engineer",
        "seniority": "senior",
        "domain": "fictional",
        "must_have": ["fictional"],
    }
    data.update(overrides)
    path = tmp_path / "profile_min.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _match_argv(profile: Path, db: Path, chroma: Path) -> list[str]:
    return ["match", "--profile", str(profile), "--db", str(db), "--chroma", str(chroma)]


def test_match_happy_path_json_shape(tmp_path, monkeypatch, capsys):
    import json

    from loomcv.cli import main
    from loomcv.state import load_retrieval_config

    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()

    rc = main(_match_argv(_write_profile(tmp_path), db, chroma))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == _MATCH_KEYS
    assert len(_MATCH_KEYS) == 8
    for retired in ("routing", "selected", "candidates", "judge"):
        assert retired not in payload
    assert payload["config"] == load_retrieval_config().model_dump()
    assert set(payload["config"]) == {"top_k", "rrf_k", "total_slots", "recency_shares"}
    assert list(payload["diagnostics"]) == _DIAGNOSTICS_KEYS
    assert payload["diagnostics"]["coverage_calls"] == 1
    # the default config's placeholder model id; `--config` runs below pin "fake-coverage"
    assert payload["diagnostics"]["coverage_model"]
    assert payload["diagnostics"]["coverage_model"] == payload["profile"]["coverage_model"]
    (line,) = payload["requirements"]
    assert list(line) == _LINE_KEYS
    assert line["text"] == "fictional"
    assert line["kind"] == "must"
    assert line["grade"] == "none"
    assert payload["gaps"] == ["fictional"]
    assert payload["profile"]["coverage"]["fictional"] == {
        "match": "none",
        "keywords": [],
        "parts": [],
    }

    # The one must line is graded `none`, so nothing is placed, unplaced,
    # covered or uncovered; every job of the index is present with its cap and `placed == 0`.
    selection = payload["selection"]
    assert list(selection) == _SELECTION_KEYS
    assert selection["total_slots"] == payload["config"]["total_slots"]
    assert selection["walk"] == []
    assert selection["unplaced"] == []
    assert selection["covered_by"] == {}
    assert selection["uncovered"] == []
    assert selection["passes"] == 0
    parsed = parse_star_vault(_vault_text()).jobs
    assert [(j["job_id"], j["job"], j["job_order"]) for j in selection["jobs"]] == [
        (j.job_id, j.name, j.order) for j in parsed
    ]
    assert len(selection["jobs"]) == 3
    for job in selection["jobs"]:
        assert list(job) == _JOB_KEYS
        assert job["placed"] == 0
        assert job["bullets"] == []
    assert sum(j["cap"] for j in selection["jobs"]) == selection["total_slots"]


class _CachingCoverageAdapter(FakeCoverageAdapter):
    """Grades every line `exact` on the index's `Caching` keyword (carried by `aur001` only)."""

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        import json

        self.calls.append((system, user, temperature))
        return json.dumps(
            {line: {"match": "exact", "keywords": ["Caching"]} for line in _requested_lines(user)}
        )


def test_match_selection_bullet_keys(tmp_path, monkeypatch, capsys):
    # Second half: a non-empty selection, so every `bullets[]` / `walk[]` entry has
    # exactly the documented keys (`pass`, not `pass_number`).
    import json

    from loomcv.cli import main

    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()
    monkeypatch.setattr(
        "loomcv.cli.build_generation_adapter",
        lambda node, provider: _CachingCoverageAdapter(model_id=node.model),
    )

    assert main(_match_argv(_write_profile(tmp_path), db, chroma)) == 0
    selection = json.loads(capsys.readouterr().out)["selection"]

    placed = [b for job in selection["jobs"] for b in job["bullets"]]
    assert [b["bullet_id"] for b in placed] == [w["bullet_id"] for w in selection["walk"]]
    assert len(placed) == 1
    assert placed[0]["bullet_id"].endswith("::aur001")
    for bullet in placed:
        assert list(bullet) == _SELECTED_BULLET_KEYS
        assert bullet["line"] == "fictional"
        assert bullet["kind"] == "must"
        assert bullet["pass"] == 1
        assert bullet["covers"] == ["fictional"]
    for step in selection["walk"]:
        assert list(step) == _WALK_KEYS
    assert selection["covered_by"] == {"fictional": [placed[0]["bullet_id"]]}
    assert selection["uncovered"] == []
    assert selection["passes"] == 1


def test_match_retrieval_config_flag(tmp_path, monkeypatch, capsys):
    # Test 14: `--retrieval-config` reaches the merge; save then replay stays byte-equal.
    import json

    import yaml

    from loomcv.cli import main

    db, chroma, _built = _synced_index(tmp_path, monkeypatch, capsys)
    retrieval = tmp_path / "retrieval_even.yaml"
    retrieval.write_text(
        yaml.safe_dump({"top_k": 12, "rrf_k": 60, "total_slots": 42, "recency_shares": [1]}),
        encoding="utf-8",
    )
    saved = tmp_path / "profile_saved.json"
    flag = ["--retrieval-config", str(retrieval)]

    assert (
        main(
            _match_argv(_write_profile(tmp_path), db, chroma)
            + flag
            + ["--save-profile", str(saved)]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)

    assert first["config"] == {
        "top_k": 12,
        "rrf_k": 60,
        "total_slots": 42,
        "recency_shares": [1],
    }
    assert first["selection"]["total_slots"] == 42
    assert [j["cap"] for j in first["selection"]["jobs"]] == [14, 14, 14]

    assert main(_match_argv(saved, db, chroma) + flag) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["diagnostics"].pop("coverage_calls") == 1
    assert second["diagnostics"].pop("coverage_calls") == 0
    assert second == first


def test_match_ranked_keys_and_null_project(tmp_path, monkeypatch, capsys):
    """`project_context` is gone; `project` is the bullet's label or null."""
    import json

    from loomcv.cli import main

    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()

    assert main(_match_argv(_write_profile(tmp_path), db, chroma)) == 0
    payload = json.loads(capsys.readouterr().out)

    ranked = [b for line in payload["requirements"] for b in line["ranked"]]
    assert ranked
    for bullet in ranked:
        assert list(bullet) == _RANKED_KEYS
    lab = next(b for b in ranked if b["bullet_id"].endswith("::lab001"))
    assert lab["project"] is None
    assert lab["job"] == "Placeholder Lab — Research Intern"
    aur = next(b for b in ranked if b["bullet_id"].endswith("::aur001"))
    assert aur["project"] == "Aurora Search Revamp"


def test_match_profile_with_min_year_null_loads_and_drops_key(tmp_path, monkeypatch, capsys):
    # CL-1: a stored profile that still carries "min_year": null loads unchanged.
    import json

    from loomcv.cli import main

    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()

    profile = _write_profile(tmp_path, min_year=None)
    assert '"min_year": null' in profile.read_text(encoding="utf-8")

    assert main(_match_argv(profile, db, chroma)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "min_year" not in payload["profile"]


def test_match_requirements_one_per_line(tmp_path, monkeypatch, capsys):
    import json

    from loomcv.cli import main

    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()
    profile = _write_profile(
        tmp_path,
        must_have=["fictional", "quantum zebra"],
        responsibilities=["ship it"],
        nice_to_have=["pretend"],
    )

    rc = main(_match_argv(profile, db, chroma))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [(ln["text"], ln["kind"]) for ln in payload["requirements"]] == [
        ("fictional", "must"),
        ("quantum zebra", "must"),
        ("ship it", "responsibility"),
        ("pretend", "nice"),
    ]
    assert "vector_missing" in payload["diagnostics"]
    # Every line is graded (the fake answers `none`); only the must lines are gaps.
    assert [ln["grade"] for ln in payload["requirements"]] == ["none"] * 4
    assert payload["gaps"] == ["fictional", "quantum zebra"]
    assert payload["diagnostics"]["rerank_pairs"] == sum(
        len(ln["ranked"]) for ln in payload["requirements"]
    )


def test_match_bad_profile_exit_1(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    rc = main(_match_argv(tmp_path / "missing.json", db, chroma))
    assert rc == 1
    assert capsys.readouterr().err.strip()

    rc = main(_match_argv(_write_profile(tmp_path, seniority="principal"), db, chroma))
    assert rc == 1
    assert capsys.readouterr().err.strip()


def test_match_before_sync_exit_3(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    rc = main(_match_argv(_write_profile(tmp_path), db, chroma))

    assert rc == 3
    assert "sync" in capsys.readouterr().err


def test_help_lists_sync_and_match(capsys):
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "sync" in out
    assert "match" in out
    assert "run" in out
    # C-19: a subcommand line, not the word inside another command's help text
    assert any(line.split()[:1] == ["review"] for line in out.splitlines())


# --------------------------------------------------- pipeline match --jd
#
# Vault = `metadata_vault.md`. The generation adapter is a scripted fake
# injected by monkeypatching `loomcv.cli.build_generation_adapter`; `--config` is a tmp yaml
# carrying both the `embedding:` section (adapter patched) and `providers`/`nodes`.
# `--jd` makes two generation calls (`parse_jd`, then `coverage`), and the index is
# needed first for the coverage keyword list.

_JD_REPLY = {
    "title": "Fake Player Engineer",
    "seniority": "senior",
    "domain": "fictional streaming",
    "must_have": ["fake buffer tuning for playback start time"],
    "nice_to_have": [],
    "ats_keywords": [],
}


class FakeGenerationAdapter:
    """Scripted, network-free stand-in for `GenerationAdapter`."""

    def __init__(self, replies: list[str], *, temperature: float = 0) -> None:
        self._replies = list(replies)
        self._temperature = temperature
        self.calls: list[tuple[str, str, float]] = []

    @property
    def model_id(self) -> str:
        return "fake-model"

    @property
    def temperature(self) -> float:
        return self._temperature

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        self.calls.append((system, user, temperature))
        return self._replies.pop(0)


def _write_gen_config(tmp_path: Path, *, nodes: dict | None = None) -> Path:
    import yaml

    data = {
        "embedding": {"adapter": "fake", "model": "fake-model", "vector_space_version": 1},
        "reranker": {"adapter": "fake", "model": "fake-reranker"},
        "providers": {
            "local_eval": {"adapter": "openai_compatible", "base_url_env": "FAKE_BASE_URL"}
        },
        "nodes": nodes
        if nodes is not None
        else {
            "parse_jd": {"provider": "local_eval", "model": "fake-model", "temperature": 0},
            "coverage": {"provider": "local_eval", "model": "fake-coverage", "temperature": 0},
        },
    }
    path = tmp_path / "models.tmp.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _patch_gen_adapter(monkeypatch, replies: list[str]) -> list:
    """Patch the generation builder; returns the `(node name, adapter)` builds, in call order."""
    built: list = []

    def build(node, provider):
        if node.name == "coverage":
            adapter = FakeCoverageAdapter(model_id=node.model)
        else:
            adapter = FakeGenerationAdapter(replies)
        built.append((node.name, adapter))
        return adapter

    monkeypatch.setattr("loomcv.cli.build_generation_adapter", build)
    return built


def _sync_metadata_vault(tmp_path: Path) -> tuple[Path, Path]:
    vault = _copy_vault(tmp_path, text=(FIXTURES / "metadata_vault.md").read_text())
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    assert _run(vault, db, chroma) == 0
    return db, chroma


def _write_jd(tmp_path: Path) -> Path:
    path = tmp_path / "jd.md"
    path.write_text("Fictional Corp seeks a fake player engineer.\n", encoding="utf-8")
    return path


def _jd_argv(jd: Path, db: Path, chroma: Path, config: Path) -> list[str]:
    return [
        "match",
        "--jd",
        str(jd),
        "--db",
        str(db),
        "--chroma",
        str(chroma),
        "--config",
        str(config),
    ]


def test_match_jd_happy_path_and_save_profile_roundtrip(tmp_path, monkeypatch, capsys):
    import json

    from loomcv.cli import main
    from loomcv.state import JDProfile

    _patch_adapter(monkeypatch)
    db, chroma = _sync_metadata_vault(tmp_path)
    capsys.readouterr()
    built = _patch_gen_adapter(monkeypatch, [json.dumps(_JD_REPLY)])
    config = _write_gen_config(tmp_path)
    saved = tmp_path / "out.json"

    rc = main(_jd_argv(_write_jd(tmp_path), db, chroma, config) + ["--save-profile", str(saved)])

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert list(payload) == _MATCH_KEYS
    assert payload["parse_warnings"] == []
    # TR16: two generation nodes, in order — parse_jd, then coverage.
    assert [name for name, _adapter in built] == ["parse_jd", "coverage"]
    assert payload["diagnostics"]["coverage_calls"] == 1
    assert saved.exists()
    saved_profile = JDProfile.model_validate_json(saved.read_text(encoding="utf-8"))
    assert saved_profile == JDProfile(**payload["profile"])
    # the saved profile carries its coverage: every line graded, the model recorded
    assert list(saved_profile.coverage) == ["fake buffer tuning for playback start time"]
    assert saved_profile.coverage_model == "fake-coverage"

    # replay from the saved profile: a config with no coverage node still works (zero calls)
    replay_config = _write_gen_config(tmp_path, nodes=_PARSE_JD_ONLY)
    rc = main(
        ["match", "--profile", str(saved), "--db", str(db), "--chroma", str(chroma)]
        + ["--config", str(replay_config)]
    )

    assert rc == 0
    second = json.loads(capsys.readouterr().out)
    assert len(built) == 2
    assert second["parse_warnings"] == []
    assert second["diagnostics"]["coverage_calls"] == 0
    assert second["requirements"] == payload["requirements"]
    assert second["gaps"] == payload["gaps"]
    assert second["selection"] == payload["selection"]
    assert second["profile"] == payload["profile"]


def test_match_jd_profile_mutually_exclusive(tmp_path, capsys):
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["match", "--jd", "a.md", "--profile", "b.json"])
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        main(["match"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--jd" in err
    assert "--profile" in err


def test_match_jd_generation_failure_exit_4(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    _patch_adapter(monkeypatch)
    db, chroma = _sync_metadata_vault(tmp_path)
    capsys.readouterr()
    jd = _write_jd(tmp_path)

    _patch_gen_adapter(monkeypatch, ["nope", "{}"])
    rc = main(_jd_argv(jd, db, chroma, _write_gen_config(tmp_path)))
    assert rc == 4
    captured = capsys.readouterr()
    assert "parse_jd" in captured.err
    assert captured.out == ""

    no_node = _write_gen_config(
        tmp_path,
        nodes={"verify": {"provider": "local_eval", "model": "fake-model", "temperature": 0}},
    )
    rc = main(_jd_argv(jd, db, chroma, no_node))
    assert rc == 4
    assert "parse_jd" in capsys.readouterr().err


def test_match_jd_before_sync_exit_3(tmp_path, monkeypatch, capsys):
    # TR16: `--jd` needs the index for the coverage keyword list before it spends a parse call.
    from loomcv.cli import main

    _patch_adapter(monkeypatch)
    built = _patch_gen_adapter(monkeypatch, [])
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"

    rc = main(_jd_argv(_write_jd(tmp_path), db, chroma, _write_gen_config(tmp_path)))

    assert rc == 3
    assert "sync" in capsys.readouterr().err
    assert all(not adapter.calls for _name, adapter in built)


def test_match_jd_missing_file_exit_1(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    _patch_adapter(monkeypatch)
    missing = tmp_path / "no_such_jd.md"
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"

    rc = main(_jd_argv(missing, db, chroma, _write_gen_config(tmp_path)))

    assert rc == 1
    assert str(missing) in capsys.readouterr().err


# ------------------------------------------------ v2 profile fields


def test_match_profile_v2_fields_echoed(tmp_path, monkeypatch, capsys):
    # The "selected / gaps identical without the v2 fields" half is deleted —
    # responsibilities are scored now.
    import json

    from loomcv.cli import main

    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()

    v2_fields = {
        "responsibilities": ["Ship the fictional player"],
        "constraints": ["Authorized to work in Canada"],
        "traits": ["Communicate clearly"],
        "skills": {"fictional": ["fictional player"]},
    }
    profile = _write_profile(tmp_path, ats_keywords=["Kafka"], **v2_fields)
    assert main(_match_argv(profile, db, chroma)) == 0
    with_fields = json.loads(capsys.readouterr().out)

    for key, value in {**v2_fields, "ats_keywords": ["Kafka"]}.items():
        assert with_fields["profile"][key] == value

    # TR17: a stored profile still carrying the retired keys loads; the payload echoes none of them.
    old = _write_profile(
        tmp_path, domains=["streaming-media"], tech=["Shaka Player"], min_year=2020
    )
    assert main(_match_argv(old, db, chroma)) == 0
    echoed = json.loads(capsys.readouterr().out)["profile"]
    for retired in ("domains", "tech", "min_year"):
        assert retired not in echoed


# ------------------------------------------------------ the coverage path
#
# Vault = `index_vault.md`; `--config` = `_write_gen_config(...)`. The coverage mapping
# lives on the profile: one call when the profile carries none, zero when it does.

_PARSE_JD_ONLY = {"parse_jd": {"provider": "local_eval", "model": "fake-model", "temperature": 0}}


def _synced_index(tmp_path: Path, monkeypatch, capsys) -> tuple[Path, Path, list]:
    vault = _copy_vault(tmp_path)
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    built = _patch_adapter(monkeypatch)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()
    return db, chroma, built


def test_match_save_profile_then_replay_zero_calls(tmp_path, monkeypatch, capsys):
    # TR15: the first run calls the coverage node once, the replay makes none; the two
    # payloads are equal once `diagnostics.coverage_calls` (1 then 0) is removed from both.
    import json

    from loomcv.cli import main

    db, chroma, built = _synced_index(tmp_path, monkeypatch, capsys)
    profile = _write_profile(tmp_path)
    saved = tmp_path / "profile_saved.json"
    argv = _match_argv(profile, db, chroma)

    assert (
        main(argv + ["--config", str(_write_gen_config(tmp_path)), "--save-profile", str(saved)])
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert saved.exists()
    stored = json.loads(saved.read_text(encoding="utf-8"))
    assert list(stored["coverage"]) == ["fictional"]
    assert stored["coverage_model"] == "fake-coverage"
    assert [name for name, _adapter in built] == ["coverage"]

    replay_argv = _match_argv(saved, db, chroma)
    no_node_cfg = _write_gen_config(tmp_path, nodes=_PARSE_JD_ONLY)
    assert main(replay_argv + ["--config", str(no_node_cfg)]) == 0
    second = json.loads(capsys.readouterr().out)

    assert len(built) == 1  # no generation adapter was built for the replay
    assert first["diagnostics"].pop("coverage_calls") == 1
    assert second["diagnostics"].pop("coverage_calls") == 0
    assert second == first


def test_match_without_coverage_node_exit_4(tmp_path, monkeypatch, capsys):
    # TR16
    from loomcv.cli import main

    db, chroma, _built = _synced_index(tmp_path, monkeypatch, capsys)
    called: list = []
    monkeypatch.setattr("loomcv.cli.match", lambda *a, **k: called.append(1))
    config = _write_gen_config(tmp_path, nodes=_PARSE_JD_ONLY)

    rc = main(_match_argv(_write_profile(tmp_path), db, chroma) + ["--config", str(config)])

    assert rc == 4
    captured = capsys.readouterr()
    assert captured.err.startswith("coverage:")
    assert captured.out == ""
    assert called == []


def test_match_coverage_for_other_profile_exit_1(tmp_path, monkeypatch, capsys):
    # TR16: a stored coverage that belongs to another profile's lines.
    import json

    from loomcv.cli import main

    db, chroma, _built = _synced_index(tmp_path, monkeypatch, capsys)
    config = _write_gen_config(tmp_path)
    saved = tmp_path / "profile_saved.json"
    argv = _match_argv(_write_profile(tmp_path), db, chroma) + ["--config", str(config)]
    assert main(argv + ["--save-profile", str(saved)]) == 0
    capsys.readouterr()

    stored = json.loads(saved.read_text(encoding="utf-8"))
    stored["must_have"] = ["something else entirely"]
    other = tmp_path / "other_profile.json"
    other.write_text(json.dumps(stored), encoding="utf-8")
    rc = main(_match_argv(other, db, chroma) + ["--config", str(config)])

    assert rc == 1
    captured = capsys.readouterr()
    assert "something else entirely" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "extra",
    [["--judge", "x.json"], ["--save-judge", "x.json"], ["--no-judge"]],
)
def test_match_judge_flags_rejected(extra, capsys):
    # TR16: the judge is gone whole; argparse rejects its flags.
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["match", "--profile", "p.json", *extra])
    assert exc.value.code == 2


# ------------------------------------------------ per-call usage line on stderr
#
# CL-1. The other fakes in this file carry a tuple-valued `calls`;
# only `CallRecord` items are printed.


class _FakeCoverageWithRecords(FakeCoverageAdapter):
    """A coverage node whose `calls` holds one usage record, as the real adapter's does."""

    def __init__(self, *, model_id: str = "fake-coverage") -> None:
        from loomcv.llm import CallRecord

        super().__init__(model_id=model_id)
        self.calls = [CallRecord(1.5, 11, 5, 4)]


def test_match_prints_coverage_call_record_on_stderr(tmp_path, monkeypatch, capsys):
    # CL-1
    import re

    from loomcv.cli import main

    db, chroma, _built = _synced_index(tmp_path, monkeypatch, capsys)
    argv = _match_argv(_write_profile(tmp_path), db, chroma) + [
        "--config",
        str(_write_gen_config(tmp_path)),
    ]

    assert main(argv) == 0
    plain = capsys.readouterr()
    assert "call 1:" not in plain.err

    monkeypatch.setattr(
        "loomcv.cli.build_generation_adapter",
        lambda node, provider: _FakeCoverageWithRecords(model_id=node.model),
    )
    assert main(argv) == 0
    recorded = capsys.readouterr()

    lines = [ln for ln in recorded.err.splitlines() if "call 1:" in ln]
    assert len(lines) == 1
    assert re.fullmatch(
        r"coverage call 1: .+ s · prompt .+ · completion .+ · reasoning .+", lines[0]
    )
    assert recorded.out == plain.out


# --------------------------------------------------- pipeline ingest
#
# C-1 … C-6. The short résumé and the canned replies duplicate `tests/test_ingest.py`'s
# (`tests/` is not a package); `tests/fixtures/ingest_expected_vault.md` is the one shared
# expected output, so the two copies cannot drift apart unnoticed. The `ingest` node gets the
# scripted `FakeGenerationAdapter` through `_patch_gen_adapter`: the structure reply (the
# keyword call and its flags moved to `pipeline complete`). All content is
# fictional.

_INGEST_PROMPTS = resources.path("prompts")

_INGEST_RESUME = """# Sam Example

Example City · sam@example.invalid

## Experience

### Acme Widgets — Platform Engineer
March 2021 – present

Technologies: Python, Django, PostgreSQL

- Cut the p95 latency of the order API from 40 s to 6 s by moving report generation to a task queue.
- Reduced checkout errors on the public storefront by 40%.

**Billing Revamp**
- Rebuilt the invoice pipeline on PostgreSQL with idempotent writes and daily reconciliation.

### Freelance Developer

- Built booking websites for small clinics with online forms and email reminders.

## Education

B.Sc. Computer Science, Example University
"""

_INGEST_STRUCTURE = {
    "jobs": [
        {
            "employer": "Acme Widgets",
            "title": "Platform Engineer",
            "date_from": "2021-03",
            "date_to": "present",
            "defaults": ["Python", "Django", "PostgreSQL"],
            "bullets": [
                {
                    "text": "Cut the p95 latency of the order API from 40 s to 6 s by moving "
                    "report generation to a task queue.",
                    "project": None,
                },
                {
                    "text": "Reduced checkout errors on the public storefront by 40%.",
                    "project": None,
                },
                {
                    "text": "Rebuilt the invoice pipeline on PostgreSQL with idempotent writes "
                    "and daily reconciliation.",
                    "project": "Billing Revamp",
                },
            ],
        },
        {
            "employer": None,
            "title": "Freelance Developer",
            "date_from": None,
            "date_to": None,
            "defaults": [],
            "bullets": [
                {
                    "text": "Built booking websites for small clinics with online forms and "
                    "email reminders.",
                    "project": None,
                }
            ],
        },
    ]
}

_INGEST_NODE = {"ingest": {"provider": "local_eval", "model": "fake-ingest", "temperature": 0}}


def _ingest_argv(tmp_path: Path, *extra: str, config: Path | None = None) -> list[str]:
    resume = tmp_path / "resume.md"
    resume.write_text(_INGEST_RESUME, encoding="utf-8")
    cfg = config if config is not None else _write_gen_config(tmp_path, nodes=_INGEST_NODE)
    return [
        "ingest",
        str(resume),
        "--out",
        str(tmp_path / "out" / "draft_vault.md"),
        "--config",
        str(cfg),
        "--prompt-structure",
        str(_INGEST_PROMPTS / "ingest_structure.md"),
        *extra,
    ]


def test_ingest_happy_path_writes_the_expected_vault(tmp_path, monkeypatch, capsys):
    # C-1
    import json
    from datetime import date

    from loomcv.cli import main

    built = _patch_gen_adapter(monkeypatch, [json.dumps(_INGEST_STRUCTURE)])
    out = tmp_path / "out" / "draft_vault.md"

    assert main(_ingest_argv(tmp_path)) == 0
    captured = capsys.readouterr()

    # The CLI passes no `today`: the fixture's date is today's.
    expected = (FIXTURES / "ingest_expected_vault.md").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == expected.replace(
        "2026-10-10", date.today().isoformat()
    )
    assert captured.out.splitlines() == [
        "jobs=2",
        "bullets=4",
        "jobs_without_bullets=0",
        "dropped=0",
        "jobs_missing_dates=1",
        "parser_warnings=0",
        f"written={out}",
    ]
    assert [name for name, _ in built] == ["ingest"]
    assert len(built[0][1].calls) == 1

    # A job returned without bullets is kept and counted.
    structure = json.loads(json.dumps(_INGEST_STRUCTURE))
    structure["jobs"][1]["bullets"] = []
    _patch_gen_adapter(monkeypatch, [json.dumps(structure)])
    assert main(_ingest_argv(tmp_path)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "bullets=3" in lines
    assert "jobs_without_bullets=1" in lines
    assert "parser_warnings=1" in lines  # the empty-job warning; the counter's meaning is unchanged


def test_ingest_missing_resume_exit_1(tmp_path, monkeypatch, capsys):
    # C-2
    from loomcv.cli import main

    built = _patch_gen_adapter(monkeypatch, [])
    argv = _ingest_argv(tmp_path)
    argv[1] = str(tmp_path / "no_such_resume.md")

    assert main(argv) == 1
    assert capsys.readouterr().err.startswith("invalid resume")
    assert built == []


def test_ingest_without_ingest_node_exit_4(tmp_path, monkeypatch, capsys):
    # C-3
    from loomcv.cli import main

    built = _patch_gen_adapter(monkeypatch, [])
    config = _write_gen_config(tmp_path)  # parse_jd + coverage only

    assert main(_ingest_argv(tmp_path, config=config)) == 4
    err = capsys.readouterr().err
    assert err.startswith("ingest:")
    assert "'ingest'" in err
    assert built == []
    assert not (tmp_path / "out" / "draft_vault.md").exists()


def test_ingest_no_jobs_extracted_exit_1(tmp_path, monkeypatch, capsys):
    # C-4
    from loomcv.cli import main

    _patch_gen_adapter(monkeypatch, ['{"jobs": []}'])

    assert main(_ingest_argv(tmp_path)) == 1
    assert "ingest: no jobs extracted" in capsys.readouterr().err
    assert not (tmp_path / "out" / "draft_vault.md").exists()


# (C-5 `test_ingest_keyword_runs_flag` is deleted with the flag.)


def test_help_lists_ingest_and_its_options(capsys):
    # C-6 (the keyword flags left `ingest`)
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "ingest" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--out", "--config", "--prompt-structure"):
        assert flag in out


def test_ingest_help_has_no_keyword_options(capsys):
    # C-8
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--keyword-runs" not in out
    assert "--prompt-keywords" not in out

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    top = capsys.readouterr().out
    assert "complete" in top
    assert "ui" in top


# ------------------------------------------------ pipeline complete
#
# C-1 … C-5. The headless path is one `--checklist` and one `--accept-all`; the loop's
# logic is tested through the module in `tests/test_complete.py`. The Sam Example draft is the
# shared expected L1 output (`ingest_expected_vault.md`); the claims draft is the second persona
# (`claims_expected_vault.md`). The canned selection replies duplicate `test_complete.py`'s.

_SEL_SAM = {
    "J1b1": {"defaults": ["Python", "django"], "own": ["task queue", "API performance"]},
    "J1b2": {"defaults": [], "own": ["error reduction"]},
    "J1b3": {"defaults": ["PostgreSQL", "Redis"], "own": ["invoicing", "a; b"]},
    "J2b1": {"defaults": [], "own": ["web development"]},
}


def _sel_claims() -> dict:
    """A canned selection reply for the claims draft: jobs 1 and 2 have nine bullets each, the
    third job none; no job carries `Defaults`, so every `defaults` list is empty."""
    return {
        f"J{j}b{k}": {"defaults": [], "own": [f"claims topic {j}.{k}", f"method {j}.{k}"]}
        for j in (1, 2)
        for k in range(1, 10)
    }


def _complete_draft(tmp_path: Path, fixture: str = "ingest_expected_vault.md") -> Path:
    draft = tmp_path / "draft.md"
    shutil.copy(FIXTURES / fixture, draft)
    return draft


def _complete_argv(tmp_path: Path, draft: Path, *modes: str) -> list[str]:
    cfg = _write_gen_config(tmp_path, nodes=_INGEST_NODE)
    return ["complete", str(draft), *modes, "--config", str(cfg)]


def test_complete_checklist_json_lines(tmp_path, monkeypatch, capsys):
    # C-1
    import json

    from loomcv.cli import main

    monkeypatch.chdir(tmp_path)
    built = _patch_gen_adapter(monkeypatch, [])
    draft = _complete_draft(tmp_path)
    before = draft.read_bytes()

    assert main(_complete_argv(tmp_path, draft, "--checklist")) == 0
    lines = capsys.readouterr().out.splitlines()
    items = [json.loads(line) for line in lines]

    acme, free = "acme-widgets-platform-engineer", "freelance-developer"
    # The header items first — the draft has no header block.
    assert [(i["kind"], i["job_id"], i["bullet_id"]) for i in items] == [
        ("header_field", None, None),
        ("header_field", None, None),
        ("header_optional", None, None),
        ("header_optional", None, None),
        ("header_optional", None, None),
        ("job_defaults", acme, None),
        ("bullet_draft", acme, f"{acme}::395f62"),
        ("bullet_draft", acme, f"{acme}::e8eb3d"),
        ("bullet_draft", acme, f"{acme}::b37b54"),
        ("job_field", free, None),
        ("job_field", free, None),
        ("job_field", free, None),
        ("job_defaults", free, None),
        ("bullet_draft", free, f"{free}::174378"),
    ]
    assert all(set(i) == {"kind", "job_id", "bullet_id", "message"} for i in items)
    assert all(i["message"] for i in items)
    assert built == []
    assert draft.read_bytes() == before
    assert not (tmp_path / "02_database").exists()


def test_complete_accept_all_writes_default_vault(tmp_path, monkeypatch, capsys):
    # C-2
    import json

    from loomcv.cli import main
    from loomcv.sync.parse_assets import parse_star_vault

    monkeypatch.chdir(tmp_path)
    built = _patch_gen_adapter(monkeypatch, [json.dumps(_SEL_SAM)])
    draft = _complete_draft(tmp_path)
    before = draft.read_bytes()

    assert main(_complete_argv(tmp_path, draft, "--accept-all")) == 0
    captured = capsys.readouterr()
    out = tmp_path / "02_database" / "vault.md"

    assert captured.out.splitlines() == [
        "jobs=2",
        "jobs_without_bullets=0",
        "bullets=4",
        "confirmed=4",
        "skills_unassigned=0",
        "written=02_database/vault.md",
    ]
    assert [name for name, _ in built] == ["ingest"]
    assert len(built[0][1].calls) == 1
    assert draft.read_bytes() == before
    parsed = parse_star_vault(out.read_text(encoding="utf-8"))
    assert parsed.errors() == []
    assert sum(len(j.bullets) for j in parsed.jobs) == 4
    assert all(b.status == "confirmed" for j in parsed.jobs for b in j.bullets)
    first = parsed.jobs[0].bullets[0]
    assert first.keywords == ["Python", "Django", "task queue", "API performance"]

    # The same on the second persona: one bullet-less job, a Skills line left unassigned.
    claims = _complete_draft(tmp_path, "claims_expected_vault.md")
    out_claims = tmp_path / "claims_vault.md"
    built = _patch_gen_adapter(monkeypatch, [json.dumps(_sel_claims())])
    argv = [*_complete_argv(tmp_path, claims, "--accept-all"), "--out", str(out_claims)]
    assert main(argv) == 0
    assert capsys.readouterr().out.splitlines() == [
        "jobs=3",
        "jobs_without_bullets=1",
        "bullets=18",
        "confirmed=18",
        "skills_unassigned=14",
        f"written={out_claims}",
    ]
    assert len(built[0][1].calls) == 1
    reread = parse_star_vault(out_claims.read_text(encoding="utf-8"))
    assert len(reread.skills) == 14
    assert reread.errors() == []


def test_complete_parse_errors_exit_1(tmp_path, monkeypatch, capsys):
    # C-3
    import json

    from loomcv.cli import main
    from loomcv.complete import workbook

    monkeypatch.chdir(tmp_path)

    # (a) an input draft with parser errors (a bullet without an id): a partial parse would
    # drop bullets silently, so both modes exit 1 naming the line and write nothing.
    built = _patch_gen_adapter(monkeypatch, [])
    text = (FIXTURES / "ingest_expected_vault.md").read_text(encoding="utf-8")
    text = text.replace(" {id: e8eb3d, status: draft, updated: 2026-10-10}", "")
    broken = tmp_path / "broken.md"
    broken.write_text(text, encoding="utf-8")
    line_no = next(n for n, ln in enumerate(text.splitlines(), 1) if "Reduced checkout" in ln)
    for mode in ("--checklist", "--accept-all"):
        assert main(_complete_argv(tmp_path, broken, mode)) == 1
        assert f"line {line_no}" in capsys.readouterr().err
    assert built == []
    assert not (tmp_path / "02_database").exists()

    # (b) the rendered output fails the strict parse (forced, as in ingest's I-12) →
    # exit 1, the file is written, stderr names its line.
    _patch_gen_adapter(monkeypatch, [json.dumps(_SEL_SAM)])
    original = workbook.render_workbook
    monkeypatch.setattr(
        "loomcv.complete.workbook.render_workbook",
        lambda wb, *, today: original(wb, today=today).replace(
            "* **From**: 2021-03", "* **From**: 2020-00", 1
        ),
    )
    draft = _complete_draft(tmp_path)

    assert main(_complete_argv(tmp_path, draft, "--accept-all")) == 1
    err = capsys.readouterr().err
    written = (tmp_path / "02_database" / "vault.md").read_text(encoding="utf-8")
    line_no = written.splitlines().index("* **From**: 2020-00") + 1
    assert f"line {line_no}" in err


def test_complete_without_mode_exit_2(tmp_path, capsys):
    # C-4
    from loomcv.cli import main

    draft = _complete_draft(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["complete", str(draft)])
    assert exc.value.code == 2
    assert "one of the arguments --checklist --accept-all is required" in capsys.readouterr().err


def test_complete_accept_all_without_ingest_node_exit_4(tmp_path, monkeypatch, capsys):
    # C-5
    from loomcv.cli import main

    monkeypatch.chdir(tmp_path)
    built = _patch_gen_adapter(monkeypatch, [])
    draft = _complete_draft(tmp_path)
    config = _write_gen_config(tmp_path)  # parse_jd + coverage only

    argv = ["complete", str(draft), "--accept-all", "--config", str(config)]
    assert main(argv) == 4
    err = capsys.readouterr().err
    assert err.startswith("complete:")
    assert "'ingest'" in err
    assert built == []
    assert not (tmp_path / "02_database").exists()


# ------------------------------------------------ the claims call through the CLI
#
# Vault = `metadata_vault.md` + a vault-level `Skills` line (inline). The claims set is
# [TypeScript @ helios + skills, Kubernetes @ skills] (see tests/test_search.py).

_CLAIMS_SKILLS = "* **Skills**: Kubernetes; TypeScript; Alerting\n"


def _claims_vault_text() -> str:
    head, rest = (FIXTURES / "metadata_vault.md").read_text(encoding="utf-8").split("\n", 1)
    return head + "\n" + _CLAIMS_SKILLS + rest


class _ClaimsCoverageAdapter(FakeCoverageAdapter):
    """Grades every line `none`, except a call over the claims (`Kubernetes` listed): `exact`."""

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        import json

        self.calls.append((system, user, temperature))
        claims_call = "- Kubernetes\n" in user + "\n"
        reply = (
            {"match": "exact", "keywords": ["Kubernetes"]}
            if claims_call
            else {"match": "none", "keywords": []}
        )
        return json.dumps({line: reply for line in _requested_lines(user)})


def _patch_claims_adapter(monkeypatch) -> list:
    built: list = []

    def build(node, provider):
        adapter = _ClaimsCoverageAdapter(model_id=node.model)
        built.append((node.name, adapter))
        return adapter

    monkeypatch.setattr("loomcv.cli.build_generation_adapter", build)
    return built


def _claims_index(tmp_path: Path, monkeypatch, capsys) -> tuple[Path, Path, list]:
    vault = _copy_vault(tmp_path, text=_claims_vault_text())
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(vault, db, chroma) == 0
    capsys.readouterr()
    return db, chroma, _patch_claims_adapter(monkeypatch)


def test_c1_sync_prints_skills_after_jobs_deleted(tmp_path, monkeypatch, capsys):
    _patch_adapter(monkeypatch)
    vault = _copy_vault(tmp_path, text=_claims_vault_text())

    assert _run(vault, tmp_path / "db.sqlite", tmp_path / "chroma") == 0

    keys = [line.split("=", 1)[0] for line in capsys.readouterr().out.splitlines()]
    assert keys[keys.index("jobs_deleted") + 1] == "skills"
    assert keys == [
        "added",
        "updated",
        "metadata_only",
        "deleted",
        "unchanged",
        "jobs_written",
        "jobs_deleted",
        "skills",
        "embedded",
        "db",
        "chroma",
    ]
    plain = _copy_vault(tmp_path, name="plain.md")
    assert _run(plain, tmp_path / "db2.sqlite", tmp_path / "chroma2") == 0
    assert "skills=0" in capsys.readouterr().out


def test_c2_match_payload_claims_key(tmp_path, monkeypatch, capsys):
    import json

    from loomcv.cli import main

    db, chroma, _built = _claims_index(tmp_path, monkeypatch, capsys)
    config = _write_gen_config(tmp_path)

    rc = main(_match_argv(_write_profile(tmp_path), db, chroma) + ["--config", str(config)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == _MATCH_KEYS
    assert list(payload).index("claims") == list(payload).index("gaps") + 1
    assert payload["gaps"] == ["fictional"]  # the grade did not move
    (item,) = payload["claims"]
    assert list(item) == ["line", "kind", "match", "keywords"]
    assert (item["line"], item["kind"], item["match"]) == ("fictional", "must", "exact")
    (keyword,) = item["keywords"]
    assert list(keyword) == ["keyword", "at"]
    assert keyword == {"keyword": "Kubernetes", "at": ["skills"]}
    assert list(payload["diagnostics"]) == _DIAGNOSTICS_KEYS
    assert payload["diagnostics"]["coverage_calls"] == 1
    assert payload["diagnostics"]["claims_calls"] == 1
    assert payload["diagnostics"]["claims_model"] == "fake-coverage"


def test_c3_save_profile_replays_claims_byte_identical(tmp_path, monkeypatch, capsys):
    import json

    from loomcv.cli import main

    db, chroma, built = _claims_index(tmp_path, monkeypatch, capsys)
    saved = tmp_path / "profile_saved.json"
    argv = _match_argv(_write_profile(tmp_path), db, chroma)

    assert (
        main(argv + ["--config", str(_write_gen_config(tmp_path)), "--save-profile", str(saved)])
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    stored = json.loads(saved.read_text(encoding="utf-8"))
    assert list(stored["claims"]) == ["fictional"]
    assert stored["claims_model"] == "fake-coverage"
    assert len(built) == 1

    no_node_cfg = _write_gen_config(tmp_path, nodes=_PARSE_JD_ONLY)
    assert main(_match_argv(saved, db, chroma) + ["--config", str(no_node_cfg)]) == 0
    second = json.loads(capsys.readouterr().out)

    assert len(built) == 1  # no generation adapter was built for the replay
    assert first["diagnostics"].pop("coverage_calls") == 1
    assert first["diagnostics"].pop("claims_calls") == 1
    assert second["diagnostics"].pop("coverage_calls") == 0
    assert second["diagnostics"].pop("claims_calls") == 0
    assert second == first
    assert first["claims"] != []


def test_c4_pre_r5_profile_one_claims_call_and_rewrite(tmp_path, monkeypatch, capsys):
    import json

    from loomcv.cli import main

    db, chroma, built = _claims_index(tmp_path, monkeypatch, capsys)
    config = _write_gen_config(tmp_path)
    first_saved = tmp_path / "first.json"
    argv = _match_argv(_write_profile(tmp_path), db, chroma) + ["--config", str(config)]
    assert main(argv + ["--save-profile", str(first_saved)]) == 0
    capsys.readouterr()

    stored = json.loads(first_saved.read_text(encoding="utf-8"))
    assert stored["coverage"]  # the coverage the pre-R5 shape also carries
    stored.pop("claims")
    stored.pop("claims_model")
    pre_r5 = tmp_path / "pre_r5.json"
    pre_r5.write_text(json.dumps(stored), encoding="utf-8")
    rewritten = tmp_path / "rewritten.json"
    built.clear()

    rc = main(
        _match_argv(pre_r5, db, chroma)
        + ["--config", str(config), "--save-profile", str(rewritten)]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    ((name, adapter),) = built
    assert name == "coverage"
    assert len(adapter.calls) == 1  # the claims call only; the coverage came from the profile
    assert payload["diagnostics"]["coverage_calls"] == 0
    assert payload["diagnostics"]["claims_calls"] == 1
    again = json.loads(rewritten.read_text(encoding="utf-8"))
    assert list(again["claims"]) == ["fictional"]
    assert again["claims_model"] == "fake-coverage"


# ------------------------------------------------- pipeline draft
#
# C-1 .. C-6. Index = `index_vault.md` synced through `main(["sync", ...])`; the payload is
# built from the Pydantic models over the parse's confirmed ids and texts and written as the
# JSON `match` would print; `generation.yaml` is a tmp file whose `output_dir` lives under
# `tmp_path`. `draft` builds no adapter at all.


def _draft_payload_text(vault_text: str, *, drop: str | None = None) -> tuple[str, int]:
    """The match JSON placing every confirmed bullet of the vault, and how many that is."""
    import json

    from loomcv.retrieve.rank import (
        JobSelection,
        MatchDiagnostics,
        MatchPayload,
        SelectedBullet,
        Selection,
        WalkStep,
    )
    from loomcv.state import JDProfile, RetrievalConfig

    pr = parse_star_vault(vault_text)
    jobs, walk, position = [], [], 0
    for job in pr.jobs:
        placed = []
        for b in job.bullets:
            position += 1
            placed.append(
                SelectedBullet(
                    bullet_id=b.bullet_id,
                    project=b.project,
                    text=b.text,
                    line="Fictional line",
                    kind="must",
                    pass_number=1,
                    walk_position=position,
                    overlap=1,
                    rerank_rank=1,
                    rrf_position=1,
                    covers=[],
                )
            )
            walk.append(
                WalkStep(
                    bullet_id=b.bullet_id,
                    job_id=job.job_id,
                    line="Fictional line",
                    kind="must",
                    pass_number=1,
                )
            )
        if placed:
            jobs.append(
                JobSelection(
                    job_id=job.job_id,
                    job=job.name,
                    job_order=job.order,
                    cap=len(placed),
                    placed=len(placed),
                    bullets=placed,
                )
            )
    payload = MatchPayload(
        profile=JDProfile(
            title="Fake Engineer", seniority="senior", domain="web", must_have=["Fictional line"]
        ),
        parse_warnings=[],
        requirements=[],
        gaps=[],
        claims=[],
        selection=Selection(
            total_slots=14,
            jobs=jobs,
            walk=walk,
            unplaced=[],
            covered_by={},
            uncovered=[],
            passes=1,
        ),
        config=RetrievalConfig(top_k=5, rrf_k=60, total_slots=14, recency_shares=[3, 2, 1]),
        diagnostics=MatchDiagnostics(
            vector_missing=0,
            rerank_pairs=0,
            rerank_model="fake-reranker",
            coverage_model="fake-coverage",
            coverage_calls=0,
            claims_model=None,
            claims_calls=0,
        ),
    )
    data = payload.model_dump(mode="json")
    if drop is not None:
        data.pop(drop)
    return json.dumps(data), position


def _generation_text(apps: Path, *, drop: str | None = None, extra: str = "", defaults=None) -> str:
    """A tmp `generation.yaml` with the four keys; absolute paths to the repo's
    committed template / pandoc defaults so a case does not depend on the working directory."""
    values = {
        "output_dir": str(apps),
        "resume_template": str(resources.path("templates", "resume_default.md")),
        "pandoc_defaults": str(defaults or resources.path("pandoc_resume.yaml")),
        "page_limit": "2",
    }
    if drop is not None:
        del values[drop]
    return "".join(f"{key}: {value}\n" for key, value in values.items()) + extra


def _draft_setup(tmp_path: Path, monkeypatch, *, sync: bool = True):
    """Synced index (or just its path), match.json, generation.yaml and the apps dir."""
    vault_text = _vault_text()
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    if sync:
        _patch_adapter(monkeypatch)
        assert _run(_copy_vault(tmp_path), db, chroma) == 0
    text, placed = _draft_payload_text(vault_text)
    match = tmp_path / "match.json"
    match.write_text(text, encoding="utf-8")
    apps = tmp_path / "apps"
    cfg = tmp_path / "generation.yaml"
    cfg.write_text(_generation_text(apps), encoding="utf-8")
    return db, match, cfg, apps, placed


def _draft_argv(match: Path, db: Path, cfg: Path, *extra: str) -> list[str]:
    return ["draft", str(match), "--db", str(db), "--generation-config", str(cfg), *extra]


def test_draft_writes_both_files_default_dir_and_out_dir(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    db, match, cfg, apps, placed = _draft_setup(tmp_path, monkeypatch)
    n_jobs = len(parse_star_vault(_vault_text()).jobs)
    capsys.readouterr()

    rc = main(_draft_argv(match, db, cfg))

    assert rc == 0
    default_dir = apps / "match"
    assert (default_dir / "tailored_resume.md").is_file()
    assert (default_dir / "match_report.md").is_file()
    assert f"jobs={n_jobs} bullets={placed} skills=0" in capsys.readouterr().err

    other = tmp_path / "elsewhere"
    rc = main(_draft_argv(match, db, cfg, "--out-dir", str(other)))

    assert rc == 0
    assert (other / "tailored_resume.md").is_file()
    assert (other / "match_report.md").is_file()


def test_draft_existing_output_refused_then_force(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    db, match, cfg, apps, _placed = _draft_setup(tmp_path, monkeypatch)
    assert main(_draft_argv(match, db, cfg)) == 0
    resume = apps / "match" / "tailored_resume.md"
    first = resume.read_text(encoding="utf-8")
    resume.write_text("hand edited\n", encoding="utf-8")
    capsys.readouterr()

    rc = main(_draft_argv(match, db, cfg))

    assert rc == 1
    assert "tailored_resume.md" in capsys.readouterr().err
    assert resume.read_text(encoding="utf-8") == "hand edited\n"

    rc = main(_draft_argv(match, db, cfg, "--force"))

    assert rc == 0
    assert resume.read_text(encoding="utf-8") == first


@pytest.mark.parametrize("case", ["not_json", "no_selection"])
def test_draft_invalid_payload_exit_1(case, tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    db, match, cfg, apps, _placed = _draft_setup(tmp_path, monkeypatch)
    if case == "not_json":
        match.write_text("{not json", encoding="utf-8")
    else:
        text, _ = _draft_payload_text(_vault_text(), drop="selection")
        match.write_text(text, encoding="utf-8")
    capsys.readouterr()

    rc = main(_draft_argv(match, db, cfg))

    assert rc == 1
    err = capsys.readouterr().err
    assert str(match) in err
    assert not apps.exists()


def test_draft_without_index_exit_3(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    db, match, cfg, apps, _placed = _draft_setup(tmp_path, monkeypatch, sync=False)

    rc = main(_draft_argv(match, db, cfg))

    assert rc == 3
    assert "pipeline sync" in capsys.readouterr().err
    assert not apps.exists()


def test_help_lists_draft_and_its_options(capsys):
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "draft" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["draft", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--out-dir", "--db", "--generation-config", "--force"):
        assert flag in out
    assert "--header" not in out  # retired


@pytest.mark.parametrize(
    "config_text",
    ["output_dir: somewhere\nrewrite: false\n", "{}\n"],
    ids=["extra_key", "no_output_dir"],
)
def test_draft_bad_generation_config_exit_1(config_text, tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    db, match, cfg, apps, _placed = _draft_setup(tmp_path, monkeypatch)
    cfg.write_text(config_text, encoding="utf-8")
    capsys.readouterr()

    rc = main(_draft_argv(match, db, cfg))

    assert rc == 1
    assert capsys.readouterr().err.strip()
    assert not apps.exists()


# ------------------------------------------------------------------ finalize
# C-7 .. C-12: `pipeline finalize RESUME_MD`. The real bundled pandoc + typst run in every
# case except C-8 (forced page count) and C-11 (injected converter failure).

_FINALIZE_OUTPUTS = ("final_resume.md", "final_resume.typ", "final_resume.pdf")


def _finalize_setup(tmp_path: Path, monkeypatch):
    """Run `draft` on the synthetic vault; returns the draft's `tailored_resume.md` and the cfg."""
    from loomcv.cli import main

    db, match, cfg, apps, _placed = _draft_setup(tmp_path, monkeypatch)
    assert main(_draft_argv(match, db, cfg)) == 0
    return apps / "match" / "tailored_resume.md", cfg


def _write_pdf(pages: int):
    """A stand-in for `export_pdf` that writes an N-page PDF and reports no warnings."""

    def fake(typ_path, pdf_path):
        from pypdf import PdfWriter

        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=612, height=792)
        with open(pdf_path, "wb") as handle:
            writer.write(handle)
        return []

    return fake


def test_finalize_real_chain_beside_input_and_out_dir(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main
    from loomcv.finalize import strip_src

    resume, cfg = _finalize_setup(tmp_path, monkeypatch)
    source = resume.read_text(encoding="utf-8")
    assert "src:" in source
    capsys.readouterr()

    rc = main(["finalize", str(resume), "--generation-config", str(cfg)])

    assert rc == 0
    for name in _FINALIZE_OUTPUTS:
        assert (resume.parent / name).is_file()
    final = (resume.parent / "final_resume.md").read_text(encoding="utf-8")
    assert "src:" not in final
    assert final == strip_src(source)
    assert "pages=1 page_limit=2" in capsys.readouterr().err

    other = tmp_path / "elsewhere"
    rc = main(["finalize", str(resume), "--generation-config", str(cfg), "--out-dir", str(other)])

    assert rc == 0
    for name in _FINALIZE_OUTPUTS:
        assert (other / name).is_file()


def test_finalize_page_limit_exceeded_exit_5_then_override(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    resume, cfg = _finalize_setup(tmp_path, monkeypatch)
    monkeypatch.setattr("loomcv.finalize.export.export_pdf", _write_pdf(3))
    capsys.readouterr()

    rc = main(["finalize", str(resume), "--generation-config", str(cfg)])

    assert rc == 5
    for name in _FINALIZE_OUTPUTS:
        assert (resume.parent / name).is_file()
    assert "pages=3 page_limit=2" in capsys.readouterr().err

    rc = main(
        ["finalize", str(resume), "--generation-config", str(cfg), "--page-limit", "3", "--force"]
    )

    assert rc == 0
    assert "pages=3 page_limit=3" in capsys.readouterr().err


def test_finalize_existing_outputs_refused_then_force(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    resume, cfg = _finalize_setup(tmp_path, monkeypatch)
    argv = ["finalize", str(resume), "--generation-config", str(cfg)]
    assert main(argv) == 0
    final = resume.parent / "final_resume.md"
    final.write_text("sentinel\n", encoding="utf-8")
    capsys.readouterr()

    rc = main(argv)

    assert rc == 1
    assert "final_resume.md" in capsys.readouterr().err
    assert final.read_text(encoding="utf-8") == "sentinel\n"

    rc = main([*argv, "--force"])

    assert rc == 0
    assert "sentinel" not in final.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "case",
    ["missing_input", "not_utf8", "no_page_limit", "extra_key", "no_defaults_file"],
)
def test_finalize_bad_input_or_config_exit_1(case, tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    resume, cfg = _finalize_setup(tmp_path, monkeypatch)
    apps = tmp_path / "apps"
    if case == "missing_input":
        resume = tmp_path / "no_such_resume.md"
    elif case == "not_utf8":
        resume = tmp_path / "bad_bytes.md"
        resume.write_bytes(b"\xff\xfe## not utf-8 \x80\n")
    elif case == "no_page_limit":
        cfg.write_text(_generation_text(apps, drop="page_limit"), encoding="utf-8")
    elif case == "extra_key":
        cfg.write_text(_generation_text(apps, extra="rewrite: false\n"), encoding="utf-8")
    else:
        cfg.write_text(
            _generation_text(apps, defaults=tmp_path / "no_such_defaults.yaml"), encoding="utf-8"
        )
    capsys.readouterr()

    rc = main(["finalize", str(resume), "--generation-config", str(cfg)])

    assert rc == 1
    assert capsys.readouterr().err.strip()
    for name in _FINALIZE_OUTPUTS:
        assert not (resume.parent / name).exists()


@pytest.mark.parametrize("case", ["pandoc", "typst"])
def test_finalize_converter_failure_exit_6(case, tmp_path, monkeypatch, capsys):
    from loomcv.cli import main
    from loomcv.finalize import PandocFailed, TypstFailed

    resume, cfg = _finalize_setup(tmp_path, monkeypatch)

    def refuse(_exc, message):
        def fail(*_args, **_kwargs):
            raise _exc(message)

        return fail

    if case == "pandoc":
        monkeypatch.setattr(
            "loomcv.finalize.export.export_typ", refuse(PandocFailed, "pandoc says no")
        )
    else:
        monkeypatch.setattr(
            "loomcv.finalize.export.export_pdf", refuse(TypstFailed, "typst says no")
        )
    capsys.readouterr()

    rc = main(["finalize", str(resume), "--generation-config", str(cfg)])

    assert rc == 6
    assert f"{case} says no" in capsys.readouterr().err
    assert (resume.parent / "final_resume.md").is_file()
    assert (resume.parent / "final_resume.typ").exists() is (case == "typst")
    assert not (resume.parent / "final_resume.pdf").exists()


def test_help_lists_finalize_and_its_options(capsys):
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "finalize" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["finalize", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--out-dir", "--page-limit", "--generation-config", "--force"):
        assert flag in out


# ------------------------------------------------------------------------ run
# C-13 .. C-18: `pipeline run --jd` (match → draft → stop before the review) and
# `pipeline run --app-id --resume-edits` (accept the file as given → finalize). Fakes as in the
# match block: the `parse_jd` reply fake and `_CachingCoverageAdapter` behind
# `loomcv.cli.build_generation_adapter`; the real pandoc + typst chain except C-15 / C-17.


def _run_setup(tmp_path: Path, monkeypatch, capsys, *, vault_text: str | None = None) -> dict:
    """Synced index, tmp models + generation configs, a JD file, patched builders (no
    header file — `vault_text` may carry a header block instead)."""
    import json

    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(_copy_vault(tmp_path, vault_text), db, chroma) == 0

    def build(node, provider):
        if node.name == "coverage":
            return _CachingCoverageAdapter(model_id=node.model)
        return FakeGenerationAdapter([json.dumps(_JD_REPLY)])

    monkeypatch.setattr("loomcv.cli.build_generation_adapter", build)
    apps = tmp_path / "apps"
    generation = tmp_path / "generation.yaml"
    generation.write_text(_generation_text(apps), encoding="utf-8")
    capsys.readouterr()
    return {
        "db": db,
        "chroma": chroma,
        "apps": apps,
        "generation": generation,
        "models": _write_gen_config(tmp_path),
        "jd": _write_jd(tmp_path),
        "checkpoints": tmp_path / "cp" / "checkpoints.sqlite",
    }


def _start_argv(env: dict, *extra: str) -> list[str]:
    return [
        "run",
        "--jd",
        str(env["jd"]),
        "--db",
        str(env["db"]),
        "--chroma",
        str(env["chroma"]),
        "--config",
        str(env["models"]),
        "--generation-config",
        str(env["generation"]),
        "--out-dir",
        str(env["apps"]),
        "--checkpoints",
        str(env["checkpoints"]),
        *extra,
    ]


def _submit_argv(env: dict, edited: Path, *extra: str, app_id: str = "jd") -> list[str]:
    return [
        "run",
        "--app-id",
        app_id,
        "--resume-edits",
        str(edited),
        "--generation-config",
        str(env["generation"]),
        "--out-dir",
        str(env["apps"]),
        "--checkpoints",
        str(env["checkpoints"]),
        *extra,
    ]


def _main_rc(argv: list[str]) -> int:
    """`main(argv)`'s exit code; argparse usage errors surface as `SystemExit` (code 2)."""
    from loomcv.cli import main

    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code


def _last_line(text: str) -> str:
    return text.strip().splitlines()[-1]


def test_run_jd_stops_at_review(tmp_path, monkeypatch, capsys):
    # C-13
    env = _run_setup(tmp_path, monkeypatch, capsys)

    rc = _main_rc(_start_argv(env))

    assert rc == 0
    folder = env["apps"] / "jd"
    resume = folder / "tailored_resume.md"
    assert _last_line(capsys.readouterr().err) == f"app_id=jd status=reviewing resume={resume}"
    for name in (
        "jd.md",
        "profile.json",
        "match.json",
        "tailored_resume.md",
        "match_report.md",
    ):
        assert (folder / name).is_file(), name
    assert not (folder / "header.md").exists()

    assert _main_rc(_start_argv(env, "--app-id", "other")) == 0
    assert (env["apps"] / "other" / "tailored_resume.md").is_file()


def test_run_resume_edits_finishes(tmp_path, monkeypatch, capsys):
    # C-14
    env = _run_setup(tmp_path, monkeypatch, capsys)
    assert _main_rc(_start_argv(env)) == 0
    folder = env["apps"] / "jd"
    edited = tmp_path / "edited.md"
    edited.write_bytes((folder / "tailored_resume.md").read_bytes())
    capsys.readouterr()

    rc = _main_rc(_submit_argv(env, edited))

    assert rc == 0
    assert _last_line(capsys.readouterr().err) == "pages=1 page_limit=2 status=done"
    assert (folder / "reviewed_resume.md").read_bytes() == edited.read_bytes()
    for name in _FINALIZE_OUTPUTS:
        assert (folder / name).is_file()


def test_run_over_limit_exit_5_then_page_limit(tmp_path, monkeypatch, capsys):
    # C-15
    env = _run_setup(tmp_path, monkeypatch, capsys)
    assert _main_rc(_start_argv(env)) == 0
    folder = env["apps"] / "jd"
    monkeypatch.setattr("loomcv.finalize.export.export_pdf", _write_pdf(3))
    capsys.readouterr()

    rc = _main_rc(_submit_argv(env, folder / "tailored_resume.md"))

    assert rc == 5
    assert _last_line(capsys.readouterr().err) == "pages=3 page_limit=2 status=done"
    for name in _FINALIZE_OUTPUTS:
        assert (folder / name).is_file()

    rc = _main_rc(_submit_argv(env, folder / "tailored_resume.md", "--page-limit", "3"))

    assert rc == 0
    assert _last_line(capsys.readouterr().err) == "pages=3 page_limit=3 status=done"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unknown_id", 1),
        ("existing_id", 1),
        ("both_modes", 2),
        ("neither_mode", 2),
        ("resume_without_app_id", 2),
        ("bad_id", 1),
    ],
)
def test_run_usage_errors(case, expected, tmp_path, monkeypatch, capsys):
    # C-16
    env = _run_setup(tmp_path, monkeypatch, capsys)
    some_file = tmp_path / "some.md"
    some_file.write_text("x\n", encoding="utf-8")
    if case == "unknown_id":
        argv = _submit_argv(env, some_file, app_id="nope")
    elif case == "existing_id":
        assert _main_rc(_start_argv(env)) == 0
        before = sorted(p.name for p in (env["apps"] / "jd").iterdir())
        argv = _start_argv(env)
    elif case == "both_modes":
        argv = [*_start_argv(env), "--resume-edits", str(some_file)]
    elif case == "neither_mode":
        argv = ["run"]
    elif case == "resume_without_app_id":
        argv = ["run", "--resume-edits", str(some_file), "--checkpoints", str(env["checkpoints"])]
    else:
        argv = _start_argv(env, "--app-id", "../x")
    capsys.readouterr()

    assert _main_rc(argv) == expected

    err = capsys.readouterr().err
    assert err.strip()
    # Exit 2 is also what an unknown sub-command gives: pin that the `run` parser spoke.
    if case in ("both_modes", "neither_mode"):
        assert "usage: pipeline run" in err
    if case == "resume_without_app_id":
        assert "--app-id" in err
    if case == "existing_id":
        assert sorted(p.name for p in (env["apps"] / "jd").iterdir()) == before
    if case in ("unknown_id", "bad_id"):
        assert not (env["apps"] / "nope").exists()
        assert not (tmp_path / "x").exists()


def test_run_converter_failure_exit_6_then_resubmit(tmp_path, monkeypatch, capsys):
    # C-17
    from loomcv.finalize import PandocFailed

    env = _run_setup(tmp_path, monkeypatch, capsys)
    assert _main_rc(_start_argv(env)) == 0
    folder = env["apps"] / "jd"
    argv = _submit_argv(env, folder / "tailored_resume.md")

    def fail(*_args, **_kwargs):
        raise PandocFailed("pandoc says no")

    capsys.readouterr()
    with monkeypatch.context() as patched:
        patched.setattr("loomcv.finalize.export.export_typ", fail)
        rc = _main_rc(argv)

    assert rc == 6
    assert "pandoc says no" in capsys.readouterr().err
    assert (folder / "final_resume.md").is_file()
    assert not (folder / "final_resume.pdf").exists()

    assert _main_rc(argv) == 0
    assert (folder / "final_resume.pdf").is_file()


def test_run_help_lists_its_flags(capsys):
    # C-18 (the top-help half is the `run` line in `test_help_lists_sync_and_match`)
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--jd",
        "--resume-edits",
        "--app-id",
        "--out-dir",
        "--page-limit",
        "--checkpoints",
    ):
        assert flag in out
    assert "--header" not in out  # retired


# ------------------------------------------------------ pipeline review
#
# C-19 … C-22. The page itself is not under pytest (the `ui` group is not installed in CI); these
# cases cover the subcommand's flags and the lazy `gradio` boundary.


def test_review_help_lists_the_runtime_flags_and_nothing_per_application(capsys):
    # C-20
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["review", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--out-dir",
        "--db",
        "--chroma",
        "--config",
        "--retrieval-config",
        "--generation-config",
        "--checkpoints",
    ):
        assert flag in out
    for flag in ("--app-id", "--jd", "--header", "--page-limit", "--resume-edits"):
        assert flag not in out


def test_review_without_the_ui_group_exits_1(monkeypatch, capsys):
    # C-21 — `gradio` blocked in `sys.modules`, `loomcv.ui.review` evicted so the import re-runs
    from loomcv.cli import main

    monkeypatch.setitem(sys.modules, "gradio", None)
    monkeypatch.delitem(sys.modules, "loomcv.ui.review", raising=False)
    assert main(["review"]) == 1
    assert "uv sync --group ui" in capsys.readouterr().err


def test_importing_the_cli_loads_neither_gradio_nor_the_review_page():
    # C-22 — fresh interpreter; the page module exists and is reached from inside `_run_review`
    code = (
        "import importlib.util, sys\n"
        "assert importlib.util.find_spec('loomcv.ui.review') is not None\n"
        "import loomcv.cli\n"
        "assert 'gradio' not in sys.modules\n"
        "assert 'loomcv.ui.review' not in sys.modules\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------- the header in the vault (C-2 … C-6)

_L3A_BLOCK = (
    "* **Name**: Fictional Person\n"
    "* **Headline**: Platform Engineer\n"
    "* **Contact**: Example City; fictional@example.invalid\n"
)


def _with_block(text: str, block: str = _L3A_BLOCK) -> str:
    first, rest = text.split("\n", 1)
    return f"{first}\n{block}{rest}"


def test_l3a_draft_header_flag_is_a_usage_error(tmp_path, capsys):
    # C-2 — retired outright: argparse refuses it before any file is read.
    rc = _main_rc(["draft", str(tmp_path / "match.json"), "--header", "x"])
    assert rc == 2
    assert "unrecognized arguments: --header" in capsys.readouterr().err
    assert _main_rc(["run", "--jd", str(tmp_path / "jd.md"), "--header", "x"]) == 2


def test_l3a_run_prints_the_vault_header_and_writes_no_header_file(tmp_path, monkeypatch, capsys):
    # C-3
    vault = _with_block((FIXTURES / VAULT_NAME).read_text(encoding="utf-8"))
    env = _run_setup(tmp_path, monkeypatch, capsys, vault_text=vault)

    assert _main_rc(_start_argv(env)) == 0

    folder = env["apps"] / "jd"
    assert not (folder / "header.md").exists()
    resume = (folder / "tailored_resume.md").read_text(encoding="utf-8")
    assert resume.startswith(
        "# Fictional Person\n\nExample City | fictional@example.invalid\n\n**Platform Engineer**\n"
    )


def test_l3a_complete_accept_all_keeps_the_header_and_writes_the_prefill(
    tmp_path, monkeypatch, capsys
):
    # C-4 — the headless path: ingest's header as extracted, the pre-fill as one group.
    import json

    from loomcv.cli import main
    from loomcv.sync.parse_assets import parse_star_vault

    monkeypatch.chdir(tmp_path)
    built = _patch_gen_adapter(monkeypatch, [json.dumps(_SEL_SAM)])
    draft = tmp_path / "draft.md"
    draft_text = _with_block(
        (FIXTURES / "ingest_expected_vault.md").read_text(encoding="utf-8"),
        _L3A_BLOCK + "* **Skills**: Go\n",
    )
    draft.write_text(draft_text, encoding="utf-8")
    out = tmp_path / "vault.md"

    assert main([*_complete_argv(tmp_path, draft, "--accept-all"), "--out", str(out)]) == 0
    assert "written=" in capsys.readouterr().out
    assert len(built[0][1].calls) == 1  # the selection call only — no grouping call

    before = parse_star_vault(draft_text, include_draft=True)
    written = parse_star_vault(out.read_text(encoding="utf-8"))
    assert written.errors() == []
    assert written.header == before.header
    assert [(g.name, g.items) for g in written.skill_groups] == [
        ("", ["Go", "Python", "Django", "PostgreSQL"])
    ]


def test_l3a_complete_checklist_marks_optional_header_lines(tmp_path, monkeypatch, capsys):
    # C-5 — required first, then the optional ones, marked as such.
    import json

    from loomcv.cli import main

    monkeypatch.chdir(tmp_path)
    _patch_gen_adapter(monkeypatch, [])
    draft = tmp_path / "draft.md"
    draft.write_text(
        _with_block((FIXTURES / "ingest_expected_vault.md").read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    assert main(_complete_argv(tmp_path, draft, "--checklist")) == 0
    items = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    header = [(i["kind"], i["message"]) for i in items if i["job_id"] is None]
    assert header == [
        ("header_optional", "Summary is empty (optional)"),
        ("header_optional", "Education is empty (optional)"),
    ]

    draft.write_text((FIXTURES / "ingest_expected_vault.md").read_text("utf-8"), encoding="utf-8")
    assert main(_complete_argv(tmp_path, draft, "--checklist")) == 0
    items = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [i["message"] for i in items[:2]] == ["Name is empty", "Contact is empty"]


def test_l3a_sync_line_shape_unchanged_skills_is_the_flat_count(tmp_path, monkeypatch, capsys):
    # C-6 — no new counter; `skills=N` counts the groups' distinct items.
    vault = tmp_path / "vault.md"
    vault.write_text((FIXTURES / "header_vault.md").read_text(encoding="utf-8"), encoding="utf-8")
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)

    assert _run(vault, db, chroma) == 0

    out = capsys.readouterr().out
    assert "skills=7" in out
    assert "jobs_written=1" in out
    assert "header" not in out and "groups" not in out


def test_l3b_complete_accept_all_makes_no_drafting_call(tmp_path, monkeypatch, capsys):
    # C-1 — the headless path builds the selection adapter only; the `rewrite` node
    # is configured but never loaded. Green at red by design: it guards unchanged behaviour.
    import json

    from loomcv.cli import main

    monkeypatch.chdir(tmp_path)
    built = _patch_gen_adapter(monkeypatch, [json.dumps(_SEL_SAM)])
    draft = _complete_draft(tmp_path)
    cfg = _write_gen_config(
        tmp_path,
        nodes={
            **_INGEST_NODE,
            "rewrite": {"provider": "local_eval", "model": "fake-rewrite", "temperature": 0.7},
        },
    )

    assert main(["complete", str(draft), "--accept-all", "--config", str(cfg)]) == 0
    assert "written=" in capsys.readouterr().out
    assert [name for name, _ in built] == ["ingest"]
    assert len(built[0][1].calls) == 1


def test_l3b_complete_and_ui_flags_unchanged():
    # C-2 — no new flag, no new subcommand.
    from loomcv.cli import _build_parser

    sub = next(a for a in _build_parser()._actions if getattr(a, "choices", None))
    assert {"complete", "ui"} <= set(sub.choices)

    def flags(name: str) -> set[str]:
        return {o for o in sub.choices[name]._option_string_actions if o.startswith("--")}

    assert flags("complete") == {"--accept-all", "--checklist", "--config", "--help", "--out"}
    assert flags("ui") == {"--config", "--draft", "--help", "--out"}


# ---------------------------------------------------------- pipeline app
#
# C-1 … C-3. The page
# itself is not under pytest (the `ui` group is not installed in CI); these cases cover the
# subcommand's flags, the two older entries' help (byte-equal) and the lazy `gradio` boundary.

_APP_FLAGS = {
    "--out-dir",
    "--checkpoints",
    "--db",
    "--chroma",
    "--config",
    "--retrieval-config",
    "--generation-config",
}

# Recorded at `845a353` (taken at HEAD `55a8ea3` with COLUMNS=80; `loomcv/cli.py` unchanged).
_UI_HELP_AT_845A353 = (
    "usage: pipeline ui [-h] [--draft DRAFT] [--out OUT] [--config CONFIG]\n"
    "\n"
    "options:\n"
    "  -h, --help       show this help message and exit\n"
    "  --draft DRAFT    Draft vault to work on (default: the --out path, so a bare\n"
    "                   `pipeline ui` resumes work on the confirmed vault)\n"
    "  --out OUT        Vault path the UI writes (default: 02_database/vault.md)\n"
    "  --config CONFIG  Model config path: its `providers:`/`nodes:` for `ingest`\n"
    "                   (default: config/models.yaml)\n"
)
_REVIEW_HELP_AT_845A353 = (
    "usage: pipeline review [-h] [--out-dir OUT_DIR] [--checkpoints CHECKPOINTS]\n"
    "                       [--db DB] [--chroma CHROMA] [--config CONFIG]\n"
    "                       [--retrieval-config RETRIEVAL_CONFIG]\n"
    "                       [--generation-config GENERATION_CONFIG]\n"
    "\n"
    "options:\n"
    "  -h, --help            show this help message and exit\n"
    "  --out-dir OUT_DIR     Applications root the page lists and starts\n"
    "                        applications in (default: output_dir from\n"
    "                        config/generation.yaml)\n"
    "  --checkpoints CHECKPOINTS\n"
    "                        LangGraph checkpoint database (default:\n"
    "                        02_database/checkpoints.sqlite); kept apart from the\n"
    "                        index database\n"
    "  --db DB               SQLite source-of-record path (default:\n"
    "                        02_database/loomcv.sqlite)\n"
    "  --chroma CHROMA       Chroma persistent store directory (default:\n"
    "                        02_database/vector_cache)\n"
    "  --config CONFIG       Model config path, as for `pipeline match` (default:\n"
    "                        config/models.yaml)\n"
    "  --retrieval-config RETRIEVAL_CONFIG\n"
    "                        Retrieval config path (default: config/retrieval.yaml)\n"
    "  --generation-config GENERATION_CONFIG\n"
    "                        Generation config path (default:\n"
    "                        config/generation.yaml)\n"
)


def test_c1_app_help_lists_the_union_minus_draft_and_out(capsys):
    from loomcv.cli import _build_parser, main

    with pytest.raises(SystemExit) as exc:
        main(["app", "--help"])
    assert exc.value.code == 0
    capsys.readouterr()

    sub = next(a for a in _build_parser()._actions if getattr(a, "choices", None))
    assert "app" in sub.choices
    options = {o for o in sub.choices["app"]._option_string_actions if o.startswith("--")}
    assert _APP_FLAGS <= options
    assert "--draft" not in options
    assert "--out" not in options
    assert options == _APP_FLAGS | {"--help"}


def test_c2_ui_and_review_help_byte_equal(monkeypatch, capsys):
    from loomcv.cli import main

    monkeypatch.setenv("COLUMNS", "80")
    for command, expected in (("ui", _UI_HELP_AT_845A353), ("review", _REVIEW_HELP_AT_845A353)):
        with pytest.raises(SystemExit) as exc:
            main([command, "--help"])
        assert exc.value.code == 0
        assert capsys.readouterr().out == expected, command


def test_c3_app_without_the_ui_group_exits_1(monkeypatch, capsys):
    # `gradio` blocked in `sys.modules`, `loomcv.ui.shell` evicted so the import re-runs.
    from loomcv.cli import main

    monkeypatch.setitem(sys.modules, "gradio", None)
    monkeypatch.delitem(sys.modules, "loomcv.ui.shell", raising=False)
    assert main(["app"]) == 1
    assert capsys.readouterr().err == (
        "pipeline app needs the optional ui group: uv sync --group ui\n"
    )


# ---- `ingest` reads PDF and Word through `loomcv.ingest.read` ----


def test_p1_c1_ingest_on_a_pdf_prints_the_same_counters_as_on_the_markdown(
    tmp_path, monkeypatch, capsys
):
    import json

    import typst

    from loomcv.cli import main

    reply = json.dumps(_INGEST_STRUCTURE)

    _patch_gen_adapter(monkeypatch, [reply])
    md_argv = _ingest_argv(tmp_path)
    assert main(md_argv) == 0
    md_lines = capsys.readouterr().out.splitlines()

    # The same text as a PDF: one typst `raw` block, wide enough that no line wraps.
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    (pdf_dir / "resume.md").write_text(_INGEST_RESUME, encoding="utf-8")
    (pdf_dir / "resume.typ").write_text(
        '#set page(width: 60cm, height: auto)\n#raw(read("resume.md"), block: true)\n',
        encoding="utf-8",
    )
    typst.compile(str(pdf_dir / "resume.typ"), output=str(pdf_dir / "resume.pdf"))
    pdf_argv = list(md_argv)
    pdf_argv[1] = str(pdf_dir / "resume.pdf")
    pdf_argv[pdf_argv.index("--out") + 1] = str(pdf_dir / "draft_vault.md")

    _patch_gen_adapter(monkeypatch, [reply])
    assert main(pdf_argv) == 0
    pdf_lines = capsys.readouterr().out.splitlines()

    assert [ln for ln in pdf_lines if not ln.startswith("written=")] == [
        ln for ln in md_lines if not ln.startswith("written=")
    ]
    assert "jobs=2" in pdf_lines
    assert "dropped=0" in pdf_lines


def test_p1_c2_ingest_refuses_an_unsupported_suffix(tmp_path, monkeypatch, capsys):
    from loomcv.cli import main

    built = _patch_gen_adapter(monkeypatch, [])
    argv = _ingest_argv(tmp_path)
    rtf = tmp_path / "resume.rtf"
    rtf.write_text(_INGEST_RESUME, encoding="utf-8")
    argv[1] = str(rtf)

    assert main(argv) == 1
    assert capsys.readouterr().err.startswith("invalid resume")
    assert built == []
    assert not (tmp_path / "out" / "draft_vault.md").exists()


def test_p1_c3_ingest_help_names_pdf_and_word(capsys):
    from loomcv.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "PDF" in out
    assert "Word" in out


# ------------------------------------------------- the sidecar's constant
#
# C-1 / C-2.


def test_p2a_c1_ui_help_keeps_its_flags():
    from loomcv.cli import _build_parser

    sub = next(a for a in _build_parser()._actions if getattr(a, "choices", None))
    flags = {o for o in sub.choices["ui"]._option_string_actions if o.startswith("--")}
    assert flags == {"--draft", "--out", "--config", "--help"}


def test_p2a_c2_default_sidecar_sits_beside_the_working_copy():
    from pathlib import Path

    from loomcv import cli
    from loomcv.complete.sidecar import sidecar_for

    assert cli._DEFAULT_SIDECAR == "02_database/draft_vault.state.json"
    assert cli._DEFAULT_SIDECAR == str(sidecar_for(Path(cli._DEFAULT_WORKING_COPY)))
