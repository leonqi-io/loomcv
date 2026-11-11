"""Step-function tests S-1 .. S-4.

`loomcv/steps.py` holds the bodies of `pipeline match` / `draft` / `finalize` as functions the graph
(and the CLI) call. Fixtures are the synthetic `index_vault.md`; the generation / embedding /
rerank adapters are the fakes of `test_cli.py` (imported, not copied — pytest's default `prepend`
import mode puts `tests/` on `sys.path`). The finalize cases run the real bundled pandoc + typst.

SP-1 .. SP-4 cover `save_profile`: the vault written, then the index built from
it, with a fake embedding adapter.

All fixture content is fictional — no real career data.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from test_cli import (  # noqa: F401  (the autouse fixture must be in this module's namespace)
    _JD_REPLY,
    FakeEmbeddingAdapter,
    FakeGenerationAdapter,
    FakeRerankAdapter,
    _CachingCoverageAdapter,
    _draft_argv,
    _draft_setup,
    _finalize_setup,
    _no_default_embedding_function,
)

REPO = Path(__file__).resolve().parents[1]
COUNTERS = re.compile(r"jobs=(\d+) bullets=(\d+) skills=(\d+)")


def _runtime(db: Path, chroma: Path, cfg: Path):
    from loomcv.steps import Runtime

    return Runtime(
        db=db,
        chroma=chroma,
        generation_config_path=cfg,
        adapters={
            "parse_jd": FakeGenerationAdapter([json.dumps(_JD_REPLY)]),
            "coverage": _CachingCoverageAdapter(model_id="fake-coverage"),
            "embedding": FakeEmbeddingAdapter(vector_space_version=1),
            "reranker": FakeRerankAdapter(),
        },
    )


# ------------------------------------------------------------------- draft_step


def test_draft_step_counts_match_cli_and_refuses_existing(tmp_path, monkeypatch, capsys):
    # S-1
    from loomcv.cli import main
    from loomcv.retrieve.rank import MatchPayload
    from loomcv.steps import OutputExists, draft_step

    db, match, cfg, _apps, _placed = _draft_setup(tmp_path, monkeypatch)
    cli_dir = tmp_path / "cli_out"
    capsys.readouterr()
    assert main(_draft_argv(match, db, cfg, "--out-dir", str(cli_dir))) == 0
    jobs, bullets, skills = map(int, COUNTERS.search(capsys.readouterr().err).groups())

    runtime = _runtime(db, tmp_path / "chroma", cfg)
    payload = MatchPayload.model_validate_json(match.read_text(encoding="utf-8"))
    out = tmp_path / "step_out"
    result = draft_step(runtime, payload, out_dir=out)

    assert (result.jobs, result.bullets, result.skills) == (jobs, bullets, skills)
    assert result.resume_path == out / "tailored_resume.md"
    assert result.report_path == out / "match_report.md"
    assert result.resume_path.is_file() and result.report_path.is_file()

    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (result.resume_path, result.report_path)
    }
    with pytest.raises(OutputExists):
        draft_step(runtime, payload, out_dir=out, force=False)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert after == before


# ---------------------------------------------------------------- finalize_step


def test_finalize_step_real_chain_and_force(tmp_path, monkeypatch):
    # S-2 — the real pandoc + typst chain
    from loomcv.steps import OutputExists, finalize_step

    resume, cfg = _finalize_setup(tmp_path, monkeypatch)
    runtime = _runtime(tmp_path / "db.sqlite", tmp_path / "chroma", cfg)
    out = tmp_path / "final_out"

    result = finalize_step(runtime, resume, out_dir=out)

    assert result.pages == 1
    assert result.page_limit == 2
    assert isinstance(result.warnings, list)
    for path in (result.md, result.typ, result.pdf):
        assert path.is_file()
        assert path.parent == out

    with pytest.raises(OutputExists):
        finalize_step(runtime, resume, out_dir=out, force=False)

    before = result.pdf.stat().st_mtime_ns
    again = finalize_step(runtime, resume, out_dir=out, force=True)

    assert again.pages == 1
    assert again.pdf.stat().st_mtime_ns > before


# ----------------------------------------------------------------- match_step


def test_match_step_jd_calls_then_profile_replay_zero_calls(tmp_path, monkeypatch, capsys):
    # S-3
    from test_cli import _copy_vault, _patch_adapter, _run

    from loomcv.steps import match_step

    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    _patch_adapter(monkeypatch)
    assert _run(_copy_vault(tmp_path), db, chroma) == 0
    capsys.readouterr()
    cfg = tmp_path / "generation.yaml"
    cfg.write_text("", encoding="utf-8")

    runtime = _runtime(db, chroma, cfg)
    first = match_step(runtime, jd_text="Fictional Corp seeks a fake player engineer.\n")

    assert len(runtime.adapters["parse_jd"].calls) == 1
    assert (
        len(runtime.adapters["coverage"].calls) == 1
    )  # the fixture has no claims → no second call
    assert first.profile.coverage
    assert first.result.profile == first.profile

    replay = _runtime(db, chroma, cfg)
    match_step(replay, profile=first.profile)

    assert replay.adapters["parse_jd"].calls == []
    assert replay.adapters["coverage"].calls == []


# --------------------------------------------------------------- import boundary


def test_steps_imports_no_langchain_or_langgraph_and_sources_clean():
    # S-4a — the step functions are LangGraph-free; no LoomCV module imports
    # `langchain*`; `langgraph*` is imported only by the graph and the application boundary.
    code = (
        "import sys\n"
        "import loomcv.steps\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('langchain', 'langchain_core', "
        "'langgraph') or m.startswith('langchain') or m.startswith('langgraph')]\n"
        "assert not loaded, loaded[:5]\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    langgraph_users = []
    for path in sorted((REPO / "loomcv").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("langchain"), f"{path}: imports {name}"
                if name.startswith("langgraph"):
                    langgraph_users.append(path.relative_to(REPO).as_posix())
    assert sorted(set(langgraph_users)) == ["loomcv/application.py", "loomcv/graph.py"]


def test_import_draft_loads_no_llm_or_search():
    # S-4b — the first assertion of the `loomcv.draft` boundary (it imports
    # only `loomcv.retrieve.rank` at runtime).
    code = (
        "import sys\n"
        "import loomcv.draft\n"
        "loaded = [m for m in sys.modules if m in ('loomcv.llm', 'loomcv.retrieve.search')]\n"
        "assert not loaded, loaded\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------- save_profile

_SAM = REPO / "tests" / "fixtures" / "ingest_expected_vault.md"
_TODAY = date(2026, 11, 8)


def _sam_confirmed():
    """The synthetic Sam workbook with every job's tools list and every bullet confirmed."""
    from loomcv.complete import accept_all, apply_bullets, apply_jobs, load_workbook

    wb = load_workbook(_SAM.read_text(encoding="utf-8"))
    jobs, bullets = accept_all(wb)
    return apply_bullets(apply_jobs(wb, jobs), bullets)


def _fingerprint_row(db: Path) -> str | None:
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'vault_sha256'").fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def _ready(saved: Path, db: Path, tmp_path: Path) -> bool:
    from loomcv.app_state import readiness

    return readiness(
        saved, db=db, out_root=tmp_path / "apps", checkpoints=tmp_path / "ck.sqlite"
    ).ready


def test_sp1_save_profile_writes_the_vault_and_builds_the_index(tmp_path):
    import sqlite3

    from loomcv.steps import save_profile
    from loomcv.sync.index_assets import vault_fingerprint
    from loomcv.sync.parse_assets import parse_star_vault

    wb = _sam_confirmed()
    n = sum(len(j.bullets) for j in wb.jobs)
    saved, db, chroma = tmp_path / "vault.md", tmp_path / "db.sqlite", tmp_path / "chroma"

    report = save_profile(
        wb,
        saved,
        db=db,
        chroma=chroma,
        adapter=lambda: FakeEmbeddingAdapter(vector_space_version=1),
        today=_TODAY,
    )

    text = saved.read_text(encoding="utf-8")
    assert parse_star_vault(text, include_draft=True).errors() == []
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM bullets").fetchone()[0]
    finally:
        conn.close()
    assert rows == n
    assert _fingerprint_row(db) == vault_fingerprint(text)
    assert _ready(saved, db, tmp_path) is True
    assert report.written.confirmed == n
    assert report.written.problems == []
    assert report.sync is not None
    assert report.sync.counts()["added"] == n
    assert report.index_error is None


def test_sp2_a_vault_with_parse_problems_is_written_and_not_indexed(tmp_path):
    from loomcv.steps import save_profile

    wb = _sam_confirmed()
    job = wb.jobs[0]
    # Past the module's own brace check (apply_bullets refuses it): a render the strict parser
    # rejects. (The plan named `}`; a lone `}` parses clean, so the case uses `{`.)
    bad = job.bullets[0].model_copy(update={"text": "Broke { things."})
    wb = wb.model_copy(
        update={"jobs": [job.model_copy(update={"bullets": [bad, *job.bullets[1:]]}), *wb.jobs[1:]]}
    )
    saved, db, chroma = tmp_path / "vault.md", tmp_path / "db.sqlite", tmp_path / "chroma"

    report = save_profile(
        wb,
        saved,
        db=db,
        chroma=chroma,
        adapter=lambda: FakeEmbeddingAdapter(vector_space_version=1),
        today=_TODAY,
    )

    assert saved.is_file()
    assert report.written.problems
    assert report.sync is None
    assert report.index_error
    assert not db.exists()


def test_sp3_an_adapter_failure_leaves_the_vault_written_and_no_index(tmp_path):
    from loomcv.steps import save_profile
    from loomcv.sync.parse_assets import parse_star_vault

    saved, db, chroma = tmp_path / "vault.md", tmp_path / "db.sqlite", tmp_path / "chroma"

    report = save_profile(
        _sam_confirmed(),
        saved,
        db=db,
        chroma=chroma,
        adapter=lambda: FakeEmbeddingAdapter(raise_on_call=True),
        today=_TODAY,
    )

    assert parse_star_vault(saved.read_text(encoding="utf-8"), include_draft=True).errors() == []
    assert report.sync is None
    assert report.index_error
    assert _ready(saved, db, tmp_path) is False


def test_sp4_a_second_save_of_the_same_workbook_embeds_nothing(tmp_path):
    from loomcv.steps import save_profile

    wb = _sam_confirmed()
    saved, db, chroma = tmp_path / "vault.md", tmp_path / "db.sqlite", tmp_path / "chroma"

    def again():
        return save_profile(
            wb,
            saved,
            db=db,
            chroma=chroma,
            adapter=lambda: FakeEmbeddingAdapter(vector_space_version=1),
            today=_TODAY,
        )

    again()
    first = _fingerprint_row(db)
    report = again()

    assert report.sync is not None
    assert report.sync.embedded == 0
    assert report.sync.added == []
    assert report.sync.updated == []
    assert _fingerprint_row(db) == first
