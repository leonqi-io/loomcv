"""Graph / application tests G-1 .. G-10.

The graph (`loomcv/graph.py`) runs match → draft → [interrupt_before human_review] → finalize over
a `SqliteSaver`; `loomcv/application.py` is the boundary the UI and the CLI call (`start` /
`submit` / `status`). The index is the synthetic `index_vault.md` synced into `tmp_path`; the
adapters are the fakes of `test_cli.py` injected through `Runtime.adapters`. The finalize cases
run the real bundled pandoc + typst except G-6 (forced page count) and G-7 (injected failure).

All fixture content is fictional — no real career data.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest
from test_cli import (  # noqa: F401  (the autouse fixture must be in this module's namespace)
    _JD_REPLY,
    FakeEmbeddingAdapter,
    FakeGenerationAdapter,
    FakeRerankAdapter,
    _CachingCoverageAdapter,
    _copy_vault,
    _draft_argv,
    _generation_text,
    _jd_argv,
    _no_default_embedding_function,
    _patch_adapter,
    _run,
    _write_gen_config,
    _write_jd,
    _write_pdf,
)

COUNTERS = re.compile(r"jobs=(\d+) bullets=(\d+) skills=(\d+)")
HEADER = "# Fictional Person\nfictional@example.invalid\n\n"


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


def _vault_with_header_block() -> str:
    """`index_vault.md` with a header block under its H1 (the bullets unchanged)."""
    text = (Path(__file__).resolve().parent / "fixtures" / "index_vault.md").read_text("utf-8")
    first, rest = text.split("\n", 1)
    return f"{first}\n{VAULT_HEADER_BLOCK}{rest}"


VAULT_HEADER_BLOCK = (
    "* **Name**: Fictional Person\n"
    "* **Headline**: Platform Engineer\n"
    "* **Contact**: Example City; fictional@example.invalid\n"
    "* **Summary**: Ten fictional years.\n"
    "* **Education**: Example University — BSc, 2014\n"
    "* **Skills**: SQL\n"
    "  * **Languages**: Python; Go\n"
)


class _App:
    """A synced tmp index, a tmp `generation.yaml` (page_limit 2) and a JD file. No
    header file — `header_block=True` syncs a vault whose header block the résumé prints."""

    def __init__(self, tmp_path: Path, monkeypatch, capsys, *, header_block: bool = False) -> None:
        self.tmp = tmp_path
        self.db, self.chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
        _patch_adapter(monkeypatch)
        vault = _copy_vault(tmp_path, _vault_with_header_block() if header_block else None)
        assert _run(vault, self.db, self.chroma) == 0
        capsys.readouterr()
        self.apps = tmp_path / "apps"
        self.cfg = tmp_path / "generation.yaml"
        self.cfg.write_text(_generation_text(self.apps), encoding="utf-8")
        self.jd = _write_jd(tmp_path)
        self.checkpoints = tmp_path / "cp" / "checkpoints.sqlite"

    def runtime(self):
        return _runtime(self.db, self.chroma, self.cfg)

    def start(self, app_id: str | None = None, *, runtime=None):
        from loomcv import application

        return application.start(
            runtime or self.runtime(),
            self.jd,
            app_id=app_id,
            out_root=self.apps,
            checkpoints=self.checkpoints,
        )

    def submit(self, edited: Path, app_id: str = "jd", *, page_limit=None, runtime=None):
        from loomcv import application

        return application.submit(
            runtime or self.runtime(),
            app_id,
            edited_path=edited,
            page_limit=page_limit,
            out_root=self.apps,
            checkpoints=self.checkpoints,
        )

    def status(self, app_id: str = "jd"):
        from loomcv import application

        return application.status(app_id, out_root=self.apps, checkpoints=self.checkpoints)

    def snapshot(self, app_id: str = "jd"):
        """The raw graph state through a fresh connection (closed before returning)."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        from loomcv.graph import build_graph, thread_config

        self.checkpoints.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.checkpoints), check_same_thread=False)
        try:
            saver = SqliteSaver(conn)
            saver.setup()
            return build_graph(self.runtime(), saver).get_state(thread_config(app_id))
        finally:
            conn.close()

    def edited_copy(self, name: str = "edited.md") -> Path:
        """The draft with the placed bullet's text changed and its `src` comment kept."""
        draft = (self.apps / "jd" / "tailored_resume.md").read_text(encoding="utf-8")
        assert "800ms to 200ms" in draft and "<!-- src:" in draft
        path = self.tmp / name
        path.write_bytes(draft.replace("800ms to 200ms", "800ms to 100ms").encode("utf-8"))
        return path


@pytest.fixture
def app(tmp_path, monkeypatch, capsys):
    return _App(tmp_path, monkeypatch, capsys)


def _pdf_pages(path: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


# ------------------------------------------------------------------------ graph


def test_graph_nodes_and_interrupt(tmp_path):
    # G-1
    from langgraph.checkpoint.sqlite import SqliteSaver

    from loomcv.graph import build_graph

    conn = sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        graph = build_graph(
            _runtime(tmp_path / "db.sqlite", tmp_path / "chroma", tmp_path / "g"), saver
        )
        assert [n for n in graph.nodes if n != "__start__"] == [
            "match",
            "draft",
            "human_review",
            "finalize",
        ]
        assert list(graph.interrupt_before_nodes) == ["human_review"]
    finally:
        conn.close()


def test_start_stops_at_review_with_the_files(app, capsys):
    # G-2
    from loomcv.cli import main
    from loomcv.retrieve.rank import MatchPayload
    from loomcv.steps import match_step

    result = app.start()

    assert result.status == "reviewing"
    snap = app.snapshot()
    assert snap.next == ("human_review",)
    values = snap.values
    assert values["review_confirmed"] is False
    for key in (
        "app_dir",
        "jd_path",
        "profile_path",
        "match_path",
        "resume_path",
        "report_path",
    ):
        assert Path(values[key]).exists(), key
    assert "header_path" not in values

    folder = app.apps / "jd"
    assert (folder / "jd.md").read_bytes() == app.jd.read_bytes()
    assert not (folder / "header.md").exists()  # the header lives in the vault
    payload = MatchPayload.model_validate_json((folder / "match.json").read_text(encoding="utf-8"))
    assert payload.selection.jobs

    replay = app.runtime()
    profile = type(payload.profile).model_validate_json(
        (folder / "profile.json").read_text("utf-8")
    )
    match_step(replay, profile=profile)
    assert replay.adapters["parse_jd"].calls == [] and replay.adapters["coverage"].calls == []

    other = app.tmp / "cli_draft"
    capsys.readouterr()
    argv = _draft_argv(folder / "match.json", app.db, app.cfg, "--out-dir", str(other))
    assert main(argv) == 0
    counters = tuple(map(int, COUNTERS.search(capsys.readouterr().err).groups()))
    for name in ("tailored_resume.md", "match_report.md"):
        assert (folder / name).read_bytes() == (other / name).read_bytes()
    assert (
        values["counts"]["jobs"],
        values["counts"]["bullets"],
        values["counts"]["skills"],
    ) == counters


def test_submit_edited_copy_finalizes_as_given(app):
    # G-3
    from loomcv.finalize import strip_src

    app.start()
    edited = app.edited_copy()

    result = app.submit(edited)

    folder = app.apps / "jd"
    assert (folder / "reviewed_resume.md").read_bytes() == edited.read_bytes()
    final = (folder / "final_resume.md").read_text(encoding="utf-8")
    assert final == strip_src(edited.read_text(encoding="utf-8"))
    assert "800ms to 100ms" in final and "src:" not in final
    assert _pdf_pages(folder / "final_resume.pdf") == 1
    assert (result.pages, result.page_limit, result.over_limit) == (1, 2, False)
    assert result.status == "done"
    assert app.status().status == "done"
    assert app.snapshot().next == ()


def test_submit_draft_unchanged(app):
    # G-4
    app.start()
    draft = app.apps / "jd" / "tailored_resume.md"

    result = app.submit(draft)

    assert result.status == "done"
    folder = app.apps / "jd"
    assert (folder / "reviewed_resume.md").read_bytes() == draft.read_bytes()
    assert (folder / "final_resume.pdf").is_file()


def test_restart_between_start_and_submit(app):
    # G-5 — every call opens its own connection; a new Runtime (new fakes) resumes the thread.
    app.start()
    assert app.snapshot().next == ("human_review",)  # a third connection sees the interrupt
    edited = app.edited_copy()

    result = app.submit(edited, runtime=app.runtime())

    assert result.status == "done"
    assert (result.pages, result.over_limit) == (1, False)
    assert app.snapshot().next == ()


def test_over_limit_is_done_and_second_submit_replaces(app, monkeypatch):
    # G-6: `done` means the user's file was exported as given.
    app.start()
    edited = app.edited_copy()
    folder = app.apps / "jd"

    with monkeypatch.context() as patched:
        patched.setattr("loomcv.finalize.export.export_pdf", _write_pdf(3))
        first = app.submit(edited)

    assert (first.pages, first.page_limit, first.over_limit) == (3, 2, True)
    assert first.status == "done" and app.status().status == "done"
    for name in ("final_resume.md", "final_resume.typ", "final_resume.pdf"):
        assert (folder / name).is_file()
    over_limit_pdf = (folder / "final_resume.pdf").read_bytes()

    second = app.submit(edited)

    assert (second.pages, second.over_limit) == (1, False)
    assert (folder / "final_resume.pdf").read_bytes() != over_limit_pdf
    assert _pdf_pages(folder / "final_resume.pdf") == 1


def test_converter_failure_leaves_finalizing_then_resubmit(app, monkeypatch):
    # G-7
    from loomcv.finalize import PandocFailed

    app.start()
    edited = app.edited_copy()
    folder = app.apps / "jd"

    def fail(*_args, **_kwargs):
        raise PandocFailed("pandoc says no")

    with monkeypatch.context() as patched:
        patched.setattr("loomcv.finalize.export.export_typ", fail)
        with pytest.raises(PandocFailed):
            app.submit(edited)

    assert app.snapshot().next == ("finalize",)
    assert app.status().status == "finalizing"
    assert (folder / "reviewed_resume.md").is_file()
    assert (folder / "final_resume.md").is_file()
    assert not (folder / "final_resume.typ").exists()
    assert not (folder / "final_resume.pdf").exists()

    result = app.submit(edited)

    assert result.status == "done" and result.pages == 1
    assert (folder / "final_resume.pdf").is_file()


def test_submit_unknown_id_and_missing_edit(app):
    # G-8
    from loomcv.application import UnknownApplication
    from loomcv.steps import InvalidInput

    with pytest.raises(UnknownApplication):
        app.submit(app.jd, "nope")
    assert not (app.apps / "nope").exists()

    app.start()
    with pytest.raises(InvalidInput):
        app.submit(app.tmp / "no_such_edit.md")

    assert not (app.apps / "jd" / "reviewed_resume.md").exists()
    assert app.snapshot().next == ("human_review",)


def test_start_refuses_existing_and_bad_id(app):
    # G-9
    from loomcv.application import ApplicationExists

    folder = app.apps / "jd"
    folder.mkdir(parents=True)
    (folder / "sentinel.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(ApplicationExists):
        app.start()

    assert [p.name for p in folder.iterdir()] == ["sentinel.txt"]
    assert app.snapshot().values == {}  # no thread was started for the id

    fresh = app.tmp / "apps_fresh"
    from loomcv import application

    with pytest.raises(ValueError):
        application.start(
            app.runtime(),
            app.jd,
            app_id="../x",
            out_root=fresh,
            checkpoints=app.checkpoints,
        )
    assert not fresh.exists()


def _exists_message(app, start) -> str:
    """The `ApplicationExists` message with the two paths blanked (tmp dirs carry test names)."""
    from loomcv.application import ApplicationExists

    with pytest.raises(ApplicationExists) as caught:
        start()
    message = str(caught.value)
    return message.replace(str(app.apps / "jd"), "<dir>").replace(str(app.checkpoints), "<cp>")


def test_application_exists_names_the_folder(app):
    # X-1
    folder = app.apps / "jd"
    folder.mkdir(parents=True)

    from loomcv.application import ApplicationExists

    with pytest.raises(ApplicationExists) as caught:
        app.start()
    assert str(folder) in str(caught.value)
    message = _exists_message(app, app.start)
    assert "folder" in message and "<dir>" in message
    assert "checkpoint" not in message
    assert "new name" in message


def test_application_exists_names_the_checkpoint(app):
    # X-2
    import shutil

    app.start()
    shutil.rmtree(app.apps / "jd")
    message = _exists_message(app, app.start)
    assert "checkpoint" in message and "<cp>" in message
    assert "folder" not in message
    assert "new name" in message
    assert not (app.apps / "jd").exists()


def test_application_exists_names_both(app):
    # X-3
    app.start()
    before = sorted(p.name for p in (app.apps / "jd").iterdir())
    message = _exists_message(app, app.start)
    assert "folder" in message and "checkpoint" in message
    assert "<dir>" in message and "<cp>" in message
    assert "new name" in message
    assert sorted(p.name for p in (app.apps / "jd").iterdir()) == before


def test_graph_adds_no_model_call_and_imports_no_langchain(app, monkeypatch, capsys):
    # G-10 — the graph adds no model call; no LoomCV module imports `langchain*`.
    from loomcv.cli import main

    runtime = app.runtime()
    app.start(runtime=runtime)
    graph_calls = (
        len(runtime.adapters["parse_jd"].calls),
        len(runtime.adapters["coverage"].calls),
    )

    built: list = []

    def build(node, provider):
        adapter = (
            _CachingCoverageAdapter(model_id=node.model)
            if node.name == "coverage"
            else FakeGenerationAdapter([json.dumps(_JD_REPLY)])
        )
        built.append((node.name, adapter))
        return adapter

    monkeypatch.setattr("loomcv.cli.build_generation_adapter", build)
    capsys.readouterr()
    models = _write_gen_config(app.tmp)
    assert main(_jd_argv(app.jd, app.db, app.chroma, models)) == 0
    cli_calls = {name: len(adapter.calls) for name, adapter in built}

    assert graph_calls == (cli_calls["parse_jd"], cli_calls["coverage"]) == (1, 1)

    assert "langgraph" in sys.modules
    assert (
        "langchain" not in sys.modules
    )  # the umbrella package; `langchain_core` comes with langgraph
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not name.startswith("loomcv") or not path or not path.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            assert not any(i.startswith("langchain") for i in imported), f"{name}: {imported}"


# --------------------------------------------- `--header` retired (G-1 … G-3)


def test_l3a_no_header_path_in_state_or_start():
    # G-1 — no kept-but-ignored parameter anywhere.
    import inspect

    from loomcv import application, graph

    assert "header_path" not in graph.PipelineState.__annotations__
    assert "header_path" not in inspect.signature(application.start).parameters
    assert not hasattr(graph, "HEADER_NAME")


def test_l3a_draft_node_prints_the_index_header(tmp_path, monkeypatch, capsys):
    # G-2 — the started application's résumé starts with the vault's name.
    app = _App(tmp_path, monkeypatch, capsys, header_block=True)
    assert app.start().status == "reviewing"
    resume = (app.apps / "jd" / "tailored_resume.md").read_text(encoding="utf-8")
    assert resume.startswith("# Fictional Person\n\nExample City | fictional@example.invalid\n\n")
    assert "**Platform Engineer**\n" in resume
    assert "## Education\n\nExample University — BSc, 2014\n" in resume


def test_l3a_a_left_over_header_file_is_ignored(tmp_path, monkeypatch, capsys):
    # G-3 — an older `header.md` in the application directory is read by nothing.
    app = _App(tmp_path, monkeypatch, capsys, header_block=True)
    app.start()
    folder = app.apps / "jd"
    (folder / "header.md").write_text(HEADER, encoding="utf-8")
    status = app.submit(folder / "tailored_resume.md")
    assert status.status == "done"
    final = (folder / "final_resume.md").read_text(encoding="utf-8")
    assert final.startswith("# Fictional Person\n")
    assert "fictional@example.invalid\n\n\n" not in final
    assert HEADER not in final
