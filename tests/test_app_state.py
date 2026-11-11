"""Tests for `loomcv.app_state` — the landing switch, the readiness gate and Start over.

Cases: S-1 … S-5 (the switch and the module's isolation from `gradio`), G-1 … G-7 (the gate),
O-1 … O-3 (Start over), E-1 … E-4 (`elapsed_text`), T-1 (two tabs only). The module is
imported inside `_mod()` so each case fails on its own at red (no collection error). An
index for the gate is built with the real `sync_vault` over the synthetic fixtures
`index_vault.md` (8 bullets, 7 confirmed) and
`claims_expected_vault.md` (all draft), so the writer and the reader of the `vault_sha256` row
meet here; the embedding adapter is the in-process fake below — zero network, zero model
downloads. All paths live under `tmp_path`.

All fixture content is fictional — no real career data.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _mod():
    import loomcv.app_state as app_state

    return app_state


@pytest.fixture(autouse=True)
def _no_default_embedding_function(monkeypatch):
    """No code path may ever call Chroma's default (ONNX) embedding fn (copied in shape from
    `tests/test_index.py`, re-declared so this suite stays independent of it)."""
    import chromadb.utils.embedding_functions as ef

    def _raise(*_args, **_kwargs):
        raise RuntimeError("default Chroma embedding function must never be called")

    for cls_name in ("DefaultEmbeddingFunction", "ONNXMiniLM_L6_V2"):
        cls = getattr(ef, cls_name, None)
        if cls is not None:
            monkeypatch.setattr(cls, "__call__", _raise)


class FakeEmbeddingAdapter:
    """Deterministic, network-free stand-in for `EmbeddingAdapter`."""

    model_id = "fake-model"
    vector_space_version = 1

    def embed(self, texts):
        return [[b / 255.0 for b in hashlib.sha1(t.encode("utf-8")).digest()[:8]] for t in texts]


# ------------------------------------------------------------------------------------ helpers


@pytest.fixture()
def where(tmp_path):
    """The paths the app fixes, all under one tmp working directory."""
    db = tmp_path / "02_database"
    db.mkdir()
    return {
        "working_copy": db / "draft_vault.md",
        "saved": db / "vault.md",
        "db": db / "loomcv.sqlite",
        "chroma": db / "vector_cache",
        "checkpoints": db / "checkpoints.sqlite",
        "out_root": tmp_path / "04_applications",
    }


def _sync(where, fixture: str) -> None:
    from loomcv.sync.index_assets import sync_vault

    shutil.copyfile(FIXTURES / fixture, where["saved"])
    sync_vault(
        where["saved"],
        db_path=where["db"],
        chroma_path=where["chroma"],
        adapter=FakeEmbeddingAdapter(),
    )


def _application(where) -> None:
    (where["out_root"] / "fennimore_p0").mkdir(parents=True)


def _readiness(where):
    return _mod().readiness(
        where["saved"],
        db=where["db"],
        out_root=where["out_root"],
        checkpoints=where["checkpoints"],
    )


# ------------------------------------------------------------------------- S: the switch


def test_s1_neither_file_is_empty(where):
    m = _mod()
    land = m.landing(where["working_copy"], where["saved"])
    assert land.state == m.ProfileState.EMPTY
    assert land.tab == m.PROFILE_TAB
    assert land.page_over is None
    assert land.line == m.EMPTY_LINE


def test_s2_working_copy_only_is_started(where):
    m = _mod()
    where["working_copy"].write_text("# draft\n", encoding="utf-8")
    land = m.landing(where["working_copy"], where["saved"])
    assert land.state == m.ProfileState.STARTED
    assert land.tab == m.PROFILE_TAB
    assert land.page_over == where["working_copy"]
    assert land.line == m.STOPPED_LINE


def test_s3_saved_only_is_saved(where):
    m = _mod()
    where["saved"].write_text("# saved\n", encoding="utf-8")
    land = m.landing(where["working_copy"], where["saved"])
    assert land.state == m.ProfileState.SAVED
    assert land.tab == m.APPLICATIONS_TAB
    assert land.page_over == where["saved"]
    assert land.line == ""


def test_s4_both_files_differing_is_still_saved(where):
    m = _mod()
    where["working_copy"].write_text("# draft, edited\n", encoding="utf-8")
    where["saved"].write_text("# saved\n", encoding="utf-8")
    land = m.landing(where["working_copy"], where["saved"])
    assert land.state == m.ProfileState.SAVED
    assert land.tab == m.APPLICATIONS_TAB
    # Built over the working copy when there is one.
    assert land.page_over == where["working_copy"]
    assert set(m.ProfileState) == {
        m.ProfileState.EMPTY,
        m.ProfileState.STARTED,
        m.ProfileState.SAVED,
    }


def test_s5_module_imports_without_gradio():
    # Fresh interpreter with gradio blocked (the `test_complete.py` U-2 pattern).
    code = (
        "import sys\n"
        "sys.modules['gradio'] = None\n"
        "import loomcv.app_state\n"
        "assert 'gradio' not in [m for m, v in sys.modules.items() if v is not None]\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------- G: the gate


def test_g1_nothing_is_closed_with_the_line(where):
    r = _readiness(where)
    assert r.existing is False
    assert r.ready is False
    assert r.posting_side_open is False
    assert r.line == "Finish and save your profile first"


def test_g2_one_application_no_saved_vault_opens_existing_only(where):
    m = _mod()
    _application(where)
    r = _readiness(where)
    assert r.existing is True
    assert r.ready is False
    assert r.posting_side_open is True
    assert r.line == m.GATE_LINE


def test_g3_saved_and_synced_no_applications_is_ready(where):
    _sync(where, "index_vault.md")
    r = _readiness(where)
    assert r.existing is False
    assert r.ready is True
    assert r.posting_side_open is True
    assert r.line == ""


def test_g4_ready_and_an_application_is_all_open(where):
    _sync(where, "index_vault.md")
    _application(where)
    r = _readiness(where)
    assert r.existing is True
    assert r.ready is True
    assert r.posting_side_open is True
    assert r.line == ""


def test_g5_index_without_the_fingerprint_row_is_closed(where):
    m = _mod()
    _sync(where, "index_vault.md")
    conn = sqlite3.connect(where["db"])
    conn.execute("DELETE FROM meta WHERE key = 'vault_sha256'")
    conn.commit()
    conn.close()
    r = _readiness(where)
    assert r.ready is False
    assert r.line == m.GATE_LINE


def test_g6_saved_edited_after_the_sync_is_closed(where):
    _sync(where, "index_vault.md")
    with where["saved"].open("a", encoding="utf-8") as fh:
        fh.write("x")
    r = _readiness(where)
    assert r.ready is False


def test_g7_drafts_only_vault_synced_is_closed(where):
    _sync(where, "claims_expected_vault.md")
    conn = sqlite3.connect(where["db"])
    try:
        assert conn.execute("SELECT COUNT(*) FROM bullets").fetchone()[0] == 0
        row = conn.execute("SELECT value FROM meta WHERE key = 'vault_sha256'").fetchone()
    finally:
        conn.close()
    assert row is not None
    r = _readiness(where)
    assert r.ready is False


# ------------------------------------------------------------------------ O: Start over


@pytest.fixture()
def others(where):
    """The stores Start over must leave alone, each with a file in it."""
    where["db"].write_bytes(b"index database bytes")
    where["chroma"].mkdir()
    (where["chroma"] / "chroma.sqlite3").write_bytes(b"chroma bytes")
    app_dir = where["out_root"] / "fennimore_p0"
    app_dir.mkdir(parents=True)
    (app_dir / "jd.md").write_bytes(b"a posting")
    where["checkpoints"].write_bytes(b"checkpoint bytes")
    files = [
        where["db"],
        where["chroma"] / "chroma.sqlite3",
        app_dir / "jd.md",
        where["checkpoints"],
    ]
    return {p: p.read_bytes() for p in files}


def _untouched(snapshot: dict[Path, bytes]) -> None:
    for path, data in snapshot.items():
        assert path.is_file(), path
        assert path.read_bytes() == data, path


def _sidecar(where) -> Path:
    """The sidecar sits beside the working copy."""
    return where["working_copy"].with_name("draft_vault.state.json")


def test_o1_both_files_are_deleted(where, others):
    where["working_copy"].write_text("# draft\n", encoding="utf-8")
    where["saved"].write_text("# saved\n", encoding="utf-8")
    _sidecar(where).write_text("{}", encoding="utf-8")
    removed = _mod().start_over(where["working_copy"], where["saved"], _sidecar(where))
    assert not where["working_copy"].exists()
    assert not where["saved"].exists()
    assert not _sidecar(where).exists()
    assert set(removed) == {where["working_copy"], where["saved"], _sidecar(where)}
    assert len(removed) == 3
    _untouched(others)


@pytest.mark.parametrize("present", ["working_copy", "saved"])
def test_o2_one_file_is_deleted(where, others, present):
    where[present].write_text("# one\n", encoding="utf-8")
    removed = _mod().start_over(where["working_copy"], where["saved"], _sidecar(where))
    assert not where[present].exists()
    assert removed == [where[present]]
    _untouched(others)


def test_o3_neither_file_is_no_error(where, others):
    removed = _mod().start_over(where["working_copy"], where["saved"], _sidecar(where))
    assert removed == []
    _untouched(others)


def test_p2a_o4_the_sidecar_alone_is_deleted(where, others):
    _sidecar(where).write_text('{"j": true}', encoding="utf-8")
    removed = _mod().start_over(where["working_copy"], where["saved"], _sidecar(where))
    assert not _sidecar(where).exists()
    assert removed == [_sidecar(where)]
    _untouched(others)


# ------------------------------------------------- the landing rule


def test_p2a_l1_both_files_build_the_page_over_the_working_copy(where):
    m = _mod()
    where["working_copy"].write_text("# draft, edited\n", encoding="utf-8")
    where["saved"].write_text("# saved\n", encoding="utf-8")
    land = m.landing(where["working_copy"], where["saved"])
    assert land.state == m.ProfileState.SAVED
    assert land.tab == m.APPLICATIONS_TAB
    assert land.page_over == where["working_copy"]
    assert land.line == ""


def test_p2a_l2_saved_only_builds_the_page_over_saved(where):
    m = _mod()
    where["saved"].write_text("# saved\n", encoding="utf-8")
    land = m.landing(where["working_copy"], where["saved"])
    assert land.state == m.ProfileState.SAVED
    assert land.page_over == where["saved"]


def test_p2a_l3_working_copy_only_is_unchanged_started(where):
    m = _mod()
    where["working_copy"].write_text("# draft\n", encoding="utf-8")
    land = m.landing(where["working_copy"], where["saved"])
    assert land.state == m.ProfileState.STARTED
    assert land.tab == m.PROFILE_TAB
    assert land.page_over == where["working_copy"]
    assert land.line == m.STOPPED_LINE


# ------------------------------- the veil's clock and the two tabs


def test_p2c0_e1_zero_seconds():
    assert _mod().elapsed_text(0) == "0:00"


def test_p2c0_e2_one_minute_one():
    assert _mod().elapsed_text(61) == "1:01"


def test_p2c0_e3_minutes_unbounded():
    assert _mod().elapsed_text(3665) == "61:05"


def test_p2c0_e4_floor_never_rounds_up():
    assert _mod().elapsed_text(59.9) == "0:59"


def test_lk23_saved_stamp_is_the_time_today_and_the_date_on_another_day():
    # an old save must not read as tonight's
    from datetime import datetime

    stamp = _mod().saved_stamp
    now = datetime(2026, 11, 10, 0, 5)
    assert stamp(datetime(2026, 11, 10, 0, 1), now) == "00:01"
    assert stamp(datetime(2026, 11, 9, 22, 52), now) == "Nov 9 22:52"  # yesterday
    assert stamp(datetime(2026, 10, 29, 22, 52), now) == "Oct 29 22:52"
    assert stamp(datetime(2025, 12, 31, 8, 0), now) == "Dec 31 2025 08:00"  # another year
    assert stamp(datetime(2026, 11, 11, 9, 0), now) == "Nov 11 09:00"  # a clock ahead: dated


def test_p2c0_t1_two_tabs_only():
    m = _mod()
    assert not hasattr(m, "APPLY_TAB")
    assert not hasattr(m, "APPLICATIONS_LINE")
    assert m.APPLICATIONS_TAB == "Applications"
