"""The application boundary: what the review UI and `pipeline run` call.

An application is one run of the graph for one JD, identified by its id (`thread_id` = the id):
`<out_root>/<app_id>/` holds its files and `checkpoints.sqlite` holds where it stands.

    start(...)   match → draft, stops before `human_review`        → status "reviewing"
    submit(...)  accepts the edited file as given, runs `finalize`  → status "done"
    status(...)  derived from the checkpoint — nothing is stored beside it

Every function opens and closes its own checkpointer connection; nothing is a long-lived object
(a Gradio callback may run on any thread). The user's edited résumé is copied verbatim to
`reviewed_resume.md` and is the only input of `finalize`; nothing compares it with the draft.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ConfigDict

from loomcv.graph import (
    JD_NAME,
    REVIEWED_NAME,
    PipelineState,
    build_graph,
    thread_config,
)
from loomcv.steps import InvalidInput, Runtime

_APP_ID = re.compile(r"[A-Za-z0-9._-]+")


class InvalidAppId(InvalidInput, ValueError):
    """An id is a directory name and a thread id: letters, digits, `.`, `_`, `-` only."""


class ApplicationExists(InvalidInput):
    """A start is never an overwrite: pick another id, or delete the directory by hand."""


class UnknownApplication(InvalidInput):
    """No checkpoint for the id under this `out_root`."""


class NotReviewable(InvalidInput):
    """The application is not at a point where a file can be submitted."""


class Status(BaseModel):
    """The one shape `start`, `submit` and `status` return."""

    model_config = ConfigDict(frozen=True)

    app_id: str
    # unknown | reviewing | finalizing | done | failed_at_match | failed_at_draft
    status: str
    app_dir: str | None = None
    resume_path: str | None = None
    report_path: str | None = None
    reviewed_path: str | None = None
    counts: dict[str, int] = {}
    final_pdf: str | None = None
    pages: int | None = None
    page_limit: int | None = None
    over_limit: bool | None = None
    typst_warnings: list[str] = []


@contextmanager
def _saver(checkpoints: Path) -> Iterator[SqliteSaver]:
    checkpoints.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoints), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()  # idempotent
        yield saver
    finally:
        conn.close()


def _check_id(app_id: str) -> str:
    if not _APP_ID.fullmatch(app_id):
        raise InvalidAppId(f"invalid application id {app_id!r}: use letters, digits, . _ - only")
    return app_id


def _is_here(values: dict[str, Any], app_id: str, out_root: Path) -> bool:
    """A thread belongs to this `out_root` when its recorded directory is `<out_root>/<app_id>`."""
    recorded = values.get("app_dir")
    return bool(recorded) and Path(recorded).resolve() == (out_root / app_id).resolve()


def _view(snapshot: Any, app_id: str, out_root: Path) -> Status:
    values: PipelineState = snapshot.values
    if not values or not _is_here(values, app_id, out_root):
        return Status(app_id=app_id, status="unknown")
    nxt = tuple(snapshot.next)
    if nxt == ("human_review",):
        state = "reviewing"
    elif nxt == ("finalize",):
        state = "finalizing"
    elif nxt == () and values.get("final_pdf"):
        state = "done"
    elif nxt in (("match",), ("draft",)):
        state = f"failed_at_{nxt[0]}"
    else:
        state = "unknown"
    return Status(
        app_id=app_id,
        status=state,
        app_dir=values.get("app_dir"),
        resume_path=values.get("resume_path"),
        report_path=values.get("report_path"),
        reviewed_path=values.get("reviewed_path"),
        counts=values.get("counts") or {},
        final_pdf=values.get("final_pdf"),
        pages=values.get("pages"),
        page_limit=values.get("page_limit"),
        over_limit=values.get("over_limit"),
        typst_warnings=values.get("typst_warnings") or [],
    )


def start(
    runtime: Runtime,
    jd_path: Path,
    *,
    app_id: str | None = None,
    out_root: Path,
    checkpoints: Path,
) -> Status:
    """Run match → draft and stop before the review. Nothing is written when the id is invalid or
    taken, or an input file is unreadable. Any later failure propagates (the thread keeps the last
    completed node; the id is spent — start again under a new one)."""
    app_id = _check_id(app_id or jd_path.stem)
    app_dir = out_root / app_id
    config = thread_config(app_id)
    with _saver(checkpoints) as saver:
        graph = build_graph(runtime, saver)
        found = [f"the folder {app_dir}"] if app_dir.exists() else []
        if saver.get_tuple(config) is not None:
            found.append(f"a checkpoint thread in {checkpoints}")
        if found:
            raise ApplicationExists(
                f"application {app_id} already exists — {' and '.join(found)}; "
                "pick a new name (--app-id)"
            )
        try:
            jd_bytes = jd_path.read_bytes()
        except OSError as exc:
            raise InvalidInput(f"invalid jd {jd_path}: {exc}") from exc
        # The directory and the JD copy come first, so a failed run leaves a readable record.
        # The header is the vault's: nothing else is copied in.
        try:
            app_dir.mkdir(parents=True)
            (app_dir / JD_NAME).write_bytes(jd_bytes)
        except OSError as exc:
            raise InvalidInput(f"cannot write {app_dir}: {exc}") from exc
        initial: PipelineState = {
            "app_id": app_id,
            "app_dir": str(app_dir),
            "jd_path": str(app_dir / JD_NAME),
            "review_confirmed": False,
            "reviewed_path": None,
            "final_md": None,
            "final_typ": None,
            "final_pdf": None,
            "pages": None,
            "page_limit": None,
            "over_limit": None,
            "typst_warnings": [],
        }
        graph.invoke(initial, config)
        return _view(graph.get_state(config), app_id, out_root)


def submit(
    runtime: Runtime,
    app_id: str,
    *,
    edited_path: Path,
    page_limit: int | None = None,
    out_root: Path,
    checkpoints: Path,
) -> Status:
    """Accept `edited_path` as the reviewed résumé, byte for byte, and run `finalize`. Allowed at
    `reviewing`, `finalizing` (a converter failed: re-submit) and `done` (a user-driven second
    pass: `finalize` re-runs over the new file). Over the page limit is a finding in the returned
    status (`over_limit`), not an error. Nothing is written until the application and the file
    have been checked."""
    _check_id(app_id)
    config = thread_config(app_id)
    with _saver(checkpoints) as saver:
        graph = build_graph(runtime, saver)
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise UnknownApplication(f"no application {app_id} in {checkpoints}")
        if not _is_here(snapshot.values, app_id, out_root):
            raise UnknownApplication(
                f"application {app_id} is at {snapshot.values.get('app_dir')}, not under {out_root}"
            )
        current = _view(snapshot, app_id, out_root).status
        if current not in ("reviewing", "finalizing", "done"):
            raise NotReviewable(f"application {app_id} is {current}; there is nothing to submit")
        try:
            edited = edited_path.read_bytes()
        except OSError as exc:
            raise InvalidInput(f"invalid résumé {edited_path}: {exc}") from exc
        reviewed = out_root / app_id / REVIEWED_NAME
        try:
            reviewed.write_bytes(edited)
        except OSError as exc:
            raise InvalidInput(f"cannot write {reviewed}: {exc}") from exc
        # Reset the results so a `finalizing` state never shows a previous run's PDF.
        graph.update_state(
            config,
            {
                "review_confirmed": True,
                "reviewed_path": str(reviewed),
                "page_limit": page_limit,
                "final_md": None,
                "final_typ": None,
                "final_pdf": None,
                "pages": None,
                "over_limit": None,
                "typst_warnings": [],
            },
            as_node="human_review" if current == "done" else None,
        )
        graph.invoke(None, config)
        return _view(graph.get_state(config), app_id, out_root)


def status(app_id: str, *, out_root: Path, checkpoints: Path) -> Status:
    """Where the application stands, read from the checkpoint."""
    _check_id(app_id)
    with _saver(checkpoints) as saver:
        # `get_state` runs no node, so the graph needs no adapters; `next` comes from its
        # topology, hence the graph. The Runtime is lazy and is never used.
        graph = build_graph(Runtime(db=Path(), chroma=Path()), saver)
        return _view(graph.get_state(thread_config(app_id)), app_id, out_root)
