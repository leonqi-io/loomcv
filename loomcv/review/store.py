"""The review document's disk and index side.

The one module under `loomcv/review/` that touches the application directory, and a caller of
`loomcv.application` like the CLI: `open_review` reads the application (the match payload, the
index — the header and the Skills groups are the vault's —, the saved state),
`save_review` writes the user's version beside the
draft — `review.json` (the document) and `edited_resume.md` (its rendering, the file `submit`
takes) — and nothing else. It never writes `tailored_resume.md`, `match_report.md`,
`reviewed_resume.md` or any `final_*` file, and deletes nothing.
`start_application` also writes the user's name for a new application (`name.txt`), and the left
list's facts (`started_at`, `read_name`, `taken_ids`, `list_entries`) are read here.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from loomcv import application
from loomcv.application import (
    InvalidAppId,
    NotReviewable,
    Status,
    UnknownApplication,
)
from loomcv.coverage import CoverageError, StoredCoverageError
from loomcv.draft.render import REPORT_NAME
from loomcv.graph import JD_NAME, MATCH_NAME, PROFILE_NAME
from loomcv.parse_jd import JDParseError
from loomcv.retrieve.rank import Candidate, MatchPayload
from loomcv.retrieve.search import (
    JobRow,
    load_candidates,
    load_job_rows,
    load_skill_groups,
    load_vault_header,
)
from loomcv.review.doc import ReviewDoc, from_draft, render
from loomcv.state import resolve_config_file
from loomcv.steps import InvalidInput, NodeCallError, Runtime

STATE_NAME = "review.json"
EDITED_NAME = "edited_resume.md"
NAME_FILE = "name.txt"  # the user's name for the application

# The top-level keys of an older `review.json` shape that the header fields replaced.
OLD_SHAPE_KEYS = frozenset({"header", "skills"})
OLD_SHAPE_NOTICE = (
    "This posting was saved before the résumé fields moved into your vault; it was rebuilt "
    "from the draft — earlier edits are not shown, and Save replaces the old file."
)

# What the page shows as text instead of a traceback: bad input, a failed node call, a missing
# index, an unreadable file. (`InvalidInput` covers every `loomcv.application` refusal.)
FAILURES: tuple[type[BaseException], ...] = (
    InvalidInput,
    NodeCallError,
    StoredCoverageError,
    CoverageError,
    JDParseError,
    ValueError,
    RuntimeError,
    OSError,
)


@dataclass(frozen=True, eq=False)  # identity hash: a `gr.State` value must be hashable
class Opened:
    app_id: str
    out_root: Path
    app_dir: Path
    doc: ReviewDoc
    payload: MatchPayload
    candidates: list[Candidate]
    rows: list[JobRow]
    status: Status
    report: str  # match_report.md, verbatim
    template_path: Path
    notice: str | None = None  # a one-line message for the page (an old-shape review.json)


@dataclass(frozen=True)
class Saved:
    state_path: Path
    resume_path: Path


def open_review(runtime: Runtime, app_id: str, *, out_root: Path, checkpoints: Path) -> Opened:
    """The application as the review page needs it: the saved document when there is one, else
    the document built from the draft's inputs. Writes nothing. A saved document is the user's
    and is not compared with the index (the vault may have moved on)."""
    if not app_id:
        raise UnknownApplication("Pick a posting first.")
    state = application.status(app_id, out_root=out_root, checkpoints=checkpoints)
    if state.status == "unknown":
        raise UnknownApplication(f"no application {app_id} under {out_root}")
    if state.status not in ("reviewing", "finalizing", "done"):
        raise NotReviewable(f"application {app_id} is {state.status}; there is nothing to review")
    app_dir = out_root / app_id

    match_path = app_dir / MATCH_NAME
    try:
        payload = MatchPayload.model_validate_json(match_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise InvalidInput(f"invalid {match_path}: {exc}") from exc
    candidates = load_candidates(runtime.db)
    rows = load_job_rows(runtime.db)
    drafted = from_draft(
        payload, rows, load_skill_groups(runtime.db), load_vault_header(runtime.db)
    )

    state_path = app_dir / STATE_NAME
    notice = None
    if state_path.is_file():
        try:
            text = state_path.read_text(encoding="utf-8")
            saved = json.loads(text)
            if isinstance(saved, dict) and (OLD_SHAPE_KEYS & saved.keys()):
                # A document saved in the older shape: rebuilt, never migrated.
                doc, notice = drafted, OLD_SHAPE_NOTICE
            else:
                doc = ReviewDoc.model_validate_json(text)
        except (OSError, ValueError, ValidationError) as exc:
            raise InvalidInput(f"invalid {state_path}: {exc}") from exc
    else:
        doc = drafted

    report_path = app_dir / REPORT_NAME
    return Opened(
        app_id=app_id,
        out_root=out_root,
        app_dir=app_dir,
        doc=doc,
        payload=payload,
        candidates=candidates,
        rows=rows,
        status=state,
        report=report_path.read_text(encoding="utf-8") if report_path.is_file() else "",
        template_path=resolve_config_file(runtime.generation().resume_template),
        notice=notice,
    )


def save_review(doc: ReviewDoc, app_id: str, *, out_root: Path, template_path: Path) -> Saved:
    """Write `edited_resume.md` and `review.json` into the application directory. The Markdown is
    rendered first, so a template error leaves the directory as it was."""
    resume = render(doc, template_path)
    app_dir = out_root / app_id
    if not app_dir.is_dir() or app_dir.resolve().parent != out_root.resolve():
        raise UnknownApplication(f"no application {app_id} under {out_root}")
    resume_path, state_path = app_dir / EDITED_NAME, app_dir / STATE_NAME
    try:
        resume_path.write_text(resume, encoding="utf-8", newline="")
        state_path.write_text(doc.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="")
    except OSError as exc:
        raise InvalidInput(f"cannot write {app_dir}: {exc}") from exc
    return Saved(state_path=state_path, resume_path=resume_path)


def _pages(count: int) -> str:
    return f"{count} page" if count == 1 else f"{count} pages"


def stage_label(status: Status) -> str:
    """Where an application stands, in the page's words: no stage token, no
    counter line; a status the page does not know reads `(no record)`."""
    state = status.status
    if state == "reviewing":
        return "Draft ready — review it"
    if state == "done":
        return "✓ PDF ready" if status.pages is None else f"✓ PDF ready · {_pages(status.pages)}"
    if state == "finalizing":
        return "PDF not finished — make it again"
    if state.startswith("failed_at_"):
        return "Couldn't build a draft — start a new application"
    return "(no record)"


def posting_title(app_dir: Path) -> str:
    """The job title `parse_jd` read from the posting (`profile.json`); the folder's name when
    the profile is missing, unreadable or has no title."""
    try:
        title = json.loads((app_dir / PROFILE_NAME).read_text(encoding="utf-8")).get("title")
    except (OSError, ValueError, AttributeError):
        return app_dir.name
    return title.strip() if isinstance(title, str) and title.strip() else app_dir.name


def started_at(app_dir: Path) -> datetime | None:
    """When the application was started: the modification time of its JD copy (`jd.md`, the
    first file `application.start` writes), local time; `None` when there is no copy."""
    try:
        return datetime.fromtimestamp((app_dir / JD_NAME).stat().st_mtime)
    except OSError:
        return None


def read_name(app_dir: Path) -> str | None:
    """The name the user gave the application; `None` when the file is absent,
    empty or unreadable — an application started without a name, or by `pipeline run`."""
    try:
        text = (app_dir / NAME_FILE).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


def taken_ids(out_root: Path, checkpoints: Path) -> set[str]:
    """Every name under `out_root` and every thread id in the checkpoints file — the ids a start
    would refuse. The thread ids are read with the stdlib `sqlite3`, read-only (no `langgraph`
    import under `loomcv/review/`); a missing or unreadable file gives the folder names alone.
    Writes and creates nothing."""
    taken = {entry.name for entry in out_root.iterdir()} if out_root.is_dir() else set()
    if not checkpoints.is_file():
        return taken
    try:
        conn = sqlite3.connect(f"{checkpoints.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return taken
    return taken | {str(row[0]) for row in rows}


@dataclass(frozen=True, eq=False)  # identity hash: kept in the page's state, like `Opened`
class ListEntry:
    """One application as the page's left list needs it."""

    app_id: str
    title: str  # `posting_title`
    name: str | None  # `read_name`
    started: datetime | None  # `started_at`
    status: Status


def list_entries(out_root: Path, checkpoints: Path) -> list[ListEntry]:
    """Every application under `out_root`, newest first by `started_at`; one with no JD copy last;
    ties by id."""
    entries = [
        ListEntry(
            app_id=s.app_id,
            title=posting_title(out_root / s.app_id),
            name=read_name(out_root / s.app_id),
            started=started_at(out_root / s.app_id),
            status=s,
        )
        for s in list_applications(out_root, checkpoints)
    ]
    dated = sorted(
        (e for e in entries if e.started is not None),
        key=lambda e: (-e.started.timestamp(), e.app_id),
    )
    return dated + sorted((e for e in entries if e.started is None), key=lambda e: e.app_id)


def list_applications(out_root: Path, checkpoints: Path) -> list[Status]:
    """One `status()` per subdirectory of `out_root`, sorted by name; a directory whose name is
    not a valid id is skipped, one with no thread shows `unknown`."""
    if not out_root.is_dir():
        return []
    rows: list[Status] = []
    for entry in sorted(out_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            rows.append(application.status(entry.name, out_root=out_root, checkpoints=checkpoints))
        except InvalidAppId:
            continue
    return rows


def start_application(
    runtime: Runtime,
    jd_text: str,
    app_id: str,
    *,
    out_root: Path,
    checkpoints: Path,
    name: str | None = None,
) -> Status:
    """`application.start` over pasted JD text: the text goes to a temporary file `start` copies
    into the application directory. The id is required on the page; the header is the vault's.
    With `name`, the user's name for the application is written
    to `name.txt` in the directory this call created — after a return, and also before a failure
    that left the directory is re-raised (R1); never into a directory that existed before.

    After a successful start the freshly opened document is saved once (`save_review`: the
    build-time `review.json` + `edited_resume.md`), so an
    application never edited afterwards keeps its own name, jobs and bullets when the profile
    changes, and the index-mismatch line can fire. A failure there is raised like the start's."""
    if not app_id.strip():
        raise InvalidAppId("give the application an id (letters, digits, . _ -)")
    app_dir = out_root / app_id.strip()
    existed = app_dir.exists()
    with tempfile.TemporaryDirectory() as tmp:
        jd_path = Path(tmp) / "jd.md"
        jd_path.write_text(jd_text, encoding="utf-8", newline="")
        try:
            status = application.start(
                runtime,
                jd_path,
                app_id=app_id.strip(),
                out_root=out_root,
                checkpoints=checkpoints,
            )
        except BaseException:
            if not existed and app_dir.is_dir():
                with contextlib.suppress(InvalidInput):  # the start's own failure is the news
                    _write_name(app_dir, name)
            raise
    _write_name(app_dir, name)
    if status.status == "reviewing":
        built = open_review(runtime, app_id.strip(), out_root=out_root, checkpoints=checkpoints)
        save_review(built.doc, built.app_id, out_root=out_root, template_path=built.template_path)
    return status


def _write_name(app_dir: Path, name: str | None) -> None:
    if name is None or not name.strip():
        return
    try:
        (app_dir / NAME_FILE).write_text(name.strip() + "\n", encoding="utf-8", newline="")
    except OSError as exc:
        raise InvalidInput(f"cannot write {app_dir}: {exc}") from exc
