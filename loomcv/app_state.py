"""The `pipeline app` frame's decisions: which tab the app lands on, whether the posting side
is open, and Start over.

Pure functions over paths — no UI import, no model call. The page
(`loomcv/ui/shell.py`) renders what these return and decides nothing itself.

- `landing` — the switch: the saved vault → *saved*, else the working copy →
  *started*, else *empty*. Existence only; no file is read.
- `readiness` — the gate: (a) at least one application folder; (b) the saved vault's
  fingerprint equals the one the last `pipeline sync` wrote and the index holds ≥ 1 bullet.
- `start_over` — deletes the two vault files (and the sidecar beside
  the working copy) and nothing else.
- `elapsed_text` — the veil's clock, formatting only. There are two tabs: My profile and
  Applications.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from loomcv.review.store import list_applications
from loomcv.sync.index_assets import vault_fingerprint

PROFILE_TAB = "My profile"
APPLICATIONS_TAB = "Applications"
GATE_LINE = "Finish and save your profile first"
STOPPED_LINE = "You stopped part-way; continue."
EMPTY_LINE = "No résumé yet."
START_OVER_QUESTION = "This deletes your profile. Your past applications stay. Continue?"


class ProfileState(StrEnum):
    EMPTY = "empty"
    STARTED = "started"
    SAVED = "saved"


@dataclass(frozen=True)
class Landing:
    state: ProfileState
    tab: str  # PROFILE_TAB | APPLICATIONS_TAB
    page_over: Path | None  # the file the My profile page is built over; None when EMPTY
    line: str  # EMPTY_LINE | STOPPED_LINE | ""


def landing(working_copy: Path, saved: Path) -> Landing:
    """Saved exists → SAVED (APPLICATIONS_TAB, no line), else working copy exists →
    STARTED (PROFILE_TAB, built over `working_copy`, STOPPED_LINE), else EMPTY (PROFILE_TAB,
    None, EMPTY_LINE). Existence only; contents never read.

    The page is built over the working copy whenever one
    exists — in the SAVED state too, since every edit writes it — else over `saved`."""
    if saved.is_file():
        over = working_copy if working_copy.is_file() else saved
        return Landing(ProfileState.SAVED, APPLICATIONS_TAB, over, "")
    if working_copy.is_file():
        return Landing(ProfileState.STARTED, PROFILE_TAB, working_copy, STOPPED_LINE)
    return Landing(ProfileState.EMPTY, PROFILE_TAB, None, EMPTY_LINE)


@dataclass(frozen=True)
class Readiness:
    existing: bool  # (a) at least one application folder (`list_applications` non-empty)
    ready: bool  # (b) saved vault exists, index fingerprint equals it, ≥ 1 bullet row
    line: str  # "" when `ready`, else GATE_LINE (Start is refused whenever not ready)

    @property
    def posting_side_open(self) -> bool:
        return self.existing or self.ready


def readiness(saved: Path, *, db: Path, out_root: Path, checkpoints: Path) -> Readiness:
    """(b) reads the index read-only (`sqlite3` URI `mode=ro`, so a missing
    database is never created): the fingerprint row against `vault_fingerprint` of the saved
    vault's text, then whether `bullets` has a row. No database, no row, a different hash, an
    empty `bullets`, or an `sqlite3.Error` → not ready."""
    existing = bool(list_applications(out_root, checkpoints))
    ready = _index_built_from(saved, db)
    return Readiness(existing=existing, ready=ready, line="" if ready else GATE_LINE)


def _index_built_from(saved: Path, db: Path) -> bool:
    if not saved.is_file() or not db.is_file():
        return False
    expected = vault_fingerprint(saved.read_text(encoding="utf-8"))
    try:
        conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'vault_sha256'").fetchone()
        if row is None or row[0] != expected:
            return False
        return conn.execute("SELECT EXISTS(SELECT 1 FROM bullets)").fetchone()[0] == 1
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def start_over(working_copy: Path, saved: Path, sidecar: Path) -> list[Path]:
    """Unlink the two vault files and the working copy's sidecar
    (`missing_ok=True`), nothing else; returns the ones that existed."""
    removed: list[Path] = []
    for path in (working_copy, saved, sidecar):
        if path.is_file():
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def elapsed_text(seconds: float) -> str:
    """`m:ss`, floor, minutes unbounded (0 → "0:00", 59.9 → "0:59", 3665 → "61:05");
    a negative input reads as 0. Formatting only (kept here, not in a UI module,
    so pytest covers it)."""
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def saved_stamp(when: datetime, now: datetime | None = None) -> str:
    """When something was saved, for a save slot:
    `HH:MM` on the day of `now`; on another day `Mon D HH:MM`, and `Mon D YYYY HH:MM` in another
    year — so an old save never reads as tonight's. Formatting only."""
    now = now or datetime.now()
    hhmm = f"{when:%H:%M}"
    if when.date() == now.date():
        return hhmm
    day = f"{_MONTHS[when.month - 1]} {when.day}"
    return f"{day} {hhmm}" if when.year == now.year else f"{day} {when.year} {hhmm}"
