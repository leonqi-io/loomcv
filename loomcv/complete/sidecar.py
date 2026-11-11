"""The working copy's sidecar.

The vault grammar has no marker for a job's `Defaults` being confirmed, so a
reload of the working copy would untick every job. A small JSON file beside the working copy —
`{job_id: true|false}` and nothing else — carries it. This module is the disk side of that and
of the per-commit working-copy write; `workbook.py` stays free of I/O except `write_workbook`.

No UI import, no model call. A malformed sidecar reads as absent with one warning, never an
exception; an `OSError` from a write propagates (the page shows it and keeps going).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from loomcv.complete.workbook import Workbook, WriteReport, load_workbook, write_workbook


def sidecar_for(working_copy: Path) -> Path:
    """`<dir>/<stem>.state.json` — `02_database/draft_vault.md` → `…/draft_vault.state.json`."""
    return working_copy.with_name(f"{working_copy.stem}.state.json")


def write_sidecar(wb: Workbook, path: Path) -> None:
    """`{job_id: defaults_confirmed}` for every job, workbook order, a JSON object of booleans
    and nothing else (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {job.job_id: job.defaults_confirmed for job in wb.jobs}
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def read_sidecar(path: Path) -> tuple[dict[str, bool], list[str]]:
    """Missing file → `({}, [])`. Not JSON, not an object, or a value that is not a bool →
    `({}, [one warning])`; never an exception."""
    if not path.is_file():
        return {}, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        return {}, [f"{path.name} could not be read ({err}); every job reads as not confirmed."]
    if not isinstance(data, dict) or not all(isinstance(v, bool) for v in data.values()):
        return {}, [
            f"{path.name} is not an object of true / false values; "
            "every job reads as not confirmed."
        ]
    return dict(data), []


def with_confirmed(wb: Workbook, confirmed: dict[str, bool]) -> Workbook:
    """Each job's `defaults_confirmed` = `confirmed.get(job_id, False)`; ids not in the
    workbook are ignored."""
    jobs = [
        job.model_copy(update={"defaults_confirmed": confirmed.get(job.job_id, False)})
        for job in wb.jobs
    ]
    return wb.model_copy(update={"jobs": jobs})


def load_profile(path: Path, *, working_copy: Path, sidecar: Path) -> tuple[Workbook, list[str]]:
    """`load_workbook` of `path`; then, when `path` is the working copy (`resolve()` equal), the
    sidecar decides each job (absent file → every job False); otherwise — the saved vault —
    every job True (its Defaults were accepted at save) and the sidecar is not read. A parser
    error raises `CompleteError`."""
    wb = load_workbook(path.read_text(encoding="utf-8"))
    if path.resolve() == working_copy.resolve():
        confirmed, warnings = read_sidecar(sidecar)
        return with_confirmed(wb, confirmed), warnings
    return with_confirmed(wb, {job.job_id: True for job in wb.jobs}), []


def save_working_copy(
    wb: Workbook, working_copy: Path, *, sidecar: Path, today: date
) -> WriteReport:
    """`write_workbook(wb, working_copy, today=today)` then `write_sidecar`. An
    `OSError` propagates; the report's `problems` never block."""
    report = write_workbook(wb, working_copy, today=today)
    write_sidecar(wb, sidecar)
    return report
