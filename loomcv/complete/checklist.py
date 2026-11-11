"""The checklist: what is still open in a `Workbook`.

Four kinds, per job in workbook order: `job_field` (an Employer / Title that is empty, a From /
To that is `0000`), `job_defaults` (the job's `Defaults` not confirmed), then per bullet
`bullet_draft` (still a draft) or `bullet_keywords` (confirmed, but its `Keywords` list is empty —
it would have no keyword side in the funnel). The vault-level `Skills` are not an item: the
unassigned ones are the unchecked boxes of stage A and `skills_unassigned`. A bullet-less job
yields no `bullet_*` item. Pure.

The header items come first, in field order — `header_field` when
`Name` or `Contact` is empty (a résumé without them is not sendable), `header_optional` when
`Headline`, `Summary` or `Education` is; no item blocks a write. They have no `job_id`.

`open_counts` is a filter over `checklist()` — what still
needs the user's eyes, per page section (About you, each job, Your skills). Pure.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from loomcv.complete.workbook import Workbook


class ChecklistItem(BaseModel):
    kind: Literal[
        "header_field",
        "header_optional",
        "job_field",
        "job_defaults",
        "bullet_keywords",
        "bullet_draft",
    ]
    job_id: str | None = None  # None for the header items
    bullet_id: str | None = None  # "<job_id>::<id>"
    message: str


def checklist(wb: Workbook) -> list[ChecklistItem]:
    items: list[ChecklistItem] = []
    header = wb.header
    for label, value, required in (
        ("Name", header.name, True),
        ("Contact", header.contact, True),
        ("Headline", header.headline, False),
        ("Summary", header.summary, False),
        ("Education", header.education, False),
    ):
        if not value:
            items.append(
                ChecklistItem(kind="header_field", message=f"{label} is empty")
                if required
                else ChecklistItem(kind="header_optional", message=f"{label} is empty (optional)")
            )
    for job in wb.jobs:
        for label, value in (("Employer", job.employer), ("Title", job.title)):
            if value is None:
                items.append(
                    ChecklistItem(kind="job_field", job_id=job.job_id, message=f"{label} is empty")
                )
        for label, value in (("From", job.date_from), ("To", job.date_to)):
            if value == "0000":
                items.append(
                    ChecklistItem(
                        kind="job_field",
                        job_id=job.job_id,
                        message=f"{label} is 0000 (unknown)",
                    )
                )
        if not job.defaults_confirmed:
            items.append(
                ChecklistItem(
                    kind="job_defaults", job_id=job.job_id, message="Defaults not confirmed"
                )
            )
        for bullet in job.bullets:
            key = f"{job.job_id}::{bullet.id}"
            if bullet.status == "draft":
                items.append(
                    ChecklistItem(
                        kind="bullet_draft",
                        job_id=job.job_id,
                        bullet_id=key,
                        message="bullet is a draft",
                    )
                )
            elif not bullet.keywords:
                items.append(
                    ChecklistItem(
                        kind="bullet_keywords",
                        job_id=job.job_id,
                        bullet_id=key,
                        message="confirmed bullet has no Keywords",
                    )
                )
    return items


class OpenCounts(BaseModel):
    header: int  # `header_field` items only (Name / Contact empty)
    jobs: dict[str, int]  # every job id, workbook order, 0 included
    skills: int  # 1 when the Skills list is empty, else 0


def open_counts(wb: Workbook) -> OpenCounts:
    """What still needs the user's eyes, per section. A filter
    over `checklist(wb)`: a job's count = its items (`job_field`, `job_defaults`,
    `bullet_draft`, `bullet_keywords` — every item that carries a `job_id`); the header's =
    the `header_field` items (`header_optional` never counts); skills = 0 when the Skills list
    has an item, else 1. Pure."""
    items = checklist(wb)
    per_job = {job.job_id: 0 for job in wb.jobs}
    for item in items:
        if item.job_id is not None:
            per_job[item.job_id] += 1
    return OpenCounts(
        header=sum(1 for item in items if item.kind == "header_field"),
        jobs=per_job,
        skills=0 if wb.skills else 1,
    )


def confirmed_counts(wb: Workbook) -> dict[str, tuple[int, int]]:
    """Every job id → `(confirmed bullets, bullets)`, workbook order (the
    left list's "x of n confirmed"). A bullet counts as confirmed by its
    status only — its keywords do not enter the count. A bullet-less job is `(0, 0)`.
    Pure."""
    return {
        job.job_id: (sum(1 for b in job.bullets if b.status == "confirmed"), len(job.bullets))
        for job in wb.jobs
    }
