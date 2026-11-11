"""The `Workbook`: the model the completion loop works on.

A `Workbook` is built from a draft vault (`parse_star_vault(text, include_draft=True)`) and
changed only by pure functions that return a new one: `apply_jobs` (stage A — job fields,
`Defaults`, skills assigned from the vault-level list), `apply_bullets` (stage C — keywords,
text, project, confirm) and, between them, `loomcv.complete.select.select_keywords` (stage B).
`render_workbook` writes it back as a v3 vault through the L1 renderer; `verify_rendered` is the
acceptance oracle (the strict parser, twice); `write_workbook` is render + write + verify.

What the loop never does: create, recompute or renumber an `{id: …}`, rename a job
(`bullet_id = <job_slug>::<id>`, so the heading stays), or reorder jobs / bullets — except
`add_bullet`, which creates one id by ingest's rule for a bullet that
starts from nothing, and `delete_bullet` (the one removal, by the
user's own click after a confirmation). A bullet's
text may be edited (the id stays, the sync layer re-embeds it). `defaults_confirmed` lives here
only — the vault grammar has no marker for it. No I/O except `write_workbook`; no
model call; no UI.

Stage D is `apply_header` (the vault-level header lines) and
`apply_skill_groups` (the Skills groups), with `skills_prefill` / `with_prefill` (the
three-source union offered in the unnamed group); `Workbook.skills` is now the groups' flat
union, read-only. The grouping proposal (a model call) is `loomcv.complete.grouping`.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from loomcv.ingest.extract import _sha6
from loomcv.ingest.keywords import _clean_items, normalise_ws
from loomcv.ingest.render import DraftBullet, DraftJob, DraftVault, render_vault
from loomcv.sync.parse_assets import SkillGroup, VaultHeader, flat_skills, parse_star_vault

# Mirrors `parse_assets._DATE_VALUE_RE` (private); `verify_rendered` is the oracle.
_DATE_RE = re.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2]))?$")


class CompleteError(Exception):
    """A user-input or draft problem, or the selection call failing after its one retry.

    `raws` holds every model reply received so far, in call order (as `IngestError`)."""

    def __init__(self, message: str, raws: list[str] | None = None) -> None:
        super().__init__(message)
        self.raws = list(raws or [])


class WbBullet(BaseModel):
    id: str  # the local id — never created or recomputed here
    text: str
    project: str | None = None
    keywords: list[str] = []
    status: Literal["draft", "confirmed"] = "draft"
    updated: str | None = None  # ISO date; None = stamp today at render


class WbJob(BaseModel):
    job_id: str  # `ParsedJob.job_id` (the slug of `name`)
    name: str
    employer: str | None = None
    title: str | None = None
    date_from: str
    date_to: str
    defaults: list[str] = []
    defaults_confirmed: bool = False  # Workbook only; never in the vault
    bullets: list[WbBullet] = []


class Workbook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    header: VaultHeader = VaultHeader()  # stage D
    skill_groups: list[SkillGroup] = []  # the unnamed group (name "") first, when there is one
    jobs: list[WbJob]

    @property
    def skills(self) -> list[str]:
        """The vault-level skills as one flat list (read-only)."""
        return flat_skills(self.skill_groups)


class HeaderAnswers(BaseModel):
    """Stage D, the header: every key optional; a key not set (`model_fields_set`) leaves the
    field alone."""

    name: str | None = None
    headline: str | None = None
    contact: list[str] | None = None
    summary: str | None = None
    education: list[str] | None = None


class SkillsAnswers(BaseModel):
    """Stage D, the Skills groups: the whole list, in print order, as the user left it."""

    groups: list[SkillGroup]


class JobAnswer(BaseModel):
    """Every key optional; a key not set (`model_fields_set`) leaves the job's value alone."""

    employer: str | None = None
    title: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    defaults: list[str] | None = None
    assign_skills: list[str] = []
    defaults_confirmed: bool | None = None


class JobsAnswers(BaseModel):
    jobs: dict[str, JobAnswer] = {}  # key = `WbJob.job_id`


class BulletAnswer(BaseModel):
    """Every key optional; `project=None` set explicitly clears the label."""

    keywords: list[str] | None = None
    text: str | None = None
    project: str | None = None
    confirm: bool | None = None  # True → confirmed, False → back to draft


class BulletsAnswers(BaseModel):
    bullets: dict[str, BulletAnswer] = {}  # key = "<job_id>::<id>"


class BulletDraftAnswers(BaseModel):
    """The three guided answers of the drafting call; each may be empty."""

    model_config = ConfigDict(extra="forbid")

    situation: str = ""
    action: str = ""
    result: str = ""


class Groups(BaseModel):
    """What the UI shows on a bullet's card: computed, never stored."""

    selected_defaults: list[str]
    unselected_defaults: list[str]
    own: list[str]


class WriteReport(BaseModel):
    path: str
    jobs: int
    jobs_without_bullets: int
    bullets: int
    confirmed: int  # bullets in the default (draft-excluding) parse of the written text
    skills_unassigned: int
    problems: list[str]  # "line N: msg" per parse error, plus a confirmed-count mismatch line


# ----------------------------------------------------------------------------- load


def empty_workbook() -> Workbook:
    """No job, no header, no skills: the profile page before a résumé arrives."""
    return Workbook(jobs=[])


def load_workbook(text: str) -> Workbook:
    """The draft vault text → a `Workbook`. Any parser error raises `CompleteError` — a partial
    parse would drop bullets or jobs silently. Jobs and bullets keep file order (the
    parser sorts by date)."""
    result = parse_star_vault(text, include_draft=True)
    errors = result.errors()
    if errors:
        raise CompleteError(
            "draft has parse errors: " + "; ".join(f"line {i.line_no}: {i.message}" for i in errors)
        )
    jobs: list[WbJob] = []
    for job in sorted(result.jobs, key=lambda j: j.line_no):
        bullets = [
            WbBullet(
                id=b.bullet_id.removeprefix(f"{job.job_id}::"),
                # The parser strips leading `[Tag]`s; keep them, so a hand-edited vault round-trips.
                text=("".join(f"[{t}]" for t in b.tags) + " " + b.text) if b.tags else b.text,
                project=b.project,
                keywords=list(b.keywords),
                status=b.status,
                updated=b.updated.isoformat() if b.updated else None,
            )
            for b in job.bullets
        ]
        jobs.append(
            WbJob(
                job_id=job.job_id,
                name=job.name,
                employer=job.employer,
                title=job.title,
                date_from=job.date_from,
                date_to=job.date_to,
                defaults=list(job.defaults),
                bullets=bullets,
            )
        )
    return Workbook(header=result.header, skill_groups=list(result.skill_groups), jobs=jobs)


# ----------------------------------------------------------------------------- helpers


def _optional(value: str | None) -> str | None:
    return None if value is None else (normalise_ws(value) or None)


def _checked_items(items: list[str], what: str) -> list[str]:
    """User-typed list items: an item containing `;` is a `CompleteError` (it would split on the
    next parse); otherwise whitespace-normalised, empties dropped, casefold-deduplicated."""
    for item in items:
        if ";" in item:
            raise CompleteError(f"{what} item '{normalise_ws(item)}' contains ';'")
    kept, _ = _clean_items(items)
    return kept


def _date(value: str, key: str, *, allow_present: bool) -> str:
    text = value.strip()
    if _DATE_RE.match(text) or (allow_present and text == "present"):
        return text
    expected = "YYYY, YYYY-MM or present" if allow_present else "YYYY or YYYY-MM"
    raise CompleteError(f"invalid {key} '{text}' (expected {expected})")


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item.casefold() not in seen:
            seen.add(item.casefold())
            out.append(item)
    return out


# ----------------------------------------------------------------------------- stage A


def apply_jobs(wb: Workbook, answers: JobsAnswers) -> Workbook:
    """Stage A: set job fields, replace `Defaults`, append assigned skills, mark confirmed.

    A default removed from the job (old − new, case-insensitive) is removed from
    every bullet's `keywords` of that job; an added one touches no bullet — it appears in each
    bullet's unselected group (`groups`). Pure; an unknown job id, a malformed date, an unknown
    skill or a `;` inside an item raises `CompleteError`."""
    known = {j.job_id for j in wb.jobs}
    for job_id in answers.jobs:
        if job_id not in known:
            raise CompleteError(f"unknown job id '{job_id}'")
    skill_by_fold = {s.casefold(): s for s in wb.skills}
    jobs: list[WbJob] = []
    for job in wb.jobs:
        answer = answers.jobs.get(job.job_id)
        if answer is None:
            jobs.append(job)
            continue
        given = answer.model_fields_set
        update: dict = {}
        if "employer" in given:
            update["employer"] = _optional(answer.employer)
        if "title" in given:
            update["title"] = _optional(answer.title)
        if answer.date_from is not None:
            update["date_from"] = _date(answer.date_from, "From", allow_present=False)
        if answer.date_to is not None:
            update["date_to"] = _date(answer.date_to, "To", allow_present=True)

        defaults = (
            _checked_items(answer.defaults, "Defaults")
            if answer.defaults is not None
            else list(job.defaults)
        )
        for requested in answer.assign_skills:
            skill = skill_by_fold.get(normalise_ws(requested).casefold())
            if skill is None:
                raise CompleteError(f"'{requested}' is not on the vault's Skills line")
            defaults.append(skill)
        defaults = _dedup(defaults)
        update["defaults"] = defaults
        if answer.defaults_confirmed is not None:
            update["defaults_confirmed"] = answer.defaults_confirmed

        removed = {d.casefold() for d in job.defaults} - {d.casefold() for d in defaults}
        if removed:
            bullets = []
            for bullet in job.bullets:
                kept = [k for k in bullet.keywords if k.casefold() not in removed]
                if len(kept) != len(bullet.keywords):
                    bullet = bullet.model_copy(update={"keywords": kept, "updated": None})
                bullets.append(bullet)
            update["bullets"] = bullets
        jobs.append(job.model_copy(update=update))
    return wb.model_copy(update={"jobs": jobs})


# ----------------------------------------------------------------------------- stage C


def apply_bullets(wb: Workbook, answers: BulletsAnswers) -> Workbook:
    """Stage C: set a bullet's keywords / text / project and confirm it.

    Pure; an unknown key (`<job_id>::<id>`), empty text, a brace in the text or a `;` inside a
    keyword raises `CompleteError`. The id never changes. Any value that actually changes resets
    `updated` (the render stamps today); confirming an unchanged confirmed bullet keeps its
    date."""
    index = {f"{j.job_id}::{b.id}" for j in wb.jobs for b in j.bullets}
    for key in answers.bullets:
        if key not in index:
            raise CompleteError(f"unknown bullet '{key}'")
    jobs: list[WbJob] = []
    for job in wb.jobs:
        bullets: list[WbBullet] = []
        for bullet in job.bullets:
            answer = answers.bullets.get(f"{job.job_id}::{bullet.id}")
            if answer is None:
                bullets.append(bullet)
                continue
            update: dict = {}
            if answer.keywords is not None:
                update["keywords"] = _checked_items(answer.keywords, "keyword")
            if answer.text is not None:
                text = normalise_ws(answer.text)
                if not text:
                    raise CompleteError(f"bullet '{job.job_id}::{bullet.id}': the text is empty")
                if "{" in text or "}" in text:
                    raise CompleteError(
                        f"bullet '{job.job_id}::{bullet.id}': the vault grammar reserves {{}}"
                    )
                update["text"] = text
            if "project" in answer.model_fields_set:
                update["project"] = _optional(answer.project)
            if answer.confirm is not None:
                update["status"] = "confirmed" if answer.confirm else "draft"
            changed = {k: v for k, v in update.items() if getattr(bullet, k) != v}
            if changed:
                bullet = bullet.model_copy(update={**changed, "updated": None})
            bullets.append(bullet)
        jobs.append(job.model_copy(update={"bullets": bullets}))
    return wb.model_copy(update={"jobs": jobs})


def add_bullet(
    wb: Workbook, job_id: str, text: str, *, project: str | None = None
) -> tuple[Workbook, str]:
    """A bullet that starts from nothing → `(the workbook, "<job_id>::<id>")`.

    Pure; an unknown job, empty text or a brace in the text raises `CompleteError`. The id is
    L1's rule (`sha1(<normalised text>)[:6]`, the ordinal rule on a collision inside the job);
    no existing id changes. The bullet is appended last in its job as a `draft` with no keyword:
    stage B gives it keywords on its next run."""
    if all(job.job_id != job_id for job in wb.jobs):
        raise CompleteError(f"unknown job '{job_id}'")
    norm = normalise_ws(text)
    if not norm:
        raise CompleteError(f"job '{job_id}': the new bullet's text is empty")
    if "{" in norm or "}" in norm:
        raise CompleteError(f"job '{job_id}': the vault grammar reserves {{}}")

    jobs: list[WbJob] = []
    key = ""
    for job in wb.jobs:
        if job.job_id != job_id:
            jobs.append(job)
            continue
        used = {b.id for b in job.bullets}
        ident = _sha6(norm)
        ordinal = 2
        while ident in used:
            ident = _sha6(f"{norm}\n{ordinal}")
            ordinal += 1
        bullet = WbBullet(id=ident, text=norm, project=_optional(project), status="draft")
        jobs.append(job.model_copy(update={"bullets": [*job.bullets, bullet]}))
        key = f"{job_id}::{ident}"
    return wb.model_copy(update={"jobs": jobs}), key


def delete_bullet(wb: Workbook, job_id: str, bullet_id: str) -> Workbook:
    """Remove one bullet (`bullet_id` = the local id, `WbBullet.id`) by the user's click
    (the one removal; the loop otherwise never drops a bullet).

    Pure; every other bullet, job, id, the header and the skill groups are untouched, and a job
    left with no bullet stays (a bullet-less job is a parser warning, not an error). An unknown
    job or bullet raises `CompleteError`."""
    if all(job.job_id != job_id for job in wb.jobs):
        raise CompleteError(f"unknown job '{job_id}'")
    jobs: list[WbJob] = []
    found = False
    for job in wb.jobs:
        if job.job_id != job_id:
            jobs.append(job)
            continue
        kept = [b for b in job.bullets if b.id != bullet_id]
        found = len(kept) != len(job.bullets)
        jobs.append(job.model_copy(update={"bullets": kept}))
    if not found:
        raise CompleteError(f"job '{job_id}': unknown bullet '{bullet_id}'")
    return wb.model_copy(update={"jobs": jobs})


# `layout` below: ((job_id, ("<job_id>::<id>", …)), …)
Layout = tuple[tuple[str, tuple[str, ...]], ...]


def layout(wb: Workbook) -> Layout:
    """The page's shape: every job id in workbook order, each
    with its bullet keys (`<job_id>::<id>`) in order. Nothing else — no text, status, keyword or
    `defaults_confirmed`, so a field commit never changes it."""
    return tuple(
        (job.job_id, tuple(f"{job.job_id}::{b.id}" for b in job.bullets)) for job in wb.jobs
    )


# ----------------------------------------------------------------------------- stage D


def _one_line(value: str | None) -> str | None:
    """The only normalisation of a one-line field: a newline becomes a space."""
    if value is None:
        return None
    text = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
    return text or None


def apply_header(wb: Workbook, answers: HeaderAnswers) -> Workbook:
    """Stage D: set the header fields as given. Pure; a `;` inside a
    `Contact` / `Education` item raises `CompleteError` (it would split on the next parse)."""
    given = answers.model_fields_set
    update: dict = {}
    for key in ("name", "headline", "summary"):
        if key in given:
            update[key] = _one_line(getattr(answers, key))
    for key, what in (("contact", "Contact"), ("education", "Education")):
        if key in given:
            update[key] = _checked_items(getattr(answers, key) or [], what)
    return wb.model_copy(update={"header": wb.header.model_copy(update=update)})


def apply_skill_groups(wb: Workbook, answers: SkillsAnswers) -> Workbook:
    """Stage D: replace the Skills groups as given — names, order, items.

    Nothing is reordered or deduplicated across groups; inside one group the casefold
    duplicates are dropped, and a group left with no items is dropped. `CompleteError` for a
    name containing `*`, two groups of one name (casefold), an unnamed group that is not first
    (the vault holds it as the `Skills` line's own items), or a `;` inside an item."""
    groups: list[SkillGroup] = []
    for group in answers.groups:
        name = _one_line(group.name) or ""
        if "*" in name:
            raise CompleteError(f"Skills group name '{name}' contains '*'")
        items = _checked_items(group.items, f"Skills group '{name or 'Ungrouped'}'")
        if not items:
            continue
        if not name and groups:
            raise CompleteError("only the first Skills group may be unnamed")
        if any(g.name.casefold() == name.casefold() for g in groups):
            raise CompleteError(f"two Skills groups are named '{name}'")
        groups.append(SkillGroup(name=name, items=items))
    return wb.model_copy(update={"skill_groups": groups})


def skills_prefill(wb: Workbook) -> list[str]:
    """The three-source union: the groups' items — the résumé's own
    skills plus what the user typed —, then every confirmed job's `Defaults` in job order;
    casefold-deduplicated, the first spelling kept. An unconfirmed job's `Defaults` are not in
    it."""
    confirmed = [d for job in wb.jobs if job.defaults_confirmed for d in job.defaults]
    return _dedup([*wb.skills, *confirmed])


def with_prefill(wb: Workbook) -> Workbook:
    """The pre-fill items that are in no group yet, appended to the unnamed group (created
    first when absent); the named groups are untouched. No model call."""
    missing = [
        s for s in skills_prefill(wb) if s.casefold() not in {i.casefold() for i in wb.skills}
    ]
    if not missing:
        return wb
    groups = list(wb.skill_groups)
    if groups and groups[0].name == "":
        groups[0] = SkillGroup(name="", items=[*groups[0].items, *missing])
    else:
        groups.insert(0, SkillGroup(name="", items=missing))
    return wb.model_copy(update={"skill_groups": groups})


# ----------------------------------------------------------------------------- views


def groups(job: WbJob, bullet: WbBullet) -> Groups:
    """The three groups of a bullet's card: the job's defaults the bullet carries, the defaults
    it does not (one click away), and its own keywords. Case-insensitive; the defaults keep the
    job's spelling and order."""
    carried = {k.casefold() for k in bullet.keywords}
    default_folds = {d.casefold() for d in job.defaults}
    return Groups(
        selected_defaults=[d for d in job.defaults if d.casefold() in carried],
        unselected_defaults=[d for d in job.defaults if d.casefold() not in carried],
        own=[k for k in bullet.keywords if k.casefold() not in default_folds],
    )


def skills_unassigned(wb: Workbook) -> list[str]:
    """The vault-level `Skills` items that are in no job's `Defaults` (computed: the `Skills`
    line keeps every item)."""
    assigned = {d.casefold() for j in wb.jobs for d in j.defaults}
    return [s for s in wb.skills if s.casefold() not in assigned]


def accept_all(wb: Workbook) -> tuple[JobsAnswers, BulletsAnswers]:
    """The answers that confirm everything as it stands: every job's `Defaults` as they are
    (nothing assigned from `Skills`), every bullet with its current keywords."""
    return (
        JobsAnswers(jobs={j.job_id: JobAnswer(defaults_confirmed=True) for j in wb.jobs}),
        BulletsAnswers(
            bullets={
                f"{j.job_id}::{b.id}": BulletAnswer(confirm=True)
                for j in wb.jobs
                for b in j.bullets
            }
        ),
    )


# ----------------------------------------------------------------------------- render / write


def render_workbook(wb: Workbook, *, today: date) -> str:
    """The workbook as v3 vault text, through the L1 renderer: `confirmed` / `draft` per bullet,
    the `Skills` line under the H1, a bullet-less job as its header and an empty `Bullets`
    block."""
    unnamed = [g for g in wb.skill_groups if g.name == ""]
    draft = DraftVault(
        header=wb.header,
        skills=unnamed[0].items if unnamed else [],
        skill_groups=[g for g in wb.skill_groups if g.name != ""],
        jobs=[
            DraftJob(
                name=job.name,
                employer=job.employer,
                title=job.title,
                date_from=job.date_from,
                date_to=job.date_to,
                defaults=job.defaults,
                bullets=[
                    DraftBullet(
                        id=b.id,
                        text=b.text,
                        project=b.project,
                        keywords=b.keywords,
                        status=b.status,
                        updated=b.updated,
                    )
                    for b in job.bullets
                ],
            )
            for job in wb.jobs
        ],
    )
    return render_vault(
        draft,
        today=today,
        heading=f"STAR Vault — written by pipeline complete, {today.isoformat()}",
    )


def verify_rendered(rendered: str, wb: Workbook) -> tuple[list[str], int]:
    """The strict parser, twice. With `include_draft=True`: every error becomes a
    problem line (the empty-job warning is allowed). With the default `include_draft=False`: the
    confirmed bullet count must equal the number of bullets the workbook has confirmed.
    Returns `(problems, confirmed)`."""
    problems = [
        f"line {i.line_no}: {i.message}"
        for i in parse_star_vault(rendered, include_draft=True).errors()
    ]
    confirmed = sum(len(j.bullets) for j in parse_star_vault(rendered).jobs)
    expected = sum(1 for j in wb.jobs for b in j.bullets if b.status == "confirmed")
    if confirmed != expected:
        problems.append(
            f"confirmed bullets in the written vault = {confirmed}, expected {expected}"
        )
    return problems, confirmed


def write_workbook(wb: Workbook, out: Path, *, today: date) -> WriteReport:
    """Render, write (parents created) and verify. The file is written even when the strict
    parse reports problems; an `OSError` from the write propagates."""
    rendered = render_workbook(wb, today=today)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    problems, confirmed = verify_rendered(rendered, wb)
    return WriteReport(
        path=str(out),
        jobs=len(wb.jobs),
        jobs_without_bullets=sum(1 for j in wb.jobs if not j.bullets),
        bullets=sum(len(j.bullets) for j in wb.jobs),
        confirmed=confirmed,
        skills_unassigned=len(skills_unassigned(wb)),
        problems=problems,
    )


# ----------------------------------------------------------------------------- views


def maybe_also(job: WbJob, bullet: WbBullet) -> list[str]:
    """The job's tools list items the bullet's keywords lack (the "Maybe also" hint under the
    keywords box). Case-insensitive; the job's spelling and
    order; the bullet's own keywords are never listed. Pure."""
    carried = {k.casefold() for k in bullet.keywords}
    return [d for d in job.defaults if d.casefold() not in carried]


def unchanged_since_entry(bullet: WbBullet) -> bool:
    """True when the bullet's text is still the text its id was made from:
    `sha1(<normalised text>)[:6] == id` — ingest's and `add_bullet`'s id rule. An
    id that took the ordinal on a collision reads as changed (the safe direction). Pure."""
    return _sha6(normalise_ws(bullet.text)) == bullet.id


def confirm_unedited(wb: Workbook, job_id: str) -> BulletsAnswers:
    """The answers of "Confirm all N bullets I haven't edited":
    `confirm=True` for each **unconfirmed** bullet of the job that is unchanged since entry, and
    nothing else — never an edited bullet, the tools list or another job. A job with none →
    empty answers. Pure; an unknown job raises `CompleteError`."""
    job = next((j for j in wb.jobs if j.job_id == job_id), None)
    if job is None:
        raise CompleteError(f"unknown job '{job_id}'")
    return BulletsAnswers(
        bullets={
            f"{job.job_id}::{b.id}": BulletAnswer(confirm=True)
            for b in job.bullets
            if b.status != "confirmed" and unchanged_since_entry(b)
        }
    )
