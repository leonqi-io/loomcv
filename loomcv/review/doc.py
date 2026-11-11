"""The review document: the template context of `render_resume`
with ids, as a value the page can edit.

Pure functions, no file, no index, no model call. The document is built from the inputs
`draft_step` uses (the payload, the `jobs` rows, the vault's Skills groups and header fields)
and rendered through the same template, so the untouched document renders byte-equal to
`tailored_resume.md` under any `resume_template`; the Markdown is never parsed back. Every
operation returns a new document and keeps a bullet inside the job it was confirmed under.
No score, rank or bullet id reaches a string the page prints (ids are dict keys and dropdown
values only).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from loomcv.draft.render import (
    bullet_label,
    header_slots,
    job_heading,
    render_context,
    resume_context,
)
from loomcv.draft.verdict import claim_sources, compute_verdict, render_verdict
from loomcv.retrieve.merge import carriers_of
from loomcv.retrieve.rank import Candidate, LineResult, MatchPayload
from loomcv.sync.parse_assets import SkillGroup, VaultHeader, flat_skills

if TYPE_CHECKING:
    from loomcv.retrieve.search import JobRow

_SKILLS_LINE = "Skills line"
_LABEL_CUT = 90


class _Keep:
    """Sentinel type: `edit` leaves the bullet's project as it is."""


KEEP = _Keep()


class ReviewBullet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Required: a bullet with no vault origin is not a shape the review document has.
    bullet_id: str
    project: str | None
    text: str


class ReviewJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    heading: str
    date_from: str
    date_to: str
    bullets: list[ReviewBullet]


class ReviewDoc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # The vault's header fields. Headline and Summary are the per-résumé
    # edits the page offers; Name / Contact / Education are printed as the vault has them.
    name: str | None
    headline: str | None
    contact: list[str]
    summary: str | None
    education: list[str]
    skill_groups: list[SkillGroup]  # vault order; the items are editable, the names are not
    jobs: list[ReviewJob]

    def __hash__(self) -> int:
        # The generated hash of a frozen model fails on its list fields; the page keeps the
        # document in a `gr.State`, which Gradio hashes to detect a change.
        return hash(self.model_dump_json())


# ---------------------------------------------------------------- build / render


def from_draft(
    payload: MatchPayload,
    rows: Sequence[JobRow],
    skill_groups: Sequence[SkillGroup],
    header: VaultHeader | None,
) -> ReviewDoc:
    """The document `render_resume` would print: `resume_context` supplies the headings, dates
    and bullets in walk order (one source of the order); ids and projects come from the payload."""
    slots = resume_context(payload, rows, skill_groups, header)["jobs"]
    project_of = {b.bullet_id: b.project for job in payload.selection.jobs for b in job.bullets}
    return ReviewDoc(
        **header_slots(header),
        skill_groups=list(skill_groups),
        jobs=[
            ReviewJob(
                job_id=row.job_id,
                heading=slot["heading"],
                date_from=slot["date_from"],
                date_to=slot["date_to"],
                bullets=[
                    ReviewBullet(
                        bullet_id=b["bullet_id"], project=project_of[b["bullet_id"]], text=b["text"]
                    )
                    for b in slot["bullets"]
                ],
            )
            for row, slot in zip(rows, slots, strict=True)
        ],
    )


def context(doc: ReviewDoc) -> dict[str, Any]:
    """The dict `resume_context` builds, from the document."""
    return {
        "name": doc.name,
        "headline": doc.headline,
        "contact": list(doc.contact),
        "summary": doc.summary,
        "education": list(doc.education),
        "skill_groups": [{"name": g.name, "items": list(g.items)} for g in doc.skill_groups],
        "skills": flat_skills(doc.skill_groups),
        "jobs": [
            {
                "heading": job.heading,
                "date_from": job.date_from,
                "date_to": job.date_to,
                "bullets": [
                    {
                        "label": bullet_label(b.project),
                        "text": b.text,
                        "bullet_id": b.bullet_id,
                    }
                    for b in job.bullets
                ],
            }
            for job in doc.jobs
        ],
    }


def render(doc: ReviewDoc, template_path: Path) -> str:
    return render_context(context(doc), template_path=template_path)


# ------------------------------------------------------------------- operations


def _job(doc: ReviewDoc, job_id: str) -> tuple[int, ReviewJob]:
    for position, job in enumerate(doc.jobs):
        if job.job_id == job_id:
            return position, job
    raise ValueError(f"no job {job_id!r} on the document")


def _in_range(job: ReviewJob, index: int) -> None:
    if not 0 <= index < len(job.bullets):
        raise ValueError(f"job {job.job_id!r} has {len(job.bullets)} bullet(s); no index {index}")


def _with_bullets(doc: ReviewDoc, position: int, bullets: list[ReviewBullet]) -> ReviewDoc:
    jobs = list(doc.jobs)
    jobs[position] = jobs[position].model_copy(update={"bullets": bullets})
    return doc.model_copy(update={"jobs": jobs})


def move(doc: ReviewDoc, job_id: str, index: int, delta: int) -> ReviewDoc:
    """Swap a bullet with its neighbour inside its job; at an end, the same document."""
    if delta not in (-1, 1):
        raise ValueError(f"delta must be -1 or +1, not {delta}")
    position, job = _job(doc, job_id)
    _in_range(job, index)
    target = index + delta
    if not 0 <= target < len(job.bullets):
        return doc
    bullets = list(job.bullets)
    bullets[index], bullets[target] = bullets[target], bullets[index]
    return _with_bullets(doc, position, bullets)


def edit(
    doc: ReviewDoc,
    job_id: str,
    index: int,
    text: str,
    project: str | None | _Keep = KEEP,
) -> ReviewDoc:
    """Replace a bullet's text, and its project label unless `KEEP`; `""` / `None` removes the
    label. The bullet keeps its id: `src` records where it came from."""
    position, job = _job(doc, job_id)
    _in_range(job, index)
    old = job.bullets[index]
    label = old.project if isinstance(project, _Keep) else (project or None)
    bullets = list(job.bullets)
    bullets[index] = ReviewBullet(bullet_id=old.bullet_id, project=label, text=text)
    return _with_bullets(doc, position, bullets)


def delete(doc: ReviewDoc, job_id: str, index: int) -> ReviewDoc:
    position, job = _job(doc, job_id)
    _in_range(job, index)
    return _with_bullets(doc, position, [b for i, b in enumerate(job.bullets) if i != index])


def add(doc: ReviewDoc, job_id: str, candidate: Candidate) -> ReviewDoc:
    """Append a confirmed vault bullet of this job; one already on the document is refused."""
    position, job = _job(doc, job_id)
    if candidate.job_id != job_id:
        raise ValueError(f"bullet {candidate.bullet_id!r} belongs to job {candidate.job_id!r}")
    if any(b.bullet_id == candidate.bullet_id for j in doc.jobs for b in j.bullets):
        raise ValueError(f"bullet {candidate.bullet_id!r} is already on the document")
    added = ReviewBullet(
        bullet_id=candidate.bullet_id, project=candidate.project, text=candidate.text
    )
    return _with_bullets(doc, position, [*job.bullets, added])


def add_checked(
    doc: ReviewDoc,
    job_id: str,
    candidates: Sequence[Candidate],
    offered: Sequence[str],
    checked: Sequence[str],
) -> ReviewDoc:
    """Add every ticked bullet of the job's offer, in the order the offer lists them (not the
    order of the ticks). Nothing ticked, or a tick that is not in the offer, is refused; so is a
    bullet already on the document. All or nothing: the argument document is never changed."""
    if not checked:
        raise ValueError("tick at least one bullet first")
    if any(bid not in offered for bid in checked):
        raise ValueError("a ticked bullet is not on offer for this job; reopen the posting")
    for bullet_id in (bid for bid in offered if bid in set(checked)):
        doc = add(doc, job_id, pick(candidates, bullet_id))
    return doc


def index_mismatch(doc: ReviewDoc, rows: Sequence[JobRow]) -> bool:
    """True when the document has jobs and none of them is in the index's `jobs` rows: the
    posting was made on another vault index, so nothing can be added to it here."""
    ids = {row.job_id for row in rows}
    return bool(doc.jobs) and not any(job.job_id in ids for job in doc.jobs)


def set_headline(doc: ReviewDoc, text: str) -> ReviewDoc:
    return doc.model_copy(update={"headline": text or None})


def set_summary(doc: ReviewDoc, text: str) -> ReviewDoc:
    return doc.model_copy(update={"summary": text or None})


def set_skill_groups(doc: ReviewDoc, items: Sequence[Sequence[str]]) -> ReviewDoc:
    """One items list per group, by position; the group names stay. A group left with no
    items stays on the document — the template prints only groups that have items."""
    if len(items) != len(doc.skill_groups):
        raise ValueError(f"expected {len(doc.skill_groups)} Skills boxes, got {len(items)}")
    groups = [
        SkillGroup(name=g.name, items=list(new))
        for g, new in zip(doc.skill_groups, items, strict=True)
    ]
    return doc.model_copy(update={"skill_groups": groups})


def skills_to_text(items: Sequence[str]) -> str:
    """The Skills items as the page's one-line box shows them: `;`-separated, the vault's own
    item separator, so an item may hold a comma."""
    return "; ".join(items)


def skills_from_text(text: str) -> list[str]:
    """The box's text back to items: split on `;`, strip, drop empties, keep the order."""
    return [part.strip() for part in text.split(";") if part.strip()]


def layout(doc: ReviewDoc) -> list[tuple[str, str]]:
    """`(job_id, bullet_id)` in page order: what a page was drawn from."""
    return [(job.job_id, b.bullet_id) for job in doc.jobs for b in job.bullets]


def commit(
    doc: ReviewDoc,
    drawn: Sequence[Sequence[str]],
    headline: str,
    summary: str,
    group_items: Sequence[Sequence[str]],
    values: Sequence[str],
) -> ReviewDoc:
    """Apply every box of the page at once: the Headline, the Summary, one items list
    per Skills group and, for each bullet in `drawn` order, its text then its project (`values` =
    text0, project0, text1, …). Name / Contact / Education are not boxes: they are
    carried over from `doc`. Refused, nothing applied, when the page was drawn from another layout
    than the document's or the box counts do not match it."""
    current = layout(doc)
    if [tuple(d) for d in drawn] != current:
        raise ValueError("the page changed under this click; reopen the application and try again")
    if len(values) != 2 * len(current):
        raise ValueError(f"expected {2 * len(current)} box values, got {len(values)}")
    edited = set_skill_groups(doc, group_items)
    jobs: list[ReviewJob] = []
    box = 0
    for job in doc.jobs:
        bullets = []
        for b in job.bullets:
            bullets.append(
                ReviewBullet(
                    bullet_id=b.bullet_id, project=values[box + 1] or None, text=values[box]
                )
            )
            box += 2
        jobs.append(job.model_copy(update={"bullets": bullets}))
    return edited.model_copy(
        update={"headline": headline or None, "summary": summary or None, "jobs": jobs}
    )


# ------------------------------------------------------------------ annotations


def _line_label(line: LineResult) -> str:
    return f"{line.text} ({line.kind})"


def _on_doc(doc: ReviewDoc) -> set[str]:
    return {b.bullet_id for job in doc.jobs for b in job.bullets}


def coverage_of(
    doc: ReviewDoc, payload: MatchPayload, candidates: Sequence[Candidate]
) -> dict[str, list[str]]:
    """Bullet id → the JD lines it carries, as `"<line> (<kind>)"` in requirements order; every
    bullet on the document is a key. For a placed bullet this is the inverse of
    `selection.covered_by`; for an added one, the lines it would have covered (P3)."""
    covered: dict[str, list[str]] = {bullet_id: [] for _, bullet_id in layout(doc)}
    pool = list(candidates)
    for line in payload.requirements:
        for carrier in carriers_of(line, pool):
            if carrier.bullet_id in covered:
                covered[carrier.bullet_id].append(_line_label(line))
    return covered


def edited_bullets(doc: ReviewDoc, candidates: Sequence[Candidate]) -> set[str]:
    """The ids of the bullets on the document whose text differs from the confirmed vault bullet
    with that id (the draft's wording). A bullet with no candidate — made on another index — is
    never in the set: there is no wording to restore."""
    vault = {c.bullet_id: c.text for c in candidates}
    return {
        b.bullet_id
        for job in doc.jobs
        for b in job.bullets
        if b.bullet_id in vault and b.text != vault[b.bullet_id]
    }


class Addable(BaseModel):
    bullet_id: str
    text: str
    project: str | None
    label: str  # the dropdown text: cut at 90 characters
    lines: list[str]  # the lines this bullet was ranked on, as "<line> (<kind>)"


def addable(
    doc: ReviewDoc, payload: MatchPayload, candidates: Sequence[Candidate]
) -> dict[str, list[Addable]]:
    """Job id → that job's confirmed bullets not on the document: those some line ranked first (in
    the order the lines rank them), then the rest in vault order. Every job of the document is a
    key; nothing from another job."""
    placed = _on_doc(doc)
    ranked_on: dict[str, list[str]] = {}
    for line in payload.requirements:
        for ranked in line.ranked:
            ranked_on.setdefault(ranked.bullet_id, []).append(_line_label(line))
    first_seen = {bullet_id: i for i, bullet_id in enumerate(ranked_on)}
    result: dict[str, list[Addable]] = {}
    for job in doc.jobs:
        mine = [c for c in candidates if c.job_id == job.job_id and c.bullet_id not in placed]
        ordered = sorted(
            (c for c in mine if c.bullet_id in ranked_on), key=lambda c: first_seen[c.bullet_id]
        ) + [c for c in mine if c.bullet_id not in ranked_on]
        result[job.job_id] = [
            Addable(
                bullet_id=c.bullet_id,
                text=c.text,
                project=c.project,
                label=c.text if len(c.text) <= _LABEL_CUT else c.text[:_LABEL_CUT] + "…",
                lines=ranked_on.get(c.bullet_id, []),
            )
            for c in ordered
        ]
    return result


def pick(candidates: Sequence[Candidate], bullet_id: str | None) -> Candidate:
    """The candidate a dropdown value names."""
    for candidate in candidates:
        if candidate.bullet_id == bullet_id:
            return candidate
    raise ValueError("pick a bullet from the list first" if bullet_id is None else "unknown bullet")


class GapEntry(BaseModel):
    text: str
    claimed_at: list[str]  # job headings, "Skills line"; [] = no matching material in the vault


class UncoveredEntry(BaseModel):
    text: str
    kind: str
    carriers: list[str]  # texts of the vault bullets that carry the line


class ClaimKeyword(BaseModel):
    keyword: str
    at: list[str]


class ClaimEntry(BaseModel):
    text: str
    kind: str
    match: str
    keywords: list[ClaimKeyword]


class Missing(BaseModel):
    headline: str  # the Verdict's first line, as the report prints it (for the draft as matched)
    gaps: list[GapEntry]
    uncovered: list[UncoveredEntry]
    claims: list[ClaimEntry]


def missing(
    doc: ReviewDoc,
    payload: MatchPayload,
    rows: Sequence[JobRow],
    candidates: Sequence[Candidate],
) -> Missing:
    """Sections 2 / 3 / 5 of the report as data. `gaps` and `claims` are the payload's (no edit
    can change them); `uncovered` is live over the document (P3): the graded lines no bullet on
    it carries."""
    by_id = {row.job_id: row for row in rows}
    headings = {job.job_id: job.heading for job in doc.jobs}

    def named(source: str) -> str:
        # The document's own headings first: a posting opened on another index must not show
        # slugs (⑧ f); the index only names a job the document does not hold.
        if source == "skills":
            return _SKILLS_LINE
        if source in headings:
            return headings[source]
        return job_heading(by_id[source]) if source in by_id else source

    claimed = {c.line: c for c in payload.claims}
    on_doc = _on_doc(doc)
    pool = list(candidates)
    uncovered = []
    for line in payload.requirements:
        if line.grade == "none":
            continue
        carriers = carriers_of(line, pool)
        if not any(c.bullet_id in on_doc for c in carriers):
            uncovered.append(
                UncoveredEntry(text=line.text, kind=line.kind, carriers=[c.text for c in carriers])
            )
    return Missing(
        headline=render_verdict(compute_verdict(payload, rows))[0],
        gaps=[
            GapEntry(
                text=gap,
                claimed_at=[named(s) for s in claim_sources(claimed[gap])]
                if gap in claimed
                else [],
            )
            for gap in payload.gaps
        ],
        uncovered=uncovered,
        claims=[
            ClaimEntry(
                text=claim.line,
                kind=claim.kind,
                match=claim.match,
                keywords=[
                    ClaimKeyword(keyword=k.keyword, at=[named(s) for s in k.at])
                    for k in claim.keywords
                ],
            )
            for claim in payload.claims
        ],
    )
