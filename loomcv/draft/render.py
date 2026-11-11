"""The tailored résumé, rendered deterministically from a match payload.

No model call and no rewriting: the body is vault text byte-for-byte — the
placed bullets of `selection.jobs[]` in walk order under every `jobs` row in `job_order`, the
vault's Skills groups and its header fields (name, headline, contact,
summary, education). The layout (section order, headings,
date line, bullet line) is a Jinja template file named by `generation.yaml` `resume_template`;
this module only fills its slots. Pure functions: the index reads
(`JobRow`s, the header, the groups, the `{bullet_id: text}` map for `check_fresh`) are done by
the caller. Only `loomcv.retrieve.rank` (and `loomcv.resources`, for the default template's
path) is imported at runtime, so `import loomcv.draft` never pulls in `search.py` and through it
`llm.py`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2

from loomcv import resources
from loomcv.retrieve.rank import MatchPayload

if TYPE_CHECKING:
    from loomcv.retrieve.search import JobRow
    from loomcv.sync.parse_assets import SkillGroup, VaultHeader

RESUME_NAME = "tailored_resume.md"
REPORT_NAME = "match_report.md"
DEFAULT_RESUME_TEMPLATE = resources.path("templates", "resume_default.md")


class StaleMatchError(ValueError):
    """The payload names a placed bullet the index no longer has as a confirmed bullet with the
    same text: the vault changed since the match."""


def check_fresh(payload: MatchPayload, indexed: Mapping[str, str]) -> None:
    """`indexed` = the confirmed bullet_id -> text of the index.

    Raises `StaleMatchError` naming every placed bullet whose id is not indexed or whose text
    differs from the indexed text.
    """
    stale = [
        b.bullet_id
        for job in payload.selection.jobs
        for b in job.bullets
        if indexed.get(b.bullet_id) != b.text
    ]
    if stale:
        raise StaleMatchError(
            f"payload names {len(stale)} bullet(s) the index no longer has as confirmed or "
            f"with the same text: {', '.join(stale)}; payload is stale, re-run match"
        )


class ResumeTemplateError(ValueError):
    """The résumé template is unreadable or names a slot the data lacks (CLI exit 1)."""


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_YEAR_MONTH = re.compile(r"(\d{4})-(\d{2})")


def resume_date(value: str) -> str:
    """`YYYY-MM` -> `Mon YYYY` (fixed English abbreviations, not the locale); `present` ->
    `Present`; anything else (`YYYY`, the `0000` marker) as stored."""
    if value == "present":
        return "Present"
    match = _YEAR_MONTH.fullmatch(value)
    if match and 1 <= int(match.group(2)) <= 12:
        return f"{_MONTHS[int(match.group(2)) - 1]} {match.group(1)}"
    return value


def job_heading(job: JobRow) -> str:
    if job.employer is None or job.title is None:
        return job.name
    return f"{job.title} — {job.employer}"


def bullet_label(project: str | None) -> str:
    return f"**{project}:** " if project is not None else ""


def _flat(groups: Sequence[SkillGroup]) -> list[str]:
    """Every group's items in order, casefold-deduplicated (the vault's flat `Skills` list)."""
    seen: set[str] = set()
    flat: list[str] = []
    for group in groups:
        for item in group.items:
            if item.casefold() not in seen:
                seen.add(item.casefold())
                flat.append(item)
    return flat


def header_slots(header: VaultHeader | None) -> dict[str, Any]:
    """The five header slots, as stored in the vault (`None` / `[]` when absent)."""
    if header is None:
        return {"name": None, "headline": None, "contact": [], "summary": None, "education": []}
    return {
        "name": header.name,
        "headline": header.headline,
        "contact": list(header.contact),
        "summary": header.summary,
        "education": list(header.education),
    }


def resume_context(
    payload: MatchPayload,
    jobs: Sequence[JobRow],
    skill_groups: Sequence[SkillGroup] = (),
    header: VaultHeader | None = None,
) -> dict[str, Any]:
    """The slots the template fills: the vault's header fields (name, headline, contact,
    summary, education), its Skills groups (`skill_groups`, in vault order, and
    `skills`, their flat union for templates written before the groups), and the `jobs` rows in
    the order given, each with its placed bullets in walk order (split out
    of `render_resume` so the review document can render through the same template). Every slot
    is independent: a template may print them in any order or leave one out."""
    placed = {job.job_id: job.bullets for job in payload.selection.jobs}
    return {
        **header_slots(header),
        "skill_groups": [{"name": g.name, "items": list(g.items)} for g in skill_groups],
        "skills": _flat(skill_groups),
        "jobs": [
            {
                "heading": job_heading(row),
                "date_from": row.date_from,
                "date_to": row.date_to,
                "bullets": [
                    {"label": bullet_label(b.project), "text": b.text, "bullet_id": b.bullet_id}
                    for b in sorted(placed.get(row.job_id, []), key=lambda b: b.walk_position)
                ],
            }
            for row in jobs
        ],
    }


def render_resume(
    payload: MatchPayload,
    jobs: Sequence[JobRow],
    skill_groups: Sequence[SkillGroup] = (),
    header: VaultHeader | None = None,
    *,
    template_path: Path = DEFAULT_RESUME_TEMPLATE,
) -> str:
    """`tailored_resume.md`, rendered through the Jinja template at `template_path`:
    the template owns section order, headings and line formats; the data is the
    vault's header fields and Skills groups (verbatim), and the `jobs` rows in the
    order given with their placed bullets in walk order. Nothing is rewritten."""
    return render_context(
        resume_context(payload, jobs, skill_groups, header), template_path=template_path
    )


def render_context(
    context: Mapping[str, Any], *, template_path: Path = DEFAULT_RESUME_TEMPLATE
) -> str:
    """Fill the Jinja template at `template_path` with `context` (the `resume_context` shape)."""
    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined, keep_trailing_newline=True, autoescape=False
    )
    env.filters["resume_date"] = resume_date
    try:
        source = Path(template_path).read_text(encoding="utf-8")
        return env.from_string(source).render(context)
    except (OSError, UnicodeDecodeError, jinja2.TemplateError) as exc:
        raise ResumeTemplateError(f"resume template {template_path}: {exc}") from exc
