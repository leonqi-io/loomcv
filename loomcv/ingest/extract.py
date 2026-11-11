"""`ingest_resume`: one old résumé → a v3 draft vault, zero new facts.

Top layer of `loomcv.ingest`. The flow: the
structure call (jobs — with or without bullets —, prose dates normalised by the model, bullets
verbatim, the job's declared skills as `defaults`, the vault-level `skills` list; one retry on an
invalid reply) → `_post_validate` (drop-only) → `assign_ids` →
`render_vault` → `check_rendered` (the strict parser on ingest's own output). There is no keyword
call here any more: it runs in `loomcv.complete` behind the user's confirmation of each job's
`Defaults`. The same call also returns the header fields (name, headline,
contact, summary, education), each under the same drop-only check (`_post_validate_header`).
The model returns JSON only; the code renders the vault. Nothing here prints a
warning — `IngestResult.warnings` is the record and the CLI prints it.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator

from loomcv import resources
from loomcv.ingest.keywords import _clean_items, complete_validated, normalise_ws, render_split
from loomcv.ingest.render import DraftBullet, DraftJob, DraftVault, check_rendered, render_vault
from loomcv.llm import GenerationAdapter
from loomcv.sync.parse_assets import ParseIssue, VaultHeader, slugify

# The prompt ships with the package, so its path is package-relative.
DEFAULT_STRUCTURE_PROMPT = resources.path("prompts", "ingest_structure.md")

# Mirrors `parse_assets._DATE_VALUE_RE` (not imported: private); `check_rendered` is the oracle.
_DATE_RE = re.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2]))?$")
UNKNOWN_DATE = "0000"


def _null_as_empty(value: Any) -> Any:
    """A `null` for a list is permitted: read as `[]`, so it costs no retry."""
    return [] if value is None else value


class _BulletReply(BaseModel):
    text: str
    project: str | None = None


class _JobReply(BaseModel):
    employer: str | None = None
    title: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    defaults: Annotated[list[str], BeforeValidator(_null_as_empty)] = []
    bullets: Annotated[list[_BulletReply], BeforeValidator(_null_as_empty)]


class _StructureReply(BaseModel):
    skills: Annotated[list[str], BeforeValidator(_null_as_empty)] = []
    jobs: list[_JobReply]
    # The header fields, each optional — the model may leave one out.
    name: str | None = None
    headline: str | None = None
    contact: Annotated[list[str], BeforeValidator(_null_as_empty)] = []
    summary: str | None = None
    education: Annotated[list[str], BeforeValidator(_null_as_empty)] = []


class IngestResult(BaseModel):
    draft: DraftVault
    rendered: str
    parse_issues: list[ParseIssue]
    warnings: list[str]
    raws: list[str]
    skills: list[str]


def render_structure_prompt(
    resume_text: str, *, today: date | str, template_path: Path = DEFAULT_STRUCTURE_PROMPT
) -> tuple[str, str]:
    """Render the structure prompt → (system, user); the résumé goes in between `<<<` / `>>>`."""
    stamp = today.isoformat() if isinstance(today, date) else str(today)
    return render_split(template_path, resume_text=resume_text, today=stamp)


def _sha6(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]


def assign_ids(texts: list[str]) -> list[str]:
    """One job's bullet texts, in order → their ids: the first 6 hex of
    `sha1(<whitespace-normalised text>)`; on a collision inside the job, of
    `sha1(<text> + "\\n" + <ordinal>)` with the smallest free ordinal ≥ 2."""
    used: set[str] = set()
    ids: list[str] = []
    for text in texts:
        norm = normalise_ws(text)
        ident = _sha6(norm)
        ordinal = 2
        while ident in used:
            ident = _sha6(f"{norm}\n{ordinal}")
            ordinal += 1
        used.add(ident)
        ids.append(ident)
    return ids


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    return normalise_ws(value) or None


def _date(
    value: str | None, key: str, job: str, warnings: list[str], *, allow_present: bool
) -> str:
    """A valid `YYYY[-MM]` (or `present` for `To`) is kept; `None` is `0000` silently (the
    résumé gave none); anything else is `0000` with a warning."""
    if value is None:
        return UNKNOWN_DATE
    text = value.strip()
    if _DATE_RE.match(text) or (allow_present and text == "present"):
        return text
    warnings.append(f"unparsed {key} '{text}' for job '{job}' (marked 0000)")
    return UNKNOWN_DATE


def _post_validate(reply: _StructureReply, resume_text: str) -> tuple[list[DraftJob], list[str]]:
    """Drop-only: returns the surviving jobs — ids assigned, defaults cleaned,
    keywords still empty — and the warnings in the `parse_jd` wording."""
    source = normalise_ws(resume_text)
    warnings: list[str] = []
    jobs: list[DraftJob] = []
    used_slugs: set[str] = set()
    for n, job in enumerate(reply.jobs, start=1):
        employer, title = _optional(job.employer), _optional(job.title)
        if employer and title:
            base = f"{employer} — {title}"
        elif employer or title:
            base = employer or title or ""
        else:
            base = f"Job {n}"
            warnings.append(f"job {n} has neither employer nor title")

        texts: list[str] = []
        projects: list[str | None] = []
        for bullet in job.bullets:
            text = normalise_ws(bullet.text)
            if not text:
                warnings.append("dropped bullet '' (empty text)")
            elif "{" in text or "}" in text:
                warnings.append(
                    f"dropped bullet '{text[:40]}' (brace in text; the vault grammar reserves {{}})"
                )
            elif text not in source:
                warnings.append(f"dropped bullet '{text[:40]}' (text not in source)")
            else:
                texts.append(text)
                projects.append(_optional(bullet.project))
        # A job with no surviving bullets is kept, bullet-less.

        date_from = _date(job.date_from, "From", base, warnings, allow_present=False)
        date_to = _date(job.date_to, "To", base, warnings, allow_present=True)
        defaults, dropped = _clean_items(job.defaults)
        warnings.extend(dropped)

        name = base
        if slugify(name) in used_slugs:
            ordinal = 2
            while slugify(f"{base} ({ordinal})") in used_slugs:
                ordinal += 1
            name = f"{base} ({ordinal})"
            warnings.append(f"duplicate job name '{base}' renamed to '{name}'")
        used_slugs.add(slugify(name))

        jobs.append(
            DraftJob(
                name=name,
                employer=employer,
                title=title,
                date_from=date_from,
                date_to=date_to,
                defaults=defaults,
                bullets=[
                    DraftBullet(id=ident, text=text, project=project)
                    for ident, text, project in zip(assign_ids(texts), texts, projects, strict=True)
                ],
            )
        )
    return jobs, warnings


def _post_validate_header(
    reply: _StructureReply, resume_text: str
) -> tuple[VaultHeader, list[str]]:
    """Drop-only, the bullet rule applied to each header value: whitespace
    normalised, empty → absent, a brace → dropped, not a substring of the résumé → dropped, and a
    list item containing `;` (it would split on the next parse) → dropped. One warning each."""
    source = normalise_ws(resume_text)
    warnings: list[str] = []

    def kept(field: str, raw: str | None) -> str | None:
        value = normalise_ws(raw or "")
        if not value:
            return None
        reason = None
        if "{" in value or "}" in value:
            reason = "brace in text; the vault grammar reserves {}"
        elif value not in source:
            reason = "text not in source"
        elif field in ("contact", "education") and ";" in value:
            reason = "contains ;"
        if reason is not None:
            warnings.append(f"dropped {field} '{value[:40]}' ({reason})")
            return None
        return value

    def kept_list(field: str, raws: list[str]) -> list[str]:
        items: list[str] = []
        for raw in raws:
            value = kept(field, raw)
            if value is not None and value.casefold() not in {i.casefold() for i in items}:
                items.append(value)
        return items

    header = VaultHeader(
        name=kept("name", reply.name),
        headline=kept("headline", reply.headline),
        contact=kept_list("contact", reply.contact),
        summary=kept("summary", reply.summary),
        education=kept_list("education", reply.education),
    )
    return header, warnings


def ingest_resume(
    resume_text: str,
    *,
    adapter: GenerationAdapter,
    today: date | None = None,
    structure_template: Path = DEFAULT_STRUCTURE_PROMPT,
) -> IngestResult:
    """One structure call (+ at most one retry)."""
    today = today or date.today()
    system, user = render_structure_prompt(
        resume_text, today=today, template_path=structure_template
    )
    raws: list[str] = []
    reply = complete_validated(
        adapter, system, user, _StructureReply.model_validate_json, raws, label="structure"
    )
    header, warnings = _post_validate_header(reply, resume_text)
    jobs, job_warnings = _post_validate(reply, resume_text)
    warnings.extend(job_warnings)
    skills, dropped = _clean_items(reply.skills)
    warnings.extend(dropped)

    draft = DraftVault(header=header, skills=skills, jobs=jobs)
    rendered = render_vault(draft, today=today)
    return IngestResult(
        draft=draft,
        rendered=rendered,
        parse_issues=check_rendered(rendered),
        warnings=warnings,
        raws=raws,
        skills=skills,
    )
