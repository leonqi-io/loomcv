"""v3 job-grammar STAR-vault parser (cold→hot sync).

Parses a STAR vault into typed objects with stable bullet IDs and content
hashes for incremental re-indexing. The structure is *job → bullet*:
a ``## [Job]`` block carries employer / title / dates and an optional
``Defaults`` list; every bullet carries its own ``{id: ...}``, an optional
``Project`` label and its own complete ``Keywords`` list.

Locked behaviour:
- Delimiter is exactly ``## [Job]`` (case-sensitive). The older
  ``## [Project]`` grammar is an *error* (never migrated).
  Parsing stops before the "Verification & Sourcing Notes" section.
- ``bullet_id = <job_slug>::<id>`` where ``id`` is the explicit 4–8 character
  ``[a-z0-9]`` key of the bullet's trailing ``{...}`` block (never derived
  from the tags). Stable under every edit of text, tags,
  keywords, label and order. The parser reports a missing id; it never
  invents one.
- ``content_hash`` covers the whitespace-normalised bullet text only: the
  vector side reads the text alone, so only a text change re-embeds.
- Keywords live on the bullet only; the job's ``Defaults`` list is stored on
  the job and is never merged into a bullet (entry-time step of Stage 0).
- Job order is by date (``To`` descending, ``present`` first, then ``From``
  descending, then file order); the position in the file carries no meaning.
- A job with no bullets is a *warning*, not an error (it is a
  job in the vault and takes no part in matching); one optional vault-level
  ``* **Skills**: a; b`` line is read before the first job (``VaultParseResult.skills``,
  ignored by the index); a second one, or one inside a job, is an error.
- Five more optional vault-level lines before the first job —
  ``Name`` / ``Headline`` / ``Summary`` (one line each) and ``Contact`` / ``Education``
  (``;``-lists) — as ``VaultParseResult.header``; each at most once, and an error after the
  first job, like ``Skills``. The ``Skills`` line may be followed by indented group lines
  ``  * **<Group name>**: a; b``; the line's own items are the unnamed group (always first).
  ``skill_groups`` keeps the groups, ``skills`` stays their flat, casefold-deduplicated union.
  An unknown ``* **Key**:`` line before any ``##`` heading is a warning.
- Strict mode — structural defects are collected as
  ``ParseIssue`` records (with line numbers); nothing is dropped silently.
  Callers (``pipeline sync``) must treat any ``error`` as fatal.
- Bullet state lives in a trailing
  ``{id: ..., status: ..., updated: ...}`` block — ``status`` / ``updated``
  feed neither ``bullet_id`` nor ``content_hash``. ``[user_confirmed]`` is
  retired as a leading tag; the parser rejects it with a ``ParseIssue``
  error.

No LLM, no I/O, no config reads — pure text → objects. This module is also
the round-trip validator for the onboarding (ingest) output.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import date
from typing import Literal

from pydantic import BaseModel

VERIFICATION_SECTION_TITLE = "Verification & Sourcing Notes"

_JOB_RE = re.compile(r"^##\s*\[Job\]\s*(?P<name>.+?)\s*$")
_LEGACY_PROJECT_RE = re.compile(r"^##\s*\[Project\]")
_SECTION_RE = re.compile(r"^##\s+(?P<title>[^#].*?)\s*$")
_JOB_LINE_RE = re.compile(r"^\*\s*\*\*(?P<key>[A-Za-z]+)\*\*\s*:\s*(?P<body>.*)$")
_ATTR_LINE_RE = re.compile(r"^\s{2,}\*\s*\*\*(?P<key>[A-Za-z]+)\*\*\s*:\s*(?P<body>.*)$")
# A Skills group line: the name is free text (spaces and commas allowed; `*` is rejected).
_GROUP_LINE_RE = re.compile(r"^\s{2,}\*\s*\*\*(?P<name>.+?)\*\*\s*:\s*(?P<body>.*)$")
_BULLETS_HEADER_RE = re.compile(r"^\*\s*\*\*Bullets\*\*\s*:\s*$")
_BULLET_LINE_RE = re.compile(r"^\s*-\s+(?P<body>.+?)\s*$")
_LEADING_TAGS_RE = re.compile(r"^(?P<tags>(?:\[[^\[\]]+\])+)\s*(?P<text>.*)$")
_TAG_TOKEN_RE = re.compile(r"\[([^\[\]]+)\]")
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_STATUS_BLOCK_RE = re.compile(r"\{([^}]*)\}\s*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_VALUE_RE = re.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2]))?$")
_ID_RE = re.compile(r"^[a-z0-9]{4,8}$")

_JOB_KEYS = ("Employer", "Title", "From", "To", "Defaults")
_RETIRED_KEYS = ("Tags", "Context", "Domain", "Year", "Tech", "Dates", "Keywords")
_ATTR_KEYS = ("Project", "Keywords")
_VAULT_KEYS = ("Name", "Headline", "Contact", "Summary", "Education", "Skills")
_VAULT_LIST_KEYS = ("Contact", "Education")


class ParsedBullet(BaseModel):
    bullet_id: str
    text: str
    tags: list[str]
    content_hash: str
    order: int
    line_no: int
    status: Literal["confirmed", "draft"] = "confirmed"
    updated: date | None = None
    project: str | None = None
    keywords: list[str] = []


class ParsedJob(BaseModel):
    job_id: str
    name: str
    employer: str | None
    title: str | None
    date_from: str
    date_to: str
    defaults: list[str]
    bullets: list[ParsedBullet]
    order: int
    line_no: int


class ParseIssue(BaseModel):
    severity: Literal["error", "warning"]
    line_no: int
    job: str | None
    message: str


class VaultHeader(BaseModel):
    """The vault-level header lines: written at entry, printed verbatim."""

    name: str | None = None
    headline: str | None = None
    contact: list[str] = []
    summary: str | None = None
    education: list[str] = []


class SkillGroup(BaseModel):
    name: str  # "" = the unnamed group: the `* **Skills**:` line's own items
    items: list[str]


class VaultParseResult(BaseModel):
    jobs: list[ParsedJob]
    issues: list[ParseIssue]
    # The flat union of `skill_groups`.
    skills: list[str] = []
    header: VaultHeader = VaultHeader()
    skill_groups: list[SkillGroup] = []

    def errors(self) -> list[ParseIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[ParseIssue]:
        return [i for i in self.issues if i.severity == "warning"]


def slugify(value: str) -> str:
    """Lowercase ASCII slug: non-alphanumeric runs collapse to single ``-``."""
    slug = _SLUG_STRIP_RE.sub("-", value.lower())
    return slug.strip("-")


def content_hash(text: str) -> str:
    """Whitespace-normalized sha1 over the bullet text only.

    Tags, keywords, the project label and the status block are not part of the
    hash: the embedding reads the text alone, so only a text change re-embeds.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    return "sha1:" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def split_list(body: str) -> list[str]:
    """Split a ``;``-separated line: strip each item, drop empties, remove
    case-insensitive duplicates keeping the first spelling. Nothing else —
    no lower-casing, stemming, stop words or splitting on spaces."""
    seen: set[str] = set()
    items: list[str] = []
    for raw in body.split(";"):
        item = raw.strip()
        if not item or item.lower() in seen:
            continue
        seen.add(item.lower())
        items.append(item)
    return items


def flat_skills(groups: Sequence[SkillGroup]) -> list[str]:
    """Every group's items in group order, casefold-deduplicated, the first spelling kept."""
    seen: set[str] = set()
    items: list[str] = []
    for group in groups:
        for item in group.items:
            if item.casefold() not in seen:
                seen.add(item.casefold())
                items.append(item)
    return items


def _date_key(value: str) -> tuple[int, int]:
    """Sort key of a validated ``From`` / ``To`` value: ``YYYY`` sorts as ``YYYY-00``,
    ``present`` after every date."""
    if value == "present":
        return (9999, 99)
    year, _, month = value.partition("-")
    return (int(year), int(month) if month else 0)


_BANNED_LEADING_TAGS = {"user_confirmed", "status", "updated"}


def _parse_status_block(
    text: str, line_no: int, job: str
) -> ParseIssue | tuple[str, str | None, str, date | None]:
    """Parse a trailing status block from a bullet line.

    Returns either a ``ParseIssue`` (reject the bullet) or
    ``(remaining_text, id, status, updated)``; ``id`` is the raw value of the
    block's ``id`` key (``None`` when absent) — its format and uniqueness are
    checked by the caller.
    """
    match = _STATUS_BLOCK_RE.search(text)

    if match:
        inner = match.group(1)
        remaining = text[: match.start()].rstrip()
    else:
        inner = None
        remaining = text

    # Mid-line check: a `{` in remaining text means the block is misplaced
    # (or no block matched but `{` exists mid-line).
    if "{" in remaining:
        return ParseIssue(
            severity="error",
            line_no=line_no,
            job=job,
            message="status block not at line end",
        )

    # Empty-text check
    if not remaining.strip():
        return ParseIssue(
            severity="error",
            line_no=line_no,
            job=job,
            message="bullet has no text content",
        )

    # If no block matched, return defaults
    if inner is None:
        return (remaining, None, "confirmed", None)

    # Parse key-value pairs
    seen_keys: set[str] = set()
    bullet_id: str | None = None
    status: str = "confirmed"
    updated: date | None = None

    for pair in inner.split(","):
        kv = pair.split(":", 1)
        if len(kv) != 2:
            return ParseIssue(
                severity="error",
                line_no=line_no,
                job=job,
                message=f"malformed status block entry '{pair.strip()}'",
            )
        key = kv[0].strip()
        value = kv[1].strip()

        if key not in ("id", "status", "updated"):
            return ParseIssue(
                severity="error",
                line_no=line_no,
                job=job,
                message=f"unknown key '{key}' in status block",
            )
        if key in seen_keys:
            return ParseIssue(
                severity="error",
                line_no=line_no,
                job=job,
                message=f"duplicate key '{key}' in status block",
            )
        seen_keys.add(key)

        if key == "id":
            bullet_id = value
        elif key == "status":
            if value not in ("confirmed", "draft"):
                return ParseIssue(
                    severity="error",
                    line_no=line_no,
                    job=job,
                    message=f"invalid status value '{value}' (expected confirmed or draft)",
                )
            status = value
        elif key == "updated":
            try:
                updated = date.fromisoformat(value)
            except ValueError:
                return ParseIssue(
                    severity="error",
                    line_no=line_no,
                    job=job,
                    message=f"invalid date format '{value}' (expected YYYY-MM-DD)",
                )

    return (remaining, bullet_id, status, updated)


class _BulletDraft:
    """One accepted bullet line while its attribute lines are still being read."""

    def __init__(
        self,
        line_no: int,
        local_id: str,
        tags: list[str],
        text: str,
        status: str,
        updated: date | None,
    ) -> None:
        self.line_no = line_no
        self.local_id = local_id
        self.tags = tags
        self.text = text
        self.status = status
        self.updated = updated
        self.project: str | None = None
        self.keywords: list[str] = []
        self.seen_attrs: set[str] = set()


class _JobDraft:
    """Mutable accumulator while scanning one job block."""

    def __init__(self, name: str, line_no: int) -> None:
        self.name = name
        self.line_no = line_no
        self.employer: str | None = None
        self.title: str | None = None
        self.date_from: str | None = None
        self.date_to: str | None = None
        self.defaults: list[str] = []
        self.seen_keys: set[str] = set()
        self.header_ok = True  # False after a From / To format error (already reported)
        self.in_bullets = False
        self.bullets: list[_BulletDraft] = []
        self.used_ids: set[str] = set()  # over all bullets incl. drafts
        self.attr_target: _BulletDraft | None = None  # most recent *accepted* bullet line
        self.after_bullet = False  # True after ANY bullet line, accepted or rejected


def parse_star_vault(text: str, *, include_draft: bool = False) -> VaultParseResult:
    """Parse a v3 job-grammar STAR vault document. Pure function; strict by default."""
    jobs: list[ParsedJob] = []
    issues: list[ParseIssue] = []
    seen_slugs: set[str] = set()
    draft: _JobDraft | None = None
    header: dict[str, str | list[str] | None] = {}
    groups: list[SkillGroup] = []
    vault_seen: set[str] = set()  # the vault-level keys already read
    # "open" after an accepted Skills line (its group lines follow), "rejected" after a refused
    # one (its group lines are skipped without a second error), None otherwise.
    skills_block: str | None = None
    job_seen = False  # a ``## [Job]`` line was read: the preamble is over
    heading_seen = False  # any ``##`` line was read: the true preamble is over

    def error(line_no: int, current: _JobDraft | None, message: str) -> None:
        issues.append(
            ParseIssue(
                severity="error",
                line_no=line_no,
                job=current.name if current is not None else None,
                message=message,
            )
        )

    def warning(line_no: int, current: _JobDraft | None, message: str) -> None:
        issues.append(
            ParseIssue(
                severity="warning",
                line_no=line_no,
                job=current.name if current is not None else None,
                message=message,
            )
        )

    def finalize(current: _JobDraft | None) -> None:
        if current is None:
            return
        if not current.header_ok:
            return  # a From / To format error was already reported
        missing = False
        for key, value in (("From", current.date_from), ("To", current.date_to)):
            if value is None:
                error(current.line_no, current, f"job is missing its **{key}** line")
                missing = True
        if missing:
            return
        assert current.date_from is not None and current.date_to is not None
        if not current.bullets:
            # A warning: a bullet-less job is a job in the vault.
            warning(current.line_no, current, "job declares no bullets (empty job)")
        slug = slugify(current.name)
        if slug in seen_slugs:
            error(
                current.line_no,
                current,
                f"duplicate job name/slug '{slug}' — later job dropped",
            )
            return
        seen_slugs.add(slug)

        bullets = [
            ParsedBullet(
                bullet_id=f"{slug}::{b.local_id}",
                text=b.text,
                tags=b.tags,
                content_hash=content_hash(b.text),
                order=order,
                line_no=b.line_no,
                status=b.status,  # type: ignore[arg-type]
                updated=b.updated,
                project=b.project,
                keywords=b.keywords,
            )
            for order, b in enumerate(current.bullets)
        ]
        if not include_draft:
            bullets = [b for b in bullets if b.status != "draft"]
        jobs.append(
            ParsedJob(
                job_id=slug,
                name=current.name,
                employer=current.employer,
                title=current.title,
                date_from=current.date_from,
                date_to=current.date_to,
                defaults=current.defaults,
                bullets=bullets,
                order=0,
                line_no=current.line_no,
            )
        )

    def read_job_line(current: _JobDraft, key: str, body: str, line_no: int) -> None:
        if key in _RETIRED_KEYS:
            error(line_no, current, f"'{key}' line is not part of the v3 job header")
            return
        if key not in _JOB_KEYS:
            return  # unknown column-0 keys are ignored, as before
        if key in current.seen_keys:
            error(line_no, current, f"duplicate **{key}** line for this job")
            return
        current.seen_keys.add(key)
        value = body.strip()
        if key == "Employer":
            current.employer = value or None
        elif key == "Title":
            current.title = value or None
        elif key == "Defaults":
            current.defaults = split_list(body)
        elif key == "From":
            if not _DATE_VALUE_RE.match(value):
                error(
                    line_no, current, f"invalid **From** value '{value}' (expected YYYY or YYYY-MM)"
                )
                current.header_ok = False
                return
            current.date_from = value
        elif key == "To":
            if value != "present" and not _DATE_VALUE_RE.match(value):
                error(
                    line_no,
                    current,
                    f"invalid **To** value '{value}' (expected YYYY, YYYY-MM or present)",
                )
                current.header_ok = False
                return
            current.date_to = value

    def read_attr_line(current: _JobDraft, key: str, body: str, line_no: int) -> None:
        if not current.after_bullet:
            error(line_no, current, f"'{key}' attribute line does not follow a bullet")
            return
        target = current.attr_target
        if target is None:
            return  # the preceding bullet was rejected and already carries its error
        if key not in _ATTR_KEYS:
            error(
                line_no,
                current,
                f"unknown bullet attribute '{key}' (expected Project or Keywords)",
            )
            return
        if key in target.seen_attrs:
            error(line_no, current, f"duplicate **{key}** line for this bullet")
            return
        target.seen_attrs.add(key)
        if key == "Project":
            target.project = body.strip() or None
        else:
            target.keywords = split_list(body)

    def read_bullet_line(current: _JobDraft, body: str, line_no: int) -> None:
        current.after_bullet = True
        current.attr_target = None

        tags_match = _LEADING_TAGS_RE.match(body)
        if tags_match is None:
            btags: list[str] = []
            raw_text = body.strip()
        else:
            btags = _TAG_TOKEN_RE.findall(tags_match.group("tags"))
            raw_text = tags_match.group("text").strip()

        # Reject retired/reserved leading tags
        for t in btags:
            if t.lower() in _BANNED_LEADING_TAGS:
                error(line_no, current, f"retired/reserved token '{t}' used as a leading tag")
                return
            if _DATE_RE.match(t):
                error(
                    line_no,
                    current,
                    f"retired/reserved token '{t}' used as a leading tag (date-shaped)",
                )
                return

        # Parse trailing status block (strips it from text)
        result = _parse_status_block(raw_text, line_no, current.name)
        if isinstance(result, ParseIssue):
            issues.append(result)
            return
        clean_text, local_id, status, updated = result

        # Explicit id, required, 4–8 [a-z0-9], unique within the job
        if local_id is None:
            error(line_no, current, "bullet has no id (add {id: …} to its block)")
            return
        if not _ID_RE.match(local_id):
            error(
                line_no,
                current,
                f"invalid id '{local_id}' (expected 4–8 characters from a-z0-9)",
            )
            return
        if local_id in current.used_ids:
            error(line_no, current, f"duplicate id '{local_id}' in this job")
            return
        current.used_ids.add(local_id)

        accepted = _BulletDraft(line_no, local_id, btags, clean_text, status, updated)
        current.bullets.append(accepted)
        current.attr_target = accepted

    def read_group_line(name: str, body: str, line_no: int) -> None:
        name = name.strip()
        if "*" in name:
            error(line_no, None, f"Skills group name '{name}' contains '*'")
            return
        items = split_list(body)
        if not items:
            error(line_no, None, f"Skills group '{name}' has no items")
            return
        if any(g.name.casefold() == name.casefold() for g in groups):
            error(line_no, None, f"duplicate Skills group '{name}'")
            return
        groups.append(SkillGroup(name=name, items=items))

    def read_vault_line(key: str, body: str, line_no: int) -> str | None:
        """One vault-level line; returns the new Skills-block state for a Skills line."""
        if job_seen:
            error(line_no, draft, f"**{key}** is a vault-level line (before the first job)")
            return "rejected"
        if key in vault_seen:
            error(line_no, None, f"duplicate **{key}** line")
            return "rejected"
        vault_seen.add(key)
        if key == "Skills":
            items = split_list(body)
            if items:
                groups.append(SkillGroup(name="", items=items))
            return "open"
        header[key.lower()] = split_list(body) if key in _VAULT_LIST_KEYS else body.strip() or None
        return None

    for line_no, line in enumerate(text.splitlines(), start=1):
        group_match = _GROUP_LINE_RE.match(line)
        if skills_block is not None:
            if group_match is not None:
                if skills_block == "open":
                    read_group_line(group_match.group("name"), group_match.group("body"), line_no)
                continue
            if line.strip():
                skills_block = None  # any other non-blank line ends the block
        elif group_match is not None and not heading_seen and draft is None:
            error(line_no, None, "Skills group line is not under a **Skills** line")
            continue

        if line.startswith("##"):
            heading_seen = True

        if _LEGACY_PROJECT_RE.match(line):
            finalize(draft)
            draft = None  # skip the old block's body without a cascade of errors
            error(
                line_no,
                None,
                "'## [Project]' is the pre-R1 vault grammar (archived, not migrated);"
                " write the vault in the v3 job grammar",
            )
            continue

        job_match = _JOB_RE.match(line)
        if job_match:
            finalize(draft)
            draft = _JobDraft(job_match.group("name"), line_no)
            job_seen = True
            continue

        section_match = _SECTION_RE.match(line)
        if section_match:
            title = section_match.group("title")
            finalize(draft)
            draft = None
            if title == VERIFICATION_SECTION_TITLE:
                break  # documented non-parseable tail section
            issues.append(
                ParseIssue(
                    severity="warning",
                    line_no=line_no,
                    job=None,
                    message=f"unknown section '## {title}' skipped",
                )
            )
            continue

        # The vault-level lines, before the first job.
        key_match = _JOB_LINE_RE.match(line)
        if key_match is not None and key_match.group("key") in _VAULT_KEYS:
            state = read_vault_line(key_match.group("key"), key_match.group("body"), line_no)
            if state is not None:
                skills_block = state
            continue

        if draft is None:
            if key_match is not None and not heading_seen:
                warning(line_no, None, f"unknown vault-level line '**{key_match.group('key')}**'")
            continue  # preamble or skipped section body

        if _BULLETS_HEADER_RE.match(line):
            draft.in_bullets = True
            draft.attr_target = None
            draft.after_bullet = False
            continue
        if (m := _JOB_LINE_RE.match(line)) is not None:
            draft.attr_target = None
            draft.after_bullet = False
            read_job_line(draft, m.group("key"), m.group("body"), line_no)
            continue
        if (m := _ATTR_LINE_RE.match(line)) is not None:
            read_attr_line(draft, m.group("key"), m.group("body"), line_no)
            continue
        if draft.in_bullets and (m := _BULLET_LINE_RE.match(line)) is not None:
            read_bullet_line(draft, m.group("body"), line_no)

    finalize(draft)

    # Job order is by date, not file position. `reverse=True`
    # keeps ties in file order (the sort stays stable).
    jobs.sort(key=lambda j: (_date_key(j.date_to), _date_key(j.date_from)), reverse=True)
    jobs = [job.model_copy(update={"order": order}) for order, job in enumerate(jobs)]
    return VaultParseResult(
        jobs=jobs,
        issues=issues,
        skills=flat_skills(groups),
        header=VaultHeader.model_validate(header),
        skill_groups=groups,
    )
