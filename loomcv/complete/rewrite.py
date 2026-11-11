"""The drafting call: one STAR-shaped résumé bullet from a plain bullet or from guided answers
(the model drafts, the user edits with no model call, and may ask
again).

One call on `nodes.rewrite` (the node's temperature, never a literal). The prompt is given
exactly the facts it may use — the job's name / employer / title, the bullet's current text
(empty for a bullet that starts from nothing), the keywords of a **confirmed** bullet (a draft
bullet's keywords are stage B's proposal, not the user's confirmation), the user's Situation /
Action / Result answers, the length budget and the earlier proposals of this session — and the
reply is `{"bullet": string}`. Pure: the workbook is not changed. The proposal reaches the vault
only through `apply_bullets` (an edit) or `add_bullet` (a new bullet), on the user's click, and
the `--accept-all` path never calls this.

No call is made, and `CompleteError` is raised, when the job or the bullet key is unknown, or
when there is nothing to write from (no bullet text and all three answers empty).

Zero new facts, entry-side form: the code checks the one class of invention that is both common
and mechanically checkable — figures — by conservation. A figure is an optional currency sign
(`$ € £ ¥ ₹`), a digit run (groups separated by `,` / thin / no-break space, one `.` decimal
part) and at most one attached suffix: `%`, `+`, `x`, `k` / `m` / `b` (when no letter follows)
or a CJK unit `万 亿 千 百`. Its normal form drops whitespace and separators and is casefolded:
`900 ms` and `900ms` are both `900`, `70 %` is `70%`, `1,200` is `1200`, `10K` is `10k`,
`30 万` is `30万`, `3X` is `3x`; digits inside a name (`p95`, `S3`) count, the inputs being
scanned the same way. A proposal figure passes when its normal form equals an input figure's,
or that figure with its currency sign and / or `+` removed (dropping a sign is not inventing):
input `$1.2M` accepts `1.2m`, input `40+` accepts `40`; input `40` does not accept `40+`, input
`900` does not accept `70%`, input `10k` does not accept `10`. Numbers written as words are not
checked (the prompt forbids them; the user's Apply is the gate). A miss makes the first attempt
invalid (the retry names the figure); a miss that survives the retry is a warning and the
proposal is still returned — nothing is dropped, cut or edited. A proposal over `budget`
characters is returned with a warning too.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loomcv import resources
from loomcv.complete.workbook import BulletDraftAnswers, CompleteError, Workbook
from loomcv.ingest.keywords import _InvalidReply, complete_validated, normalise_ws, render_split
from loomcv.ingest.render import IngestError
from loomcv.llm import GenerationAdapter

# The prompt ships with the package, so its path is package-relative.
DEFAULT_DRAFT_PROMPT = resources.path("prompts", "draft_bullet.md")

_SEPARATORS = ",  "
_FIGURE_RE = re.compile(
    r"(?P<cur>[$€£¥₹]\s*)?"
    rf"(?P<num>\d+(?:[{_SEPARATORS}]\d{{3}}(?!\d))*(?:\.\d+)?)"
    r"(?P<suffix>\s*%|\+|x(?![^\W\d_])|[kmb](?![^\W\d_])|\s*[万亿千百])?",
    re.IGNORECASE,
)


def _parts(match: re.Match[str]) -> tuple[str, str, str]:
    currency = (match.group("cur") or "").strip()
    number = re.sub(f"[{_SEPARATORS}]", "", match.group("num"))
    suffix = re.sub(r"\s", "", match.group("suffix") or "").casefold()
    return currency, number, suffix


def _accepted(match: re.Match[str]) -> set[str]:
    """The normal forms an input figure lets a proposal carry (dropping a sign is not inventing)."""
    currency, number, suffix = _parts(match)
    forms = {currency + number + suffix, number + suffix}
    if suffix == "+":
        forms |= {currency + number, number}
    return forms


def _missing_figures(proposal: str, inputs: str) -> list[str]:
    """The figures of `proposal`, as written, whose normal form no input figure accepts; one per
    normal form, in order of appearance."""
    allowed: set[str] = set()
    for match in _FIGURE_RE.finditer(inputs):
        allowed |= _accepted(match)
    missing: list[str] = []
    seen: set[str] = set()
    for match in _FIGURE_RE.finditer(proposal):
        form = "".join(_parts(match))
        if form not in allowed and form not in seen:
            seen.add(form)
            missing.append(match.group(0))
    return missing


def _validate_shape(text: str) -> str:
    try:
        reply: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _InvalidReply(f"not valid JSON ({exc.msg})") from exc
    if not isinstance(reply, dict) or not isinstance(reply.get("bullet"), str):
        raise _InvalidReply("the reply is not an object with a string 'bullet'")
    bullet = normalise_ws(reply["bullet"])
    if not bullet:
        raise _InvalidReply("the bullet is empty")
    return bullet


def draft_bullet(
    wb: Workbook,
    job_id: str,
    *,
    bullet_key: str | None,
    answers: BulletDraftAnswers,
    adapter: GenerationAdapter,
    budget: int,
    template_path: Path = DEFAULT_DRAFT_PROMPT,
    previous: Sequence[str] = (),
) -> tuple[str, list[str]]:
    """One call → `(the proposal, warnings)`; `bullet_key` is `<job_id>::<id>`, or `None` for a
    bullet that starts from nothing.

    Raises `CompleteError` without a call when the job / bullet is unknown or there is nothing
    to write from, and when no valid reply arrives after one retry (`raws` carries every
    reply)."""
    job = next((j for j in wb.jobs if j.job_id == job_id), None)
    if job is None:
        raise CompleteError(f"unknown job '{job_id}'")
    text = ""
    keywords: list[str] = []
    if bullet_key is not None:
        bullet = next((b for b in job.bullets if f"{job_id}::{b.id}" == bullet_key), None)
        if bullet is None:
            raise CompleteError(f"unknown bullet '{bullet_key}' in job '{job_id}'")
        text = bullet.text
        if bullet.status == "confirmed":
            keywords = list(bullet.keywords)
    given = [
        {"label": label, "value": normalise_ws(value)}
        for label, value in (
            ("Situation", answers.situation),
            ("Action", answers.action),
            ("Result", answers.result),
        )
        if normalise_ws(value)
    ]
    if not text and not given:
        raise CompleteError("nothing to draft from: type at least one answer")

    system, user = render_split(
        template_path,
        job_name=job.name,
        employer=job.employer,
        title=job.title,
        text=text,
        answers=given,
        keywords=keywords,
        budget=budget,
        previous=list(previous),
    )
    inputs = "\n".join(
        [job.name, job.employer or "", job.title or "", text]
        + [a["value"] for a in given]
        + keywords
    )

    attempts = 0

    def validate(reply: str) -> str:
        nonlocal attempts
        attempts += 1
        bullet = _validate_shape(reply)
        if attempts == 1:
            missing = _missing_figures(bullet, inputs)
            if missing:
                raise _InvalidReply(
                    "the bullet contains a figure that is not in the input: "
                    + ", ".join(missing)
                    + ". Use only the figures given; if none is given, write the sentence "
                    "without one"
                )
        return bullet

    raws: list[str] = []
    try:
        proposal = complete_validated(adapter, system, user, validate, raws, label="drafting")
    except IngestError as exc:
        raise CompleteError(str(exc), raws=exc.raws) from exc

    warnings = [
        f"contains a figure not in your input: {figure}"
        for figure in _missing_figures(proposal, inputs)
    ]
    if len(proposal) > budget:
        warnings.append(f"over the budget ({len(proposal)} > {budget})")
    return proposal, warnings
