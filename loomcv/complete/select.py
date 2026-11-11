"""The selection call: per draft bullet, which of the job's `Defaults` apply, plus the
bullet's own keywords.

The keyword call of ingest, moved behind the user's review of each job's `Defaults` and
re-shaped: one call per vault over the **draft** bullets only (a confirmed bullet is neither in
the prompt nor touched — re-selecting it could remove a keyword the user confirmed), each job
listed with its `Defaults`. The reply is `{handle: {"defaults": […], "own": […]}}`;
it is complete or retried once (`complete_validated`). The model proposes which defaults apply
and adds its own; it never removes anything: a bullet's new list = the selected defaults (in
the job's `Defaults` order and spelling) + `own` (in reply order) + whatever the bullet already
carried, case-insensitive dedup keeping the first spelling. A `defaults` item not in the job's
list is dropped with a warning, never added. No word list, no threshold. Node:
`nodes.ingest` — the same as the structure call.

`suggestion_due` is the pure guard the page may put in front of the same call
when a tools-line tick could start it by itself; `select_keywords` is unchanged.

Each job's `Defaults` are offered as they stand, confirmed or not (on the upload
path they are the extracted, unconfirmed list). `first_keywords` is the upload's second call: the
same `select_keywords`, and a failure never refuses the upload. `suggestion_due` has no caller
on the page any more (no run on a tools confirm); it stays, like `open_counts`. The
call also takes a confirmed bullet that has no keyword (`_takes`); its status is never changed.
`only=` narrows the call to named bullets: a bullet's "suggest" is
one short call for that bullet; "Suggest search keywords again" passes nothing and is unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

from loomcv import resources
from loomcv.complete.workbook import CompleteError, WbBullet, Workbook
from loomcv.ingest.keywords import (
    _clean_items,
    _InvalidReply,
    _reject_repeated_keys,
    complete_validated,
    handle,
    normalise_ws,
    render_split,
)
from loomcv.ingest.render import IngestError
from loomcv.llm import GenerationAdapter

# The prompt ships with the package, so its path is package-relative.
DEFAULT_SELECT_PROMPT = resources.path("prompts", "select_keywords.md")


def _string_list(value: Any, handle_name: str, key: str) -> list[str]:
    """`null` or an absent key is `[]`; anything else must be strings."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _InvalidReply(f"{handle_name}: '{key}' is not a list of strings")
    return value


def _validate_selection(text: str, expected: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    try:
        reply = json.loads(text, object_pairs_hook=_reject_repeated_keys)
    except json.JSONDecodeError as exc:
        raise _InvalidReply(f"not valid JSON ({exc.msg})") from exc
    if not isinstance(reply, dict):
        raise _InvalidReply("the reply is not a JSON object")
    for key in expected:
        if key not in reply:
            raise _InvalidReply(f"missing bullet id {key}")
    wanted = set(expected)
    for key in reply:
        if key not in wanted:
            raise _InvalidReply(f"unexpected bullet id {key!r}")
    parsed: dict[str, tuple[list[str], list[str]]] = {}
    for key in expected:
        value = reply[key]
        if not isinstance(value, dict):
            raise _InvalidReply(f"{key}: the value is not an object with 'defaults' and 'own'")
        parsed[key] = (
            _string_list(value.get("defaults"), key, "defaults"),
            _string_list(value.get("own"), key, "own"),
        )
    return parsed


def _takes(bullet: WbBullet) -> bool:
    """A draft bullet, or any bullet with no keyword (a confirmed bullet left without keywords
    is never found by a posting). A confirmed bullet
    with keywords is not taken — re-selecting it could remove a keyword the user confirmed."""
    return bullet.status == "draft" or not bullet.keywords


def select_keywords(
    wb: Workbook,
    *,
    adapter: GenerationAdapter,
    template_path: Path = DEFAULT_SELECT_PROMPT,
    only: Collection[str] | None = None,
) -> tuple[Workbook, list[str]]:
    """One call over every draft bullet and every bullet with no keyword (`_takes`) →
    `(the workbook with the proposed keywords, warnings)`.

    Each job's `Defaults` are offered as they stand, confirmed or not (the
    flags are never changed here). `only` (bullet keys `<job_id>::<id>`) narrows the call to
    those bullets — a bullet's "suggest"; a key the call would not
    take, or one that names no bullet, adds nothing. Raises `CompleteError` when the call fails
    or no valid reply arrives after one retry (`raws` carries every reply). With no such bullet
    there is no call and `(wb, [])` is returned."""
    blocks: list[dict[str, Any]] = []
    # handle -> (job index in wb.jobs, bullet index in that job)
    where: dict[str, tuple[int, int]] = {}
    for ji, job in enumerate(wb.jobs):
        drafts = [
            (bi, b)
            for bi, b in enumerate(job.bullets)
            if _takes(b) and (only is None or f"{job.job_id}::{b.id}" in only)
        ]
        if not drafts:
            continue
        number = len(blocks) + 1
        bullets = []
        for k, (bi, bullet) in enumerate(drafts, start=1):
            name = handle(number, k)
            where[name] = (ji, bi)
            bullets.append({"handle": name, "text": bullet.text})
        blocks.append(
            {
                "index": number,
                "name": job.name,
                "defaults_line": "; ".join(job.defaults) if job.defaults else "(none)",
                "bullets": bullets,
            }
        )
    if not where:
        return wb, []

    system, user = render_split(template_path, jobs=blocks)
    expected = list(where)
    raws: list[str] = []
    try:
        reply = complete_validated(
            adapter,
            system,
            user,
            lambda text: _validate_selection(text, expected),
            raws,
            label="selection",
        )
    except IngestError as exc:
        raise CompleteError(str(exc), raws=exc.raws) from exc

    warnings: list[str] = []
    jobs = [job.model_copy(update={"bullets": list(job.bullets)}) for job in wb.jobs]
    for name in expected:
        ji, bi = where[name]
        job = jobs[ji]
        bullet: WbBullet = job.bullets[bi]
        chosen, own_raw = reply[name]

        by_fold = {d.casefold(): d for d in job.defaults}
        selected: set[str] = set()
        for item in chosen:
            fold = normalise_ws(item).casefold()
            if fold in by_fold:
                selected.add(fold)
            else:
                warnings.append(
                    f"dropped default '{normalise_ws(item)}' for {name} (not in the job's Defaults)"
                )
        own, dropped = _clean_items(own_raw)
        warnings.extend(dropped)

        merged: list[str] = []
        seen: set[str] = set()
        for word in [
            *(d for d in job.defaults if d.casefold() in selected),
            *own,
            *bullet.keywords,
        ]:
            if word.casefold() not in seen:
                seen.add(word.casefold())
                merged.append(word)
        if merged != bullet.keywords:
            job.bullets[bi] = bullet.model_copy(update={"keywords": merged, "updated": None})
    return wb.model_copy(update={"jobs": jobs}), warnings


def first_keywords(
    wb: Workbook,
    *,
    adapter: GenerationAdapter,
    template_path: Path = DEFAULT_SELECT_PROMPT,
) -> tuple[Workbook, list[str], str | None]:
    """The upload's second call: `select_keywords` over every
    draft bullet with the extracted, unconfirmed `Defaults` → `(workbook, warnings, None)`.
    A failure never refuses the upload: `(wb unchanged, [], the failure's sentence)` — the
    profile is written without keywords and each bullet offers "suggest"."""
    try:
        new, warnings = select_keywords(wb, adapter=adapter, template_path=template_path)
    except CompleteError as exc:
        return wb, [], str(exc)
    return new, warnings, None


def suggestion_due(wb: Workbook) -> bool:
    """The keyword suggestion should run by itself: the
    workbook has a job, every job's `Defaults` are confirmed, and no bullet of any job has a
    keyword — the suggestion has never produced anything. Pure; no call."""
    if not wb.jobs:
        return False
    if not all(job.defaults_confirmed for job in wb.jobs):
        return False
    return not any(bullet.keywords for job in wb.jobs for bullet in job.bullets)
