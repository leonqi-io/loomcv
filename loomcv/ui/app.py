"""The profile page: default components, no logic.

Every handler is one call into `loomcv.complete` plus the write of the working copy; the `for`
loops only lay components out. Whatever the module refuses — a malformed date, a `;` inside a
keyword — is shown as the module wrote it (a warning toast and a small line under the box that
was refused).

The upload fills every bullet's keywords
(the shell); the tools list is no longer a gate — no run on a tools confirm, no warning line, no
waiting line on a bullet; "Suggest search keywords again" and a bullet's "suggest" are the user's
and do not depend on the tools lists.

The edit model: every editable box commits on its own event (a text box on `blur`, a one-line
box also on `submit`) as a *partial* answer to the module function that owns the field; each
accepted commit writes the working copy and its sidecar (`loomcv.complete.sidecar`); a refused
one writes nothing and the box keeps what was typed. A field commit never redraws anything: its
outputs are targeted components.

The page is task-shaped. A left list
(one entry per job, About you, Your skills, Save profile) is the only navigation; the pane beside
it is one `@gr.render` whose triggers are the page load, the layout state and `selected` — a
navigation draws one job, an edit never redraws. Everything editable is text with an "edit" link;
"edit" turns that one line into a box in its own place with Done / cancel under it (a targeted
visibility toggle; cancel is itself a commit of the opening value). Two commits send a pair: a
tools-list edit also un-confirms the list, a text edit also un-confirms the bullet — each only
when the field actually changes. Every long call (the keyword step, drafting a stronger bullet,
Sort into groups, Save profile) runs under the page's veil from `loomcv.ui.common`, and every
model call inside `common.model_turn()` (one rule: the page is locked while a model call
runs). A bullet's
"suggest" asks for that bullet only; "Suggest search keywords again" for every bullet the
step takes. Save profile is
an injected callable (`save_profile=`): the shell builds it from the runtime; `pipeline ui` passes
a write-only one. The CSS this page names is `CHIP_CSS` (the name kept for the shell); no JS.
Not covered by pytest (the `ui` group is not installed in CI).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError

import gradio as gr

from loomcv.app_state import saved_stamp
from loomcv.complete import (
    BulletAnswer,
    BulletDraftAnswers,
    BulletsAnswers,
    CompleteError,
    HeaderAnswers,
    JobAnswer,
    JobsAnswers,
    SkillGroup,
    SkillsAnswers,
    Workbook,
    add_bullet,
    apply_bullets,
    apply_header,
    apply_jobs,
    apply_skill_groups,
    checklist,
    confirm_unedited,
    confirmed_counts,
    delete_bullet,
    draft_bullet,
    layout,
    load_profile,
    maybe_also,
    propose_groups,
    save_working_copy,
    select_keywords,
    split_list,
    with_prefill,
)
from loomcv.complete.workbook import empty_workbook
from loomcv.llm import build_generation_adapter, load_node_config
from loomcv.state import load_generation_config
from loomcv.ui import common

if TYPE_CHECKING:
    from loomcv.steps import SaveReport

# Every handler of the profile side runs one at a time and in order: nothing reads `state` before
# a queued commit has written it. A commit is quick, so it shows no pending frame.
SERIAL = {"concurrency_limit": 1, "concurrency_id": "profile"}
QUIET = {**SERIAL, "show_progress": "hidden"}
# A commit event may fire again while the previous one is still queued: run every one of them.
EVENT = {**QUIET, "trigger_mode": "multiple"}
# (Drafting a stronger bullet has no queue of its own — it
# runs on this one under the page's veil, and its model call takes `common.model_turn()`.)
# The left list is a static pool of job entries: a job past this one has no entry.
MAX_JOBS = 30

# This page's named CSS (the name kept so the shell
# imports it unchanged): the left-list entry (three lines, left-aligned, a done job green); the
# card (an accent bar on the current card, the edit links and actions shown on it only, a
# confirmed card faded); the dashed Add card; Save light until everything is confirmed; the
# grey hint and the refusal line. The card's bar / tint / hidden actions live in
# `common.LOOKS_CSS`; "Looks right?" sits against its Confirm at the right (`lc-state`); a
# refusal reads as a warning, not a hint; the drafting panel is tinted and joined to its card.
CHIP_CSS = (
    ".lc-list .lc-entry {white-space: pre-line; text-align: left; "
    "justify-content: flex-start;} "
    ".lc-list .lc-entry.ok {color: #14532d; background: #dcfce7;} "
    ".lc-card.done {opacity: 0.6; background: #f0fdf4;} "
    ".lc-state {justify-content: flex-end; align-items: center; flex-wrap: nowrap;} "
    ".lc-state > * {flex: 0 0 auto !important; min-width: 0 !important; width: auto;} "
    ".lc-add {border: 1px dashed var(--border-color-primary); border-radius: 6px;} "
    ".gradio-container button.lc-save.primary:not(.ready) "
    "{background: var(--button-secondary-background-fill); color: var(--body-text-color); "
    "border-color: var(--border-color-primary);} "
    ".lc-panel {position: relative; background: var(--lc-tint); "
    "border-left: 4px solid var(--lc-accent); border-radius: 0 0 6px 6px; padding: 8px 12px; "
    "margin-top: calc(-1 * var(--layout-gap));} "
    ".lc-grey {color: var(--body-text-color-subdued); font-size: var(--text-sm);} "
    ".gradio-container .lc-refused p {color: #7a4b00; background: #fbefd3; "
    "border-left: 3px solid #9a6400; padding: 4px 8px; margin: 0; border-radius: 0 4px 4px 0;} "
    # Layout only: an entry's first line is the name, the rest small mono details;
    ".lc-list > button.lc-entry {display: block; font-size: 11px; font-family: var(--font-mono); "
    "font-weight: 400;} "
    ".lc-list > button.lc-entry::first-line {font-size: 13px; font-family: var(--font); "
    "font-weight: 600;} "
    # the heading, its dates and "edit title / dates" on one line;
    ".lc-headrow {flex-wrap: wrap; align-items: baseline; gap: 4px 12px;} "
    ".lc-headrow > * {flex: 0 1 auto !important; min-width: 0 !important; width: auto;} "
    ".lc-headrow h3 {margin: 0;} "
    # "edit" (and "suggest") right after the last word of the text, in the text's flow;
    ".lc-inline {display: block !important;} "
    ".lc-inline > .block, .lc-inline > .block > div:not(.wrap), .lc-inline .prose, "
    ".lc-inline .md, .lc-inline .md > p:last-child {display: inline !important;} "
    ".lc-inline > .block {padding: 0 !important;} "
    ".lc-inline .wrap.hide {display: none !important;} "
    ".lc-inline > button:not(.hidden) {display: inline-flex !important; width: auto !important; "
    "min-width: 0 !important; margin-left: 6px; vertical-align: baseline;} "
    # the tools card and each bullet card framed (left edge: the current-card bar);
    ".lc-box {border-top: 1px solid var(--border-color-primary); "
    "border-right: 1px solid var(--border-color-primary); "
    "border-bottom: 1px solid var(--border-color-primary); border-radius: 0 4px 4px 0; "
    "padding: 8px 12px 8px 8px;} "
    ".lc-tools:not(.done) {background: var(--background-fill-secondary);} "
    # the job's foot line under a rule.
    ".lc-foot {border-top: 1px solid var(--border-color-primary); padding-top: 10px;}"
)

_FAILURES = (
    CompleteError,
    KeyError,
    ValueError,
    RuntimeError,
    NotImplementedError,
    URLError,
    OSError,
)

# ----------------------------------------------------------------------------- words
# The wording is judged on the screen.
LANDING_HEAD = "## We read your résumé: {jobs}, {bullets}."
# The landing says how the whole page works, in order.
LANDING_BODY = (
    "Now check that we read it right, then save. Work down the list on the left:\n\n"
    "1. **Each job.** Look at *Tools you used here* first — fix the list if it's wrong, then "
    "**Confirm** it. Then read each bullet: right → **Confirm**; wrong → *edit* the text "
    "first. *Confirm all … I haven't edited* confirms the bullets you left as we read them.\n"
    "2. **About you** — your name and contact are printed at the top of every résumé; the "
    "headline, summary and education are optional.\n"
    "3. **Your skills** — the Skills section of the résumé, in groups.\n"
    "4. **Save profile** — you can save at any time, even before everything is confirmed.\n\n"
    "Everything you change here is kept as you go — come back any time.\n\n"
    "#### Search keywords\n"
    "Each bullet carries search keywords. When you apply to a posting, LoomCV uses them to "
    "find the bullets that match what the posting asks for — **a bullet with no keywords is "
    "never picked**. We pick them when we read your résumé; change a bullet's with *edit* on "
    "its keyword line.\n\n"
    "- **suggest** (on a bullet that has no keywords) picks keywords for that one bullet — a "
    "short wait.\n"
    "- **Suggest search keywords again** (at the end of each job) picks them again for every "
    "unconfirmed bullet and every bullet without keywords, in all your jobs — this can take "
    "several minutes. Keywords are only ever added, never taken away.\n\n"
    "#### The tools list\n"
    "*(none listed)* means your résumé didn't name the tools for that job in a list of their "
    "own. It's worth adding the ones you used there: they're offered as keywords for that "
    "job's bullets, and *Add the tools from your jobs that are missing here* (in Your skills) "
    "copies them into your Skills. If there's nothing worth listing, leave it empty and "
    "confirm it.\n\n"
    "#### ✨ improve this bullet\n"
    "Asks three short questions — what was going on, what you did, what changed — and drafts "
    "a stronger version of the bullet for you to use, try again, or ignore. **It never "
    "invents numbers:** only figures you type or that are already in the bullet appear.\n\n"
    "#### After Save profile\n"
    "LoomCV writes your profile and builds its search index from your confirmed bullets — a "
    "short wait. When that's done you're taken to **Applications**, where you paste a posting "
    "and get a résumé built for it. Only confirmed bullets are used there.\n\n"
    "**Pick a job on the left to start.** *How this page works* (on the left) brings you back "
    "here."
)
HOW_IT_WORKS = "How this page works"
LANDING_NOTE = (
    "An old or two-column résumé can come out split oddly — fix what reads wrong as you go."
)
LANDING_EMPTY = "Nothing here yet."

EDIT = "edit"
DONE = "Done"
CANCEL = "cancel"
CONFIRM = "Confirm"
LOOKS_RIGHT = "Looks right?"
CONFIRMED = "✓ Confirmed"
UNDO = "undo"

EDIT_HEAD = "edit title / dates"
HEAD_HINT = (
    "Saved when you leave a box. A date that isn't YYYY or YYYY-MM is refused and the box keeps "
    "what you typed."
)
HEAD_LABELS = {
    "title": "Title",
    "employer": "Employer",
    "date_from": "From (YYYY-MM)",
    "date_to": "To (YYYY-MM or present)",
}

TOOLS_HEAD = "**TOOLS YOU USED HERE**"
TOOLS_NONE = "(none listed)"
TOOLS_LABEL = "Tools — separate with ;"

COUNT_LINE = "**{n} BULLETS · {x} CONFIRMED**"
CONFIRM_ALL = "Confirm all {n} bullets I haven't edited"
CONFIRM_ONE = "Confirm the 1 bullet I haven't edited"
CONFIRMED_TOAST = "{n} bullets confirmed"
CONFIRMED_TOAST_ONE = "1 bullet confirmed"
# The job's foot line.
FOOT_LEFT = "{n} bullets left to check in this job · pick the next job on the left when you're done"
FOOT_ONE = "1 bullet left to check in this job · pick the next job on the left when you're done"
FOOT_DONE = "Every bullet in this job is confirmed · pick the next job on the left"
FOOT_EMPTY = "No bullets in this job yet · add one above, or pick the next job on the left"

TEXT_HINT = "Saved when you leave the box, too."
KW_LINE = "Keywords for matching postings: {keywords}"
KW_NONE_YET = "Keywords for matching postings: none yet"
KW_NONE_CONFIRMED = "Keywords for matching postings: none — this bullet won't be found by a posting"
KW_LABEL = "Keywords — separate with ;"
KW_SHOWN = 5
MAYBE = "Maybe also: {tools}"
SUGGEST = "suggest"
# The hint says what the link does — this bullet only.
SUGGEST_HINT = "suggest picks search keywords for this bullet only"
SUGGEST_AGAIN = "Suggest search keywords again"
KEYWORDS_DONE = "Keywords added to your bullets."
KEYWORDS_FAILED = "The keyword step didn't come through: {exc}"

IMPROVE_LINK = "✨ improve this bullet"
REMOVE = "remove"
REMOVE_QUESTION = "Remove this bullet? This cannot be undone."
REMOVED = "Bullet removed."

PANEL_HEAD = "#### Make this bullet stronger"
PANEL_INTRO = (
    "Answer what you can, one line each; an empty box means we work from the bullet as written. "
    "Numbers are never invented: only figures you type here or that are already in the bullet "
    "will appear."
)
STAR_LABELS = {
    "situation": "What was going on?",
    "action": "What did you do?",
    "result": "What changed?",
}
DRAFT = "Draft a better version"
SUGGESTED = "SUGGESTED"
USE = "Use this"
TRY_AGAIN = "try again"
KEEP = "keep mine"
NO_PROPOSAL = "Draft a better version first."
DRAFT_FAILED = "The draft didn't come through: {exc}"

ADD_HEAD = "**+ Something missing from this job?**"
ADD_PLACEHOLDER = "Type what you did, in one sentence…"
ADD = "Add"
ADDED = "Added. Confirm it when it reads right — or improve it first."

ABOUT = "About you"
ABOUT_HEAD = "## About you"
ABOUT_GREY = "Printed at the top of every résumé."
ABOUT_LABELS = {
    "name": "Name",
    "contact": "Contact — City, Province; email; phone; linkedin.com/in/you; github.com/you",
    "headline": "Headline (optional) — one line under your name",
    "summary": "Summary (optional)",
}
EDU_LABEL = "Education — degree, school, year (printed as written)"
ADD_EDU = "+ Add new"
ABOUT_SAVED = "Saved as you leave each box."
ABOUT_OK = "✓ name and contact filled"
NAME_MISSING = "name missing"
NO_CONTACT = "no contact"

SKILLS = "Your skills"
SKILLS_HEAD = "## Your skills"
SKILLS_GREY = "Printed as the Skills section of the résumé."
SKILLS_SENTENCE = (
    "Each group prints as one line, in this order. A group with no name prints as a plain list "
    "with no label."
)
GROUP_NAME = "Group name (optional)"
GROUP_SKILLS = "Skills — separate with ;"
REMOVE_GROUP = "remove this group"
ADD_GROUP = "+ Add a group"
SORT = "✨ Sort these into groups for me"
FILL = "Add the tools from your jobs that are missing here"
FILL_NONE = "Every tool from your jobs is already here."
GROUPS_DONE = "Sorted into groups — change anything you like."
GROUPS_FAILED = "Sorting didn't come through: {exc}"
NO_SKILLS = "no skills yet"

SAVE = "Save profile"
SAVE_LINE = "{x} of {n} confirmed — you can save any time"
SAVED_AT = " · saved {hhmm}"
SAVED_READY = "Profile saved — ready to apply."
SAVED_ONLY = "Profile saved."
SAVED_NOT_READY = "Saved — but it can't be used for applications yet: {reason}"
SAVE_FAILED = "Your profile could not be saved to disk."

BAR_LINE = "{m} bullets · {x} confirmed"
BAR_SAVED = "Saved ✓ {hhmm}"
BAR_PROFILE_SAVED = "Profile saved ✓ {hhmm}"
BAR_NOT_SAVED = "Not saved to disk"
NOT_WRITTEN = "Your change could not be written to disk."
PROBLEMS = "Saved, but this may not print right: {problems}"
TOO_MANY_JOBS = f"Only the first {MAX_JOBS} jobs are listed on the left."

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# ----------------------------------------------------------------------------- formatting


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _now() -> str:
    return saved_stamp(datetime.now())


def _mtime(path: Path) -> str | None:
    """The file's save time; dated when it is not today."""
    try:
        return saved_stamp(datetime.fromtimestamp(path.stat().st_mtime))
    except OSError:
        return None


UNKNOWN_DATE = "date unknown"


def _date_text(value: str) -> str:
    """`2019-03` → `Mar 2019`; `2019`, `present` as they are; the unknown `0000` in words."""
    if value == "0000":
        return UNKNOWN_DATE
    year, _, month = value.partition("-")
    if month.isdigit() and 1 <= int(month) <= 12:
        return f"{_MONTHS[int(month) - 1]} {year}"
    return value


def _year(value: str) -> str:
    return UNKNOWN_DATE if value == "0000" else value if value == "present" else value[:4]


def _job_of(wb: Workbook, job_id: str):
    return next((job for job in wb.jobs if job.job_id == job_id), None)


def _bullet_of(wb: Workbook, job_id: str, bullet_id: str):
    job = _job_of(wb, job_id)
    if job is None:
        return None
    return next((b for b in job.bullets if b.id == bullet_id), None)


def _first_selected(wb: Workbook) -> str | None:
    """Nothing confirmed yet → the landing (`None`); otherwise the
    first job with an unconfirmed bullet or tools list, else the first job."""
    touched = any(
        job.defaults_confirmed or any(b.status == "confirmed" for b in job.bullets)
        for job in wb.jobs
    )
    if not touched:
        return None
    for job in wb.jobs:
        if not job.defaults_confirmed or any(b.status != "confirmed" for b in job.bullets):
            return job.job_id
    return wb.jobs[0].job_id


def _job_done(job, counts: tuple[int, int]) -> bool:
    confirmed, total = counts
    return job.defaults_confirmed and confirmed == total


def _all_done(wb: Workbook) -> bool:
    counts = confirmed_counts(wb)
    return all(_job_done(job, counts[job.job_id]) for job in wb.jobs)


def _totals(wb: Workbook) -> tuple[int, int]:
    counts = confirmed_counts(wb).values()
    return sum(c for c, _ in counts), sum(n for _, n in counts)


def _job_label(job, counts: tuple[int, int]) -> str:
    confirmed, total = counts
    name = job.employer or job.title or job.name
    title = job.title if job.employer else None  # without an employer the title is the name
    sub = " · ".join(p for p in (title, f"{_year(job.date_from)}–{_year(job.date_to)}") if p)
    if total == 0:
        state = (
            "✓ tools list confirmed"
            if job.defaults_confirmed
            else "no bullets · tools list to confirm"
        )
    elif _job_done(job, counts):
        state = f"✓ all {total} confirmed"
    else:
        state = f"{confirmed} of {total} confirmed"
    return f"{name}\n{sub}\n{state}"


def _entry_classes(on: bool, ok: bool) -> list[str]:
    return ["lc-entry", *(["on"] if on else []), *(["ok"] if ok else [])]


def _job_entry(wb: Workbook, job_id: str, *, on: bool):
    job = _job_of(wb, job_id)
    if job is None:
        return gr.skip()
    counts = confirmed_counts(wb)[job_id]
    return gr.Button(
        value=_job_label(job, counts),
        elem_classes=_entry_classes(on, _job_done(job, counts)),
        visible=True,
    )


def _about_missing(wb: Workbook) -> list[str]:
    """What the header still lacks, in the user's words, from `checklist`'s `header_field`."""
    empty = {i.message.split(" ")[0] for i in checklist(wb) if i.kind == "header_field"}
    return [w for f, w in (("Name", NAME_MISSING), ("Contact", NO_CONTACT)) if f in empty]


def _about_entry(wb: Workbook, *, on: bool):
    head = wb.header
    who = " · ".join(p for p in (head.name, head.contact[0] if head.contact else None) if p)
    missing = _about_missing(wb)
    lines = [ABOUT, who or "—", " · ".join(missing) or ABOUT_OK]
    return gr.Button(value="\n".join(lines), elem_classes=_entry_classes(on, not missing))


def _skills_entry(wb: Workbook, *, on: bool):
    n, m = len(wb.skills), len(wb.skill_groups)
    line = f"{_plural(n, 'skill')} · {_plural(m, 'group')}" if n else NO_SKILLS
    return gr.Button(value=f"{SKILLS}\n{line}", elem_classes=_entry_classes(on, n > 0))


def _save_button(wb: Workbook):
    return gr.Button(elem_classes=["lc-save", "ready"] if _all_done(wb) else ["lc-save"])


def _save_line(wb: Workbook, mem: dict) -> str:
    confirmed, total = _totals(wb)
    stamp = mem.get("saved_at")
    return SAVE_LINE.format(x=confirmed, n=total) + (SAVED_AT.format(hhmm=stamp) if stamp else "")


def _bar_line(wb: Workbook) -> str:
    confirmed, total = _totals(wb)
    return BAR_LINE.format(m=total, x=confirmed)


def _heading(job) -> str:
    title = " — ".join(p for p in (job.title or job.name, job.employer) if p)
    return f"### {title}"


def _dates(job) -> str:
    return f"{_date_text(job.date_from)} – {_date_text(job.date_to)}"


def _foot(wb: Workbook, job_id: str) -> str:
    """The job's foot line: what is left here, in words."""
    confirmed, total = confirmed_counts(wb)[job_id]
    left = total - confirmed
    if total == 0:
        return FOOT_EMPTY
    if left == 0:
        return FOOT_DONE
    return (FOOT_ONE if left == 1 else FOOT_LEFT).format(n=left)


def _tools_text(job) -> str:
    return "; ".join(job.defaults) or TOOLS_NONE


def _suggestable(wb: Workbook, bullet) -> bool:
    """The "suggest" link: any bullet with no keyword, confirmed or not, whatever the tools lists
    say."""
    return not bullet.keywords


def _kw_line(wb: Workbook, bullet) -> str:
    if bullet.keywords:
        shown = " · ".join(bullet.keywords[:KW_SHOWN])
        return KW_LINE.format(keywords=shown + (" …" if len(bullet.keywords) > KW_SHOWN else ""))
    if bullet.status == "confirmed":
        return KW_NONE_CONFIRMED
    return KW_NONE_YET


def _maybe_text(job, bullet) -> str:
    tools = maybe_also(job, bullet)
    return MAYBE.format(tools="; ".join(tools)) if tools else ""


def _card_classes(done: bool, *extra: str) -> list[str]:
    base = ["lc-card", "lc-box", *extra]
    return [*base, "done"] if done else base


@dataclass
class _Pair:
    """One editable line: its text + "edit" (`show`) and its box group (`edit`)."""

    key: str
    show: Any
    edit: Any
    boxes: list
    refusal: Any
    snap: Callable[[Workbook], tuple[Any, list[str]]]  # (opening snapshot, the boxes' values)
    restore: Callable[[Workbook, Any], Workbook]  # the snapshot as a commit


# ----------------------------------------------------------------------------- the page


def build(
    page_over: Path | None,
    saved: Path,
    *,
    working_copy: Path,
    sidecar: Path,
    config: Path | None = None,
    bar: gr.State | None = None,
    save_profile: Callable[..., SaveReport] | None = None,
    left_foot: Callable[[], None] | None = None,
) -> gr.Blocks:
    """The profile page over `page_over` (`None`: an empty workbook); every commit writes
    `working_copy` and `sidecar`; Save profile calls `save_profile(wb, today=…)` (not drawn when
    `None`). `saved` is read for its time only. Returns the page, not launched. `left_foot` is
    called inside the sticky left list, directly under "How this page works", for the caller to
    mount its own components there (the shell's Start over); this page wires nothing on
    them.

    The shell loads a workbook into the page after launch through `.hand_off(wb, warnings)`, which
    returns one value per component of `.hand_off_targets`; `.saved_signal` is a counter bumped
    after a Save that made the profile ready for applications."""
    # The length budget of the drafting call: the shipped `generation.yaml` (no flag for it).
    budget = load_generation_config().bullet_max_chars
    epochs = itertools.count(1)
    bars = [bar] if bar is not None else []

    if page_over is None:
        wb0, warnings0 = empty_workbook(), []
    else:
        wb0, warnings0 = load_profile(page_over, working_copy=working_copy, sidecar=sidecar)
    sel0 = _first_selected(wb0)

    # ------------------------------------------------------------------ commits

    def persist(new: Workbook) -> str:
        """Write the working copy and its sidecar; the bar's save slot (problems never block)."""
        try:
            report = save_working_copy(new, working_copy, sidecar=sidecar, today=date.today())
        except OSError:
            gr.Warning(NOT_WRITTEN)
            return BAR_NOT_SAVED
        if report.problems:
            gr.Warning(PROBLEMS.format(problems="; ".join(report.problems)))
        return BAR_SAVED.format(hhmm=_now())

    def commit(wb: Workbook, change: Callable[[Workbook], Workbook]):
        """One commit: `(workbook, refusal sentence or "", bar save text or None)`. A refusal
        keeps the workbook and the file and shows a toast; no change writes nothing."""
        try:
            new = change(wb)
        except CompleteError as exc:
            gr.Warning(str(exc))
            return wb, str(exc), None
        if new == wb:
            return wb, "", None
        return new, "", persist(new)

    def bar_out(new: Workbook, save: str | None) -> list:
        if not bars:
            return []
        return [gr.skip() if save is None else (_bar_line(new), save)]

    def bar_skip() -> list:
        return [gr.skip()] * len(bars)

    def adapter(node: str):
        node_cfg, provider = load_node_config(node, Path(config) if config else None)
        return build_generation_adapter(node_cfg, provider)

    def all_entries(wb: Workbook, sel) -> list:
        """Every left-list entry as the workbook says (label, classes, shown)."""
        jobs = [
            _job_entry(wb, wb.jobs[i].job_id, on=wb.jobs[i].job_id == sel)
            if i < len(wb.jobs)
            else gr.Button(value="", visible=False)
            for i in range(MAX_JOBS)
        ]
        return [
            *jobs,
            _about_entry(wb, on=sel == "about"),
            _skills_entry(wb, on=sel == "skills"),
        ]

    def entry_classes(wb: Workbook, sel) -> list:
        """The entries' classes only (a navigation): the `on` mark moves, nothing is relabelled."""
        counts = confirmed_counts(wb)
        jobs = []
        for i in range(MAX_JOBS):
            if i < len(wb.jobs):
                job = wb.jobs[i]
                ok = _job_done(job, counts[job.job_id])
                jobs.append(gr.Button(elem_classes=_entry_classes(job.job_id == sel, ok)))
            else:
                jobs.append(gr.skip())
        about_ok = not _about_missing(wb)
        return [
            *jobs,
            gr.Button(elem_classes=_entry_classes(sel == "about", about_ok)),
            gr.Button(elem_classes=_entry_classes(sel == "skills", bool(wb.skills))),
        ]

    # --------------------------------------------------------------------- the layout

    with gr.Blocks() as app:
        state = gr.State(wb0)
        # Earlier proposals of this session, by bullet key: fed back into the drafting prompt,
        # never written anywhere.
        previous = gr.State({})
        # Page-side values the workbook does not hold: the three drafting boxes and the Add box
        # ("<key>|<field>"), the open line ("open" = (line key, opening snapshot)), the time of
        # the last Save ("saved_at"). Never on disk.
        memory = gr.State({})
        cards_layout = gr.State((0, layout(wb0)))  # (epoch, layout): a trigger of the pane
        selected = gr.State(sel0)  # a job id, "about", "skills" or None (the landing)
        skills_view = gr.State((0, 0))  # (version, blank cards): the skills block's trigger
        edu_view = gr.State((0, 0))  # (version, blank boxes): the Education list's trigger
        saved_signal = gr.State(0)  # bumped after a Save that made the profile usable
        veil = common.veil()

        with gr.Row(equal_height=False):
            # -------------------------------------------------------- the left list
            with gr.Column(scale=1, min_width=250, elem_classes=["lc-list"]):
                counts0 = confirmed_counts(wb0)
                entries = []
                for i in range(MAX_JOBS):
                    job = wb0.jobs[i] if i < len(wb0.jobs) else None
                    entries.append(
                        gr.Button(
                            _job_label(job, counts0[job.job_id]) if job else "",
                            visible=job is not None,
                            elem_classes=_entry_classes(
                                job is not None and job.job_id == sel0,
                                job is not None and _job_done(job, counts0[job.job_id]),
                            ),
                        )
                    )
                gr.Markdown("---")
                about_entry = _about_entry(wb0, on=False)
                skills_entry = _skills_entry(wb0, on=False)
                gr.Markdown("---")
                save_btn = common.primary(
                    SAVE,
                    visible=save_profile is not None,
                    elem_classes=["lc-save", "ready"] if _all_done(wb0) else ["lc-save"],
                )
                save_line = gr.Markdown(
                    _save_line(wb0, {}), visible=save_profile is not None, elem_classes=["lc-grey"]
                )
                help_entry = common.link(HOW_IT_WORKS)  # back to the landing
                if left_foot is not None:
                    left_foot()  # the shell's Start over, in view while scrolling

            with gr.Column(scale=3):
                # ---------------------------------------------------- About you (static)
                head0 = wb0.header
                with gr.Column(visible=False) as about_col:
                    gr.Markdown(ABOUT_HEAD)
                    gr.Markdown(ABOUT_GREY, elem_classes=["lc-grey"])
                    about_boxes = {
                        "name": gr.Textbox(
                            label=ABOUT_LABELS["name"], value=head0.name or "", interactive=True
                        ),
                        "contact": gr.Textbox(
                            label=ABOUT_LABELS["contact"],
                            value="; ".join(head0.contact),
                            interactive=True,
                        ),
                        "headline": gr.Textbox(
                            label=ABOUT_LABELS["headline"],
                            value=head0.headline or "",
                            interactive=True,
                        ),
                        "summary": gr.Textbox(
                            label=ABOUT_LABELS["summary"],
                            value=head0.summary or "",
                            lines=3,
                            interactive=True,
                        ),
                    }
                    about_refused = {
                        field: gr.Markdown("", elem_classes=["lc-refused"]) for field in about_boxes
                    }
                    gr.Markdown(f"**{EDU_LABEL}**")

                    @gr.render(
                        inputs=[edu_view, state], triggers=[app.load, edu_view.change], **QUIET
                    )
                    def education(view, wb: Workbook):
                        values = [*wb.header.education, *[""] * view[1]]
                        boxes = [
                            gr.Textbox(show_label=False, value=v, interactive=True) for v in values
                        ]
                        for box in boxes:
                            for event in (box.blur, box.submit):
                                event(
                                    education_commit,
                                    inputs=[state, *boxes],
                                    outputs=[state, edu_refused, about_entry, *bars],
                                    **EVENT,
                                )

                    edu_refused = gr.Markdown("", elem_classes=["lc-refused"])
                    with gr.Row():
                        add_edu = common.link(ADD_EDU, scale=0)
                    gr.Markdown(ABOUT_SAVED, elem_classes=["lc-grey"])

                # -------------------------------------------------- Your skills (static)
                with gr.Column(visible=False) as skills_col:
                    gr.Markdown(SKILLS_HEAD)
                    gr.Markdown(SKILLS_GREY, elem_classes=["lc-grey"])
                    gr.Markdown(SKILLS_SENTENCE)

                    @gr.render(
                        inputs=[skills_view, state],
                        triggers=[app.load, skills_view.change],
                        **QUIET,
                    )
                    def skills_block(view, wb: Workbook):
                        cards = [*wb.skill_groups, *[SkillGroup(name="", items=[])] * view[1]]
                        boxes: list = []
                        removes: list = []
                        for group in cards:
                            with gr.Column(elem_classes=["lc-card"]):
                                boxes.append(
                                    gr.Textbox(label=GROUP_NAME, value=group.name, interactive=True)
                                )
                                boxes.append(
                                    gr.Textbox(
                                        label=GROUP_SKILLS,
                                        value="; ".join(group.items),
                                        interactive=True,
                                    )
                                )
                                removes.append(common.link(REMOVE_GROUP, elem_classes=["lc-act"]))
                        for box in boxes:
                            for event in (box.blur, box.submit):
                                event(
                                    skills_commit,
                                    inputs=[state, *boxes],
                                    outputs=[state, skills_refused, skills_entry, *bars],
                                    **EVENT,
                                )
                        for index, remove in enumerate(removes):
                            remove.click(
                                remove_group(index),
                                inputs=[state, skills_view, *boxes],
                                outputs=[state, skills_view, skills_refused, skills_entry, *bars],
                                **QUIET,
                            )

                    skills_refused = gr.Markdown("", elem_classes=["lc-refused"])
                    with gr.Row():
                        add_group = common.link(ADD_GROUP, scale=0)
                        sort_groups = common.link(SORT, scale=0)
                        fill_tools = common.link(FILL, scale=0)

                # ------------------------------------------------------------ the pane
                @gr.render(
                    inputs=[cards_layout, selected, state, memory, previous],
                    triggers=[app.load, cards_layout.change, selected.change],
                    **QUIET,
                )
                def pane(lay, sel, wb: Workbook, mem: dict, earlier: dict):
                    if sel in ("about", "skills"):
                        return
                    job = _job_of(wb, sel) if sel is not None else None
                    if job is None:
                        landing(wb)
                        return
                    draw_job(wb, job, mem, earlier)

        # ------------------------------------------------------------- the pane, drawn

        def landing(wb: Workbook) -> None:
            if not wb.jobs:
                gr.Markdown(LANDING_EMPTY)
                return
            bullets = sum(len(job.bullets) for job in wb.jobs)
            gr.Markdown(
                LANDING_HEAD.format(
                    jobs=_plural(len(wb.jobs), "job"), bullets=_plural(bullets, "bullet")
                )
            )
            gr.Markdown(LANDING_BODY)
            gr.Markdown(LANDING_NOTE, elem_classes=["lc-grey"])

        def draw_job(wb: Workbook, job, mem: dict, earlier: dict) -> None:
            """One job: heading, tools card, count line, bullet cards, the Add card. Every
            handler bound here outputs targeted components only (`watched`), never the layout
            or `selected` — except Add and remove, which change the layout."""
            jid = job.job_id
            index = [j.job_id for j in wb.jobs].index(jid)
            watched: list[tuple[Any, Callable[[Workbook, dict], Any]]] = []
            pairs: list[_Pair] = []

            def watch(comp, fn):
                watched.append((comp, fn))
                return comp

            def job_now(w: Workbook):
                return _job_of(w, jid)

            # --- heading row
            with gr.Column() as head_show:
                with gr.Row(elem_classes=["lc-headrow"]):  # one line
                    watch(
                        gr.Markdown(_heading(job)),
                        lambda w, m: _heading(job_now(w)) if job_now(w) else gr.skip(),
                    )
                    watch(
                        gr.Markdown(_dates(job), elem_classes=["lc-grey"]),
                        lambda w, m: _dates(job_now(w)) if job_now(w) else gr.skip(),
                    )
                    head_link = common.link(EDIT_HEAD, scale=0)
            with gr.Column(visible=False) as head_edit:
                with gr.Row():
                    head_boxes = {
                        field: gr.Textbox(
                            label=label, value=getattr(job, field) or "", interactive=True
                        )
                        for field, label in HEAD_LABELS.items()
                    }
                with gr.Row():
                    head_done = common.secondary(DONE, scale=0)
                    head_cancel = common.link(CANCEL, scale=0)
                gr.Markdown(HEAD_HINT, elem_classes=["lc-grey"])
                head_refused = gr.Markdown("", elem_classes=["lc-refused"])

            def head_snap(w: Workbook):
                j = job_now(w)
                values = tuple(getattr(j, f) for f in HEAD_LABELS)
                return values, [v or "" for v in values]

            def head_restore(w: Workbook, snap) -> Workbook:
                j = job_now(w)
                diff = {f: v for f, v in zip(HEAD_LABELS, snap, strict=True) if getattr(j, f) != v}
                if not diff:
                    return w
                return apply_jobs(w, JobsAnswers(jobs={jid: JobAnswer(**diff)}))

            pairs.append(
                _Pair(
                    "head",
                    head_show,
                    head_edit,
                    list(head_boxes.values()),
                    head_refused,
                    head_snap,
                    head_restore,
                )
            )

            # --- the tools card (the label above a framed, tinted card; edit inline)
            gr.Markdown(TOOLS_HEAD, elem_classes=["lc-grey"])
            with watch(
                gr.Column(elem_classes=_card_classes(job.defaults_confirmed, "lc-tools")),
                lambda w, m: gr.Column(
                    elem_classes=_card_classes(job_now(w).defaults_confirmed, "lc-tools")
                ),
            ):
                with gr.Row(elem_classes=["lc-inline"]) as tools_show:
                    watch(gr.Markdown(_tools_text(job)), lambda w, m: _tools_text(job_now(w)))
                    tools_link = common.link(EDIT, elem_classes=["lc-act"], scale=0)
                with gr.Column(visible=False) as tools_edit:
                    tools_box = gr.Textbox(
                        label=TOOLS_LABEL, value="; ".join(job.defaults), interactive=True
                    )
                    with gr.Row():
                        tools_done = common.secondary(DONE, scale=0)
                        tools_cancel = common.link(CANCEL, scale=0)
                    tools_refused = gr.Markdown("", elem_classes=["lc-refused"])
                with gr.Row(equal_height=True):
                    gr.Column(scale=3, min_width=0)  # the left of the row: the state sits right
                    with gr.Column(scale=2, min_width=0):
                        with watch(
                            gr.Row(visible=not job.defaults_confirmed, elem_classes=["lc-state"]),
                            lambda w, m: gr.Row(visible=not job_now(w).defaults_confirmed),
                        ):
                            gr.Markdown(LOOKS_RIGHT)
                            tools_confirm = common.secondary(CONFIRM, scale=0)
                        with watch(
                            gr.Row(visible=job.defaults_confirmed, elem_classes=["lc-state"]),
                            lambda w, m: gr.Row(visible=job_now(w).defaults_confirmed),
                        ):
                            gr.Markdown(CONFIRMED)
                            tools_undo = common.link(UNDO, scale=0)

            def tools_snap(w: Workbook):
                j = job_now(w)
                return (tuple(j.defaults), j.defaults_confirmed), ["; ".join(j.defaults)]

            def tools_restore(w: Workbook, snap) -> Workbook:
                j = job_now(w)
                defaults, flag = snap
                answer: dict = {}
                if tuple(j.defaults) != defaults:
                    answer["defaults"] = list(defaults)
                if j.defaults_confirmed != flag:
                    answer["defaults_confirmed"] = flag
                if not answer:
                    return w
                return apply_jobs(w, JobsAnswers(jobs={jid: JobAnswer(**answer)}))

            pairs.append(
                _Pair(
                    "tools",
                    tools_show,
                    tools_edit,
                    [tools_box],
                    tools_refused,
                    tools_snap,
                    tools_restore,
                )
            )

            # --- the count line
            def confirm_all_label(w: Workbook) -> tuple[str, int]:
                n = len(confirm_unedited(w, jid).bullets)
                return (CONFIRM_ONE if n == 1 else CONFIRM_ALL.format(n=n)), n

            def confirm_all_link(w: Workbook):
                label, n = confirm_all_label(w)
                return gr.Button(value=label, visible=n > 0)

            def count_text(w: Workbook) -> str:
                confirmed, total = confirmed_counts(w)[jid]
                return COUNT_LINE.format(n=total, x=confirmed)

            label0, n0 = confirm_all_label(wb)
            with gr.Row():
                watch(gr.Markdown(count_text(wb)), lambda w, m: count_text(w))
                confirm_all = watch(
                    common.link(label0, scale=0, visible=n0 > 0),
                    lambda w, m: confirm_all_link(w),
                )

            # --- the bullet cards
            cards: list[dict] = []
            for bullet in job.bullets:
                cards.append(draw_bullet(wb, job, bullet, mem, earlier, watch, pairs))

            # --- the Add card
            add_key = f"{jid}|add"
            with gr.Column(elem_classes=["lc-card", "lc-add"]):
                gr.Markdown(ADD_HEAD)
                add_box = gr.Textbox(
                    show_label=False,
                    placeholder=ADD_PLACEHOLDER,
                    value=mem.get(add_key, ""),
                    lines=2,
                    interactive=True,
                )
                add_refused = gr.Markdown("", elem_classes=["lc-refused"])
                with gr.Row():
                    gr.Markdown("")
                    add_button = common.secondary(ADD, scale=0)

            suggest_again = common.link(SUGGEST_AGAIN, scale=0)
            watch(  # the job's foot line, moved by every confirm / undo / add
                gr.Markdown(_foot(wb, jid), elem_classes=["lc-grey", "lc-foot"]),
                lambda w, m: _foot(w, jid),
            )

            # The static components a change of this job moves (targeted, never redrawn).
            if index < MAX_JOBS:
                watch(entries[index], lambda w, m: _job_entry(w, jid, on=True))
            watch(save_btn, lambda w, m: _save_button(w))
            watch(save_line, lambda w, m: _save_line(w, m))

            outs = [comp for comp, _ in watched]

            def refresh(w: Workbook, m: dict) -> list:
                return [fn(w, m) for _, fn in watched]

            def no_refresh() -> list:
                return [gr.skip()] * len(outs)

            toggles = [comp for pair in pairs for comp in (pair.show, pair.edit)]

            def closed() -> list:
                return [gr.update(visible=i % 2 == 0) for i in range(len(toggles))]

            # ------------------------------------------------ the in-place toggles (#6)
            def opener(pair: _Pair):
                def handler(w: Workbook, m: dict):
                    snap, values = pair.snap(w)
                    vis = [
                        gr.update(visible=(p is pair) == (k == 1)) for p in pairs for k in (0, 1)
                    ]
                    return ({**m, "open": (pair.key, snap)}, *vis, *values, "")

                return handler

            def done_handler(m: dict):
                return ({k: v for k, v in m.items() if k != "open"}, *closed())

            def canceller(pair: _Pair):
                def handler(w: Workbook, m: dict):
                    kept = {k: v for k, v in m.items() if k != "open"}
                    opened = m.get("open")
                    if not opened or opened[0] != pair.key:
                        _snap, values = pair.snap(w)
                        return (w, kept, "", *closed(), *values, *no_refresh(), *bar_skip())
                    new, refusal, save = commit(w, lambda x: pair.restore(x, opened[1]))
                    _snap, values = pair.snap(new)
                    return (
                        new,
                        kept,
                        refusal,
                        *closed(),
                        *values,
                        *refresh(new, kept),
                        *bar_out(new, save),
                    )

                return handler

            open_links = [head_link, tools_link, *[link for c in cards for link in c["links"]]]
            for pair, link in zip(pairs, open_links, strict=True):
                link.click(
                    opener(pair),
                    inputs=[state, memory],
                    outputs=[memory, *toggles, *pair.boxes, pair.refusal],
                    **QUIET,
                )
            done_buttons = [head_done, tools_done, *[d for c in cards for d in c["dones"]]]
            cancel_buttons = [
                head_cancel,
                tools_cancel,
                *[c_ for c in cards for c_ in c["cancels"]],
            ]
            for pair, done, cancel in zip(pairs, done_buttons, cancel_buttons, strict=True):
                done.click(done_handler, inputs=[memory], outputs=[memory, *toggles], **QUIET)
                cancel.click(
                    canceller(pair),
                    inputs=[state, memory],
                    outputs=[
                        state,
                        memory,
                        pair.refusal,
                        *toggles,
                        *pair.boxes,
                        *outs,
                        *bars,
                    ],
                    **QUIET,
                )

            # ------------------------------------------------------------ commits
            commit_outputs_tail = [*outs, *bars]

            def field_commit(make: Callable[[str], Callable[[Workbook], Workbook]]):
                """A box's commit: `make(value)` is the change; outputs the workbook, the box's
                refusal line, every watched component and the bar."""

                def handler(w: Workbook, m: dict, value: str):
                    new, refusal, save = commit(w, make(value))
                    if new is w:
                        return (w, refusal, *no_refresh(), *bar_skip())
                    return (new, refusal, *refresh(new, m), *bar_out(new, save))

                return handler

            def head_change(field: str):
                def make(value: str):
                    answer = JobAnswer(**{field: value})
                    return lambda w: apply_jobs(w, JobsAnswers(jobs={jid: answer}))

                return make

            for field, box in head_boxes.items():
                for event in (box.blur, box.submit):
                    event(
                        field_commit(head_change(field)),
                        inputs=[state, memory, box],
                        outputs=[state, head_refused, *commit_outputs_tail],
                        **EVENT,
                    )

            def tools_change(value: str):
                """Pair 1 (#5): an edit that changes the list also un-confirms it."""
                items = split_list(value)

                def change(w: Workbook) -> Workbook:
                    trial = apply_jobs(w, JobsAnswers(jobs={jid: JobAnswer(defaults=items)}))
                    if trial == w:
                        return w
                    answer = JobAnswer(defaults=items, defaults_confirmed=False)
                    return apply_jobs(w, JobsAnswers(jobs={jid: answer}))

                return change

            for event in (tools_box.blur, tools_box.submit):
                event(
                    field_commit(tools_change),
                    inputs=[state, memory, tools_box],
                    outputs=[state, tools_refused, *commit_outputs_tail],
                    **EVENT,
                )

            def flag_commit(change: Callable[[Workbook], Workbook]):
                def handler(w: Workbook, m: dict):
                    new, _refusal, save = commit(w, change)
                    if new is w:
                        return (w, *no_refresh(), *bar_skip())
                    return (new, *refresh(new, m), *bar_out(new, save))

                return handler

            def tools_flag(flag: bool):
                answers = JobsAnswers(jobs={jid: JobAnswer(defaults_confirmed=flag)})
                return lambda w: apply_jobs(w, answers)

            n_suggest = 1 + len(outs) + len(bars)

            def suggest_work(w: Workbook, m: dict, only: set[str] | None) -> tuple:
                try:
                    with common.model_turn():
                        new, warnings = select_keywords(w, adapter=adapter("ingest"), only=only)
                except CompleteError as exc:
                    gr.Warning(str(exc))
                    return (gr.skip(),) * n_suggest
                except _FAILURES as exc:
                    gr.Warning(KEYWORDS_FAILED.format(exc=exc))
                    return (gr.skip(),) * n_suggest
                save = persist(new) if new != w else None
                gr.Info(KEYWORDS_DONE)
                if warnings:
                    gr.Warning("; ".join(warnings))
                return (new, *refresh(new, m), *bar_out(new, save))

            def suggester(only: set[str] | None, phrase: str):
                """The keyword step under the page veil — the user's click only: `None` =
                the whole profile (Suggest again), else one bullet's key."""

                def suggest(w: Workbook, m: dict):
                    yield (*[gr.skip()] * n_suggest, *veil.on(phrase))
                    landed = (gr.skip(),) * n_suggest
                    try:
                        landed = suggest_work(w, m, only)
                    finally:
                        yield (*landed, *veil.off())

                return suggest

            suggest_outputs = [state, *outs, *bars, *veil.outputs]
            tools_confirm.click(
                flag_commit(tools_flag(True)),
                inputs=[state, memory],
                outputs=[state, *commit_outputs_tail],
                **QUIET,
            )
            tools_undo.click(
                flag_commit(tools_flag(False)),
                inputs=[state, memory],
                outputs=[state, *commit_outputs_tail],
                **QUIET,
            )
            suggest_again.click(
                suggester(None, common.PHRASE_KEYWORDS),
                inputs=[state, memory],
                outputs=suggest_outputs,
                **QUIET,
            )

            def confirm_all_handler(w: Workbook, m: dict):
                answers = confirm_unedited(w, jid)
                n = len(answers.bullets)
                new, _refusal, save = commit(w, lambda x: apply_bullets(x, answers))
                if new is w:
                    return (w, *no_refresh(), *bar_skip())
                gr.Info(CONFIRMED_TOAST_ONE if n == 1 else CONFIRMED_TOAST.format(n=n))
                return (new, *refresh(new, m), *bar_out(new, save))

            confirm_all.click(
                confirm_all_handler,
                inputs=[state, memory],
                outputs=[state, *commit_outputs_tail],
                **QUIET,
            )

            # Add (#9): the layout changes, so the pane is drawn again on the same job.
            static_outs = [entries[index]] if index < MAX_JOBS else []
            static_outs += [save_btn, save_line]

            def static_view(w: Workbook, m: dict) -> list:
                view = [_job_entry(w, jid, on=True)] if index < MAX_JOBS else []
                return [*view, _save_button(w), _save_line(w, m)]

            def remember(key: str):
                def store(m: dict, value: str):
                    return {**m, key: value}

                return store

            add_box.blur(remember(add_key), inputs=[memory, add_box], outputs=[memory], **EVENT)

            def add_handler(w: Workbook, m: dict, value: str):
                new, refusal, save = commit(w, lambda x: add_bullet(x, jid, value)[0])
                if new is w:
                    return (w, m, gr.skip(), refusal, *[gr.skip()] * len(static_outs), *bar_skip())
                gr.Info(ADDED)
                kept = {k: v for k, v in m.items() if k not in (add_key, "open")}
                return (
                    new,
                    kept,
                    (next(epochs), layout(new)),
                    "",
                    *static_view(new, kept),
                    *bar_out(new, save),
                )

            add_button.click(
                add_handler,
                inputs=[state, memory, add_box],
                outputs=[state, memory, cards_layout, add_refused, *static_outs, *bars],
                **QUIET,
            )

            # --- each card's own events
            for card in cards:
                bind_bullet(card, jid, refresh, no_refresh, outs, static_outs, static_view)
                card["suggest"].click(
                    suggester({card["key"]}, common.PHRASE_KEYWORDS_ONE),
                    inputs=[state, memory],
                    outputs=suggest_outputs,
                    **QUIET,
                )

        def draw_bullet(wb: Workbook, job, bullet, mem: dict, earlier: dict, watch, pairs) -> dict:
            """One bullet card and its folded drafting panel; returns its components."""
            jid, bid = job.job_id, bullet.id
            key = f"{jid}::{bid}"

            def now(w: Workbook):
                return _bullet_of(w, jid, bid)

            def job_now(w: Workbook):
                return _job_of(w, jid)

            def card_view(fn):
                def view(w: Workbook, m: dict):
                    b = now(w)
                    return gr.skip() if b is None else fn(w, b)

                return view

            done0 = bullet.status == "confirmed"
            with watch(
                gr.Column(elem_classes=_card_classes(done0)),
                card_view(
                    lambda w, b: gr.Column(elem_classes=_card_classes(b.status == "confirmed"))
                ),
            ):
                # the text line ("edit" right after the last word)
                with gr.Row(elem_classes=["lc-inline"]) as text_show:
                    watch(gr.Markdown(bullet.text), card_view(lambda w, b: b.text))
                    text_link = common.link(EDIT, elem_classes=["lc-act"], scale=0)
                with gr.Column(visible=False) as text_edit:
                    text_box = gr.Textbox(
                        show_label=False, value=bullet.text, lines=2, interactive=True
                    )
                    with gr.Row():
                        text_done = common.secondary(DONE, scale=0)
                        text_cancel = common.link(CANCEL, scale=0)
                    gr.Markdown(TEXT_HINT, elem_classes=["lc-grey"])
                    text_refused = gr.Markdown("", elem_classes=["lc-refused"])
                # the keywords line
                with gr.Column() as kw_show:
                    with gr.Row(elem_classes=["lc-inline"]):
                        watch(gr.Markdown(_kw_line(wb, bullet)), card_view(_kw_line))
                        suggest = watch(
                            common.link(
                                SUGGEST,
                                elem_classes=["lc-act"],
                                scale=0,
                                visible=_suggestable(wb, bullet),
                            ),
                            card_view(lambda w, b: gr.Button(visible=_suggestable(w, b))),
                        )
                        kw_link = common.link(EDIT, elem_classes=["lc-act"], scale=0)
                    watch(
                        gr.Markdown(
                            SUGGEST_HINT,
                            visible=_suggestable(wb, bullet),
                            elem_classes=["lc-grey", "lc-act"],
                        ),
                        card_view(lambda w, b: gr.Markdown(visible=_suggestable(w, b))),
                    )
                with gr.Column(visible=False) as kw_edit:
                    kw_box = gr.Textbox(
                        label=KW_LABEL, value="; ".join(bullet.keywords), interactive=True
                    )
                    watch(
                        gr.Markdown(_maybe_text(job, bullet), elem_classes=["lc-grey"]),
                        card_view(lambda w, b: _maybe_text(job_now(w), b)),
                    )
                    with gr.Row():
                        kw_done = common.secondary(DONE, scale=0)
                        kw_cancel = common.link(CANCEL, scale=0)
                    kw_refused = gr.Markdown("", elem_classes=["lc-refused"])
                # the bottom row: actions left, the state right (#7)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=0):
                        with gr.Row():
                            improve = common.link(IMPROVE_LINK, elem_classes=["lc-act"], scale=0)
                            remove = common.link(REMOVE, elem_classes=["lc-act"], scale=0)
                    with gr.Column(scale=2, min_width=0):
                        with watch(
                            gr.Row(visible=not done0, elem_classes=["lc-state"]),
                            card_view(lambda w, b: gr.Row(visible=b.status != "confirmed")),
                        ):
                            gr.Markdown(LOOKS_RIGHT)
                            confirm = common.secondary(CONFIRM, scale=0)
                        with watch(
                            gr.Row(visible=done0, elem_classes=["lc-state"]),
                            card_view(lambda w, b: gr.Row(visible=b.status == "confirmed")),
                        ):
                            gr.Markdown(CONFIRMED)
                            undo = common.link(UNDO, scale=0)
                remove_ask = common.ask(REMOVE_QUESTION)  # one line, no grey block

            # the drafting panel, folded, directly under the card (#10)
            proposals = earlier.get(key) or []
            with gr.Column(visible=False, elem_classes=["lc-panel"]) as panel:
                gr.Markdown(PANEL_HEAD)
                gr.Markdown(PANEL_INTRO, elem_classes=["lc-grey"])
                star = {
                    field: gr.Textbox(
                        label=label, value=mem.get(f"{key}|{field}", ""), interactive=True
                    )
                    for field, label in STAR_LABELS.items()
                }
                with gr.Row():
                    draft = common.primary(DRAFT, scale=0, min_width=240)
                with gr.Column(visible=bool(proposals)) as suggested:
                    gr.Markdown(SUGGESTED, elem_classes=["lc-grey"])
                    proposal_md = gr.Markdown(proposals[-1] if proposals else "")
                    with gr.Row():
                        use = common.secondary(USE, scale=0)
                        try_again = common.link(TRY_AGAIN, scale=0)
                draft_note = gr.Markdown("", elem_classes=["lc-refused"])
                keep = common.link(KEEP, scale=0)

            def text_snap(w: Workbook):
                b = now(w)
                return (b.text, b.status), [b.text]

            def text_restore(w: Workbook, snap) -> Workbook:
                b = now(w)
                text, status = snap
                answer: dict = {}
                if b.text != text:
                    answer["text"] = text
                if b.status != status:
                    answer["confirm"] = status == "confirmed"
                if not answer:
                    return w
                return apply_bullets(w, BulletsAnswers(bullets={key: BulletAnswer(**answer)}))

            def kw_snap(w: Workbook):
                b = now(w)
                return tuple(b.keywords), ["; ".join(b.keywords)]

            def kw_restore(w: Workbook, snap) -> Workbook:
                if tuple(now(w).keywords) == snap:
                    return w
                answers = BulletsAnswers(bullets={key: BulletAnswer(keywords=list(snap))})
                return apply_bullets(w, answers)

            pairs.append(
                _Pair(
                    f"text:{key}",
                    text_show,
                    text_edit,
                    [text_box],
                    text_refused,
                    text_snap,
                    text_restore,
                )
            )
            pairs.append(
                _Pair(f"kw:{key}", kw_show, kw_edit, [kw_box], kw_refused, kw_snap, kw_restore)
            )
            return {
                "key": key,
                "bid": bid,
                "links": [text_link, kw_link],
                "dones": [text_done, kw_done],
                "cancels": [text_cancel, kw_cancel],
                "text_box": text_box,
                "text_refused": text_refused,
                "kw_box": kw_box,
                "kw_refused": kw_refused,
                "suggest": suggest,
                "confirm": confirm,
                "undo": undo,
                "improve": improve,
                "remove": remove,
                "remove_group": remove_ask.row,
                "remove_yes": remove_ask.yes,
                "remove_no": remove_ask.no,
                "panel": panel,
                "star": star,
                "draft": draft,
                "try_again": try_again,
                "suggested": suggested,
                "proposal_md": proposal_md,
                "draft_note": draft_note,
                "use": use,
                "keep": keep,
            }

        def bind_bullet(card: dict, jid: str, refresh, no_refresh, outs, static_outs, static_view):
            """The events of one bullet card (bound after the whole pane is drawn, because a
            commit's outputs are every watched component of the pane)."""
            key, bid = card["key"], card["bid"]
            tail = [*outs, *bars]

            def run(change: Callable[[Workbook], Workbook], *, with_refusal: bool):
                def handler(w: Workbook, m: dict, *values):
                    new, refusal, save = commit(w, change(*values) if values else change)
                    head = [refusal] if with_refusal else []
                    if new is w:
                        return (w, *head, *no_refresh(), *bar_skip())
                    return (new, *head, *refresh(new, m), *bar_out(new, save))

                return handler

            def text_change(value: str):
                """Pair 2 (#6): a text edit that changes the text also un-confirms the bullet."""

                def change(w: Workbook) -> Workbook:
                    trial = apply_bullets(
                        w, BulletsAnswers(bullets={key: BulletAnswer(text=value)})
                    )
                    if trial == w:
                        return w
                    answer = BulletAnswer(text=value, confirm=False)
                    return apply_bullets(w, BulletsAnswers(bullets={key: answer}))

                return change

            def kw_change(value: str):
                answer = BulletAnswer(keywords=split_list(value))
                return lambda w: apply_bullets(w, BulletsAnswers(bullets={key: answer}))

            def confirm_change(flag: bool):
                answers = BulletsAnswers(bullets={key: BulletAnswer(confirm=flag)})
                return lambda w: apply_bullets(w, answers)

            card["text_box"].blur(
                run(text_change, with_refusal=True),
                inputs=[state, memory, card["text_box"]],
                outputs=[state, card["text_refused"], *tail],
                **EVENT,
            )
            for event in (card["kw_box"].blur, card["kw_box"].submit):
                event(
                    run(kw_change, with_refusal=True),
                    inputs=[state, memory, card["kw_box"]],
                    outputs=[state, card["kw_refused"], *tail],
                    **EVENT,
                )
            for button, flag in ((card["confirm"], True), (card["undo"], False)):
                button.click(
                    run(confirm_change(flag), with_refusal=False),
                    inputs=[state, memory],
                    outputs=[state, *tail],
                    **QUIET,
                )

            # remove (2a #8): the question first; the layout changes, the pane is drawn again.
            card["remove"].click(
                lambda: gr.Row(visible=True), outputs=[card["remove_group"]], **QUIET
            )
            card["remove_no"].click(
                lambda: gr.Row(visible=False), outputs=[card["remove_group"]], **QUIET
            )

            def remove_handler(w: Workbook, m: dict):
                new, _refusal, save = commit(w, lambda x: delete_bullet(x, jid, bid))
                if new is w:
                    return (w, gr.skip(), *[gr.skip()] * len(static_outs), *bar_skip())
                gr.Info(REMOVED)
                return (new, (next(epochs), layout(new)), *static_view(new, m), *bar_out(new, save))

            card["remove_yes"].click(
                remove_handler,
                inputs=[state, memory],
                outputs=[state, cards_layout, *static_outs, *bars],
                **QUIET,
            )

            # the drafting panel (the page's veil, the page's queue)
            card["improve"].click(lambda: gr.Column(visible=True), outputs=[card["panel"]], **QUIET)
            card["keep"].click(lambda: gr.Column(visible=False), outputs=[card["panel"]], **QUIET)
            for field, box in card["star"].items():
                box.blur(
                    lambda m, value, field=field: {**m, f"{key}|{field}": value},
                    inputs=[memory, box],
                    outputs=[memory],
                    **EVENT,
                )

            def draft_handler(w: Workbook, earlier: dict, situation, action, result):
                yield (*[gr.skip()] * 4, *veil.on(common.PHRASE_IMPROVE))
                landed = (gr.skip(),) * 4
                try:
                    answers = BulletDraftAnswers(situation=situation, action=action, result=result)
                    try:
                        with common.model_turn():
                            proposal, warnings = draft_bullet(
                                w,
                                jid,
                                bullet_key=key,
                                answers=answers,
                                adapter=adapter("rewrite"),
                                budget=budget,
                                previous=earlier.get(key, []),
                            )
                    except _FAILURES as exc:
                        sentence = DRAFT_FAILED.format(exc=exc)
                        gr.Warning(sentence)
                        landed = (gr.skip(), gr.skip(), sentence, gr.skip())
                    else:
                        kept = {**earlier, key: [*earlier.get(key, []), proposal]}
                        landed = (proposal, gr.Column(visible=True), "\n\n".join(warnings), kept)
                finally:
                    yield (*landed, *veil.off())

            draft_outputs = [
                card["proposal_md"],
                card["suggested"],
                card["draft_note"],
                previous,
                *veil.outputs,
            ]
            for button in (card["draft"], card["try_again"]):
                button.click(
                    draft_handler,
                    inputs=[state, previous, *card["star"].values()],
                    outputs=draft_outputs,
                    **QUIET,
                )

            def use_handler(w: Workbook, m: dict, earlier: dict):
                """ "Use this" — the proposal as the bullet's text; on a confirmed bullet it also
                un-confirms it (#6); the panel folds."""
                proposals = earlier.get(key) or []
                if not proposals:
                    gr.Warning(NO_PROPOSAL)
                    return (w, gr.skip(), gr.skip(), *no_refresh(), *bar_skip())
                text = proposals[-1]

                def change(x: Workbook) -> Workbook:
                    b = _bullet_of(x, jid, bid)
                    answer = (
                        BulletAnswer(text=text, confirm=False)
                        if b.status == "confirmed"
                        else BulletAnswer(text=text)
                    )
                    return apply_bullets(x, BulletsAnswers(bullets={key: answer}))

                new, refusal, save = commit(w, change)
                if refusal:
                    return (w, gr.skip(), gr.skip(), *no_refresh(), *bar_skip())
                kept = {k: v for k, v in earlier.items() if k != key}
                if new is w:
                    return (w, kept, gr.Column(visible=False), *no_refresh(), *bar_skip())
                return (
                    new,
                    kept,
                    gr.Column(visible=False),
                    *refresh(new, m),
                    *bar_out(new, save),
                )

            card["use"].click(
                use_handler,
                inputs=[state, memory, previous],
                outputs=[state, previous, card["panel"], *tail],
                **QUIET,
            )

        # ------------------------------------------------------------- navigation (#1)
        nav_outputs = [selected, *entries, about_entry, skills_entry, about_col, skills_col]

        def navigate(target):
            def handler(w: Workbook):
                if isinstance(target, int):
                    if target >= len(w.jobs):
                        return (gr.skip(),) * len(nav_outputs)
                    sel = w.jobs[target].job_id
                else:
                    sel = target
                return (
                    sel,
                    *entry_classes(w, sel),
                    gr.Column(visible=sel == "about"),
                    gr.Column(visible=sel == "skills"),
                )

            return handler

        for i, entry in enumerate(entries):
            entry.click(navigate(i), inputs=[state], outputs=nav_outputs, **QUIET)
        about_entry.click(navigate("about"), inputs=[state], outputs=nav_outputs, **QUIET)
        skills_entry.click(navigate("skills"), inputs=[state], outputs=nav_outputs, **QUIET)
        help_entry.click(navigate(None), inputs=[state], outputs=nav_outputs, **QUIET)

        # ------------------------------------------------------------- About you (#11)
        def header_commit(field: str):
            def handler(w: Workbook, value: str):
                converted = split_list(value) if field == "contact" else value
                answers = HeaderAnswers(**{field: converted})
                new, refusal, save = commit(w, lambda x: apply_header(x, answers))
                if new is w:
                    return (w, refusal, gr.skip(), *bar_skip())
                return (new, refusal, _about_entry(new, on=True), *bar_out(new, save))

            return handler

        for field, box in about_boxes.items():
            events = [box.blur] if field == "summary" else [box.blur, box.submit]
            for event in events:
                event(
                    header_commit(field),
                    inputs=[state, box],
                    outputs=[state, about_refused[field], about_entry, *bars],
                    **EVENT,
                )

        def education_commit(w: Workbook, *values: str):
            entries_ = [v for v in values if v.strip()]
            answers = HeaderAnswers(education=entries_)
            new, refusal, save = commit(w, lambda x: apply_header(x, answers))
            if new is w:
                return (w, refusal, gr.skip(), *bar_skip())
            return (new, refusal, _about_entry(new, on=True), *bar_out(new, save))

        add_edu.click(lambda view: (view[0] + 1, 1), inputs=[edu_view], outputs=[edu_view], **QUIET)

        # ----------------------------------------------------------- Your skills (#12)
        def groups_of(boxes) -> list[SkillGroup]:
            pairs_ = zip(boxes[0::2], boxes[1::2], strict=True)
            return [SkillGroup(name=name, items=split_list(items)) for name, items in pairs_]

        def skills_commit(w: Workbook, *boxes: str):
            answers = SkillsAnswers(groups=groups_of(boxes))
            new, refusal, save = commit(w, lambda x: apply_skill_groups(x, answers))
            if new is w:
                return (w, refusal, gr.skip(), *bar_skip())
            return (new, refusal, _skills_entry(new, on=True), *bar_out(new, save))

        def remove_group(index: int):
            def handler(w: Workbook, view, *boxes: str):
                kept = [g for i, g in enumerate(groups_of(boxes)) if i != index]
                answers = SkillsAnswers(groups=kept)
                new, refusal, save = commit(w, lambda x: apply_skill_groups(x, answers))
                if refusal:
                    return (w, gr.skip(), refusal, gr.skip(), *bar_skip())
                return (
                    new,
                    (view[0] + 1, 0),
                    "",
                    _skills_entry(new, on=True),
                    *bar_out(new, save),
                )

            return handler

        add_group.click(
            lambda view: (view[0] + 1, view[1] + 1),
            inputs=[skills_view],
            outputs=[skills_view],
            **QUIET,
        )

        def fill_handler(w: Workbook, view):
            new, _refusal, save = commit(w, with_prefill)
            if new is w:
                gr.Info(FILL_NONE)
                return (w, gr.skip(), gr.skip(), *bar_skip())
            return (new, (view[0] + 1, 0), _skills_entry(new, on=True), *bar_out(new, save))

        fill_tools.click(
            fill_handler,
            inputs=[state, skills_view],
            outputs=[state, skills_view, skills_entry, *bars],
            **QUIET,
        )

        n_sort = 3 + len(bars)

        def sort_work(w: Workbook, view) -> tuple:
            try:
                with common.model_turn():
                    groups, warnings = propose_groups(w, adapter=adapter("ingest"))
                new = apply_skill_groups(w, SkillsAnswers(groups=groups))
            except CompleteError as exc:
                gr.Warning(str(exc))
                return (gr.skip(),) * n_sort
            except _FAILURES as exc:
                gr.Warning(GROUPS_FAILED.format(exc=exc))
                return (gr.skip(),) * n_sort
            save = persist(new) if new != w else None
            gr.Info(GROUPS_DONE)
            if warnings:
                gr.Warning("; ".join(warnings))
            return (new, (view[0] + 1, 0), _skills_entry(new, on=True), *bar_out(new, save))

        def sort_handler(w: Workbook, view):
            yield (*[gr.skip()] * n_sort, *veil.on(common.PHRASE_GROUPS))
            landed = (gr.skip(),) * n_sort
            try:
                landed = sort_work(w, view)
            finally:
                yield (*landed, *veil.off())

        sort_groups.click(
            sort_handler,
            inputs=[state, skills_view],
            outputs=[state, skills_view, skills_entry, *bars, *veil.outputs],
            **QUIET,
        )

        # ------------------------------------------------------------ Save profile (#13)
        n_save = 3 + len(bars)

        def save_work(w: Workbook, m: dict, signal: int) -> tuple:
            try:
                with common.model_turn():  # the sync's embedding calls
                    report = save_profile(w, today=date.today())
            except OSError:
                gr.Warning(SAVE_FAILED)
                return (gr.skip(),) * n_save
            stamp = _now()
            kept = {**m, "saved_at": stamp}
            if report.ready:
                gr.Info(SAVED_READY)
                bar_value = [(_bar_line(w), BAR_PROFILE_SAVED.format(hhmm=stamp))] if bars else []
                return (_save_line(w, kept), signal + 1, kept, *bar_value)
            if report.index_error:
                gr.Warning(SAVED_NOT_READY.format(reason=report.index_error))
            else:
                if report.written.problems:
                    gr.Warning(PROBLEMS.format(problems="; ".join(report.written.problems)))
                gr.Info(SAVED_ONLY)
            return (_save_line(w, kept), gr.skip(), kept, *bar_skip())

        def save_handler(w: Workbook, m: dict, signal: int):
            yield (*[gr.skip()] * n_save, *veil.on(common.PHRASE_SAVE))
            landed = (gr.skip(),) * n_save
            try:
                landed = save_work(w, m, signal)
            finally:
                yield (*landed, *veil.off())

        if save_profile is not None:
            save_btn.click(
                save_handler,
                inputs=[state, memory, saved_signal],
                outputs=[save_line, saved_signal, memory, *bars, *veil.outputs],
                **QUIET,
            )

    # ------------------------------------------------------------------ hand-off

    def hand_off(wb: Workbook, warnings: list[str]) -> tuple:
        """A workbook arrives from outside (a load, an upload): one value per component of
        `hand_off_targets`. A new epoch draws the pane anew, so nothing typed before stays on
        screen."""
        epoch = next(epochs)
        sel = _first_selected(wb)
        if warnings:
            gr.Warning(" · ".join(warnings))
        if len(wb.jobs) > MAX_JOBS:
            gr.Warning(TOO_MANY_JOBS)
        head = wb.header
        stamp = _mtime(saved)
        mem = {"saved_at": stamp} if stamp else {}
        bar_value = []
        if bars:
            wrote = _mtime(working_copy)
            if wrote:
                save = BAR_SAVED.format(hhmm=wrote)
            elif stamp:
                save = BAR_PROFILE_SAVED.format(hhmm=stamp)
            else:
                save = ""
            bar_value = [(_bar_line(wb), save)]
        return (
            wb,
            {},
            mem,
            (epoch, layout(wb)),
            sel,
            (epoch, 0),
            (epoch, 0),
            head.name or "",
            "; ".join(head.contact),
            head.headline or "",
            head.summary or "",
            *[""] * len(about_refused),
            "",
            "",
            *all_entries(wb, sel),
            _save_button(wb),
            _save_line(wb, mem),
            gr.Column(visible=False),
            gr.Column(visible=False),
            *bar_value,
        )

    app.hand_off_targets = [
        state,
        previous,
        memory,
        cards_layout,
        selected,
        skills_view,
        edu_view,
        *about_boxes.values(),
        *about_refused.values(),
        edu_refused,
        skills_refused,
        *entries,
        about_entry,
        skills_entry,
        save_btn,
        save_line,
        about_col,
        skills_col,
        *bars,
    ]
    app.hand_off = hand_off
    app.saved_signal = saved_signal
    if warnings0 and not bars:  # under the shell, its load hands the warnings over itself
        app.load(lambda: gr.Warning(" · ".join(warnings0)))
    return app


def launch(
    draft: Path,
    out: Path,
    *,
    working_copy: Path,
    sidecar: Path,
    config: Path | None = None,
    save_profile: Callable[..., SaveReport] | None = None,
) -> None:
    """Open the page over `draft`; every edit writes `working_copy` (+ `sidecar`); Save profile
    calls `save_profile`. Blocks until the server stops."""
    page = build(
        draft,
        out,
        working_copy=working_copy,
        sidecar=sidecar,
        config=config,
        save_profile=save_profile,
    )
    page.launch(css=" ".join([CHIP_CSS, common.VEIL_CSS, common.LOOKS_CSS, common.BUTTON_CSS]))
