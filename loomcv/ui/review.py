"""The applications page: one left list — "+ New application" and every application, newest
first — and one pane on the right.

**No logic here.** Every sentence comes from `loomcv.review.words`, every file from
`loomcv.review.store`, every change of the résumé from one `loomcv.review.doc` operation, the
application itself from `loomcv.application`; the `for` loops only lay components out and the
handlers only pass values between those calls and the components.

- **The list** is its own small `@gr.render` (redrawn on load, a selection, a Build, a PDF). A
  click on an entry opens it; nothing else opens anything. Its head — "+ New
  application" (a render of its own on the same trigger), the "Your applications" label and the
  link "How to manage this list", which opens and closes the folder note in place — stays
  pinned while the entries scroll; with no application one line stands in for the entries.
- **The New pane** is static: the posting, a name (the id is derived from it), one primary button
  — Build, under the veil. When the profile is not ready the form gives way to one sentence.
- **The application pane** is one `@gr.render` triggered by `pane_key` only — never by the
  document — so an edit, a move, a remove or an Add never redraws it: each job gets, at draw
  time, as many bullet rows and Add rows as it can ever show (its bullets on the résumé plus its
  bullets still in the profile) and its events update those rows as targeted outputs.
- **Saved as you go:** every action runs one operation, then `save_review`, and writes the bar's
  save slot. A box commits on leaving it (blur) and on Done; cancel commits the opening value.
- **Make the PDF** = `save_review` then `application.submit`, under the veil; once a PDF exists
  its full path shows on one line under the PDF line (the only place a path shows).

Every event carries `show_progress="hidden"` and runs in one concurrency group; Build's model
calls also take `common.model_turn()`. Default
components, the classes of `common.LOOKS_CSS` and this page's `REVIEW_CSS`;
no JavaScript of our own. Not covered by pytest (the `ui` group is not installed in CI).
"""

from __future__ import annotations

import dataclasses
import html
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import gradio as gr

from loomcv import application
from loomcv.app_state import saved_stamp
from loomcv.application import ApplicationExists
from loomcv.draft import resume_date
from loomcv.review.doc import (
    ReviewDoc,
    ReviewJob,
    add,
    addable,
    delete,
    edit,
    index_mismatch,
    move,
    pick,
    set_headline,
    set_skill_groups,
    set_summary,
    skills_from_text,
    skills_to_text,
)
from loomcv.review.store import (
    FAILURES,
    ListEntry,
    Opened,
    list_entries,
    open_review,
    save_review,
    start_application,
    taken_ids,
)
from loomcv.review.words import (
    CARD_NOTE,
    EntryWords,
    PdfWords,
    RowWords,
    app_id_from_name,
    built_words,
    entry_words,
    job_rows,
    pdf_words,
    report_body,
    tally_words,
    verdict_words,
)
from loomcv.steps import Runtime
from loomcv.sync.parse_assets import SkillGroup
from loomcv.ui import common

# One request at a time: a click reads the States, acts and writes them back.
SERIAL = {"concurrency_limit": 1, "concurrency_id": "review"}
QUIET = {"show_progress": "hidden", **SERIAL}
NEW = "new"  # the "+ New application" entry's key in `selected`
SHOWN = 2  # Add rows shown before "show all N"

# The words of the page.
NEW_ENTRY = "+ New application"
NEW_HEADING = "Apply to a new posting"
NEW_SENTENCE = (
    "Paste the job posting and give this application a name. LoomCV reads the posting, picks "
    "the bullets from your profile that fit it and drafts a résumé — you review it here before "
    "anything is printed."
)
POSTING_LABEL = "The job posting, pasted as it is"
NAME_LABEL = "Your name for this application"
NAME_PLACEHOLDER = "For example: Fennimore — web platform"
BUILD_LABEL = "Build my résumé for this posting"
BUILD_NOTE = (
    "This can take a few minutes, depending on your model. Your name, contact, summary and "
    "education come from My profile."
)
# Where the applications live, and how to tidy the list.
WHERE_SAVED = "Your applications are saved in:"
TIDY_NOTE = (
    "Each application is one folder there, with its PDF inside. LoomCV currently doesn't support "
    "archiving or deleting an application: to tidy the list, move a finished folder out of that "
    "folder entirely — not into a folder inside it, which would show up as an entry. Move it back "
    "and it shows again."
)
# The note sits in the left list, under a
# label and behind a link; with no application one line stands in place of the entries.
LIST_LABEL = "Your applications"
MANAGE_LINK = "How to manage this list"
NO_APPS = "No applications yet. Click + New application above to start one."
GATE_SENTENCE = "Finish and save your profile first — in My profile."
TAKEN = "You already have an application with this name — pick another."
REPORT_INTRO = (
    "This is LoomCV's detailed working for this posting. Nothing here is editable; the card "
    "above is its reading."
)
SHOW_REPORT = "Show the full report"
HIDE_REPORT = "Hide the full report"
PROFILE_NOTE = "Name, contact and education: change these in My profile."
EDIT_NOTE = "Changes here go into this résumé only. Your profile is not changed."
REMOVE_QUESTION = (
    "Remove this bullet from this résumé? It goes back to “Add a bullet from this job”."
)
NO_BULLET = "No bullet picked for this posting."
OTHER_INDEX = (
    "This application was made from an earlier version of your profile, so bullets can't be "
    "added from it."
)
NO_HEADLINE = "No headline — edit to add one."
NO_SUMMARY = "No summary — edit to add one."
NO_ITEMS = "No items — edit to add some."
FOOT = "Saved as you go. The PDF is made from what you see here."
MAKE_PDF = "Make the PDF"
MAKE_AGAIN = "Make the PDF again"
AGAIN_LINE = "Changed something since? Make it again."
PDF_FAILED = "The PDF could not be made — try again."
DOWNLOAD = "Download the PDF"
DOWNLOAD_ANYWAY = "Download it anyway"
SAVED_TO = "Saved to:"  # the PDF's full path follows
CANT_OPEN = "This application can't be opened — its files are missing or damaged."

# This page's named CSS. The card and the list entry — the
# current-card rule, the accent bar, the tint — are `common.LOOKS_CSS`'s; what stays here is
# this page's own: the green / amber / grey lines (the chip colours), the list entry as a
# `gr.HTML` (padding, pointer, the dashed "+ New application"), and the bullet card (compact,
# grey, the actions inline on the right of the text). Passed by `launch` here and by the
# shell's `launch`.
REVIEW_CSS = (
    ".lc-ok {color: #14532d;} "
    ".lc-warn {color: #7a4b00;} "
    ".lc-note {color: var(--body-text-color-subdued); font-size: var(--text-sm);} "
    ".lc-list .lc-entry {padding: 6px 10px; cursor: pointer;} "
    ".lc-list .lc-entry.lc-new {border: 1px dashed var(--border-color-primary);} "
    ".lc-card.lc-bullet {background: var(--background-fill-secondary); border-radius: 4px; "
    "padding: 4px 8px; gap: 2px;} "
    ".lc-card.lc-bullet:hover, .lc-card.lc-bullet:focus-within {background: var(--lc-tint);} "
    ".lc-bullet .lc-line {flex-wrap: nowrap; align-items: flex-start; gap: 8px;} "
    ".lc-bullet .lc-acts {flex-wrap: nowrap; gap: 2px; flex: 0 0 auto !important; "
    "width: auto !important; min-width: 0 !important;} "
    ".lc-bullet .lc-acts > button {flex: 0 0 auto !important; min-width: max-content !important;} "
    ".lc-bullet .html-container {padding: 2px 4px;} "
    ".lc-bullet p {margin: 0;} "
    # Layout only: an entry's title 13 px, its details small mono; "+ New application" centred;
    ".lc-list .lc-entry {font-size: 13px;} "
    ".lc-list .lc-entry .lc-note, .lc-list .lc-entry .lc-ok "
    "{font-size: 11px; font-family: var(--font-mono);} "
    ".lc-list .lc-entry.lc-new {text-align: center;} "
    # the verdict and its report link in one framed box; the résumé as a framed sheet
    # with small upper-case section headings over a rule;
    ".lc-verdict {border: 1px solid var(--border-color-primary); border-radius: 4px; "
    "background: var(--background-fill-secondary); padding: 10px 14px; gap: 6px;} "
    # (the report's tables at 960 px: small type, words never broken mid-word)
    ".lc-verdict table {font-size: var(--text-sm);} "
    ".lc-verdict td, .lc-verdict th {word-break: normal !important; "
    "overflow-wrap: normal !important;} "
    ".lc-sheet {border: 1px solid var(--border-color-primary); border-radius: 4px; "
    "padding: 14px 18px;} "
    ".lc-sheet h3.lc-sec {font-size: 12px; font-weight: 600; text-transform: uppercase; "
    "letter-spacing: 0.06em; color: var(--body-text-color-subdued); margin: 10px 0 0; "
    "padding-bottom: 2px; border-bottom: 1px solid var(--border-color-primary);} "
    # "+ Add a bullet from this job" left-aligned in a dashed box, the opened rows in one;
    # (no `.gradio-container` in front: Gradio prefixes every rule with its container and the
    # prefixed link rule — `border: none` — would outrank this one)
    "button.lnk.lc-pickl {width: 100% !important; justify-content: flex-start; "
    "border: 1px dashed var(--border-color-primary); border-radius: 4px; padding: 6px 10px;} "
    ".lc-pick {border: 1px dashed var(--border-color-primary); border-radius: 4px; "
    "padding: 8px 10px;} "
    # the foot under a rule, the note left and Make the PDF right; the PDF line and
    # Download on one line.
    ".lc-foot {border-top: 1px solid var(--border-color-primary); padding-top: 10px; "
    "align-items: center;} "
    ".lc-foot > button.lc-fit {margin-left: auto;} "
    ".lc-pdfrow {align-items: center; gap: 12px;} "
    # The remove question sits between the cards — its text inset to the cards' text
    # (the card's 4 px bar + 8 px padding + the HTML block's 4 px).
    ".lc-sheet .lc-ask {padding-left: 16px;} "
    # The PDF's path — monospace, selectable in one click, wrapped anywhere.
    ".lc-path {font-family: var(--font-mono); user-select: all; overflow-wrap: anywhere;} "
    # The head of the left list — "+ New application", the "Your applications" label,
    # "How to manage this list" and its note — stays at the top of the list's scroll box, opaque,
    # while the entries scroll under it; the label is a small heading like the sheet's sections.
    ".lc-list > .lc-lhead {position: sticky; top: 0; z-index: 2; "
    "background: var(--background-fill-primary); padding-bottom: 4px; gap: 4px;} "
    ".lc-lhead .lc-label {font-size: 12px; font-weight: 600; text-transform: uppercase; "
    "letter-spacing: 0.06em; color: var(--body-text-color-subdued); margin: 8px 0 0;} "
    ".lc-lhead .lc-note {margin: 0;}"
)


# ------------------------------------------------------------------ HTML helpers (layout only)


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _p(text: str, cls: str | None = None) -> str:
    klass = f' class="{cls}"' if cls else ""
    return f"<p{klass}>{_esc(text)}</p>"


def _section(name: str) -> str:
    """A section heading of the printed sheet: small, upper-case, over a rule."""
    return f'<h3 class="lc-sec">{_esc(name)}</h3>'


def _html(value: str = "", **kwargs) -> gr.HTML:
    """A piece of text; no click listener (`js_on_load=None` — only the list's entries take one)."""
    return gr.HTML(value, js_on_load=None, **kwargs)


def _entry_html(words: EntryWords) -> str:
    stage = "lc-ok" if words.done else "lc-note"
    return (
        f"<div><b>{_esc(words.title)}</b><br>"
        f'<span class="lc-note">{_esc(words.sub)}</span><br>'
        f'<span class="{stage}">{_esc(words.stage)}</span></div>'
    )


def _bullet_html(text: str, project: str | None) -> str:
    label = f"<b>{_esc(project)}:</b> " if project else ""
    return f"<p>{label}{_esc(text)}</p>"


def _covers_html(covers: list[str]) -> str:
    return _p("covers: " + "; ".join(covers), "lc-note") if covers else ""


def _group_html(group: SkillGroup) -> str:
    items = _esc(skills_to_text(group.items)) if group.items else ""
    label = f"<b>{_esc(group.name)}:</b> " if group.name else ""
    if not items:
        return f'<p>{label}<span class="lc-note">{_esc(NO_ITEMS)}</span></p>'
    return f"<p>{label}{items}</p>"


def _text_html(text: str | None, empty: str, *, bold: bool = False) -> str:
    if not text:
        return _p(empty, "lc-note")
    return f"<p><b>{_esc(text)}</b></p>" if bold else _p(text)


def _pdf_line_html(words: PdfWords | None) -> str:
    if words is None:
        return ""
    return _p(words.line, "lc-ok" if words.ok else "lc-warn")


def _path_html(words: PdfWords | None) -> str:
    """ "Saved to: <the PDF's full path>", the path in monospace and selectable."""
    if words is None or not words.pdf:
        return ""
    path = str(Path(words.pdf).resolve())
    return f'<p class="lc-note">{_esc(SAVED_TO)} <code class="lc-path">{_esc(path)}</code></p>'


def _where_html(out_root: Path) -> str:
    """The applications folder's full path (monospace, selectable) and how to tidy it."""
    path = str(out_root.resolve())
    return (
        f'<p class="lc-note">{_esc(WHERE_SAVED)} <code class="lc-path">{_esc(path)}</code><br>'
        f"{_esc(TIDY_NOTE)}</p>"
    )


def _notes_html(words: PdfWords | None) -> str:
    return "".join(_p(note, "lc-note") for note in words.notes) if words else ""


def _add_label(count: int) -> str:
    return f"+ Add a bullet from this job ({count} in your profile)"


def _saved_now() -> str:
    return f"Saved ✓ {saved_stamp(datetime.now())}"  # dated when not today


def _bullet_at(doc: ReviewDoc, job_id: str, index: int):
    for job in doc.jobs:
        if job.job_id == job_id:
            return job.bullets[index] if 0 <= index < len(job.bullets) else None
    return None


# ------------------------------------------------------------------------------------ the page


def build(
    runtime: Runtime,
    *,
    out_root: Path,
    checkpoints: Path,
    page_limit_default: int,
    start_gate: Callable[[], str] | None = None,
    bar: gr.State | None = None,
) -> gr.Blocks:
    """The page over `out_root`. Returns the page, not launched. `start_gate`:
    a non-empty line means the profile is not ready — the New pane shows the sentence in
    place of the form, and Build refuses. `bar`: the shell's two-slot state `(line,
    save)` of this tab; `None` under `pipeline review`, where nothing writes a bar."""
    bar_outs = [bar] if bar is not None else []

    def bar_line(line: str, save: str = "") -> tuple:
        return ((line, save),) if bar is not None else ()

    def bar_skip() -> tuple:
        return (gr.skip(),) if bar is not None else ()

    def gate_now() -> str:
        return start_gate() if start_gate is not None else ""

    def read_entries() -> list[ListEntry]:
        try:
            return list_entries(out_root, checkpoints)
        except FAILURES as exc:
            gr.Warning(str(exc))
            return []

    # --------------------------------------------------------------- navigation (page level)
    # `page_outs` order: entries, selected, list_key, opened, doc, note, app_name, pane_key,
    # the New pane, the application pane, the New form, the gate sentence, then the bar.

    def gate_view() -> tuple:
        """The New form or the gate sentence, as the gate reads now (`gate_targets`' order)."""
        gate = gate_now()
        return (gr.Column(visible=not gate), gr.Markdown(visible=bool(gate)))

    def show_new(items: list[ListEntry], list_key: int) -> tuple:
        return (
            items,
            NEW,
            list_key + 1,
            None,
            None,
            "",
            "",
            gr.skip(),
            gr.Column(visible=True),
            gr.Column(visible=False),
            *gate_view(),
            *bar_line(tally_words(items)),
        )

    def show_app(app_id: str, items: list[ListEntry], list_key: int, pane_key: int) -> tuple:
        entry = next((e for e in items if e.app_id == app_id), None)
        name = entry.name if entry is not None and entry.name else app_id
        opened: Opened | None = None
        note = ""
        try:
            opened = open_review(runtime, app_id, out_root=out_root, checkpoints=checkpoints)
        except (application.NotReviewable, application.UnknownApplication):
            note = entry_words(entry).stage if entry is not None else CANT_OPEN
        except FAILURES:
            note = CANT_OPEN
            gr.Warning(CANT_OPEN)
        if opened is not None and opened.notice:
            gr.Info(opened.notice)
        return (
            items,
            app_id,
            list_key + 1,
            opened,
            None if opened is None else opened.doc,
            note,
            name,
            pane_key + 1,
            gr.Column(visible=False),
            gr.Column(visible=True),
            gr.skip(),
            gr.skip(),
            *bar_line(name),
        )

    def on_load(list_key: int, pane_key: int):
        items = read_entries()
        if items:
            return show_app(items[0].app_id, items, list_key, pane_key)
        return show_new(items, list_key)

    def go_new(items: list[ListEntry], list_key: int):
        return show_new(items, list_key)

    def go_open(app_id: str):
        def handler(items: list[ListEntry], list_key: int, pane_key: int):
            return show_app(app_id, items, list_key, pane_key)

        return handler

    # ------------------------------------------------------------------------ Build (New pane)

    def build_work(posting: str, name: str, items, list_key: int, pane_key: int) -> tuple:
        n_page = len(page_outs)
        unchanged = (gr.skip(),) * (n_page + 2)
        if gate_now():
            gr.Warning(GATE_SENTENCE)
            return (*show_new(items, list_key), gr.skip(), gr.skip())
        try:
            app_id = app_id_from_name(name, taken_ids(out_root, checkpoints))
        except FAILURES as exc:
            gr.Warning(str(exc))
            return unchanged
        try:
            with common.model_turn():  # one model call at a time, both pages
                status = start_application(
                    runtime,
                    posting,
                    app_id,
                    out_root=out_root,
                    checkpoints=checkpoints,
                    name=name.strip(),
                )
        except ApplicationExists as exc:
            gr.Warning(f"{TAKEN} {exc}")
            return unchanged
        except FAILURES as exc:
            # The id is spent: the entry is listed with its stage words.
            gr.Warning(str(exc))
            return (read_entries(), gr.skip(), list_key + 1, *(gr.skip(),) * (n_page - 3 + 2))
        gr.Info(built_words(status))
        return (*show_app(app_id, read_entries(), list_key, pane_key), "", "")

    def build_app(posting: str, name: str, items, list_key: int, pane_key: int):
        n = len(page_outs) + 2
        yield (*(gr.skip(),) * n, *page_veil.on(common.PHRASE_BUILD))
        landed: tuple = (gr.skip(),) * n
        try:
            landed = build_work(posting, name, items, list_key, pane_key)
        finally:
            yield (*landed, *page_veil.off())

    # ----------------------------------------------------------------------- the application

    def draw_pane(opened: Opened, doc: ReviewDoc, name: str) -> None:
        """Lay out one application; every event below targets this draw's own components."""
        my_id = opened.app_id

        def fresh(o: Opened | None) -> bool:
            # A late event after a navigation acts on nothing.
            return o is not None and o.app_id == my_id

        def keep(o: Opened, old: ReviewDoc, new: ReviewDoc) -> bool:
            """Save when the operation changed the document; say whether it did."""
            if new == old:
                return False
            save_review(new, o.app_id, out_root=o.out_root, template_path=o.template_path)
            return True

        def saved_bar(changed: bool) -> tuple:
            return bar_line(name, _saved_now()) if changed else bar_skip()

        # ---- the verdict card and the full report (#4)
        verdict = verdict_words(opened.payload, opened.rows)
        head = _esc(verdict.headline)
        if verdict.fit and verdict.headline.startswith(verdict.fit):
            head = f"<b>{_esc(verdict.fit)}</b>{_esc(verdict.headline[len(verdict.fit) :])}"
        card = [f"<p>{head}</p>"]
        if verdict.matches:
            card.append(_p(verdict.matches))
        card.extend(_p(sentence, "lc-warn") for sentence in verdict.uncovered)
        if verdict.nice:
            card.append(_p(verdict.nice))
        card.append(_p(CARD_NOTE, "lc-note"))
        report_open = gr.State(False)
        with gr.Column(elem_classes=["lc-verdict"]):  # one framed box, link inside
            _html("<div>" + "".join(card) + "</div>")
            report_link = common.link(SHOW_REPORT)
            with gr.Column(visible=False) as report_col:
                _html(_p(REPORT_INTRO, "lc-note"))
                gr.Markdown(report_body(opened.report))

        def toggle_report(shown: bool):
            return (
                not shown,
                gr.Column(visible=not shown),
                gr.Button(value=SHOW_REPORT if shown else HIDE_REPORT),
            )

        report_link.click(
            toggle_report,
            inputs=[report_open],
            outputs=[report_open, report_col, report_link],
            **QUIET,
        )

        # ---- the résumé as it will print (drawn below in one framed sheet)
        def inline(
            display: Callable[[ReviewDoc], str],
            current: Callable[[ReviewDoc], str],
            apply: Callable[[ReviewDoc, str], ReviewDoc],
            lines: int = 1,
        ) -> None:
            """One line with **edit**; the line becomes a box in place (#7)."""
            with gr.Column(elem_classes=["lc-card"]):
                with gr.Row(equal_height=True) as shown_row:
                    text = _html(display(doc))
                    edit_link = common.link("edit", scale=0, min_width=60)
                with gr.Column(visible=False) as editor:
                    box = gr.Textbox(
                        show_label=False, lines=lines, interactive=True, autofocus=True
                    )
                    with gr.Row():
                        done = common.secondary("Done", scale=0, min_width=80)
                        cancel = common.link("cancel", scale=0, min_width=80)
            opening = gr.State("")

            def open_box(o: Opened | None, d: ReviewDoc):
                if not fresh(o):
                    return (gr.skip(),) * 4
                value = current(d)
                return (
                    gr.Row(visible=False),
                    gr.Column(visible=True),
                    gr.Textbox(value=value),
                    value,
                )

            def commit(o: Opened | None, d: ReviewDoc, value: str):
                if not fresh(o):
                    return (gr.skip(),) * (4 + len(bar_outs))
                try:
                    new = apply(d, value)
                    changed = keep(o, d, new)
                except FAILURES as exc:
                    gr.Warning(str(exc))
                    return (gr.skip(),) * (4 + len(bar_outs))
                return (
                    new,
                    gr.Row(visible=True),
                    gr.HTML(value=display(new)),
                    gr.Column(visible=False),
                    *saved_bar(changed),
                )

            outs = [doc_state, shown_row, text, editor, *bar_outs]
            edit_link.click(
                open_box,
                inputs=[opened_state, doc_state],
                outputs=[shown_row, editor, box, opening],
                **QUIET,
            )
            box.blur(commit, inputs=[opened_state, doc_state, box], outputs=outs, **QUIET)
            done.click(commit, inputs=[opened_state, doc_state, box], outputs=outs, **QUIET)
            cancel.click(commit, inputs=[opened_state, doc_state, opening], outputs=outs, **QUIET)

        with gr.Column(elem_classes=["lc-sheet"]):
            # the fixed header lines
            if doc.name:
                _html(f"<h2>{_esc(doc.name)}</h2>")
            if doc.contact:
                _html(_p(" | ".join(doc.contact)))
            _html(_p(PROFILE_NOTE, "lc-note"))
            inline(
                lambda d: _text_html(d.headline, NO_HEADLINE, bold=True),
                lambda d: d.headline or "",
                set_headline,
            )
            _html(_section("Summary"))
            inline(
                lambda d: _text_html(d.summary, NO_SUMMARY),
                lambda d: d.summary or "",
                set_summary,
                4,
            )
            if doc.skill_groups:
                _html(_section("Skills"))
            for g in range(len(doc.skill_groups)):

                def set_group(d: ReviewDoc, value: str, g: int = g) -> ReviewDoc:
                    items = [list(group.items) for group in d.skill_groups]
                    items[g] = skills_from_text(value)
                    return set_skill_groups(d, items)

                inline(
                    lambda d, g=g: _group_html(d.skill_groups[g]),
                    lambda d, g=g: skills_to_text(d.skill_groups[g].items),
                    set_group,
                )

            # ---- Experience: per job, position-indexed rows (#5, #6, #9)
            _html(_section("Experience"))
            mismatch = index_mismatch(doc, opened.rows)
            if mismatch:
                _html(_p(OTHER_INDEX, "lc-warn"))
            offers_at_draw = {} if mismatch else addable(doc, opened.payload, opened.candidates)
            for job in doc.jobs:
                draw_job(
                    opened,
                    doc,
                    job,
                    offers_at_draw.get(job.job_id, []),
                    mismatch,
                    fresh,
                    keep,
                    saved_bar,
                )

            if doc.education:
                _html(_section("Education") + "".join(_p(line) for line in doc.education))

        # ---- the foot: Make the PDF and the PDF line — the note and the
        # button on one line under a rule; the PDF line and Download on one (the path
        # line under them).
        words = pdf_words(opened.status)
        with gr.Row(equal_height=True, elem_classes=["lc-foot"]):
            _html(_p(FOOT, "lc-note"))
            make = common.primary(MAKE_PDF, visible=words is None, scale=0, elem_classes=["lc-fit"])
        with gr.Row(equal_height=True, elem_classes=["lc-pdfrow"]):
            pdf_line = _html(_pdf_line_html(words), visible=words is not None)
            download = gr.DownloadButton(
                label=DOWNLOAD if words is None or words.ok else DOWNLOAD_ANYWAY,
                value=None if words is None else words.pdf,
                visible=words is not None,
                size="sm",
                scale=0,
                min_width=0,
                elem_classes=["lc-fit"],
            )
        # No folder link — the PDF's full path on its own line
        # whenever a PDF exists, the one place a path shows.
        path_line = _html(_path_html(words), visible=bool(words and words.pdf))
        with gr.Row(visible=words is not None) as again_row:
            _html(_p(AGAIN_LINE))
            again = common.secondary(MAKE_AGAIN, scale=0)
        notes = _html(_notes_html(words), visible=bool(words and words.notes))

        def pdf_work(o: Opened, d: ReviewDoc, list_key: int) -> tuple:
            try:
                saved = save_review(d, o.app_id, out_root=o.out_root, template_path=o.template_path)
                result = application.submit(
                    runtime,
                    o.app_id,
                    edited_path=saved.resume_path,
                    page_limit=page_limit_default,
                    out_root=o.out_root,
                    checkpoints=checkpoints,
                )
            except FAILURES as exc:
                gr.Warning(f"{PDF_FAILED} {exc}")
                return (gr.skip(),) * n_pdf
            done = pdf_words(result)
            return (
                dataclasses.replace(o, status=result),
                read_entries(),
                list_key + 1,
                gr.Button(visible=done is None),
                gr.HTML(value=_pdf_line_html(done), visible=done is not None),
                gr.DownloadButton(
                    label=DOWNLOAD if done is None or done.ok else DOWNLOAD_ANYWAY,
                    value=None if done is None else done.pdf,
                    visible=done is not None,
                ),
                gr.HTML(value=_path_html(done), visible=bool(done and done.pdf)),
                gr.Row(visible=done is not None),
                gr.HTML(value=_notes_html(done), visible=bool(done and done.notes)),
                *saved_bar(True),
            )

        def make_pdf(o: Opened | None, d: ReviewDoc, list_key: int):
            yield (*(gr.skip(),) * n_pdf, *page_veil.on(common.PHRASE_PDF))
            landed: tuple = (gr.skip(),) * n_pdf
            try:
                if fresh(o):
                    landed = pdf_work(o, d, list_key)
            finally:
                yield (*landed, *page_veil.off())

        pdf_outs = [
            opened_state,
            entries,
            list_key_state,
            make,
            pdf_line,
            download,
            path_line,
            again_row,
            notes,
            *bar_outs,
        ]
        n_pdf = len(pdf_outs)
        for button in (make, again):
            button.click(
                make_pdf,
                inputs=[opened_state, doc_state, list_key_state],
                outputs=[*pdf_outs, *page_veil.outputs],
                **QUIET,
            )

    def draw_job(
        opened: Opened,
        doc: ReviewDoc,
        job: ReviewJob,
        offers: list,
        mismatch: bool,
        fresh: Callable[[Opened | None], bool],
        keep: Callable[[Opened, ReviewDoc, ReviewDoc], bool],
        saved_bar: Callable[[bool], tuple],
    ) -> None:
        """One job: its heading, `len(bullets) + len(offers)` bullet rows (each with its remove
        question under its card), the Add fold with as many rows. Every event updates this
        job's rows only."""
        job_id = job.job_id
        n = len(job.bullets) + len(offers)
        _html(
            f"<p><b>{_esc(job.heading)}</b><br>"
            f'<span class="lc-note">{_esc(resume_date(job.date_from))} – '
            f"{_esc(resume_date(job.date_to))}</span></p>"
        )
        empty = _html(_p(NO_BULLET, "lc-note"), visible=not job.bullets)
        show_all = gr.State(False)

        rows = job_rows(doc, job_id, opened.payload, opened.candidates)
        slots: list[dict] = []
        for i in range(n):
            row = rows[i] if i < len(rows) else None
            project = job.bullets[i].project if i < len(job.bullets) else None
            # The compact card — the text with the actions inline on its
            # right (hidden off the current card, in the same row, so no empty band), the
            # editor in the text's place, "covers" right under.
            with gr.Column(visible=row is not None, elem_classes=["lc-card", "lc-bullet"]) as card:
                with gr.Row(equal_height=False, elem_classes=["lc-line"]):
                    text = _html(
                        _bullet_html(row.text, project) if row else "", scale=1, min_width=0
                    )
                    with gr.Row(scale=0, elem_classes=["lc-act", "lc-acts"]) as acts:
                        edit_link = common.link("✎ edit", scale=0, min_width=0)
                        up = common.link("↑", scale=0, min_width=0)
                        down = common.link("↓", scale=0, min_width=0)
                        remove = common.link("remove", scale=0, min_width=0)
                        restore = common.link(
                            "restore the draft's wording",
                            visible=bool(row and row.edited),
                            scale=0,
                            min_width=0,
                        )
                with gr.Column(visible=False) as editor:
                    box = gr.Textbox(show_label=False, lines=3, interactive=True, autofocus=True)
                    with gr.Row(equal_height=True):
                        done = common.secondary("Done", scale=0, min_width=80)
                        cancel = common.link("cancel", scale=0, min_width=80)
                        _html(_p(EDIT_NOTE, "lc-note"), min_width=0)
                covers = _html(_covers_html(row.covers if row else []))
            # The question opens directly under its own card
            remove_ask = common.ask(REMOVE_QUESTION)  # one line, no grey block
            opening = gr.State("")
            slots.append(
                {
                    "card": card,
                    "text": text,
                    "acts": acts,
                    "restore": restore,
                    "covers": covers,
                    "editor": editor,
                    "edit": edit_link,
                    "up": up,
                    "down": down,
                    "remove": remove,
                    "question": remove_ask.row,
                    "yes": remove_ask.yes,
                    "no": remove_ask.no,
                    "box": box,
                    "done": done,
                    "cancel": cancel,
                    "opening": opening,
                }
            )

        questions = [slot["question"] for slot in slots]

        add_rows: list[tuple] = []
        add_link = more = fold = None
        if not mismatch:
            # The link left-aligned in a dashed box; the opened rows in one too.
            add_link = common.link(
                _add_label(len(offers)), visible=bool(offers), elem_classes=["lc-pickl"]
            )
            fold_open = gr.State(False)
            with gr.Column(visible=False, elem_classes=["lc-pick"]) as fold:
                for k in range(n):
                    offer = offers[k] if k < len(offers) else None
                    with gr.Row(visible=offer is not None and k < SHOWN) as add_row:
                        add_text = _html(_p(offer.text) if offer else "")
                        add_button = common.secondary("Add", scale=0, min_width=70)
                    add_rows.append((add_row, add_text, add_button))
                more = common.link(f"show all {len(offers)}", visible=len(offers) > SHOWN)

        # ---- the job's views: one row, and the whole job
        slot_keys = ("card", "text", "acts", "restore", "covers", "editor")

        def slot_view(d: ReviewDoc, row: RowWords | None, index: int) -> tuple:
            if row is None:
                return (gr.Column(visible=False), *(gr.skip(),) * 4, gr.Column(visible=False))
            bullet = _bullet_at(d, job_id, index)
            return (
                gr.Column(visible=True),
                gr.HTML(
                    value=_bullet_html(row.text, bullet.project if bullet else None), visible=True
                ),
                gr.Row(visible=True),
                gr.Button(visible=row.edited),
                gr.HTML(value=_covers_html(row.covers)),
                gr.Column(visible=False),
            )

        def job_view(d: ReviewDoc, o: Opened, show: bool) -> tuple:
            now = job_rows(d, job_id, o.payload, o.candidates)
            out: list = [gr.HTML(visible=not now)]
            for i in range(n):
                out.extend(slot_view(d, now[i] if i < len(now) else None, i))
            if not mismatch:
                offered = addable(d, o.payload, o.candidates).get(job_id, [])
                out.append(gr.Button(value=_add_label(len(offered)), visible=bool(offered)))
                for k in range(n):
                    offer = offered[k] if k < len(offered) else None
                    out.append(gr.Row(visible=offer is not None and (show or k < SHOWN)))
                    out.append(gr.HTML(value=_p(offer.text) if offer else ""))
                out.append(
                    gr.Button(
                        value=f"show all {len(offered)}",
                        visible=len(offered) > SHOWN and not show,
                    )
                )
            out.extend(gr.Row(visible=False) for _ in questions)  # rows move: questions close
            return tuple(out)

        job_targets: list = [empty]
        for slot in slots:
            job_targets.extend(slot[key] for key in slot_keys)
        if not mismatch:
            job_targets.append(add_link)
            for add_row, add_text, _ in add_rows:
                job_targets.extend((add_row, add_text))
            job_targets.append(more)
        job_targets.extend(questions)
        job_outs = [doc_state, *job_targets, *bar_outs]

        def job_act(change: Callable[..., ReviewDoc]):
            """A move, a remove or an Add: one operation, a save, this job's rows."""

            def handler(o: Opened | None, d: ReviewDoc, show: bool):
                skipped = (gr.skip(),) * len(job_outs)
                if not fresh(o):
                    return skipped
                try:
                    new = change(d, o)
                    changed = keep(o, d, new)
                except FAILURES as exc:
                    gr.Warning(str(exc))
                    return skipped
                return (new, *job_view(new, o, show), *saved_bar(changed))

            return handler

        def slot_act(index: int, change: Callable[..., ReviewDoc]):
            """A text change of one row (commit, cancel, restore): that row only."""
            outs_n = 1 + len(slot_keys) + len(bar_outs)

            def handler(o: Opened | None, d: ReviewDoc, *extra):
                if not fresh(o):
                    return (gr.skip(),) * outs_n
                try:
                    new = change(d, o, *extra)
                    changed = keep(o, d, new)
                except FAILURES as exc:
                    gr.Warning(str(exc))
                    return (gr.skip(),) * outs_n
                now = job_rows(new, job_id, o.payload, o.candidates)
                row = now[index] if index < len(now) else None
                return (new, *slot_view(new, row, index), *saved_bar(changed))

            return handler

        def open_box(index: int):
            def handler(o: Opened | None, d: ReviewDoc):
                bullet = _bullet_at(d, job_id, index)
                if not fresh(o) or bullet is None:
                    return (gr.skip(),) * 5
                return (
                    gr.HTML(visible=False),
                    gr.Row(visible=False),
                    gr.Column(visible=True),
                    gr.Textbox(value=bullet.text),
                    bullet.text,
                )

            return handler

        def restored(d: ReviewDoc, o: Opened, index: int) -> ReviewDoc:
            bullet = _bullet_at(d, job_id, index)
            if bullet is None:
                return d
            return edit(d, job_id, index, pick(o.candidates, bullet.bullet_id).text)

        def added(d: ReviewDoc, o: Opened, k: int) -> ReviewDoc:
            offered = addable(d, o.payload, o.candidates).get(job_id, [])
            if k >= len(offered):
                return d
            return add(d, job_id, pick(o.candidates, offered[k].bullet_id))

        for i, slot in enumerate(slots):
            row_outs = [doc_state, *(slot[key] for key in slot_keys), *bar_outs]
            slot["edit"].click(
                open_box(i),
                inputs=[opened_state, doc_state],
                outputs=[slot["text"], slot["acts"], slot["editor"], slot["box"], slot["opening"]],
                **QUIET,
            )
            commit = slot_act(i, lambda d, _o, value, i=i: edit(d, job_id, i, value))
            slot["box"].blur(
                commit, inputs=[opened_state, doc_state, slot["box"]], outputs=row_outs, **QUIET
            )
            slot["done"].click(
                commit, inputs=[opened_state, doc_state, slot["box"]], outputs=row_outs, **QUIET
            )
            slot["cancel"].click(
                commit,
                inputs=[opened_state, doc_state, slot["opening"]],
                outputs=row_outs,
                **QUIET,
            )
            slot["restore"].click(
                slot_act(i, lambda d, o, i=i: restored(d, o, i)),
                inputs=[opened_state, doc_state],
                outputs=row_outs,
                **QUIET,
            )
            for key, delta in (("up", -1), ("down", 1)):
                slot[key].click(
                    job_act(lambda d, _o, i=i, delta=delta: move(d, job_id, i, delta)),
                    inputs=[opened_state, doc_state, show_all],
                    outputs=job_outs,
                    **QUIET,
                )
            # This card's question opens, any other of the job's closes
            slot["remove"].click(
                lambda o, i=i: (
                    tuple(gr.Row(visible=j == i) for j in range(len(questions)))
                    if fresh(o)
                    else (gr.skip(),) * len(questions)
                ),
                inputs=[opened_state],
                outputs=questions,
                **QUIET,
            )
            slot["yes"].click(
                job_act(lambda d, _o, i=i: delete(d, job_id, i)),
                inputs=[opened_state, doc_state, show_all],
                outputs=job_outs,
                **QUIET,
            )
            slot["no"].click(lambda: gr.Row(visible=False), outputs=[slot["question"]], **QUIET)

        if not mismatch:
            add_link.click(
                lambda shown: (not shown, gr.Column(visible=not shown)),
                inputs=[fold_open],
                outputs=[fold_open, fold],
                **QUIET,
            )
            for k, (_, _, add_button) in enumerate(add_rows):
                add_button.click(
                    job_act(lambda d, o, k=k: added(d, o, k)),
                    inputs=[opened_state, doc_state, show_all],
                    outputs=job_outs,
                    **QUIET,
                )

            def show_everything(o: Opened | None, d: ReviewDoc):
                if not fresh(o):
                    return (gr.skip(),) * (1 + len(job_targets))
                return (True, *job_view(d, o, True))

            more.click(
                show_everything,
                inputs=[opened_state, doc_state],
                outputs=[show_all, *job_targets],
                **QUIET,
            )

    # ------------------------------------------------------------------------------- layout

    with gr.Blocks(title="LoomCV applications") as page:
        page_veil = common.veil()
        entries = gr.State([])  # list[ListEntry]
        selected = gr.State(NEW)  # NEW or an application id
        list_key_state = gr.State(0)  # bumped to redraw the list
        opened_state = gr.State(None)  # Opened | None
        doc_state = gr.State(None)  # ReviewDoc | None — the truth while a pane is open
        note_state = gr.State("")  # the line a pane shows when the application can't open
        app_name = gr.State("")  # the selected application's name, for the bar
        pane_key = gr.State(0)  # bumped to redraw the pane — the only trigger of its render

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260, elem_classes=["lc-list"]):
                with gr.Column(elem_classes=["lc-lhead"]):  # pinned while entries scroll

                    @gr.render(
                        inputs=[entries, selected],
                        triggers=[list_key_state.change],
                        show_progress="hidden",
                        **SERIAL,
                    )
                    def the_new(items: list[ListEntry], current: str):
                        new_on = ["on"] if current == NEW else []
                        new_entry = gr.HTML(
                            f"<div><b>{_esc(NEW_ENTRY)}</b></div>",
                            elem_classes=["lc-entry", "lc-new", *new_on],
                        )
                        new_entry.click(
                            go_new, inputs=[entries, list_key_state], outputs=page_outs, **QUIET
                        )

                    _html(f'<p class="lc-label">{_esc(LIST_LABEL)}</p>')
                    manage_link = common.link(MANAGE_LINK)
                    with gr.Column(visible=False) as manage_note:
                        _html(_where_html(out_root))
                    manage_open = gr.State(False)

                @gr.render(
                    inputs=[entries, selected],
                    triggers=[list_key_state.change],
                    show_progress="hidden",
                    **SERIAL,
                )
                def the_list(items: list[ListEntry], current: str):
                    if not items:
                        _html(_p(NO_APPS, "lc-note"))
                    for item in items:
                        on = ["on"] if current == item.app_id else []
                        entry = gr.HTML(
                            _entry_html(entry_words(item)), elem_classes=["lc-entry", *on]
                        )
                        entry.click(
                            go_open(item.app_id),
                            inputs=[entries, list_key_state, pane_key],
                            outputs=page_outs,
                            **QUIET,
                        )

            with gr.Column(scale=3):
                with gr.Column(visible=False) as new_pane:
                    gr.Markdown(f"### {NEW_HEADING}")
                    gate_md = gr.Markdown(GATE_SENTENCE, visible=False)
                    with gr.Column() as new_form:
                        gr.Markdown(NEW_SENTENCE)
                        posting = gr.Textbox(label=POSTING_LABEL, lines=7, interactive=True)
                        name_box = gr.Textbox(
                            label=NAME_LABEL,
                            placeholder=NAME_PLACEHOLDER,
                            max_lines=1,
                            interactive=True,
                        )
                        build_button = common.primary(BUILD_LABEL, elem_classes=["lc-fit"])  # A3
                        _html(_p(BUILD_NOTE, "lc-note"))

                with gr.Column(visible=False) as app_pane:

                    @gr.render(
                        inputs=[opened_state, doc_state, note_state, app_name],
                        triggers=[pane_key.change],
                        show_progress="hidden",
                        **SERIAL,
                    )
                    def the_pane(opened: Opened | None, doc: ReviewDoc | None, note: str, name):
                        if opened is None or doc is None:
                            if note:
                                _html(_p(note, "lc-warn"))
                            return
                        draw_pane(opened, doc, name or opened.app_id)

        page_outs = [
            entries,
            selected,
            list_key_state,
            opened_state,
            doc_state,
            note_state,
            app_name,
            pane_key,
            new_pane,
            app_pane,
            new_form,
            gate_md,
            *bar_outs,
        ]
        page.load(on_load, inputs=[list_key_state, pane_key], outputs=page_outs, **QUIET)
        manage_link.click(
            lambda shown: (not shown, gr.Column(visible=not shown)),
            inputs=[manage_open],
            outputs=[manage_open, manage_note],
            **QUIET,
        )
        # The frame re-reads the gate on its own events (a Save that opens
        # it, a tab click) through these two.
        page.gate_targets = [new_form, gate_md]
        page.gate_view = gate_view
        build_button.click(
            build_app,
            inputs=[posting, name_box, entries, list_key_state, pane_key],
            outputs=[*page_outs, posting, name_box, *page_veil.outputs],
            **QUIET,
        )

    return page


def launch(runtime: Runtime, *, out_root: Path, checkpoints: Path, page_limit_default: int) -> None:
    """Open the page alone (`pipeline review`) over `out_root`. Blocks until the server stops.
    Gradio serves a download only from the working directory, the temp directory or an allowed
    path; the applications folder is usually none of them."""
    css = " ".join([common.VEIL_CSS, common.LOOKS_CSS, common.BUTTON_CSS, REVIEW_CSS])
    build(
        runtime, out_root=out_root, checkpoints=checkpoints, page_limit_default=page_limit_default
    ).launch(css=css, allowed_paths=[str(out_root.resolve())])
