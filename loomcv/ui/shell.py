"""`pipeline app`: the frame around the two pages. Default components; no logic — every
decision comes from `loomcv.app_state`.

The frame holds the profile page (`pipeline ui`) and the whole `pipeline review` page in
tabs under a one-line status bar. The landing (which tab, which file the profile page is built
over) is decided once, when `pipeline app` starts; the gate is re-checked on every browser load
and every tab click. Start over deletes the two profile files behind one confirmation. Not covered
by pytest (the `ui` group is not installed in CI).

The profile page is built in all three states; in
*empty* it sits behind an upload block (one file control) whose handler reads the file
(`loomcv.ingest.read`), runs the existing structure call, writes the working copy and loads the
workbook into the page's `gr.State` in place. A browser load redraws the profile side from the
files, because a `gr.State` returns to its build-time value on every load.

The upload asks before it reads — a
"Read this résumé" button runs the handler, dropping a file runs nothing; `launch` passes the
profile page's chip CSS (`profile_ui.CHIP_CSS`) beside the status bar's rules; the result line
has its singular.

There are two tabs, My profile and Applications (the whole `pipeline review` page under the
gate). The
bar is `step · page line · save state`: slot 1 the selected tab; slots 2–3 each page's own,
written by the page into a two-slot `gr.State` the shell hands it as `bar=` — the shell composes
the Markdown from the selected tab's slots on a tab select and when that tab's slots change. The
gate sentence is shown inside the Applications tab, not in the bar. The upload runs under the
shell's own veil (`loomcv.ui.common`). Start over sits in the profile page's sticky left list,
under "How this page works" — mounted through the page's `left_foot`, still the shell's
handler. `launch` passes the bar's rules, the chips and `common`'s three CSS constants.

The upload fills everything once — the
structure call, then `first_keywords` over every bullet with the extracted, unconfirmed tools
lists, under the same veil; a failed keyword call is a warning, never a refusal.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any

import gradio as gr

from loomcv import app_state
from loomcv.complete import (
    CompleteError,
    first_keywords,
    load_profile,
    load_workbook,
    save_working_copy,
)
from loomcv.complete.workbook import Workbook, empty_workbook
from loomcv.ingest import IngestError, ResumeReadError, ingest_resume, resume_text
from loomcv.llm import build_generation_adapter, load_node_config
from loomcv.steps import Runtime
from loomcv.ui import app as profile_ui
from loomcv.ui import common
from loomcv.ui import review as review_ui

# The bar pinned: Gradio's container stops being the sticky
# element's scroll container — clipping unchanged.
# The bar is drawn as a light strip — position unchanged.
# A box's label (Gradio's `block-info`) is drawn at z-index 40, so it
# painted over the bar on scroll — the bar sits at 100, still under the veil's 1000.
STATUS_CSS = (
    "#loomcv-status {position: sticky; top: 0; z-index: 100; "
    "background: var(--background-fill-secondary); padding: 6px 16px; "
    "border-bottom: 1px solid var(--border-color-primary); "
    # Help entry (2026-11-10): the bar is a row — the status text, then the link at its right.
    "flex-wrap: nowrap; align-items: center; gap: 12px;} "
    # Gradio writes each child's flex and min-width inline, hence `!important` (as `common`).
    "#loomcv-status > .lc-bartext {flex: 1 1 0% !important; min-width: 0 !important;} "
    "#loomcv-status > .lc-help {flex: 0 0 auto !important; min-width: 0 !important; "
    "width: auto !important;} "
    # The two blocks' own padding off, so the bar keeps its one-line height.
    "#loomcv-status > .block, #loomcv-status .html-container {padding: 0 !important;} "
    "#loomcv-status .lc-help a {color: var(--lc-accent); text-decoration: underline; "
    "font-size: var(--text-sm); white-space: nowrap;} "
    ".gradio-container {overflow: clip;}"
)
# A placeholder until the public repo's README exists.
HELP_URL = "https://github.com/leonqi-io/loomcv#readme"
HELP_LINK = f'<a href="{HELP_URL}" target="_blank" rel="noopener">Help &amp; setup ↗</a>'
TAB_IDS = {
    app_state.PROFILE_TAB: "profile",
    app_state.APPLICATIONS_TAB: "applications",
}
# The profile side's events run one at a time (the review page's pattern, `review.py` SERIAL).
SERIAL = {"concurrency_limit": 1, "concurrency_id": "profile"}
# The wording is judged on the screen.
UPLOAD_GUIDANCE = (
    f"{app_state.EMPTY_LINE} Upload yours to start — PDF, Word, Markdown or plain text."
)
CHOOSE_FILE_LINE = "Choose your résumé file first."
RESULT_LINE = (
    "{jobs} and {bullets} found — check them below. "
    "An old-format or multi-column résumé may be split imperfectly."
)
MODEL_FAILED_LINE = (
    "The model could not be reached or did not answer. Nothing was saved; try again."
)
NO_JOBS_LINE = (
    "No work history was recognised. Try another file, or a Markdown or plain-text version."
)
WRITE_FAILED_LINE = "Your profile could not be saved here. Nothing was kept; try again."
KEYWORDS_SKIPPED_LINE = (
    "Search keywords couldn't be picked this time — use “suggest” on a bullet or "
    "“Suggest search keywords again” at the end of a job."
)
# The generation path of the `ingest` node (as `pipeline ingest`'s exit 4, plus the file system).
_FAILURES = (
    KeyError,
    ValueError,
    RuntimeError,
    NotImplementedError,
    OSError,
    IngestError,
    CompleteError,
)


def _count(n: int, word: str) -> str:
    """`1 job` / `3 jobs` — a format of the number, nothing else."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def keyword_phase(started: float) -> tuple:
    """The upload veil's second line moved to the keyword step; the clock is not reset —
    the same three outputs as `Veil.on`, the start kept."""
    phrase = common.PHRASE_KEYWORDS
    return (
        gr.HTML(value=common.veil_html(phrase, time.monotonic() - started)),
        gr.skip(),
        (phrase, started),
    )


def _number_renders(*pages: gr.Blocks) -> None:
    """Give every `@gr.render` of the frame its own number.

    Gradio (6.30) numbers a `@gr.render` by its place in the Blocks it was built in
    (`Renderable._id = len(blocks_config.renderables)`), sends that number with every redraw as
    `render_id`, and the browser then drops **every** event tagged with that number before adding
    the redraw's own (`dependencies.filter(e => e.rendered_in !== render_id)`). The two pages are
    built apart and mounted with `.render()`, which does not renumber them, so both count from 0:
    the profile's Education (0) and skills block (1) shared their numbers with the applications
    list (0) and pane (1). On every load the profile's hand-off redraws Education and the skills
    block after the list, and the list's clicks were gone ("No endpoint matching N"); after a
    Build the list redrew last and Education's boxes were the dead ones instead. Renumbered once,
    before launch, over both pages in order — no number is shared, nothing else changes."""
    renders = [render for page in pages for render in page.renderables]
    for number, render in enumerate(renders):
        render._id = number


def build(
    runtime: Runtime,
    *,
    working_copy: Path,
    saved: Path,
    sidecar: Path,
    out_root: Path,
    checkpoints: Path,
    page_limit_default: int,
    config: Path | None,
) -> gr.Blocks:
    """The frame with both pages mounted. Returns the page, not launched."""
    land = app_state.landing(working_copy, saved)

    def readiness() -> app_state.Readiness:
        return app_state.readiness(saved, db=runtime.db, out_root=out_root, checkpoints=checkpoints)

    # Each page's (line, save) slots of the bar, created before the pages and handed to them.
    slots = {
        app_state.PROFILE_TAB: gr.State(common.EMPTY_SLOTS, render=False),
        app_state.APPLICATIONS_TAB: gr.State(common.EMPTY_SLOTS, render=False),
    }
    # Built in every state (an empty workbook in *empty*): the upload lands in this page's state.
    from loomcv.steps import save_profile as save_step

    start_over_parts: dict[str, Any] = {}

    def mount_start_over() -> None:
        """Start over and its question are built inside
        the profile page's sticky left list, under "How this page works"; wired below."""
        start_over_parts["link"] = common.link(
            "Start over with a new résumé", visible=land.page_over is not None
        )
        start_over_parts["ask"] = common.ask(app_state.START_OVER_QUESTION)

    profile_page = profile_ui.build(
        land.page_over,
        saved,
        save_profile=lambda wb, *, today: save_step(
            wb,
            saved,
            db=runtime.db,
            chroma=runtime.chroma,
            adapter=lambda: runtime.adapter("embedding"),
            today=today,
        ),
        working_copy=working_copy,
        sidecar=sidecar,
        config=config,
        bar=slots[app_state.PROFILE_TAB],
        left_foot=mount_start_over,
    )
    start_over = start_over_parts["link"]
    start_over_ask = start_over_parts["ask"]
    start_over_group = start_over_ask.row
    start_over_yes, start_over_no = start_over_ask.yes, start_over_ask.no
    review_page = review_ui.build(
        runtime,
        out_root=out_root,
        checkpoints=checkpoints,
        page_limit_default=page_limit_default,
        start_gate=lambda: readiness().line,
        bar=slots[app_state.APPLICATIONS_TAB],
    )
    _number_renders(profile_page, review_page)

    def gate_outputs(step: str, shown: tuple[str, str]):
        """Bar (the selected tab and its own slots), the posting side's gate line and page, then
        the review page's New pane read against the gate again (a Save that opens the gate
        drops the page's "finish your profile first" sentence without a click)."""
        is_open = readiness().posting_side_open
        return (
            common.bar_text(step, *shown),
            gr.Markdown(visible=not is_open),
            gr.Column(visible=is_open),
            *review_page.gate_view(),
        )

    def profile_view(now: app_state.Landing):
        """The profile side as the files say it is: line, page column, Start over, upload
        block, upload Message line, then the page's hand-off values (workbook, layout, header…).
        A `gr.State` is not the truth after a browser load, so this is read from the files each
        time (the landing is `app_state.landing`; the working copy wins over the saved vault,
        and its sidecar decides each job's Defaults confirmed)."""
        if now.page_over is None:
            return (
                UPLOAD_GUIDANCE,
                gr.Column(visible=False),
                gr.Button(visible=False),
                gr.Column(visible=True),
                "",
                *profile_page.hand_off(empty_workbook(), []),
            )
        workbook, warnings = load_profile(now.page_over, working_copy=working_copy, sidecar=sidecar)
        return (
            now.line,
            gr.Column(visible=True),
            gr.Button(visible=True),
            gr.Column(visible=False),
            "",
            *profile_page.hand_off(workbook, warnings),
        )

    def on_load(profile_slots, applications_slots):
        """Each browser load: the gate, and the profile side redrawn from the files."""
        shown = profile_slots if land.tab == app_state.PROFILE_TAB else applications_slots
        return (
            *gate_outputs(land.tab, shown),
            *profile_view(app_state.landing(working_copy, saved)),
        )

    def start_over_confirmed(profile_slots):
        app_state.start_over(working_copy, saved, sidecar)
        gr.Info("Your profile was deleted. Your past applications stay.")
        return (
            *gate_outputs(app_state.PROFILE_TAB, profile_slots),
            *profile_view(app_state.landing(working_copy, saved)),
            gr.Row(visible=False),
        )

    def on_select(name: str):
        def handler(shown):
            return (name, *gate_outputs(name, shown))

        return handler

    def on_slots(name: str):
        def handler(now_selected, shown):
            return common.bar_text(name, *shown) if now_selected == name else gr.skip()

        return handler

    def read_upload(path: str | None):
        """The upload: file → text → the structure call → the keyword call over every bullet
        (the veil's second line changes between them) → the working copy and its
        sidecar → the page, in place, under the shell's veil. Every refusal leaves *empty* with
        the veil off and the control live; the status bar is not an output.
        The output after the page's targets is the file control (reset in every refusal and in
        the landing), then the veil."""
        skip = gr.skip()
        rest = len(profile_page.hand_off_targets)  # the page's own targets, all left alone
        n_landed = 5 + rest + 1
        if path is None:
            gr.Warning(CHOOSE_FILE_LINE)
            yield (skip,) * (n_landed + len(upload_veil.outputs))
            return
        veil_on = upload_veil.on(common.PHRASE_UPLOAD)
        started = veil_on[2][1]  # the kept `(phrase, start)`: the keyword phase keeps the clock
        yield (*[skip] * 4, "", *[skip] * rest, skip, *veil_on)

        def refused(sentence: str):
            gr.Warning(sentence)
            return (
                UPLOAD_GUIDANCE,
                skip,
                skip,
                skip,
                sentence,
                *[skip] * rest,
                gr.File(value=None),
            )

        def structure():
            """The first call → `(workbook, adapter)`, or a refusal's tuple."""
            try:
                text = resume_text(Path(path))
            except ResumeReadError as exc:
                return refused(str(exc))
            try:
                node_cfg, provider_cfg = load_node_config("ingest", config)
                adapter = build_generation_adapter(node_cfg, provider_cfg)
                with common.model_turn():  # one model call at a time, both pages
                    result = ingest_resume(text, adapter=adapter)
                workbook = load_workbook(result.rendered) if result.draft.jobs else None
            except _FAILURES:
                return refused(MODEL_FAILED_LINE)
            if workbook is None:
                return refused(NO_JOBS_LINE)
            return workbook, adapter

        def keywords_and_write(workbook, adapter):
            """The second call over every bullet, with the extracted
            Defaults; its failure is a warning and the profile is written without keywords."""
            try:
                with common.model_turn():
                    workbook, _, failure = first_keywords(workbook, adapter=adapter)
            except _FAILURES as exc:  # the adapter's own failures outside the call
                failure = str(exc)
            if failure is not None:
                gr.Warning(KEYWORDS_SKIPPED_LINE)
            try:
                working_copy.parent.mkdir(parents=True, exist_ok=True)
                save_working_copy(workbook, working_copy, sidecar=sidecar, today=date.today())
            except OSError:
                return refused(WRITE_FAILED_LINE)
            jobs = workbook.jobs
            bullets = sum(len(job.bullets) for job in jobs)
            line = RESULT_LINE.format(
                jobs=_count(len(jobs), "job"), bullets=_count(bullets, "bullet")
            )
            return (
                f"{line} {KEYWORDS_SKIPPED_LINE}" if failure is not None else line,
                gr.Column(visible=True),
                gr.Button(visible=True),
                gr.Column(visible=False),
                "",
                *profile_page.hand_off(workbook, []),
                gr.File(value=None),
            )

        landed = (skip,) * n_landed  # what the page shows if the work raises
        try:
            first = structure()
            if isinstance(first[0], Workbook):
                # Line 2 of the veil changes; the clock runs on from the upload's start.
                yield (*[skip] * n_landed, *keyword_phase(started))
                landed = keywords_and_write(*first)
            else:
                landed = first
        finally:
            yield (*landed, *upload_veil.off())

    with gr.Blocks(title="LoomCV") as page:
        # The bar: `status` is what `common.bar_text` recomposes; the Help link beside it is
        # static and never an output, so no bar update can drop or repeat it.
        with gr.Row(elem_id="loomcv-status"):
            status = gr.Markdown(elem_classes=["lc-bartext"], scale=1)
            gr.HTML(HELP_LINK, js_on_load=None, elem_classes=["lc-help"], scale=0)
        selected = gr.State(land.tab)
        for slot in slots.values():
            slot.render()
        upload_veil = common.veil()
        with gr.Tabs(selected=TAB_IDS[land.tab]) as tabs:
            with gr.Tab(app_state.PROFILE_TAB, id=TAB_IDS[app_state.PROFILE_TAB]) as profile_tab:
                profile_line = gr.Markdown(UPLOAD_GUIDANCE if land.page_over is None else land.line)
                with gr.Column(visible=land.page_over is None) as upload_block:
                    upload_file = gr.File(label="Your résumé", type="filepath", file_count="single")
                    with gr.Row():
                        read_button = common.secondary("Read this résumé", visible=False, scale=0)
                    upload_message = gr.Markdown()
                with gr.Column(visible=land.page_over is not None) as profile_col:
                    profile_page.render()  # holds Start over
            with gr.Tab(
                app_state.APPLICATIONS_TAB, id=TAB_IDS[app_state.APPLICATIONS_TAB]
            ) as applications_tab:
                gate_line = gr.Markdown(app_state.GATE_LINE)
                with gr.Column() as review_col:
                    review_page.render()

        gate_targets = [status, gate_line, review_col, *review_page.gate_targets]
        upload_file.change(lambda f: gr.Button(visible=bool(f)), upload_file, read_button)
        profile_page.saved_signal.change(  # Save → Applications
            lambda shown: (
                gr.Tabs(selected="applications"),
                *on_select(app_state.APPLICATIONS_TAB)(shown),
            ),
            inputs=[slots[app_state.APPLICATIONS_TAB]],
            outputs=[tabs, selected, *gate_targets],
        )
        profile_targets = [
            profile_line,
            profile_col,
            start_over,
            upload_block,
            upload_message,
            *profile_page.hand_off_targets,
        ]
        page.load(
            on_load,
            inputs=[slots[app_state.PROFILE_TAB], slots[app_state.APPLICATIONS_TAB]],
            outputs=[*gate_targets, *profile_targets],
        )
        read_button.click(
            read_upload,
            inputs=[upload_file],
            outputs=[*profile_targets, upload_file, *upload_veil.outputs],
            show_progress="hidden",
            **SERIAL,
        )
        for tab, name in (
            (profile_tab, app_state.PROFILE_TAB),
            (applications_tab, app_state.APPLICATIONS_TAB),
        ):
            tab.select(on_select(name), inputs=[slots[name]], outputs=[selected, *gate_targets])
            slots[name].change(
                on_slots(name),
                inputs=[selected, slots[name]],
                outputs=[status],
                show_progress="hidden",
            )

        start_over.click(lambda: gr.Row(visible=True), outputs=[start_over_group])
        start_over_no.click(lambda: gr.Row(visible=False), outputs=[start_over_group])
        start_over_yes.click(
            start_over_confirmed,
            inputs=[slots[app_state.PROFILE_TAB]],
            outputs=[*gate_targets, *profile_targets, start_over_group],
            **SERIAL,
        )

    return page


def launch(
    runtime: Runtime,
    *,
    working_copy: Path,
    saved: Path,
    sidecar: Path,
    out_root: Path,
    checkpoints: Path,
    page_limit_default: int,
    config: Path | None,
) -> None:
    """Open the frame. Blocks until the server stops. Downloads are served from the applications
    folder, as `pipeline review` does."""
    css = " ".join(
        [
            STATUS_CSS,
            profile_ui.CHIP_CSS,
            common.VEIL_CSS,
            common.LOOKS_CSS,
            common.BUTTON_CSS,
            review_ui.REVIEW_CSS,
        ]
    )
    build(
        runtime,
        working_copy=working_copy,
        saved=saved,
        sidecar=sidecar,
        out_root=out_root,
        checkpoints=checkpoints,
        page_limit_default=page_limit_default,
        config=config,
    ).launch(css=css, allowed_paths=[str(out_root.resolve())])
