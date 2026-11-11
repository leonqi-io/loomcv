"""What both task-shaped pages share: the veil over every long call, its phrases (eight), the
CSS of the four looks rules and the button tiers, the three button constructors and the status
bar's text. Formatting only — nothing here reads the workbook, the review document or a file;
the veil's clock is `loomcv.app_state.elapsed_text`, which pytest covers.

**The lock.** A shown veil hides the whole
`.gradio-container` through one CSS rule pair keyed on its presence (`VEIL_CSS`); a hidden
`gr.HTML` is absent from the DOM, so nothing is hidden while no veil is up. Nothing under the veil
takes a pointer or the keyboard; no class is updated, no JS runs. Every long call's event carries
`show_progress="hidden"` so no output shows Gradio's pending frame.

**The one pattern for a long call** (the veil comes off in `finally:`, never in a `.then()` —
a `.then()` does not run after the call raises):

    def handler(...):
        yield (*skips, *veil.on(PHRASE_…))
        landed = (gr.skip(),) * n          # what the page shows if the work raises
        try:
            landed = work(...)             # the call; refusals return their own tuple
        finally:
            yield (*landed, *veil.off())

    button.click(handler, inputs=…, outputs=[*targets, *veil.outputs],
                 show_progress="hidden", **SERIAL)

Each page builds its own veil inside its container (`veil()`); several on one app page are fine —
only one is ever on. No long call may start from a box's `submit` (a focused box's queued blur
would commit after the call — a known limit).

**One model call at a time.** Every handler that calls
a model — the upload's two calls, the keyword step (a bullet's suggest, Suggest again), drafting
a stronger bullet, Sort into groups, Save profile's embedding, Build — makes the call inside
`with model_turn():`, one process-wide lock. A second call (another tab, another page, a missed
entry point) waits under its own veil until the first is done; it never runs beside it. The
pages' own queues are unchanged, so an ordinary commit never waits for this lock (a single
Gradio queue for the model calls would have taken the profile page's calls off the `profile`
queue, and a commit queued just before a call could then land after it). Make the PDF calls no
model (pandoc + typst) and does not take it.

**The confirmation row.** `ask(question)` — the question left,
Cancel · Continue adjacent on the right, one line, no grey block; Continue a solid danger red.
"""

from __future__ import annotations

import html
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import gradio as gr

from loomcv.app_state import elapsed_text

# ------------------------------------------------------------------ the veil

VEIL_LINE = "One moment — LoomCV is working."  # line 1, fixed
VEIL_CLOCK = "Running for {elapsed} · how long this takes depends on your model."  # line 3, fixed

# The phrase table — line 2, one per long-call entry point of both pages.
PHRASE_UPLOAD = "Reading your résumé"  # the structure call (shell)
PHRASE_KEYWORDS = "Picking search keywords for each of your bullets"  # again / upload
# A bullet's "suggest" — that bullet only: its own phrase, so
# line 2 says what the call does (one call per entry point).
PHRASE_KEYWORDS_ONE = "Picking search keywords for this bullet"
PHRASE_IMPROVE = "Drafting a stronger version of this bullet"  # STAR drafting of one bullet
PHRASE_GROUPS = "Sorting your skills into groups"  # propose_groups
PHRASE_SAVE = "Saving your profile — this builds the search index"  # write the vault + sync
PHRASE_BUILD = "Reading the posting and picking your bullets"  # application.start
PHRASE_PDF = "Making the PDF"  # application.submit, finalize
PHRASES = (
    PHRASE_UPLOAD,
    PHRASE_KEYWORDS,
    PHRASE_KEYWORDS_ONE,
    PHRASE_IMPROVE,
    PHRASE_GROUPS,
    PHRASE_SAVE,
    PHRASE_BUILD,
    PHRASE_PDF,
)

# ------------------------------------------------------------------- the status bar

# An empty slot is not shown, nor is its " · ".
EMPTY_SLOT = ""
EMPTY_SLOTS = (EMPTY_SLOT, EMPTY_SLOT)  # a page's (line, save) before it writes them

# ----------------------------------------------------------------------------- the CSS
# The veil: viewport-wide, centred, a light sheet; the rule pair.
VEIL_CSS = (
    ".lc-veil {position: fixed; inset: 0; z-index: 1000; display: flex; "
    "align-items: center; justify-content: center; "
    "background: var(--body-background-fill);} "
    ".lc-veil-box {display: flex; flex-direction: column; align-items: center; "
    "background: var(--background-fill-primary); "
    "border: 1px solid var(--border-color-primary); border-radius: 8px;} "
    "body:has(.lc-veil) .gradio-container {visibility: hidden;} "
    ".lc-veil {visibility: visible;}"
)
# The four looks rules: ① the selected top tab as a binder tab (the page background, no line
# between it and the pane — its `::after` repaints the row's grey line — and the accent square);
# ② tabs change background on hover; ③ the selected left-list entry: a thick accent bar on its
# left edge and the page background; ④ left-list entries change background on hover. One accent.
# Every tab is bordered, the selected one bolder and larger; the selected
# entry and the current card tinted as well as barred, so their edges show; the card and
# entry rules both pages share live here, one width; a `gr.HTML` entry carries its classes on
# two nested elements, so the inner one is neutralised; the left list stays in view.
LOOKS_CSS = (
    ":root {--lc-accent: #2f6f8f; --lc-tint: rgba(47, 111, 143, 0.09);} "
    ".tab-container > button[role=tab], .lc-list .lc-entry, .lc-card "
    "{transition: background-color 120ms;} "
    # Longhands: Gradio's CSS pass drops a `border:` shorthand holding a var() when the same
    # rule then sets `border-bottom` (the 2c-0 binder tab never got its border that way).
    ".tab-container > button[role=tab] {border-style: solid solid none solid; "
    "border-width: 1px 1px 0 1px; border-color: var(--border-color-primary); "
    "border-radius: 6px 6px 0 0; margin-right: 4px;} "
    ".tab-container > button[role=tab]:hover:not(.selected) "
    "{background: var(--background-fill-secondary);} "
    ".tab-container > button[role=tab].selected {color: var(--body-text-color); "
    "background: var(--body-background-fill); font-weight: 700; font-size: var(--text-lg); "
    "border-color: var(--body-text-color-subdued);} "
    ".tab-container > button[role=tab].selected::after "
    "{background: var(--body-background-fill);} "
    '.tab-container > button[role=tab].selected::before {content: "■ "; '
    "color: var(--lc-accent);} "
    ".lc-list {position: sticky; top: 3rem; align-self: flex-start; "
    "max-height: calc(100vh - 4rem); overflow-y: auto;} "
    # Gradio's column wraps (`flex-wrap: wrap`), so a list taller than its
    # max-height put its last children in a second track behind the pane; it scrolls instead.
    ".lc-list {flex-wrap: nowrap !important;} "
    ".lc-list > * {flex-shrink: 0 !important;} "
    ".lc-list .lc-entry {border-left: 4px solid transparent;} "
    ".lc-list .lc-entry:hover {background: var(--background-fill-secondary);} "
    ".lc-list .lc-entry.on {background: var(--lc-tint); "
    "border-left: 4px solid var(--lc-accent);} "
    ".lc-list .lc-entry .lc-entry.lc-entry {border: none; background: none; padding: 0;} "
    ".lc-card {position: relative; border-left: 4px solid transparent; padding-left: 8px;} "
    ".lc-card:hover, .lc-card:focus-within {border-left-color: var(--lc-accent); "
    "background: var(--lc-tint);} "
    ".lc-card:not(:hover):not(:focus-within) .lc-act {visibility: hidden;} "
    # The page as wide as the design's frame — 960 px
    # of content (Gradio's own 32 px padding on each side), centred.
    ".gradio-container > .main {max-width: 1024px !important; margin-left: auto !important; "
    "margin-right: auto !important;}"
)
# The button tiers: primary in the page's foreground (the four theme variables — the one
# theme touch); the link tier as underlined accent text.
BUTTON_CSS = (
    ":root {--button-primary-background-fill: var(--body-text-color); "
    "--button-primary-background-fill-hover: var(--body-text-color); "
    "--button-primary-text-color: var(--body-background-fill); "
    "--button-primary-border-color: var(--body-text-color);} "
    "button.lnk.secondary, button.lnk.secondary:hover {background: none; border: none; "
    "color: var(--lc-accent); text-decoration: underline;} "
    # A link is as wide as its words, also in a column; `lc-fit`
    # does the same for a primary button.
    # Gradio gives a `scale=0` button `flex-basis: 0%`: without the old 160 px floor it would
    # collapse to its padding, so the basis is its own width.
    ".gradio-container button.lnk, .gradio-container .lc-fit {width: fit-content; "
    "flex-basis: auto !important; flex-shrink: 0; white-space: nowrap;} "
    # The confirmation row — no grey block, the question left
    # and its two buttons adjacent on the right, one size and one baseline; Continue a solid
    # danger red with white text, Cancel the secondary tier.
    ".lc-ask {flex-wrap: nowrap; align-items: center !important; gap: 8px; "
    "padding: 4px 0; background: none; border: none;} "
    ".lc-ask > .lc-askq {flex: 1 1 auto !important; min-width: 0 !important;} "
    ".lc-ask .lc-askq p {margin: 0;} "
    ".lc-ask > button {flex: 0 0 auto !important; min-width: 88px !important; "
    "align-self: center;} "
    # In the narrow left list the question stacks — the sentence on its
    # own line, Cancel · Continue under it on the right; card-level questions unchanged.
    ".lc-list .lc-ask {flex-wrap: wrap !important; justify-content: flex-end;} "
    ".lc-list .lc-ask > .lc-askq {flex: 1 0 100% !important;} "
    "button.lc-danger, button.lc-danger:hover {background: #b42318; color: #fff; "
    "border-color: #b42318;} "
    "button.lc-danger:hover {background: #912018; border-color: #912018;}"
)


def veil_html(phrase: str, seconds: float) -> str:
    """The veil's three lines in one box; the phrase escaped."""
    lines = (VEIL_LINE, phrase, VEIL_CLOCK.format(elapsed=elapsed_text(seconds)))
    body = "".join(f"<p>{html.escape(line)}</p>" for line in lines)
    return f'<div class="lc-veil-box">{body}</div>'


@dataclass(frozen=True)
class Veil:
    """One page's veil: the three lines, the one-second clock and the kept start."""

    html: gr.HTML  # visible=False, elem_classes=["lc-veil"]
    timer: gr.Timer  # gr.Timer(1, active=False)
    started: gr.State  # (phrase, time.monotonic()) while on; None while off

    @property
    def outputs(self) -> list:
        return [self.html, self.timer, self.started]

    def on(self, phrase: str) -> tuple:
        """The veil shown, the clock started, the start kept page-side."""
        return (
            gr.HTML(value=veil_html(phrase, 0), visible=True),
            gr.Timer(active=True),
            (phrase, time.monotonic()),
        )

    def off(self) -> tuple:
        """The veil hidden, the clock stopped, nothing kept."""
        return (gr.HTML(visible=False), gr.Timer(active=False), None)


def veil() -> Veil:
    """Create the veil's components in the current Blocks and bind its tick. The tick rewrites
    the value only — never `visible` — so a late tick cannot raise a veil that is off; it runs
    on Gradio's default concurrency, beside the call it times."""
    box = gr.HTML(visible=False, elem_classes=["lc-veil"])
    timer = gr.Timer(1, active=False)
    started = gr.State(None)

    def tick(kept):
        if kept is None:
            return gr.skip()
        phrase, start = kept
        return gr.HTML(value=veil_html(phrase, time.monotonic() - start))

    timer.tick(tick, inputs=[started], outputs=[box], show_progress="hidden")
    return Veil(box, timer, started)


# ---------------------------------------------------------------------- button tiers (#6)


def primary(label: str, **kwargs) -> gr.Button:
    """The one standing action of a screen: dark, large."""
    return gr.Button(label, variant="primary", size="lg", **kwargs)


def secondary(label: str, **kwargs) -> gr.Button:
    """Small and bordered: Confirm, Done, Add, Download the PDF. Beside other things
    (`scale=0`) it is as wide as its words unless a caller names a width."""
    if kwargs.get("scale") == 0:
        kwargs.setdefault("min_width", 0)
        kwargs["elem_classes"] = [*(kwargs.get("elem_classes") or []), "lc-fit"]
    return gr.Button(label, variant="secondary", size="sm", **kwargs)


def link(label: str, **kwargs) -> gr.Button:
    """Underlined accent text, no border or background: edit, undo, remove, Start over… As wide
    as its words unless a caller names a width (Gradio's default is 160 px, or the whole
    row)."""
    kwargs.setdefault("scale", 0)
    kwargs.setdefault("min_width", 0)
    extra = kwargs.pop("elem_classes", None) or []
    return gr.Button(label, size="sm", elem_classes=["lnk", *extra], **kwargs)


@dataclass(frozen=True)
class Ask:
    """A confirmation row: `row` is hidden until its question is asked."""

    row: gr.Row
    yes: gr.Button  # Continue
    no: gr.Button  # Cancel


def ask(question: str) -> Ask:
    """The question left, Cancel · Continue adjacent on the right; hidden. Show and hide it with
    `gr.Row(visible=…)` updates on `.row`."""
    with gr.Row(visible=False, elem_classes=["lc-ask"]) as row:
        gr.Markdown(question, elem_classes=["lc-askq"])
        no = secondary("Cancel", scale=0)
        yes = gr.Button(
            "Continue",
            variant="stop",
            size="sm",
            scale=0,
            min_width=0,
            elem_classes=["lc-fit", "lc-danger"],
        )
    return Ask(row, yes, no)


# --------------------------------------------------------------- one model call at a time

_MODEL_LOCK = threading.Lock()


@contextmanager
def model_turn() -> Iterator[None]:
    """Hold the process-wide model lock for the duration of one model call (or one handler's
    run of calls): a second caller waits here, under its own veil, until the first is done."""
    with _MODEL_LOCK:
        yield


def bar_text(step: str, line: str = "", save: str = "") -> str:
    """`step · page line · save state`; an empty slot is left out with its separator."""
    return " · ".join(part for part in (step, line, save) if part)
