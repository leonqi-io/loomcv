"""The orchestration graph: match → draft → [interrupt before human_review] → finalize.

LangGraph does orchestration, state, HITL and checkpointing here and nothing else:
every model call is one the step functions already make through `loomcv/llm.py`, the
adapters reach the nodes through the `Runtime` the node closures capture, and no `langchain*`
module is imported by LoomCV (`langchain-core` arrives with `langgraph` and stays unused).

The review is a file the user edits outside the process, so the breakpoint is the static
`interrupt_before=["human_review"]` on the node, not the dynamic interrupt call. `submit` (in
`loomcv/application.py`) writes `review_confirmed` + `reviewed_path` into the state and resumes;
`human_review` refuses to run without both, and reads no file — the edited résumé is accepted as
given: nothing here compares it with the draft, the payload or the index.

`PipelineState` holds paths, counts and flags — never résumé text, a bullet, the JD or the payload:
the files in the application directory are the truth, so a checkpoint is readable
with sqlite3 + json and no LoomCV class sits on the serialiser's path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from loomcv.retrieve.rank import MatchPayload
from loomcv.steps import Runtime, draft_step, finalize_step, match_step

JD_NAME = "jd.md"
PROFILE_NAME, MATCH_NAME = "profile.json", "match.json"
REVIEWED_NAME = "reviewed_resume.md"
NODES = ("match", "draft", "human_review", "finalize")


class PipelineState(TypedDict, total=False):
    """JSON-plain values only."""

    app_id: str
    app_dir: str
    jd_path: str
    profile_path: str
    match_path: str
    resume_path: str
    report_path: str
    counts: dict[str, int]  # jobs / bullets / skills
    review_confirmed: bool
    reviewed_path: str | None
    final_md: str | None
    final_typ: str | None
    final_pdf: str | None
    pages: int | None
    page_limit: (
        int | None
    )  # set by submit (None = generation.yaml's); the node writes the resolved value
    over_limit: bool | None
    typst_warnings: list[str]


class ReviewNotConfirmed(RuntimeError):
    """`human_review` ran without a confirmed file: the resume did not come through `submit`."""


def thread_config(app_id: str) -> dict[str, Any]:
    """The thread of an application is its id."""
    return {"configurable": {"thread_id": app_id}}


def build_graph(runtime: Runtime, checkpointer: BaseCheckpointSaver):
    """Compile the four-node line over `checkpointer`; the nodes close over `runtime`."""

    def match_node(state: PipelineState) -> dict[str, Any]:
        app_dir = Path(state["app_dir"])
        jd_text = Path(state["jd_path"]).read_text(encoding="utf-8")
        step = match_step(runtime, jd_text=jd_text, save_profile=app_dir / PROFILE_NAME)
        match_path = app_dir / MATCH_NAME
        match_path.write_text(
            json.dumps(step.payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return {"profile_path": str(app_dir / PROFILE_NAME), "match_path": str(match_path)}

    def draft_node(state: PipelineState) -> dict[str, Any]:
        payload = MatchPayload.model_validate_json(
            Path(state["match_path"]).read_text(encoding="utf-8")
        )
        # The header is the vault's, read by `draft_step` from the index.
        drafted = draft_step(runtime, payload, out_dir=Path(state["app_dir"]), force=False)
        return {
            "resume_path": str(drafted.resume_path),
            "report_path": str(drafted.report_path),
            "counts": {
                "jobs": drafted.jobs,
                "bullets": drafted.bullets,
                "skills": drafted.skills,
            },
        }

    def human_review_node(state: PipelineState) -> dict[str, Any]:
        if not (state.get("review_confirmed") and state.get("reviewed_path")):
            raise ReviewNotConfirmed(
                "the review was not confirmed: resume through `submit`, not a bare invoke"
            )
        return {"review_confirmed": True}

    def finalize_node(state: PipelineState) -> dict[str, Any]:
        # The final_* files in an application directory are the graph's own: a re-run replaces
        # them; the "refuse existing" rule of `pipeline finalize` guards files outside.
        finalized = finalize_step(
            runtime,
            Path(state["reviewed_path"]),
            out_dir=Path(state["app_dir"]),
            page_limit=state.get("page_limit"),
            force=True,
        )
        return {
            "final_md": str(finalized.md),
            "final_typ": str(finalized.typ),
            "final_pdf": str(finalized.pdf),
            "pages": finalized.pages,
            "page_limit": finalized.page_limit,
            "over_limit": finalized.pages > finalized.page_limit,
            "typst_warnings": finalized.warnings,
        }

    graph = StateGraph(PipelineState)
    graph.add_node("match", match_node)
    graph.add_node("draft", draft_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "match")
    graph.add_edge("match", "draft")
    graph.add_edge("draft", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
