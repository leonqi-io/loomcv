"""Deterministic generation side: the tailored résumé and the match report, rendered
from a match payload with no model call and no rewriting."""

from loomcv.draft.render import (
    REPORT_NAME,
    RESUME_NAME,
    ResumeTemplateError,
    StaleMatchError,
    check_fresh,
    render_resume,
    resume_date,
)
from loomcv.draft.report import render_report
from loomcv.draft.verdict import Verdict, compute_verdict

__all__ = [
    "REPORT_NAME",
    "RESUME_NAME",
    "ResumeTemplateError",
    "StaleMatchError",
    "Verdict",
    "check_fresh",
    "compute_verdict",
    "render_report",
    "render_resume",
    "resume_date",
]
