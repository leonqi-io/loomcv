"""The match report, rendered deterministically from a match payload.

Pure over the payload: the verdict, one line per posting item, what is not shown, what the profile
lists without a bullet, the similar items, what is not on the résumé, and — under one "For
troubleshooting" heading — the slot account and the provenance trail. Sections 0–5 are plain
English (no term of the algorithm or the pipeline); section 6 keeps the pipeline's. Nothing is a
score — none of `max_cosine`, `max_bm25`, `top_rrf_score`, `rerank_rank`, `rrf_position`,
`overlap`, `ranked`, `config` or `diagnostics` is read, and no percentage or length figure is
printed anywhere. The only transformation of a payload string is the `|` escape inside the
section 1 table cells; job ids become job names by the lookup section 0 uses (the id when a row is
missing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loomcv.draft.verdict import (
    SKILLS_SOURCE,
    claim_sources,
    compute_verdict,
    job_name,
    line_bin,
    part_groups,
    part_line,
    render_verdict,
)
from loomcv.retrieve.rank import MatchPayload

if TYPE_CHECKING:
    from collections.abc import Sequence

    from loomcv.retrieve.search import JobRow

_ADD_BULLET = "add a bullet in My profile if you've done it"
_TROUBLESHOOTING_NOTE = (
    "How this résumé was put together — for checking a problem, not for deciding whether to apply."
)
_TYPE = {"must": "Required", "nice": "Preferred", "responsibility": "Responsibility"}
_KIND_WORD = {"must": "required", "nice": "preferred", "responsibility": "responsibility"}
_BIN_WORD = {"direct": "Clearly shows", "close": "Something similar", "none": "Doesn't show"}
_REASON = {
    "no_candidates": "nothing in your profile for it",
    "all_taken": "the bullets for it are already used for other items",
    "budget": "no room left on the page",
}


def _cell(value: str) -> str:
    return value.replace("|", "\\|")


def _section(heading: str, body: list[str]) -> str:
    return "\n".join([heading, "", *(body or ["none"])])


def _where(sources: Sequence[str], by_id: dict[str, JobRow]) -> str:
    """Where a keyword is listed: `at A; B and in your skills`, `at A`, `in your skills`."""
    jobs = [job_name(s, by_id) for s in sources if s != SKILLS_SOURCE]
    in_skills = SKILLS_SOURCE in sources
    if jobs:
        return "at " + "; ".join(jobs) + (" and in your skills" if in_skills else "")
    return "in your skills" if in_skills else ""


def _coverage(payload: MatchPayload) -> list[str]:
    if not payload.requirements:
        return []
    placed = {b.bullet_id for job in payload.selection.jobs for b in job.bullets}
    rows = [
        "| Type | What the posting asks | Your résumé | Matched skills | Bullets used |",
        "|---|---|---|---|---|",
    ]
    for line in payload.requirements:
        used = [i for i in payload.selection.covered_by.get(line.text, []) if i in placed]
        cells = [
            _TYPE[line.kind],
            line.text,
            _BIN_WORD[line_bin(line)],
            ", ".join(line.mapped_keywords) or "—",
            str(len(used)) if used else "—",
        ]
        rows.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return rows


def _gaps(payload: MatchPayload, by_id: dict[str, JobRow]) -> list[str]:
    claimed = {c.line: c for c in payload.claims}
    rows = []
    for gap in payload.gaps:
        claim = claimed.get(gap)
        gap = gap.rstrip().rstrip(".")  # the requirement's own final period, before the dash
        where = _where(claim_sources(claim), by_id) if claim is not None else ""
        if not where:
            rows.append(f"- {gap} — no bullet in your profile backs this up")
        else:
            rows.append(f"- {gap} — you list it {where}")
    # The named parts the card counts as missing, so section 2 and the
    # card agree.
    rows.extend(part_line(text, parts) for text, parts in part_groups(compute_verdict(payload)))
    return rows


def _claims(payload: MatchPayload, by_id: dict[str, JobRow]) -> list[str]:
    rows = []
    for claim in payload.claims:
        rows.append(f"- {claim.line}")
        for k in claim.keywords:
            where = _where(k.at, by_id)
            rows.append(f"  - {k.keyword} — {where}" if where else f"  - {k.keyword}")
        rows.append(f"  - {_ADD_BULLET}")
    return rows


def _similar(payload: MatchPayload) -> list[str]:
    return [
        f"- {line.text} — closest skills: {', '.join(line.mapped_keywords) or '—'}"
        for line in payload.requirements
        if line_bin(line) == "close"
    ]


def _not_on_resume(payload: MatchPayload) -> str:
    uncovered = [f"- {u.line} ({_KIND_WORD[u.kind]})" for u in payload.selection.uncovered]
    unplaced = [
        f"- {u.line} ({_KIND_WORD[u.kind]}) — {_REASON[u.reason]}"
        for u in payload.selection.unplaced
    ]
    return "\n".join(
        [
            "## 5 Not on this résumé",
            "",
            "### No bullet on this résumé covers it",
            "",
            *(uncovered or ["none"]),
            "",
            "### Left off, and why",
            "",
            *(unplaced or ["none"]),
        ]
    )


def _slots(payload: MatchPayload) -> list[str]:
    selection = payload.selection
    return [
        f"- total_slots: {selection.total_slots}",
        f"- passes: {selection.passes}",
        *(f"- {job.job}: cap {job.cap}, placed {job.placed}" for job in selection.jobs),
    ]


def _provenance(payload: MatchPayload) -> list[str]:
    placed = sorted(
        ((b, job.job) for job in payload.selection.jobs for b in job.bullets),
        key=lambda pair: pair[0].walk_position,
    )
    return [
        f"- {b.bullet_id} → {job} → {b.line} → pass {b.pass_number} → walk {b.walk_position}"
        for b, job in placed
    ]


def _troubleshooting(payload: MatchPayload) -> str:
    return "\n\n".join(
        [
            "## 6 For troubleshooting",
            _TROUBLESHOOTING_NOTE,
            _section("### Slots", _slots(payload)),
            _section("### Provenance", _provenance(payload)),
        ]
    )


def render_report(payload: MatchPayload, rows: Sequence[JobRow] = ()) -> str:
    """`match_report.md`: section 0 (the Verdict), five plain-words sections, then one
    troubleshooting section; "none" under any empty heading."""
    by_id = {row.job_id: row for row in rows}
    blocks = [
        f"# Match report — {payload.profile.title}",
        _section("## 0 Verdict", render_verdict(compute_verdict(payload, rows))),
        _section("## 1 Each item in this posting", _coverage(payload)),
        _section("## 2 Required, not shown", _gaps(payload, by_id)),
        _section(
            "## 3 Listed in your profile, but no bullet describes it", _claims(payload, by_id)
        ),
        _section("## 4 Something similar — prepare for the question", _similar(payload)),
        _not_on_resume(payload),
        _troubleshooting(payload),
    ]
    return "\n\n".join(blocks) + "\n"
