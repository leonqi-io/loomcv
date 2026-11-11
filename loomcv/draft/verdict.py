"""Section 0 of `match_report.md`, the Verdict: a count of the must
lines in three bins, mapped to a letter on fixed thresholds.

Pure, zero calls. Reads `requirements[].{text, kind, grade, parts}`, `claims[].{line,
keywords[].at}` and `JobRow.{job_id, name, employer, title}` and nothing else — no `selection`,
`gaps`, `diagnostics`, `config`, `profile.must_have` and none of the diagnostic floats.

Each must line falls in exactly one bin: **direct** (graded `exact`, and no named part below
`exact`), **not covered** (graded `none`) or **close** (everything else: graded `adjacent`, or a
compound line with a part that is only close or missing). `points = 2 * direct + close` out of
`2 * M`; the letter is a comparison of integers against `THRESHOLDS`. No percentage is printed
anywhere. A part graded `none`
on a line that is otherwise covered is listed among the not-covered items by its own name.

Every sentence the user reads here is plain English: "clearly shows /
something similar / doesn't show"; how the grade is computed lives in this docstring and in code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from loomcv.retrieve.rank import ClaimLine, LineResult, MatchPayload

if TYPE_CHECKING:
    from collections.abc import Sequence

    from loomcv.retrieve.search import JobRow

# A, B, C on `points / (2 * M)`, in percent; below the last → D. Fitted on five postings —
# provisional; module constants, never config.
THRESHOLDS: tuple[int, int, int] = (90, 80, 50)
_LETTERS = ("A", "B", "C")

Bin = Literal["direct", "close", "none"]

LETTER_SENTENCES: dict[str, str] = {
    "A": "Strong match — your résumé clearly shows nearly everything this posting requires.",
    "B": "Mostly qualified — most required items are clearly shown; be ready to talk about the "
    "ones where you only have something similar.",
    "C": "Partial match — several required items are only something similar or not shown at all; "
    "a reviewer will notice them.",
    "D": "Large gap — many required items are not shown in your saved experience.",
}
NO_GRADE_HEADLINE = (
    "**?? — no grade**: we couldn't find a required item in this posting, so there's nothing "
    "to grade against. Your résumé was still built from your profile and this posting — judge "
    "the fit yourself."
)
NOT_COVERED_HEAD = "Required, but not shown on your résumé:"
COUNTS_TAIL = (
    "Only the required items count toward the rating at the top; these are here for your "
    "information."
)
LEGEND = (
    'How to read this: "Required" = what the posting says you must have. "Clearly shows" = a '
    'bullet on this résumé names that exact thing. "Something similar" = your saved experience '
    "has related work, but not the exact thing the posting names. When a required item names "
    'several things ("A and B"), each one is checked; any one your résumé doesn\'t show is '
    "listed on its own. \"You list it, but no bullet shows it\" = it's in your skills or a job's "
    "tools, but no bullet describes it — add one in My profile if you've done it; the next "
    "résumé you build will use it."
)

SKILLS_SOURCE = "skills"


class NotCovered(BaseModel):
    text: str  # the requirement line, verbatim
    part: str | None = None  # set: a named part graded none on a line that is not `none`
    listed_under: list[str] = []  # job names (employer, title), in claim-source order
    in_skills: bool = False  # "skills" among the claim's sources; a part entry never carries one


class Verdict(BaseModel):
    must_total: int  # M
    direct: int
    close: int
    close_lines: list[str]  # the must lines in the close bin, payload order
    not_covered: list[NotCovered]  # payload order; a line's part entries in parts order
    nice_covered: int
    nice_total: int
    resp_covered: int
    resp_total: int
    letter: Literal["A", "B", "C", "D"] | None  # None when M == 0

    @property
    def uncovered(self) -> int:
        """The must lines graded `none` (whole lines)."""
        return sum(1 for e in self.not_covered if e.part is None)

    @property
    def missing_parts(self) -> int:
        """The named parts with no evidence on a line that is otherwise covered."""
        return sum(1 for e in self.not_covered if e.part is not None)

    @property
    def points(self) -> int:
        return 2 * self.direct + self.close


def line_bin(line: LineResult) -> Bin:
    """The bin of one line: `none` graded none; `direct` graded exact with every part exact (or
    no parts); `close` for the rest. The same rule decides section 0, the card and section 4."""
    if line.grade == "none":
        return "none"
    if line.grade == "exact" and all(p.match == "exact" for p in line.parts):
        return "direct"
    return "close"


def claim_sources(claim: ClaimLine) -> list[str]:
    """The union of `keyword.at` in order: job ids, then `"skills"` where a keyword says so."""
    sources: list[str] = []
    for keyword in claim.keywords:
        sources.extend(s for s in keyword.at if s not in sources)
    return sources


def job_name(job_id: str, by_id: dict[str, JobRow]) -> str:
    row = by_id.get(job_id)
    if row is None:
        return job_id
    parts = [p for p in (row.employer, row.title) if p is not None]
    return ", ".join(parts) if parts else row.name


def compute_verdict(payload: MatchPayload, rows: Sequence[JobRow] = ()) -> Verdict:
    by_id = {row.job_id: row for row in rows}
    claimed = {c.line: c for c in payload.claims}
    direct = close = 0
    close_lines: list[str] = []
    not_covered: list[NotCovered] = []
    whole_lines = 0
    covered = {"nice": 0, "responsibility": 0}
    total = {"nice": 0, "responsibility": 0}
    for line in payload.requirements:
        if line.kind == "must":
            bin_ = line_bin(line)
            if bin_ == "direct":
                direct += 1
            elif bin_ == "close":
                close += 1
                close_lines.append(line.text)
                not_covered.extend(
                    NotCovered(text=line.text, part=p.term) for p in line.parts if p.match == "none"
                )
            else:
                whole_lines += 1
                claim = claimed.get(line.text)
                sources = claim_sources(claim) if claim is not None else []
                not_covered.append(
                    NotCovered(
                        text=line.text,
                        listed_under=[job_name(s, by_id) for s in sources if s != SKILLS_SOURCE],
                        in_skills=SKILLS_SOURCE in sources,
                    )
                )
        else:
            total[line.kind] += 1
            covered[line.kind] += line.grade in ("exact", "adjacent")
    must_total = direct + close + whole_lines
    letter: Literal["A", "B", "C", "D"] | None = None
    if must_total:  # M == 0 is not a grade; this branch comes before any comparison
        points = 2 * direct + close
        letter = "D"
        for candidate, threshold in zip(_LETTERS, THRESHOLDS, strict=True):
            if points * 100 >= threshold * 2 * must_total:
                letter = candidate
                break
    return Verdict(
        must_total=must_total,
        direct=direct,
        close=close,
        close_lines=close_lines,
        not_covered=not_covered,
        nice_covered=covered["nice"],
        nice_total=total["nice"],
        resp_covered=covered["responsibility"],
        resp_total=total["responsibility"],
        letter=letter,
    )


# A requirement quoted inside a sentence is cut to
# CUT characters at a word boundary, its final period dropped; parts of one line are named together.
CUT = 70


def short(text: str) -> str:
    """A requirement as quoted inside a sentence: whitespace collapsed, the final period dropped,
    cut to `CUT` characters at a word boundary with "…"."""
    text = " ".join(text.split()).rstrip(".")
    if len(text) <= CUT:
        return text
    head = text[:CUT]
    if " " in head:
        head = head[: head.rindex(" ")]
    return head.rstrip(" ,;:") + "…"


def name_list(parts: list[str]) -> str:
    """Names joined for a sentence: `A`, `A and B`, `A, B and C`."""
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def missing_sentence(n: int, where: str) -> str:
    """The headline's last sentence: `Missing from <where> entirely: N thing(s), listed below.`"""
    return f"Missing from {where} entirely: {n} {'thing' if n == 1 else 'things'}, listed below."


def part_groups(verdict: Verdict) -> list[tuple[str, list[str]]]:
    """The missing parts grouped by their requirement line, in `not_covered` order."""
    groups: list[tuple[str, list[str]]] = []
    for entry in verdict.not_covered:
        if entry.part is None:
            continue
        if groups and groups[-1][0] == entry.text:
            groups[-1][1].append(entry.part)
        else:
            groups.append((entry.text, [entry.part]))
    return groups


def _headline(verdict: Verdict) -> str:
    items = "1 required item" if verdict.must_total == 1 else f"{verdict.must_total} required items"
    text = (
        f"**{verdict.letter}** — this posting lists {items}. Your résumé clearly shows "
        f"{verdict.direct} and shows something similar for {verdict.close}."
    )
    missing = verdict.uncovered + verdict.missing_parts
    if missing:
        text += " " + missing_sentence(missing, "your résumé")
    return text


def part_line(text: str, parts: list[str]) -> str:
    """One report entry for the missing parts of one requirement line."""
    it = "it" if len(parts) == 1 else "them"
    return f'- {name_list(parts)} — the posting asks for {it} here: "{short(text)}"'


def not_covered_lines(verdict: Verdict) -> list[str]:
    """Section 0's not-covered entries in payload order: one per whole line, one per requirement
    line whose named parts are missing (its parts together)."""
    groups = dict(part_groups(verdict))
    lines: list[str] = []
    seen: set[str] = set()
    for entry in verdict.not_covered:
        if entry.part is None:
            lines.append(_not_covered_line(entry))
        elif entry.text not in seen:
            seen.add(entry.text)
            lines.append(part_line(entry.text, groups[entry.text]))
    return lines


def _not_covered_line(entry: NotCovered) -> str:
    where = ""
    if entry.listed_under:
        where = "listed under " + "; ".join(entry.listed_under)
        if entry.in_skills:
            where += " and in your Skills"
    elif entry.in_skills:
        where = "listed in your Skills"
    suffix = f" — {where}, but no bullet shows it" if where else ""
    return f"- {entry.text}{suffix}"


def _count(label: str, covered: int, total: int) -> str:
    if total == 0:
        return f"{label}: none read from this posting."
    return f"{label}: {covered} of {total} shown."


def render_verdict(verdict: Verdict) -> list[str]:
    """The section-0 body lines; blocks are separated by one empty line."""
    counts = " ".join(
        [
            _count("Preferred items", verdict.nice_covered, verdict.nice_total),
            _count("Responsibilities", verdict.resp_covered, verdict.resp_total),
            COUNTS_TAIL,
        ]
    )
    if verdict.letter is None:
        return [NO_GRADE_HEADLINE, "", counts, "", LEGEND]
    if verdict.not_covered:
        not_covered = [NOT_COVERED_HEAD, *not_covered_lines(verdict)]
    else:
        not_covered = [f"{NOT_COVERED_HEAD} none"]
    return [
        _headline(verdict),
        "",
        LETTER_SENTENCES[verdict.letter],
        "",
        *not_covered,
        "",
        counts,
        "",
        LEGEND,
    ]
