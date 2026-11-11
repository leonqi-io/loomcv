"""The merge across requirement lines and the per-job slots.

Pure functions, no I/O, no model call. Stage 3 of the match path: it reads what R3 produced —
each line's grade, mapped keywords and reranked cut (`LineResult`), the line's whole RRF order
(`LineRanking.order`), the loaded candidates with their stored keyword lists — and decides which
bullets go on the résumé and under which job.

- A line's candidates are the carriers of its mapped keywords (`casefold()` equality on the whole
  string); the grade only decides whether there are any (`none` → none, `adjacent` like `exact`).
- Rule D orders them: overlap count (descending), the line's reranker rank,
  the RRF position, `bullet_id`. Lookups, counts and ranks only — no score, no weight, no
  threshold on any signal; `job_order` and the shares never enter an order.
- The walk is two phases. The first (pass 1): every must line takes its first candidate, with no
  cap and no budget. The second: the budget left of `total_slots` is spent under per-job caps;
  pass 2 visits responsibility then nice lines, pass 3 and later visit must → responsibility →
  nice; a line takes its first untaken candidate whose job has room. The walk ends when the
  budget is spent or a pass of 3 or later takes nothing (an empty pass 2 says nothing about the
  must lines' second bullets).
- The per-job caps divide `total_slots` by `recency_shares` (by `job_order`, the last share
  repeating) with largest-remainder rounding. A job with fewer carriers than its cap stays short.
"""

from __future__ import annotations

from loomcv.retrieve.rank import (
    Candidate,
    JobSelection,
    LineResult,
    RequirementKind,
    SelectedBullet,
    Selection,
    UncoveredLine,
    UnplacedLine,
    WalkStep,
)

_KIND_ORDER: dict[RequirementKind, int] = {"must": 0, "responsibility": 1, "nice": 2}


def _fold(keywords: list[str]) -> set[str]:
    return {k.casefold() for k in keywords}


def carriers_of(line: LineResult, candidates: list[Candidate]) -> list[Candidate]:
    """The candidates carrying at least one of the line's mapped keywords, in candidate order.

    `[]` when the line is graded `none` or maps no keyword. Whole-string, case-insensitive
    equality: `cypress` carries `Cypress`; `Test Automation Framework` does not carry
    `Test Automation`.
    """
    if line.grade == "none" or not line.mapped_keywords:
        return []
    wanted = _fold(line.mapped_keywords)
    return [c for c in candidates if wanted & _fold(c.keywords)]


def _rule_d(
    line: LineResult, candidates: list[Candidate], order: list[str]
) -> list[tuple[Candidate, int, int | None, int | None]]:
    """The carriers with their rule D inputs `(overlap, rerank_rank, rrf_position)`, best first."""
    wanted = _fold(line.mapped_keywords)
    rerank_rank = {b.bullet_id: b.rerank_rank for b in line.ranked}
    rrf_position = {bid: i for i, bid in enumerate(order, start=1)}
    rows = [
        (
            c,
            len(wanted & _fold(c.keywords)),
            rerank_rank.get(c.bullet_id),
            rrf_position.get(c.bullet_id),
        )
        for c in carriers_of(line, candidates)
    ]

    def key(row):
        c, overlap, rr, pos = row
        # a missing rank / position sorts after every present one
        return (-overlap, rr is None, rr or 0, pos is None, pos or 0, c.bullet_id)

    return sorted(rows, key=key)


def line_candidates(line: LineResult, candidates: list[Candidate], order: list[str]) -> list[str]:
    """The line's carriers in rule D order, as bullet ids."""
    return [c.bullet_id for c, *_ in _rule_d(line, candidates, order)]


def caps_for(
    jobs: list[tuple[str, int]], total_slots: int, recency_shares: list[int]
) -> dict[str, int]:
    """Per-job caps: `total_slots × share / Σ shares`, largest-remainder rounding.

    `jobs` = `(job_id, job_order)`, sorted here by `(job_order, job_id)`; job *i* gets
    `recency_shares[i]`, or the last share when *i* runs past the list. Integer arithmetic, the
    leftover going to the largest remainders (ties by job order), so `Σ caps == total_slots`.
    """
    ordered = [job_id for job_id, _order in sorted(jobs, key=lambda j: (j[1], j[0]))]
    shares = [recency_shares[min(i, len(recency_shares) - 1)] for i in range(len(ordered))]
    denominator = sum(shares)
    caps = [total_slots * s // denominator for s in shares]
    remainders = [total_slots * s % denominator for s in shares]
    by_remainder = sorted(range(len(ordered)), key=lambda i: (-remainders[i], i))
    for i in by_remainder[: total_slots - sum(caps)]:
        caps[i] += 1
    return dict(zip(ordered, caps, strict=True))


def walk(
    lines: list[LineResult],
    candidates: list[Candidate],
    orders: dict[str, list[str]],
    total_slots: int,
    caps: dict[str, int],
) -> Selection:
    """Run the two-phase walk and build the `Selection`.

    `orders` maps a line's text to its whole RRF order (`LineRanking.order`); `caps` maps every
    job id of `candidates` to its cap (`caps_for`).
    """
    lines = sorted(
        lines, key=lambda ln: _KIND_ORDER[ln.kind]
    )  # stable: requirements order inside a kind
    rows = [_rule_d(ln, candidates, orders.get(ln.text, [])) for ln in lines]
    job_of = {c.bullet_id: c.job_id for c in candidates}
    placed_in: dict[str, int] = {c.job_id: 0 for c in candidates}
    taken: set[str] = set()
    steps: list[tuple[int, int, tuple[Candidate, int, int | None, int | None]]] = []

    def place(line_index: int, pass_number: int, row) -> None:
        taken.add(row[0].bullet_id)
        placed_in[row[0].job_id] += 1
        steps.append((line_index, pass_number, row))

    # First phase: every must line takes its first untaken candidate. No cap, no budget.
    for i, ln in enumerate(lines):
        if ln.kind == "must":
            row = next((r for r in rows[i] if r[0].bullet_id not in taken), None)
            if row is not None:
                place(i, 1, row)

    # Second phase: the rest of the budget, under the per-job caps.
    pass_number = 2
    while len(steps) < total_slots:
        kinds = (
            ("responsibility", "nice") if pass_number == 2 else ("must", "responsibility", "nice")
        )
        visit = [i for kind in kinds for i, ln in enumerate(lines) if ln.kind == kind]
        before = len(steps)
        for i in visit:
            if len(steps) >= total_slots:
                break
            row = next(
                (
                    r
                    for r in rows[i]
                    if r[0].bullet_id not in taken and placed_in[r[0].job_id] < caps[r[0].job_id]
                ),
                None,
            )
            if row is not None:
                place(i, pass_number, row)
        if len(steps) == before and pass_number >= 3:
            break
        pass_number += 1

    return _selection(lines, candidates, rows, steps, taken, job_of, total_slots, caps)


def _selection(lines, candidates, rows, steps, taken, job_of, total_slots, caps) -> Selection:
    graded = [ln for ln in lines if ln.grade != "none"]
    folded = {c.bullet_id: _fold(c.keywords) for c in candidates}
    mapped = {ln.text: _fold(ln.mapped_keywords) for ln in graded}

    def covers(bullet_id: str) -> list[str]:
        return [ln.text for ln in graded if mapped[ln.text] & folded[bullet_id]]

    walk_steps: list[WalkStep] = []
    placed_bullets: list[SelectedBullet] = []
    for position, (line_index, pass_number, (cand, overlap, rr, pos)) in enumerate(steps, start=1):
        ln = lines[line_index]
        walk_steps.append(
            WalkStep(
                bullet_id=cand.bullet_id,
                job_id=cand.job_id,
                line=ln.text,
                kind=ln.kind,
                pass_number=pass_number,
            )
        )
        placed_bullets.append(
            SelectedBullet(
                bullet_id=cand.bullet_id,
                project=cand.project,
                text=cand.text,
                line=ln.text,
                kind=ln.kind,
                pass_number=pass_number,
                walk_position=position,
                overlap=overlap,
                rerank_rank=rr,
                rrf_position=pos,
                covers=covers(cand.bullet_id),
            )
        )

    jobs: dict[str, JobSelection] = {}
    for c in sorted(candidates, key=lambda c: (c.job_order, c.job_id)):
        jobs.setdefault(
            c.job_id,
            JobSelection(
                job_id=c.job_id,
                job=c.job,
                job_order=c.job_order,
                cap=caps[c.job_id],
                placed=0,
                bullets=[],
            ),
        )
    for bullet in placed_bullets:
        job = jobs[job_of[bullet.bullet_id]]
        job.bullets.append(bullet)
        job.placed += 1

    placed_lines = {w.line for w in walk_steps}
    index_of = {ln.text: i for i, ln in enumerate(lines)}
    unplaced: list[UnplacedLine] = []
    for ln in graded:
        if ln.text in placed_lines:
            continue
        carriers = [r[0].bullet_id for r in rows[index_of[ln.text]]]
        if not carriers:
            reason = "no_candidates"
        elif all(bid in taken for bid in carriers):
            reason = "all_taken"
        else:
            reason = "budget"
        unplaced.append(UnplacedLine(line=ln.text, kind=ln.kind, reason=reason))

    covered_by = {
        ln.text: [b.bullet_id for b in placed_bullets if ln.text in b.covers] for ln in graded
    }
    uncovered = [
        UncoveredLine(line=ln.text, kind=ln.kind) for ln in graded if not covered_by[ln.text]
    ]
    return Selection(
        total_slots=total_slots,
        jobs=list(jobs.values()),
        walk=walk_steps,
        unplaced=unplaced,
        covered_by=covered_by,
        uncovered=uncovered,
        passes=len({w.pass_number for w in walk_steps}),
    )
