"""Tests for `loomcv/retrieve/merge.py` (the merge across lines and the per-job slots).

Pure tests: hand-built `Candidate` / `LineResult` objects and explicit RRF orders, no I/O, no
model, no fixture file.

The mechanism under test: a line's candidates are the
carriers of its mapped keywords (the grade only decides whether there are any), ordered by rule D
(overlap count, reranker rank, RRF position, `bullet_id`); phase 1 gives every must line its first
candidate with no cap; phase 2 fills `total_slots` under per-job caps derived from
`recency_shares` (pass 2: responsibilities then nice; pass 3 and later: must, responsibilities,
nice). No score, weight or threshold anywhere.

`loomcv.retrieve.merge` and the `rank` shapes are imported inside the helpers, so a missing
symbol fails each test at red on its own instead of aborting collection.
"""

from __future__ import annotations

import itertools


def _m():
    import loomcv.retrieve.merge as merge_module

    return merge_module


def _rank():
    import loomcv.retrieve.rank as rank_module

    return rank_module


def cand(bid, keywords, job_id="A", job_order=0):
    return _rank().Candidate(
        bullet_id=bid,
        text=f"text of {bid}",
        job_id=job_id,
        job=f"Job {job_id}",
        project=None,
        keywords=list(keywords),
        job_order=job_order,
    )


def line(text, kind, grade, mapped, ranked_ids=()):
    """A `LineResult` whose `ranked` carries `rerank_rank` = 1-based position in `ranked_ids`."""
    rank = _rank()
    ranked = [
        rank.RankedBullet(
            bullet_id=bid,
            job="Job",
            project=None,
            text=f"text of {bid}",
            rrf_rank=i,
            rerank_score=1.0 / i,
            rerank_rank=i,
        )
        for i, bid in enumerate(ranked_ids, start=1)
    ]
    return rank.LineResult(
        text=text,
        kind=kind,
        grade=grade,
        mapped_keywords=list(mapped),
        ranked=ranked,
        max_cosine=None,
        max_bm25=None,
        top_rrf_score=None,
    )


def jobs_of(cands):
    return sorted({(c.job_id, c.job_order) for c in cands}, key=lambda j: (j[1], j[0]))


def run_walk(lines, cands, total, caps, orders=None):
    """`walk()` with every candidate in every line's RRF order unless `orders` says otherwise."""
    if orders is None:
        orders = {ln.text: [c.bullet_id for c in cands] for ln in lines}
    return _m().walk(lines, cands, orders, total, caps)


def ids(candidates):
    return [c.bullet_id for c in candidates]


def walk_lines(sel):
    return [w.line for w in sel.walk]


def walk_ids(sel):
    return [w.bullet_id for w in sel.walk]


def walk_passes(sel):
    return [w.pass_number for w in sel.walk]


# ----------------------------------------------------------- 1. carriers


def test_carriers_casefold_whole_string():
    cands = [
        cand("a", ["cypress"]),
        cand("b", ["Test Automation Framework"]),
        cand("c", ["Other"]),
    ]
    ln = line("browser testing", "must", "exact", ["Cypress", "Test Automation"])

    assert ids(_m().carriers_of(ln, cands)) == ["a"]


def test_carriers_none_grade_has_none():
    cands = [cand("a", ["Cypress"])]
    ln = line("browser testing", "must", "none", ["Cypress"])

    assert _m().carriers_of(ln, cands) == []


def test_carriers_adjacent_like_exact():
    cands = [cand("a", ["Cypress"]), cand("b", ["Docker"]), cand("c", ["cypress", "Docker"])]
    exact = line("x", "must", "exact", ["Cypress", "Docker"])
    adjacent = line("x", "must", "adjacent", ["Cypress", "Docker"])

    assert ids(_m().carriers_of(adjacent, cands)) == ids(_m().carriers_of(exact, cands))
    assert ids(_m().carriers_of(adjacent, cands)) == ["a", "b", "c"]


def test_carriers_empty_mapping_exact_has_none():
    cands = [cand("a", ["Cypress"])]
    ln = line("x", "must", "exact", [])

    assert _m().carriers_of(ln, cands) == []


# ----------------------------------------------------------- 2. rule D


def test_rule_d_overlap_then_rerank_then_rrf():
    # Overlaps 2 / 1 / 1 / 1; the count-1 ones are (rerank 2), (rerank 1), (out of cut, RRF 15).
    # The ids sort the other way round, so `bullet_id` cannot be what orders them.
    cands = [
        cand("ww3", ["K1"]),
        cand("xx2", ["K1"]),
        cand("yy1", ["K1"]),
        cand("zz2", ["K1", "K2"]),
    ]
    ln = line("x", "must", "exact", ["K1", "K2"], ranked_ids=("yy1", "xx2"))
    order = ["zz2", "yy1", "xx2", *[f"f{i}" for i in range(11)], "ww3"]
    assert order.index("ww3") + 1 == 15

    assert _m().line_candidates(ln, cands, order) == ["zz2", "yy1", "xx2", "ww3"]


def test_rule_d_out_of_cut_by_rrf_position():
    cands = [cand("a1", ["K"]), cand("a2", ["K"])]
    ln = line("x", "must", "exact", ["K"])

    assert _m().line_candidates(ln, cands, ["a2", "a1"]) == ["a2", "a1"]


def test_rule_d_rrf_zero_after_positioned():
    # `a1` is absent from the order (RRF 0): it sorts after `b1` although its id is smaller.
    cands = [cand("a1", ["K"]), cand("b1", ["K"])]
    ln = line("x", "must", "exact", ["K"])

    assert _m().line_candidates(ln, cands, ["b1"]) == ["b1", "a1"]


def test_rule_d_bullet_id_last():
    cands = [cand("b", ["K"]), cand("a", ["K"])]
    ln = line("x", "must", "exact", ["K"])

    assert _m().line_candidates(ln, cands, []) == ["a", "b"]


def test_rule_d_ignores_job_order():
    ln = line("x", "must", "exact", ["K"], ranked_ids=("c",))
    order = ["b", "a", "c"]
    seen = set()
    for perm in itertools.permutations([0, 1, 2]):
        cands = [
            cand("a", ["K"], job_id="J0", job_order=perm[0]),
            cand("b", ["K"], job_id="J1", job_order=perm[1]),
            cand("c", ["K"], job_id="J2", job_order=perm[2]),
        ]
        seen.add(tuple(_m().line_candidates(ln, cands, order)))

    assert seen == {("c", "b", "a")}


# ----------------------------------------------------------- 3. phase 1


def test_phase1_ignores_caps():
    cands = [cand("c1", ["K"]), cand("c2", ["K"]), cand("c3", ["K"])]
    lines = [line(f"m{i}", "must", "exact", ["K"]) for i in (1, 2, 3)]

    sel = run_walk(lines, cands, 1, {"A": 1})

    assert walk_ids(sel) == ["c1", "c2", "c3"]
    assert walk_lines(sel) == ["m1", "m2", "m3"]
    assert sel.jobs[0].cap == 1
    assert sel.jobs[0].placed == 3
    assert sel.passes == 1
    assert walk_passes(sel) == [1, 1, 1]
    assert sel.unplaced == []


def test_phase1_all_taken_must_unplaced():
    cands = [cand("c1", ["K"])]
    lines = [line("m1", "must", "exact", ["K"]), line("m2", "must", "exact", ["K"])]

    sel = run_walk(lines, cands, 5, {"A": 5})

    assert walk_ids(sel) == ["c1"]
    assert [(u.line, u.kind, u.reason) for u in sel.unplaced] == [("m2", "must", "all_taken")]


def test_none_must_absent_from_walk_and_unplaced():
    cands = [cand("c1", ["K"])]
    lines = [line("m1", "must", "none", ["K"]), line("m2", "must", "exact", ["K"])]

    sel = run_walk(lines, cands, 5, {"A": 5})

    assert walk_lines(sel) == ["m2"]
    assert sel.unplaced == []
    assert "m1" not in sel.covered_by
    assert sel.uncovered == []


# ----------------------------------------------------------- 4. phase 2


def _six_line_data():
    # m1 has two carriers; every other line has one. Roomy caps.
    cands = [
        cand("a1", ["Km1"]),
        cand("a1b", ["Km1"]),
        cand("b", ["Km2"]),
        cand("c", ["Kr1"]),
        cand("d", ["Kr2"]),
        cand("e", ["Kn1"]),
    ]
    lines = [
        line("m1", "must", "exact", ["Km1"]),
        line("m2", "must", "exact", ["Km2"]),
        line("r1", "responsibility", "exact", ["Kr1"]),
        line("r2", "responsibility", "adjacent", ["Kr2"]),
        line("n1", "nice", "exact", ["Kn1"]),
    ]
    return cands, lines


def test_walk_order_and_passes_total_6():
    cands, lines = _six_line_data()

    sel = run_walk(lines, cands, 6, {"A": 10})

    assert walk_lines(sel) == ["m1", "m2", "r1", "r2", "n1", "m1"]
    assert walk_passes(sel) == [1, 1, 2, 2, 2, 3]
    assert walk_ids(sel) == ["a1", "b", "c", "d", "e", "a1b"]
    assert sel.passes == 3


def test_walk_stops_at_budget_total_5():
    cands, lines = _six_line_data()

    sel = run_walk(lines, cands, 5, {"A": 10})

    assert walk_lines(sel) == ["m1", "m2", "r1", "r2", "n1"]
    assert sel.passes == 2


def test_fill_skips_full_job():
    # Job A (cap 1) is full after m1's phase-1 pick; r1's first carrier is in A, so it takes
    # its next carrier, in B.
    cands = [
        cand("cA1", ["K1"], job_id="A", job_order=0),
        cand("cA2", ["K2"], job_id="A", job_order=0),
        cand("cB2", ["K2"], job_id="B", job_order=1),
    ]
    lines = [line("m1", "must", "exact", ["K1"]), line("r1", "responsibility", "exact", ["K2"])]

    sel = run_walk(lines, cands, 5, {"A": 1, "B": 5})

    assert walk_ids(sel) == ["cA1", "cB2"]
    assert [(w.line, w.job_id, w.pass_number) for w in sel.walk] == [
        ("m1", "A", 1),
        ("r1", "B", 2),
    ]
    assert sel.unplaced == []


def test_unplaced_budget_when_only_full_jobs():
    cands = [
        cand("cA1", ["K1"], job_id="A", job_order=0),
        cand("cA3", ["K3"], job_id="A", job_order=0),
        cand("cB", ["Z"], job_id="B", job_order=1),
    ]
    lines = [line("m1", "must", "exact", ["K1"]), line("r1", "responsibility", "exact", ["K3"])]

    sel = run_walk(lines, cands, 5, {"A": 1, "B": 5})

    assert walk_ids(sel) == ["cA1"]
    assert [(u.line, u.kind, u.reason) for u in sel.unplaced] == [
        ("r1", "responsibility", "budget")
    ]


def test_passes_counts_only_passes_that_placed():
    cands, lines = _six_line_data()
    sel = run_walk(lines, cands, 6, {"A": 10})
    assert sel.passes == len(set(walk_passes(sel))) == 3

    # only must lines: pass 2 places nothing, so the pass values skip 2
    must_cands = [cand("a1", ["K1"]), cand("a2", ["K1"]), cand("b1", ["K2"]), cand("b2", ["K2"])]
    must_lines = [line("m1", "must", "exact", ["K1"]), line("m2", "must", "exact", ["K2"])]
    sel = run_walk(must_lines, must_cands, 4, {"A": 10})
    assert walk_passes(sel) == [1, 1, 3, 3]
    assert sel.passes == len(set(walk_passes(sel))) == 2


def test_empty_pass_2_does_not_end_walk():
    # Pass 2 visits only responsibility / nice lines, so an empty pass 2 says nothing
    # about the must lines' second bullets.
    cands = [cand("a1", ["K1"]), cand("a2", ["K1"]), cand("b1", ["K2"]), cand("b2", ["K2"])]
    lines = [line("m1", "must", "exact", ["K1"]), line("m2", "must", "exact", ["K2"])]

    sel = run_walk(lines, cands, 4, {"A": 10})

    assert walk_ids(sel) == ["a1", "b1", "a2", "b2"]
    assert walk_lines(sel) == ["m1", "m2", "m1", "m2"]
    assert walk_passes(sel) == [1, 1, 3, 3]
    assert sel.passes == 2


# ----------------------------------------------------------- 5. caps


def test_caps_three_jobs_14_321():
    jobs = [("j0", 0), ("j1", 1), ("j2", 2)]

    assert _m().caps_for(jobs, 14, [3, 2, 1]) == {"j0": 7, "j1": 5, "j2": 2}
    # the caller's order does not matter: jobs sort by (job_order, job_id)
    assert _m().caps_for(list(reversed(jobs)), 14, [3, 2, 1]) == {"j0": 7, "j1": 5, "j2": 2}


def test_caps_eight_jobs():
    jobs = [(f"j{i}", i) for i in range(8)]

    caps = _m().caps_for(jobs, 14, [3, 2, 1])

    assert [caps[f"j{i}"] for i in range(8)] == [4, 3, 2, 1, 1, 1, 1, 1]


def test_caps_one_job():
    assert _m().caps_for([("j0", 0)], 14, [3, 2, 1]) == {"j0": 14}


def test_caps_two_jobs_32():
    caps = _m().caps_for([("j0", 0), ("j1", 1)], 14, [3, 2, 1])

    assert caps == {"j0": 8, "j1": 6}


def test_caps_sum_equals_total():
    for total in range(1, 21):
        for n_jobs in range(1, 9):
            for shares in ([3, 2, 1], [1], [2, 1, 1]):
                jobs = [(f"j{i}", i) for i in range(n_jobs)]
                caps = _m().caps_for(jobs, total, shares)

                assert set(caps) == {j for j, _ in jobs}
                assert sum(caps.values()) == total
                assert all(v >= 0 for v in caps.values())


def test_zero_carrier_job_present_placed_0():
    cands = [
        cand("a1", ["K1"], job_id="A", job_order=0),
        cand("z1", ["Unmapped"], job_id="Z", job_order=1),
    ]
    lines = [line("m1", "must", "exact", ["K1"])]

    sel = run_walk(lines, cands, 14, {"A": 8, "Z": 6})

    assert [(j.job_id, j.cap, j.placed, j.bullets) for j in sel.jobs][1] == ("Z", 6, 0, [])
    assert [j.job_id for j in sel.jobs] == ["A", "Z"]


# ----------------------------------------------------------- 6. grouping and coverage


def _grouping_scenario():
    cands = [
        cand("a1", ["K1"], job_id="A", job_order=0),
        cand("a2", ["K2"], job_id="A", job_order=0),
        cand("b1", ["K1"], job_id="B", job_order=1),
        cand("c1", ["Unmapped"], job_id="C", job_order=2),
    ]
    lines = [
        line("m1", "must", "exact", ["K1"], ranked_ids=("b1",)),
        line("r1", "responsibility", "adjacent", ["K2"]),
    ]
    orders = {"m1": ["a1", "b1"], "r1": []}
    return cands, lines, orders


def test_selection_grouping_and_bullet_fields():
    cands, lines, orders = _grouping_scenario()

    sel = run_walk(lines, cands, 3, {"A": 5, "B": 5, "C": 5}, orders)

    assert [(j.job_id, j.job, j.job_order, j.cap, j.placed) for j in sel.jobs] == [
        ("A", "Job A", 0, 5, 2),
        ("B", "Job B", 1, 5, 1),
        ("C", "Job C", 2, 5, 0),
    ]
    assert sel.total_slots == 3
    # m1's carriers a1 / b1 tie on overlap; b1 is in the reranker cut, so it goes first
    assert walk_ids(sel) == ["b1", "a2", "a1"]
    assert walk_lines(sel) == ["m1", "r1", "m1"]
    assert walk_passes(sel) == [1, 2, 3]
    assert [b.bullet_id for b in sel.jobs[0].bullets] == ["a2", "a1"]
    assert [b.walk_position for b in sel.jobs[0].bullets] == [2, 3]
    assert [b.walk_position for b in sel.jobs[1].bullets] == [1]
    assert sel.jobs[2].bullets == []
    for job in sel.jobs:
        assert job.placed == len(job.bullets)

    placed = sorted((b for j in sel.jobs for b in j.bullets), key=lambda b: b.walk_position)
    assert [b.bullet_id for b in placed] == walk_ids(sel)
    by_id = {b.bullet_id: b for b in placed}
    assert (by_id["b1"].rerank_rank, by_id["b1"].rrf_position) == (1, 2)
    assert (by_id["a2"].rerank_rank, by_id["a2"].rrf_position) == (None, None)
    assert (by_id["a1"].rerank_rank, by_id["a1"].rrf_position) == (None, 1)
    assert all(b.overlap == 1 for b in placed)
    assert (by_id["a2"].line, by_id["a2"].kind, by_id["a2"].pass_number) == (
        "r1",
        "responsibility",
        2,
    )
    assert by_id["b1"].covers == ["m1"]
    assert by_id["a2"].covers == ["r1"]
    assert by_id["a1"].covers == ["m1"]

    # the JSON shape: the documented keys exactly, `pass` rather than `pass_number`
    dumped = sel.model_dump(mode="json")
    assert list(dumped) == [
        "total_slots",
        "jobs",
        "walk",
        "unplaced",
        "covered_by",
        "uncovered",
        "passes",
    ]
    assert list(dumped["jobs"][0]) == ["job_id", "job", "job_order", "cap", "placed", "bullets"]
    assert list(dumped["jobs"][0]["bullets"][0]) == [
        "bullet_id",
        "project",
        "text",
        "line",
        "kind",
        "pass",
        "walk_position",
        "overlap",
        "rerank_rank",
        "rrf_position",
        "covers",
    ]
    assert list(dumped["walk"][0]) == ["bullet_id", "job_id", "line", "kind", "pass"]
    assert [w["pass"] for w in dumped["walk"]] == [1, 2, 3]


def test_covers_covered_by_uncovered():
    cands = [cand("p1", ["K1", "k2"]), cand("p2", ["K3"])]
    lines = [
        line("m1", "must", "exact", ["K1"]),
        line("m2", "must", "exact", ["K2"]),
        line("n1", "nice", "adjacent", ["K3"]),
        line("n2", "nice", "none", ["K1"]),
    ]

    sel = run_walk(lines, cands, 1, {"A": 1})

    assert walk_ids(sel) == ["p1"]
    (bullet,) = sel.jobs[0].bullets
    assert bullet.line == "m1"
    # `covers` follows the requirements order and is a casefold lookup; a `none` line is absent
    assert bullet.covers == ["m1", "m2"]
    # m2 took no slot (its only carrier was taken) but is carried by a placed bullet
    assert sel.covered_by == {"m1": ["p1"], "m2": ["p1"], "n1": []}
    assert [(u.line, u.kind) for u in sel.uncovered] == [("n1", "nice")]
    assert [(u.line, u.reason) for u in sel.unplaced] == [("m2", "all_taken"), ("n1", "budget")]


# ----------------------------------------------------------- 7. determinism


def test_walk_deterministic():
    cands, lines, orders = _grouping_scenario()
    caps = {"A": 5, "B": 5, "C": 5}

    first = run_walk(lines, cands, 3, caps, orders)
    second = run_walk(lines, cands, 3, caps, orders)

    assert first.model_dump_json() == second.model_dump_json()


# ----------------------------------------------------------- 8. the uncapped reading


def test_uncapped_first_8_is_global_8_reading():
    # Eight must lines, each with two carriers (`<i>y` is in the reranker cut and wins); `s`
    # carries the keywords of m1 and m3 and goes first on both, so m3 takes its next candidate.
    cands = [cand("s", ["K1", "K3"], job_id="A", job_order=0)]
    lines = []
    for i in range(1, 9):
        job = ("A", "B", "C")[i % 3]
        cands.append(cand(f"b{i}x", [f"K{i}"], job_id=job, job_order="ABC".index(job)))
        cands.append(cand(f"b{i}y", [f"K{i}"], job_id=job, job_order="ABC".index(job)))
        ranked = ("s", f"b{i}y") if i in (1, 3) else (f"b{i}y",)
        lines.append(line(f"m{i}", "must", "exact", [f"K{i}"], ranked_ids=ranked))
    cands.append(cand("r1x", ["Kr1"], job_id="B", job_order=1))
    lines.append(line("r1", "responsibility", "exact", ["Kr1"]))
    caps = _m().caps_for(jobs_of(cands), 42, [1])
    assert caps == {"A": 14, "B": 14, "C": 14}

    sel = run_walk(lines, cands, 42, caps)

    assert walk_ids(sel)[:8] == ["s", "b2y", "b3y", "b4y", "b5y", "b6y", "b7y", "b8y"]
    assert walk_passes(sel)[:8] == [1] * 8
    assert walk_lines(sel)[:8] == [f"m{i}" for i in range(1, 9)]
    assert walk_lines(sel)[8] == "r1"
