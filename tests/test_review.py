"""Review-document tests R-1 .. R-17 and store tests S-1 .. S-8.

`loomcv/review/doc.py` is the review document: the template context with ids, built from the
inputs `draft_step` uses (never by parsing the Markdown), the pure within-job operations and the
annotations the page prints. `loomcv/review/store.py` is its disk / index side and the caller of
`loomcv.application`. The page (`loomcv/ui/review.py`) is not under pytest (the `ui` group is not
installed in CI).

Pure cases build payloads with `test_draft.py`'s helpers (read-only) and `Candidate`s in-test;
store cases use `test_graph.py`'s `_App` (a synced tmp `index_vault.md`, the fakes of
`test_cli.py`, `start` to `reviewing`). All content is fictional. Project imports are inside
each test so the red run fails per test, not on one collection error.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_cli import (
    FIXTURES,
    _copy_vault,
    _no_default_embedding_function,  # noqa: F401  (autouse fixture)
    _run,
)
from test_draft import (
    _bullet,
    _claim,
    _index,
    _job_row,
    _job_sel,
    _line,
    _payload,
    _section,
    _src_ids,
)
from test_graph import _App

from loomcv import resources

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = resources.path("templates", "resume_default.md")

L_CACHING = "Caching experience"
L_SEARCH = "Search relevance"
L_PYTHON = "Python skills"
L_K8S = "Kubernetes operations"
L_OBS = "Observability tooling"


def _cand(bullet_id, job_id, text, keywords=(), project=None):
    from loomcv.retrieve.rank import Candidate

    return Candidate(
        bullet_id=bullet_id,
        text=text,
        job_id=job_id,
        job=f"name of {job_id}",
        project=project,
        keywords=list(keywords),
    )


def _ranked(cand, rank=1):
    from loomcv.retrieve.rank import RankedBullet

    return RankedBullet(
        bullet_id=cand.bullet_id,
        job=cand.job,
        project=cand.project,
        text=cand.text,
        rrf_rank=rank,
        rerank_score=0.5,
        rerank_rank=rank,
    )


def _vault_header(with_header: bool = True):
    """The vault's header fields; `VaultHeader()` when there are none."""
    from loomcv.sync.parse_assets import VaultHeader

    if not with_header:
        return VaultHeader()
    return VaultHeader(
        name="Fictional Person",
        headline="Platform Engineer",
        contact=["Example City", "fictional@example.invalid"],
        summary="Ten fictional years.",
        education=["Example University — BSc, 2014"],
    )


def _scenario(with_header: bool = True):
    """Three jobs (the third has no placed bullet), seven confirmed vault bullets, five JD lines.

    Placed: j0 = c1, c2 (project Aurora); j1 = d1, d2. Not placed: c3 (carries L_OBS, ranked on
    two lines), c4 (ranked nowhere), d3. `covered_by` holds placed bullets only (merge.py).
    The header is the vault's fields; the Skills are one named and one unnamed group."""
    from loomcv.retrieve.rank import UncoveredLine
    from loomcv.sync.parse_assets import SkillGroup

    rows = [
        _job_row("j0", name="n0", employer="E0", title="T0", job_order=0),
        _job_row("j1", name="n1", employer="E1", title="T1", job_order=1),
        _job_row("j2", name="n2", employer="E2", title="T2", job_order=2),
    ]
    c1 = _cand("j0::c1", "j0", "Cut latency by half.", ["Caching"])
    c2 = _cand("j0::c2", "j0", "Reduced timeouts by 30%.", ["Search"], project="Aurora")
    c4 = _cand("j0::c4", "j0", "Wrote metrics dashboards.", ["Metrics"])
    c3 = _cand("j0::c3", "j0", "Instrumented tracing across 6 services.", ["Observability"])
    d1 = _cand("j1::d1", "j1", "Built Python ETL jobs.", ["Python"])
    d2 = _cand("j1::d2", "j1", "Ran Airflow pipelines.", ["Airflow"])
    d3 = _cand("j1::d3", "j1", "Wrote Terraform modules.", ["Terraform"])
    candidates = [c1, c2, c4, c3, d1, d2, d3]
    lines = [
        _line(L_CACHING, "must", "exact", ["Caching"]),
        _line(L_SEARCH, "responsibility", "adjacent", ["Search"]).model_copy(
            update={"ranked": [_ranked(c2), _ranked(c3, 2)]}
        ),
        _line(L_PYTHON, "nice", "exact", ["Python"]).model_copy(update={"ranked": [_ranked(d1)]}),
        _line(L_K8S, "must", "none"),
        _line(L_OBS, "must", "exact", ["Observability"]).model_copy(
            update={"ranked": [_ranked(c3)]}
        ),
    ]
    payload = _payload(
        jobs=[
            _job_sel(
                "j0",
                "n0",
                0,
                [
                    _bullet(c1.bullet_id, c1.text, line=L_CACHING, walk=1, covers=["Caching"]),
                    _bullet(
                        c2.bullet_id,
                        c2.text,
                        project="Aurora",
                        line=L_SEARCH,
                        kind="responsibility",
                        walk=2,
                        covers=["Search"],
                    ),
                ],
            ),
            _job_sel(
                "j1",
                "n1",
                1,
                [
                    _bullet(d1.bullet_id, d1.text, line=L_PYTHON, kind="nice", walk=3),
                    _bullet(d2.bullet_id, d2.text, line=L_PYTHON, kind="nice", walk=4),
                ],
            ),
        ],
        lines=lines,
        gaps=[L_K8S],
        claims=[_claim(L_K8S, [("Kubernetes", ["j1", "skills"])])],
        uncovered=[UncoveredLine(line=L_OBS, kind="must")],
        covered_by={L_CACHING: [c1.bullet_id], L_SEARCH: [c2.bullet_id], L_PYTHON: [d1.bullet_id]},
    )
    return SimpleNamespace(
        payload=payload,
        rows=rows,
        skill_groups=[
            SkillGroup(name="", items=["SQL"]),
            SkillGroup(name="Languages", items=["Python", "Go"]),
        ],
        header=_vault_header(with_header),
        candidates=candidates,
        c=SimpleNamespace(c1=c1, c2=c2, c3=c3, c4=c4, d1=d1, d2=d2, d3=d3),
    )


def _doc(s):
    from loomcv.review.doc import from_draft

    return from_draft(s.payload, s.rows, s.skill_groups, s.header)


def _render(doc):
    from loomcv.review.doc import render

    return render(doc, TEMPLATE)


# ------------------------------------------------------------------ document


@pytest.mark.parametrize("with_header", [True, False], ids=["header", "no-header"])
def test_from_draft_renders_byte_equal_to_render_resume(with_header):
    # R-1 — the round-trip proof of the whole model; over the slots
    from loomcv.draft.render import render_resume

    s = _scenario(with_header)
    assert [r.job_id for r in s.rows][-1] == "j2"  # the last job has no placed bullet
    expected = render_resume(s.payload, s.rows, s.skill_groups, s.header, template_path=TEMPLATE)
    assert _render(_doc(s)) == expected
    assert expected.startswith("# Fictional Person\n") is with_header
    assert render_resume(s.payload, s.rows, s.skill_groups, None, template_path=TEMPLATE) == (
        render_resume(
            s.payload, s.rows, s.skill_groups, _vault_header(False), template_path=TEMPLATE
        )
    )


def test_context_equals_resume_context():
    # R-2
    from loomcv.draft.render import resume_context
    from loomcv.review.doc import context

    s = _scenario()
    assert context(_doc(s)) == resume_context(s.payload, s.rows, s.skill_groups, s.header)


def test_move_within_a_job_clamped_at_the_ends():
    # R-3
    from loomcv.review.doc import move

    s = _scenario()
    doc = _doc(s)
    moved = move(doc, "j0", 0, +1)
    assert [b.bullet_id for b in moved.jobs[0].bullets] == ["j0::c2", "j0::c1"]
    assert moved.jobs[1:] == doc.jobs[1:]
    assert _src_ids(_render(moved)) == ["j0::c2", "j0::c1", "j1::d1", "j1::d2"]
    assert move(moved, "j0", 1, -1) == doc
    assert move(doc, "j0", 0, -1) == doc
    assert move(doc, "j0", 1, +1) == doc


def test_edit_changes_only_that_bullet_and_keeps_its_src_comment():
    # R-4
    from loomcv.review.doc import edit

    s = _scenario()
    doc = _doc(s)
    new = edit(doc, "j0", 0, "Brand new text.")
    assert new.jobs[0].bullets[0].text == "Brand new text."
    assert new.jobs[0].bullets[0].bullet_id == "j0::c1"
    assert new.jobs[0].bullets[0].project is None
    assert new.jobs[0].bullets[1:] == doc.jobs[0].bullets[1:]
    assert new.jobs[1:] == doc.jobs[1:]
    assert new.model_dump(exclude={"jobs"}) == doc.model_dump(exclude={"jobs"})
    assert "- Brand new text. <!-- src: j0::c1 -->" in _render(new)

    kept = edit(doc, "j0", 1, "T")
    assert kept.jobs[0].bullets[1].project == "Aurora"
    assert "- **Aurora:** T <!-- src: j0::c2 -->" in _render(kept)
    relabelled = edit(doc, "j0", 1, "T", project="Zed")
    assert "- **Zed:** T <!-- src: j0::c2 -->" in _render(relabelled)
    unlabelled = edit(doc, "j0", 1, "T", project="")
    assert unlabelled.jobs[0].bullets[1].project is None
    assert "- T <!-- src: j0::c2 -->" in _render(unlabelled)


def test_delete_removes_the_bullet_and_an_empty_job_renders_heading_and_dates():
    # R-5
    from loomcv.review.doc import delete

    s = _scenario()
    doc = _doc(s)
    one = delete(doc, "j0", 0)
    assert [b.bullet_id for b in one.jobs[0].bullets] == ["j0::c2"]
    assert _src_ids(_render(one)) == ["j0::c2", "j1::d1", "j1::d2"]
    empty = delete(delete(doc, "j1", 0), "j1", 0)
    assert empty.jobs[1].bullets == []
    # An emptied job prints its heading and date line, no list
    assert "### T1 — E1\n*Jan 2020 – Dec 2021*\n\n### T2 — E2\n*Jan 2020 – Dec 2021*\n" in _render(
        empty
    )
    assert _src_ids(_render(empty)) == ["j0::c1", "j0::c2"]


def test_add_appends_a_vault_bullet_and_refuses_duplicates_and_other_jobs():
    # R-6
    from loomcv.review.doc import add, pick

    s = _scenario()
    doc = _doc(s)
    c = s.c
    added = add(doc, "j0", c.c3)
    last = added.jobs[0].bullets[-1]
    assert (last.bullet_id, last.text, last.project) == ("j0::c3", c.c3.text, None)
    assert [b.bullet_id for b in added.jobs[0].bullets] == ["j0::c1", "j0::c2", "j0::c3"]
    assert _src_ids(_render(added))[2] == "j0::c3"
    with pytest.raises(ValueError):
        add(added, "j0", c.c3)  # already placed
    with pytest.raises(ValueError):
        add(doc, "j0", c.c1)  # already placed
    with pytest.raises(ValueError):
        add(doc, "j1", c.c3)  # another job's candidate
    assert doc == _doc(s)
    assert pick(s.candidates, "j0::c3") == c.c3
    with pytest.raises(ValueError):
        pick(s.candidates, None)
    with pytest.raises(ValueError):
        pick(s.candidates, "j0::nope")


def test_add_checked_adds_every_ticked_bullet_in_the_offered_order_or_nothing():
    # ⑧ (e)
    from loomcv.review.doc import add_checked

    s = _scenario()
    doc = _doc(s)
    offered = ["j0::c3", "j0::c4"]

    added = add_checked(doc, "j0", s.candidates, offered, ["j0::c4", "j0::c3"])
    assert [b.bullet_id for b in added.jobs[0].bullets] == ["j0::c1", "j0::c2", "j0::c3", "j0::c4"]
    assert _src_ids(_render(added))[2:4] == ["j0::c3", "j0::c4"]
    one = add_checked(doc, "j0", s.candidates, offered, ["j0::c4"])
    assert [b.bullet_id for b in one.jobs[0].bullets][-1] == "j0::c4"
    assert doc == _doc(s)

    with pytest.raises(ValueError):
        add_checked(doc, "j0", s.candidates, offered, [])  # nothing ticked
    with pytest.raises(ValueError):
        add_checked(doc, "j0", s.candidates, offered, ["j0::c3", "j0::nope"])  # not offered
    with pytest.raises(ValueError):
        add_checked(added, "j0", s.candidates, offered, ["j0::c3"])  # already on the document


def test_out_of_range_unknown_job_and_bad_delta_are_value_errors():
    # R-7
    from loomcv.review.doc import delete, edit, move

    doc = _doc(_scenario())
    for index in (-1, 2):
        with pytest.raises(ValueError):
            move(doc, "j0", index, +1)
        with pytest.raises(ValueError):
            edit(doc, "j0", index, "x")
        with pytest.raises(ValueError):
            delete(doc, "j0", index)
    with pytest.raises(ValueError):
        move(doc, "j2", 0, +1)  # a job with no bullets
    with pytest.raises(ValueError):
        move(doc, "nope", 0, +1)
    with pytest.raises(ValueError):
        edit(doc, "nope", 0, "x")
    with pytest.raises(ValueError):
        delete(doc, "nope", 0)
    with pytest.raises(ValueError):
        move(doc, "j0", 0, 2)
    with pytest.raises(ValueError):
        move(doc, "j0", 0, 0)


def test_set_header_and_set_skills_change_only_their_slots():
    # R-8: Headline, Summary and the groups' items are the
    # per-résumé edits; each setter changes only its slot.
    from loomcv.review.doc import set_headline, set_skill_groups, set_summary

    s = _scenario()
    doc = _doc(s)
    base = _render(doc)
    bullet_lines = [ln for ln in base.splitlines() if ln.startswith("- ")]

    def others(changed, field):
        return changed.model_dump(exclude={field}) == doc.model_dump(exclude={field})

    headed = set_headline(doc, "Payments Platform Engineer")
    assert headed.headline == "Payments Platform Engineer"
    assert others(headed, "headline")
    assert "**Payments Platform Engineer**\n" in _render(headed)
    assert set_headline(doc, "").headline is None

    summarised = set_summary(doc, "A trimmed summary.")
    assert summarised.summary == "A trimmed summary."
    assert others(summarised, "summary")
    assert "## Summary\n\nA trimmed summary.\n" in _render(summarised)

    skilled = set_skill_groups(doc, [["Go", "Rust"], ["Zig"]])
    assert [(g.name, g.items) for g in skilled.skill_groups] == [
        ("", ["Go", "Rust"]),
        ("Languages", ["Zig"]),
    ]
    assert others(skilled, "skill_groups")
    rendered = _render(skilled)
    assert "**Languages:** Zig\\\nGo; Rust\n" in rendered  # the unnamed group prints last
    for changed in (headed, summarised, skilled):
        assert [ln for ln in _render(changed).splitlines() if ln.startswith("- ")] == bullet_lines
    with pytest.raises(ValueError):
        set_skill_groups(doc, [["Go"]])  # one box per group


def test_operations_return_new_documents_and_models_are_frozen():
    # R-9
    from pydantic import ValidationError

    from loomcv.review.doc import add, delete, edit, move, set_headline, set_skill_groups

    s = _scenario()
    doc = _doc(s)
    before = doc.model_copy(deep=True)
    results = [
        move(doc, "j0", 0, +1),
        edit(doc, "j0", 0, "x", project="p"),
        delete(doc, "j0", 0),
        add(doc, "j0", s.c.c3),
        set_headline(doc, "H"),
        set_skill_groups(doc, [["A"], ["B"]]),
    ]
    assert doc == before
    assert all(r is not doc and r != doc for r in results)
    with pytest.raises(ValidationError):
        doc.headline = "changed"
    with pytest.raises(ValidationError):
        doc.jobs[0].heading = "changed"


def test_coverage_of_is_the_inverse_of_covered_by_for_placed_bullets():
    # R-10
    from loomcv.review.doc import add, coverage_of

    s = _scenario()
    doc = _doc(s)
    covered = coverage_of(doc, s.payload, s.candidates)
    expected = {
        bullet_id: [
            f"{line.text} ({line.kind})"
            for line in s.payload.requirements
            if bullet_id in s.payload.selection.covered_by.get(line.text, [])
        ]
        for job in doc.jobs
        for bullet_id in (b.bullet_id for b in job.bullets)
    }
    assert covered == expected
    assert covered["j0::c1"] == [f"{L_CACHING} (must)"]
    assert covered["j1::d2"] == []  # placed, on no line

    with_added = coverage_of(add(doc, "j0", s.c.c3), s.payload, s.candidates)
    assert with_added["j0::c3"] == [f"{L_OBS} (must)"]  # the lines it would have covered


def test_addable_lists_the_jobs_unplaced_bullets_ranked_first(tmp_path):
    # R-11
    from loomcv.review.doc import add, addable

    s = _scenario()
    doc = _doc(s)
    result = addable(doc, s.payload, s.candidates)
    assert list(result) == ["j0", "j1", "j2"]
    assert [a.bullet_id for a in result["j0"]] == ["j0::c3", "j0::c4"]  # ranked first, then vault
    assert result["j0"][0].lines == [f"{L_SEARCH} (responsibility)", f"{L_OBS} (must)"]
    assert result["j0"][1].lines == []
    assert [a.bullet_id for a in result["j1"]] == ["j1::d3"]
    assert result["j2"] == []
    assert (result["j0"][0].text, result["j0"][0].project) == (s.c.c3.text, None)
    assert result["j0"][0].label == s.c.c3.text
    after = addable(add(doc, "j0", s.c.c3), s.payload, s.candidates)
    assert [a.bullet_id for a in after["j0"]] == ["j0::c4"]

    long_text = "word " * 60
    longer = [*s.candidates, _cand("j2::z1", "j2", long_text)]
    label = addable(doc, s.payload, longer)["j2"][0].label
    assert label == long_text[:90] + "…"
    assert len(label) <= 91

    # second half: over a real index, which holds confirmed bullets only
    from loomcv.retrieve.search import load_candidates, load_job_rows
    from loomcv.review.doc import from_draft

    db, _ = _index(tmp_path, "index_vault.md")
    candidates = load_candidates(db)
    rows = load_job_rows(db)
    by_name = {c.bullet_id.split("::")[1]: c for c in candidates}
    assert "cas003" not in by_name
    first = by_name["aur001"]
    payload = _payload(
        jobs=[_job_sel(first.job_id, first.job, 0, [_bullet(first.bullet_id, first.text, walk=1)])]
    )
    doc = from_draft(payload, rows, [], None)
    result = addable(doc, payload, candidates)
    ids = {job_id: [a.bullet_id.split("::")[1] for a in items] for job_id, items in result.items()}
    assert ids[first.job_id] == ["aur002", "bea001", "bea002"]
    assert ids[by_name["cas001"].job_id] == ["cas001", "cas002"]
    assert ids[by_name["lab001"].job_id] == ["lab001"]
    by_id = {c.bullet_id: c for c in candidates}
    assert all(
        by_id[a.bullet_id].job_id == job_id for job_id, items in result.items() for a in items
    )


def test_missing_matches_the_report_and_follows_the_document():
    # R-12
    from loomcv.draft import render_report
    from loomcv.review.doc import add, missing

    s = _scenario()
    doc = _doc(s)
    report = render_report(s.payload, s.rows)
    m = missing(doc, s.payload, s.rows, s.candidates)

    assert [g.text for g in m.gaps] == s.payload.gaps
    assert m.gaps[0].claimed_at == ["T1 — E1", "Skills line"]
    assert L_K8S in _section(report, "## 2 Required, not shown")

    assert [c.text for c in m.claims] == [c.line for c in s.payload.claims]
    claim = m.claims[0]
    assert (claim.kind, claim.match) == ("must", "exact")
    assert [(k.keyword, k.at) for k in claim.keywords] == [
        ("Kubernetes", ["T1 — E1", "Skills line"])
    ]
    assert "Kubernetes" in _section(
        report, "## 3 Listed in your profile, but no bullet describes it"
    )

    assert [(u.text, u.kind) for u in m.uncovered] == [
        (u.line, u.kind) for u in s.payload.selection.uncovered
    ]
    assert m.uncovered[0].carriers == [s.c.c3.text]
    assert L_OBS in _section(report, "## 5 Not on this résumé")

    first_line = _section(report, "## 0 Verdict").strip().splitlines()[0]
    assert m.headline == first_line

    after = missing(add(doc, "j0", s.c.c3), s.payload, s.rows, s.candidates)
    assert after.uncovered == []
    assert after.gaps == m.gaps and after.claims == m.claims


def test_index_mismatch_is_no_job_of_the_document_in_the_index_and_names_stay_readable():
    # ⑧ (f)
    from loomcv.retrieve.search import JobRow
    from loomcv.review.doc import ReviewDoc, index_mismatch, missing

    s = _scenario()
    doc = _doc(s)
    assert index_mismatch(doc, s.rows) is False
    assert index_mismatch(doc, s.rows[1:2]) is False  # one shared job is enough
    assert index_mismatch(doc, []) is True
    other: list[JobRow] = [_job_row("elsewhere", name="x", employer="X", title="Y", job_order=0)]
    assert index_mismatch(doc, other) is True
    empty = ReviewDoc(
        name=None, headline=None, contact=[], summary=None, education=[], skill_groups=[], jobs=[]
    )
    assert index_mismatch(empty, s.rows) is False

    # the job names come from the document's own headings, not from the index and not the slug
    found = missing(doc, s.payload, other, s.candidates)
    assert found.gaps[0].claimed_at == ["T1 — E1", "Skills line"]
    assert [(k.keyword, k.at) for k in found.claims[0].keywords] == [
        ("Kubernetes", ["T1 — E1", "Skills line"])
    ]


def test_a_bullet_without_an_id_is_refused_by_the_model():
    # R-13: a bullet with no vault origin is not a shape the review document has
    from pydantic import ValidationError

    from loomcv.review.doc import ReviewBullet

    with pytest.raises(ValidationError):
        ReviewBullet(bullet_id=None, project=None, text="x")
    assert ReviewBullet(bullet_id="j0::a", project=None, text="x").bullet_id == "j0::a"


def test_review_doc_json_round_trip():
    # R-14
    from loomcv.review.doc import ReviewDoc

    doc = _doc(_scenario())
    assert ReviewDoc.model_validate_json(doc.model_dump_json()) == doc


def test_render_with_a_template_missing_a_slot_raises_the_same_error(tmp_path):
    # R-15
    from loomcv.draft import ResumeTemplateError, render_resume
    from loomcv.review.doc import render

    s = _scenario()
    template = tmp_path / "broken.md"
    template.write_text("{{ nope.slot }}\n", encoding="utf-8")
    with pytest.raises(ResumeTemplateError):
        render_resume(s.payload, s.rows, s.skill_groups, s.header, template_path=template)
    with pytest.raises(ResumeTemplateError):
        render(_doc(s), template)


def test_review_package_imports_no_ui_or_orchestration_library():
    # R-16
    files = sorted((REPO_ROOT / "loomcv" / "review").glob("*.py"))
    assert {f.name for f in files} >= {"__init__.py", "doc.py", "store.py"}
    banned = ("gradio", "langgraph", "langchain", "loomcv.ui")
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith(banned), f"{path.name} imports {name}"


def test_commit_applies_every_box_by_layout_and_refuses_a_stale_page():
    # R-17
    from loomcv.review.doc import commit, layout

    doc = _doc(_scenario())
    drawn = layout(doc)
    assert drawn == [("j0", "j0::c1"), ("j0", "j0::c2"), ("j1", "j1::d1"), ("j1", "j1::d2")]

    same = [v for b in (b for j in doc.jobs for b in j.bullets) for v in (b.text, b.project or "")]
    groups = [g.items for g in doc.skill_groups]
    assert commit(doc, drawn, doc.headline or "", doc.summary or "", groups, same) == doc

    # R-5: Headline, Summary and one items list per group are the boxes.
    values = ["One.", "", "Two.", "Lab", "Three.", "", "Four.", ""]
    changed = commit(doc, drawn, "New headline", "", [["X"], ["Y", "Z"]], values)
    assert changed.headline == "New headline"
    assert changed.summary is None
    assert [(g.name, g.items) for g in changed.skill_groups] == [
        ("", ["X"]),
        ("Languages", ["Y", "Z"]),
    ]
    assert (changed.name, changed.contact, changed.education) == (
        doc.name,
        doc.contact,
        doc.education,
    )
    assert [(b.text, b.project) for j in changed.jobs for b in j.bullets] == [
        ("One.", None),
        ("Two.", "Lab"),
        ("Three.", None),
        ("Four.", None),
    ]
    assert [b.bullet_id for j in changed.jobs for b in j.bullets] == [i for _, i in drawn]

    before = doc.model_copy(deep=True)
    with pytest.raises(ValueError):
        # the page was drawn from another layout
        commit(doc, list(reversed(drawn)), "", "", groups, values)
    with pytest.raises(ValueError):
        commit(doc, drawn, "", "", groups, values[:-1])  # a wrong number of boxes
    with pytest.raises(ValueError):
        commit(doc, drawn, "", "", groups[:1], values)  # a wrong number of Skills boxes
    assert doc == before


def test_skills_text_round_trip_blanks_and_trailing_separator():
    # R-19: the Skills box is one line, `;`-separated, the
    # vault's own item separator
    from loomcv.review.doc import skills_from_text, skills_to_text

    assert skills_to_text(["Python", "SQL"]) == "Python; SQL"
    assert skills_to_text([]) == ""
    assert skills_from_text("Python; SQL") == ["Python", "SQL"]
    assert skills_from_text("") == []
    assert skills_from_text(" a ;; b ; ") == ["a", "b"]  # strip, drop empties, trailing `;`
    assert skills_from_text("Go;Rust;Zig") == ["Go", "Rust", "Zig"]  # order kept
    items = ["Go", "Rust", "Zig"]
    assert skills_from_text(skills_to_text(items)) == items


def test_skills_text_round_trip_keeps_an_item_with_a_comma():
    # A comma inside an item is text, never a separator
    from loomcv.review.doc import skills_from_text, skills_to_text

    items = ["JavaScript, TypeScript, Node.js", "Web (video-player stack, headless CMS)"]
    assert (
        skills_to_text(items)
        == "JavaScript, TypeScript, Node.js; Web (video-player stack, headless CMS)"
    )
    assert skills_from_text(skills_to_text(items)) == items


# --------------------------------------------------------------------- store


@pytest.fixture
def app(tmp_path, monkeypatch, capsys):
    return _App(tmp_path, monkeypatch, capsys)


def _open(app, app_id="jd"):
    from loomcv.review.store import open_review

    return open_review(app.runtime(), app_id, out_root=app.apps, checkpoints=app.checkpoints)


def _save(app, doc, opened, app_id="jd"):
    from loomcv.review.store import save_review

    return save_review(doc, app_id, out_root=app.apps, template_path=opened.template_path)


def _find(doc, suffix):
    for job in doc.jobs:
        for index, bullet in enumerate(job.bullets):
            if bullet.bullet_id.endswith(suffix):
                return job.job_id, index
    raise AssertionError(f"{suffix} is not on the document")


def test_open_review_without_a_state_file_builds_the_document_from_the_draft(app):
    # S-1
    from loomcv.retrieve.rank import MatchPayload
    from loomcv.retrieve.search import (
        load_candidates,
        load_job_rows,
        load_skill_groups,
        load_vault_header,
    )
    from loomcv.review.doc import from_draft, render

    app.start()
    opened = _open(app)
    folder = app.apps / "jd"
    payload = MatchPayload.model_validate_json((folder / "match.json").read_text(encoding="utf-8"))
    rows = load_job_rows(app.db)
    groups = load_skill_groups(app.db)

    # The header and the groups come from the index, not from a header file.
    assert opened.doc == from_draft(payload, rows, groups, load_vault_header(app.db))
    assert opened.notice is None
    assert opened.status.status == "reviewing"
    assert opened.payload == payload
    assert opened.candidates == load_candidates(app.db)
    assert opened.rows == rows
    assert opened.report == (folder / "match_report.md").read_text(encoding="utf-8")
    assert (opened.app_id, opened.out_root, opened.app_dir) == ("jd", app.apps, folder)
    assert (
        render(opened.doc, opened.template_path).encode("utf-8")
        == (folder / "tailored_resume.md").read_bytes()
    )


def test_save_review_writes_the_state_and_the_markdown_and_not_the_draft(app):
    # S-2
    from loomcv.review.doc import ReviewDoc, edit, render

    app.start()
    folder = app.apps / "jd"
    draft = (folder / "tailored_resume.md").read_bytes()
    opened = _open(app)
    job_id, index = _find(opened.doc, "aur001")
    doc = edit(opened.doc, job_id, index, "Edited bullet text.")

    saved = _save(app, doc, opened)

    assert saved.state_path == folder / "review.json"
    assert saved.resume_path == folder / "edited_resume.md"
    assert ReviewDoc.model_validate_json(saved.state_path.read_text(encoding="utf-8")) == doc
    assert saved.resume_path.read_text(encoding="utf-8") == render(doc, opened.template_path)
    assert (folder / "tailored_resume.md").read_bytes() == draft
    assert not (folder / "reviewed_resume.md").exists()
    assert app.status().status == "reviewing"


def test_a_saved_edit_survives_a_reopen(app):
    # S-3
    from loomcv.review.doc import edit

    app.start()
    opened = _open(app)
    job_id, index = _find(opened.doc, "aur001")
    doc = edit(opened.doc, job_id, index, "Edited bullet text.")
    _save(app, doc, opened)

    again = _open(app)
    assert again.doc == doc
    assert again.doc != opened.doc
    assert again.status.status == "reviewing"


def test_submit_after_save_takes_the_saved_markdown_through_the_real_chain(app):
    # S-5
    from loomcv import application
    from loomcv.review.doc import edit

    app.start()
    folder = app.apps / "jd"
    opened = _open(app)
    job_id, index = _find(opened.doc, "aur001")
    text = opened.doc.jobs[0].bullets[index].text
    assert "800ms to 200ms" in text
    doc = edit(opened.doc, job_id, index, text.replace("800ms to 200ms", "800ms to 100ms"))
    saved = _save(app, doc, opened)

    status = application.submit(
        app.runtime(),
        "jd",
        edited_path=saved.resume_path,
        out_root=app.apps,
        checkpoints=app.checkpoints,
    )

    assert status.status == "done"
    # The same checks via `pdf_words`.
    from loomcv.review.words import pdf_words

    assert pdf_words(opened.status) is None  # a posting still being reviewed has no result
    reopened = _open(app)
    words = pdf_words(reopened.status)
    assert words is not None and words.ok
    assert words.line.startswith("PDF ready") and "pages=" not in words.line
    pdf = words.pdf
    assert pdf == str(folder / "final_resume.pdf") and Path(pdf).is_file()
    assert (folder / "reviewed_resume.md").read_bytes() == saved.resume_path.read_bytes()
    final = (folder / "final_resume.md").read_text(encoding="utf-8")
    assert "800ms to 100ms" in final
    assert "src:" not in final


def test_list_applications_and_unknown_application(app):
    # S-6
    from loomcv.application import UnknownApplication
    from loomcv.review.store import list_applications

    app.start("jd")
    app.start("jd2")
    app.submit(app.apps / "jd2" / "tailored_resume.md", "jd2")
    (app.apps / "other").mkdir()
    (app.apps / "note.txt").write_text("not an application", encoding="utf-8")
    (app.apps / "bad name").mkdir()

    rows = list_applications(app.apps, app.checkpoints)
    assert [(s.app_id, s.status) for s in rows] == [
        ("jd", "reviewing"),
        ("jd2", "done"),
        ("other", "unknown"),
    ]
    assert list_applications(app.tmp / "no_such_root", app.checkpoints) == []

    with pytest.raises(UnknownApplication, match="Pick a posting first"):
        _open(app, "")
    with pytest.raises(UnknownApplication):
        _open(app, "ghost")
    assert not (app.apps / "ghost").exists()


def test_start_application_from_pasted_text_requires_an_id(app):
    # S-7
    from loomcv.application import ApplicationExists, InvalidAppId
    from loomcv.review.store import start_application

    text = app.jd.read_text(encoding="utf-8")

    def go(app_id):
        return start_application(
            app.runtime(),
            text,
            app_id,
            out_root=app.apps,
            checkpoints=app.checkpoints,
        )

    with pytest.raises(InvalidAppId):
        go("")
    assert not app.apps.exists()

    status = go("jd7")
    assert status.status == "reviewing"
    assert (app.apps / "jd7" / "jd.md").read_text(encoding="utf-8") == text
    assert not (app.apps / "jd7" / "header.md").exists()  # `--header` retired

    before = sorted(p.name for p in (app.apps / "jd7").iterdir())
    with pytest.raises(ApplicationExists):
        go("jd7")
    assert sorted(p.name for p in (app.apps / "jd7").iterdir()) == before


# ------------------------------------- the applications page's pure side
#
# I-1 … I-6 (`app_id_from_name`), T-1
# (`taken_ids`), N-1 … N-4 (`started_at`, the name file, `list_entries`), W-1 … W-6
# (`verdict_words`), R-1 … R-3 (`report_body`), V-1 … V-3 (`covers_words`), E-1 … E-3
# (`edited_bullets`), J-1 (`job_rows`), L-1 / L-2 (`stage_label`, `entry_words`, `tally_words`),
# P-1 … P-4 (`pdf_words`, `built_words`).


def _today():
    from datetime import date

    return date(2026, 11, 8)


def test_app_id_from_name_slugifies_a_title():
    # I-1
    from loomcv.review.words import app_id_from_name

    assert app_id_from_name("Applied AI Engineer — Lumenwright", set()) == (
        "applied-ai-engineer-lumenwright"
    )


def test_app_id_from_name_trims_and_collapses():
    # I-2
    from loomcv.review.words import app_id_from_name

    assert app_id_from_name("  Staff FE (Fennimore)!! ", set()) == "staff-fe-fennimore"
    assert app_id_from_name("a__b..c", set()) == "a__b..c"


def test_app_id_from_name_adds_the_date_then_a_counter():
    # I-3
    from loomcv.review.words import app_id_from_name

    today = _today()
    slug = "fennimore-web-platform"
    name = "Fennimore web platform"
    assert app_id_from_name(name, {slug}, today=today) == f"{slug}-20261108"
    taken = {slug, f"{slug}-20261108"}
    assert app_id_from_name(name, taken, today=today) == f"{slug}-20261108-2"
    taken.add(f"{slug}-20261108-2")
    assert app_id_from_name(name, taken, today=today) == f"{slug}-20261108-3"
    assert app_id_from_name(name, {slug.upper()}, today=today) == f"{slug}-20261108"  # casefold
    assert app_id_from_name(name, {"other"}, today=today) == slug


def test_app_id_from_name_refuses_a_name_with_nothing_to_use():
    # I-4
    from loomcv.application import InvalidAppId
    from loomcv.review.words import NAME_NEEDED, app_id_from_name

    for name in ("—", "", "  ", ".."):
        with pytest.raises(InvalidAppId) as caught:
            app_id_from_name(name, set())
        assert str(caught.value) == NAME_NEEDED


def test_app_id_from_name_always_matches_the_id_pattern():
    # I-5
    from loomcv.application import _APP_ID
    from loomcv.review.words import app_id_from_name

    names = [
        "../x",
        "a/b",
        "-x-",
        "Résumé",
        "Staff Engineer @ Example, Inc. (Remote)",
        "  .hidden_  ",
        "C++ / Rust",
        "腾讯前端",
    ]
    for name in names:
        result = app_id_from_name(name, set(), today=_today())
        assert _APP_ID.fullmatch(result), (name, result)
        assert result[0] not in "._-" and result[-1] not in "._-", (name, result)


def test_app_id_from_name_folds_accents_and_names_a_non_ascii_title():
    # I-6
    from loomcv.review.words import app_id_from_name

    assert app_id_from_name("Résumé", set()) == "resume"
    assert app_id_from_name("腾讯前端", set(), today=_today()) == "application"
    assert app_id_from_name("腾讯前端", {"application"}, today=_today()) == "application-20261108"
    assert app_id_from_name("简历 Lumenwright", set()) == "lumenwright"


def test_taken_ids_are_the_folders_and_the_checkpoint_threads(app):
    # T-1
    import shutil

    from loomcv.review.store import taken_ids

    app.start("jd")
    app.start("jd2")
    shutil.rmtree(app.apps / "jd2")  # the thread stays in the checkpoints file
    (app.apps / "note.txt").write_text("stray\n", encoding="utf-8")

    assert taken_ids(app.apps, app.checkpoints) == {"jd", "jd2", "note.txt"}

    missing = app.tmp / "no_cp" / "checkpoints.sqlite"
    assert taken_ids(app.apps, missing) == {"jd", "note.txt"}
    assert not missing.parent.exists()  # nothing created
    assert taken_ids(app.tmp / "no_root", missing) == set()


def test_list_entries_are_newest_first_by_the_jd_copy(app):
    # N-1
    import json
    import os
    from datetime import datetime

    from loomcv.review.store import list_entries, started_at

    app.start("a1")
    app.start("a2")
    older, newer = 1_700_000_000, 1_700_100_000
    os.utime(app.apps / "a2" / "jd.md", (older, older))
    os.utime(app.apps / "a1" / "jd.md", (newer, newer))

    entries = list_entries(app.apps, app.checkpoints)
    assert [e.app_id for e in entries] == ["a1", "a2"]
    assert entries[0].started == datetime.fromtimestamp(newer)
    assert started_at(app.apps / "a2") == datetime.fromtimestamp(older)
    assert entries[0].status.status == "reviewing"
    title = json.loads((app.apps / "a1" / "profile.json").read_text(encoding="utf-8"))["title"]
    assert entries[0].title == title
    assert entries[0].name is None


def test_list_entries_put_a_folder_without_a_jd_last(app):
    # N-2
    from loomcv.review.store import list_entries, started_at

    app.start("b1")
    (app.apps / "a0").mkdir()  # sorts first by name; no jd.md
    (app.apps / "zz").mkdir()

    entries = list_entries(app.apps, app.checkpoints)
    assert [e.app_id for e in entries] == ["b1", "a0", "zz"]  # no date last, ties by id
    assert entries[1].started is None and entries[1].status.status == "unknown"
    assert entries[1].title == "a0"
    assert started_at(app.apps / "a0") is None
    assert list_entries(app.tmp / "no_root", app.checkpoints) == []


def _start(app, app_id, name=None):
    from loomcv.review.store import start_application

    return start_application(
        app.runtime(),
        app.jd.read_text(encoding="utf-8"),
        app_id,
        out_root=app.apps,
        checkpoints=app.checkpoints,
        name=name,
    )


def test_start_application_writes_the_name_and_it_reads_back(app):
    # N-3
    from loomcv.review.store import NAME_FILE, list_entries, read_name
    from loomcv.review.words import entry_words

    status = _start(app, "jd7", name="Fennimore web platform")
    assert status.status == "reviewing"
    assert (app.apps / "jd7" / NAME_FILE).is_file()
    assert read_name(app.apps / "jd7") == "Fennimore web platform"

    app.start("jd")  # an application started without a name (`pipeline run`'s path)
    assert read_name(app.apps / "jd") is None
    by_id = {e.app_id: e for e in list_entries(app.apps, app.checkpoints)}
    assert by_id["jd7"].name == "Fennimore web platform"
    assert entry_words(by_id["jd"]).sub.startswith("jd")
    assert entry_words(by_id["jd7"]).sub.startswith("Fennimore web platform")


def test_build_keeps_the_build_time_document_after_the_profile_changes(app, capsys):
    # A successful Build writes `review.json` + `edited_resume.md`
    # once, so an application never edited after it opens on its own document, not the index's.
    from loomcv.review.doc import edited_bullets, index_mismatch, render
    from loomcv.review.store import EDITED_NAME, STATE_NAME, stage_label
    from loomcv.steps import Runtime

    status = _start(app, "jd7", name="Fennimore web platform")
    app_dir = app.apps / "jd7"
    assert (app_dir / STATE_NAME).is_file() and (app_dir / EDITED_NAME).is_file()
    built = _open(app, "jd7")
    assert built.notice is None
    # the untouched document renders byte-equal to the draft
    assert (app_dir / EDITED_NAME).read_text(encoding="utf-8") == render(
        built.doc, built.template_path
    )
    assert (app_dir / EDITED_NAME).read_bytes() == (app_dir / "tailored_resume.md").read_bytes()
    assert any(job.bullets for job in built.doc.jobs)
    # the same stage words, no restore link, the same list
    assert stage_label(status) == stage_label(built.status) == "Draft ready — review it"
    assert edited_bullets(built.doc, built.candidates) == set()

    # Start over + a different profile: another vault synced into another index
    db2, chroma2 = app.tmp / "db2.sqlite", app.tmp / "chroma2"
    other = _copy_vault(app.tmp, (FIXTURES / "basic_vault.md").read_text("utf-8"), "other.md")
    assert _run(other, db2, chroma2) == 0
    capsys.readouterr()
    later = Runtime(db=db2, chroma=chroma2, generation_config_path=app.cfg, adapters={})
    from loomcv.review.store import open_review

    reopened = open_review(later, "jd7", out_root=app.apps, checkpoints=app.checkpoints)
    assert reopened.doc == built.doc  # its own name, jobs and bullets
    assert index_mismatch(reopened.doc, reopened.rows) is True  # the "earlier version" line


def test_the_name_survives_a_failed_build_and_never_lands_on_a_taken_folder(app, monkeypatch):
    # N-4 (R1)
    from loomcv.application import ApplicationExists
    from loomcv.review.store import read_name

    def fail(*_args, **_kwargs):
        raise RuntimeError("the fake match failed")

    monkeypatch.setattr("loomcv.graph.match_step", fail)
    with pytest.raises(RuntimeError, match="fake match failed"):
        _start(app, "jd8", name="First name")
    assert (app.apps / "jd8").is_dir()
    assert read_name(app.apps / "jd8") == "First name"

    before = sorted(p.name for p in (app.apps / "jd8").iterdir())
    with pytest.raises(ApplicationExists):
        _start(app, "jd8", name="Second name")
    assert sorted(p.name for p in (app.apps / "jd8").iterdir()) == before
    assert read_name(app.apps / "jd8") == "First name"


def _must(texts_grades):
    return [_line(text, "must", grade) for text, grade in texts_grades]


def test_verdict_words_name_the_fit_from_the_letter():
    # W-1
    from loomcv.review.words import verdict_words

    # A close match counts half — A (1 direct) · B (3 direct + 1 close) ·
    # C (2 direct + 1 close + 2 none, 50 %) · D (1 direct + 2 none).
    a = _payload(lines=_must([("L1", "exact")]))
    b = _payload(
        lines=_must([("L1", "exact"), ("L2", "exact"), ("L3", "exact"), ("L4", "adjacent")])
    )
    c = _payload(
        lines=_must(
            [("L1", "exact"), ("L2", "exact"), ("L3", "adjacent"), ("L4", "none"), ("L5", "none")]
        )
    )
    d = _payload(lines=_must([("L1", "exact"), ("L2", "none"), ("L3", "none")]))
    # e: the case the old rule read as Good (4 of 4 hits) — two close matches count half → 75 %.
    e = _payload(
        lines=_must([("L1", "exact"), ("L2", "exact"), ("L3", "adjacent"), ("L4", "adjacent")])
    )
    assert [verdict_words(p).fit for p in (a, b, c, d, e)] == [
        "Good fit",
        "Good fit",
        "Partial fit",
        "Weak fit",
        "Partial fit",
    ]
    assert verdict_words(c).headline.startswith("Partial fit")


def test_verdict_words_headline_three_counts():
    # W-1: the three counts in plain words, singular for one.
    from loomcv.review.words import verdict_words

    many = _payload(
        lines=_must(
            [("L1", "exact"), ("L2", "exact")]
            + [(f"C{i}", "adjacent") for i in range(5)]
            + [("N1", "none")]
        )
    )
    one = _payload(lines=_must([("L1", "exact")]))

    assert verdict_words(many).headline == (
        "Partial fit — this posting requires 8 things. Your profile clearly shows 2 and shows "
        "something similar for 5. Missing from your profile entirely: 1 thing, listed below."
    )
    assert verdict_words(one).headline == (
        "Good fit — this posting requires 1 thing. Your profile clearly shows 1 and shows "
        "something similar for 0."
    )
    # a missing part only (the line is graded exact): the third sentence counts the part
    part_only = _payload(
        lines=[
            _line("Build Go and Node services", "must", "exact", parts=[("Go", "none")]),
            _line("Ship it", "must", "exact"),
        ]
    )
    assert verdict_words(part_only).headline == (
        "Partial fit — this posting requires 2 things. Your profile clearly shows 1 and shows "
        "something similar for 1. Missing from your profile entirely: 1 thing, listed below."
    )
    # two missing → plural
    two = _payload(lines=_must([("L1", "exact"), ("N1", "none"), ("N2", "none")]))
    assert verdict_words(two).headline.endswith(
        "Missing from your profile entirely: 2 things, listed below."
    )
    assert verdict_words(one).matches == "1 clearly shown."


def test_verdict_words_name_missing_parts():
    # W-2: a compound line with a part that has no evidence is a close
    # line, and the part is named among the uncovered; a whole line keeps its claims sentence.
    from loomcv.review.words import verdict_words

    rows = [_job_row("j1", name="n1", employer="E1", title="T1")]
    compound = "Build Go and Node services"
    payload = _payload(
        lines=[
            _line("Plain thing", "must", "exact"),
            _line(compound, "must", "exact", ["Go"], parts=[("Go", "exact"), ("Node", "none")]),
            _line("Kube ops", "must", "none"),
        ],
        claims=[_claim("Kube ops", [("Kubernetes", ["j1"])])],
    )

    words = verdict_words(payload, rows)

    assert words.close == [compound]
    assert words.uncovered == [
        "Node — no bullet in your profile backs this up. "
        f'The posting asks for it here: "{compound}"',
        "Kube ops — listed at E1, T1, but no bullet shows it",
    ]
    assert words.headline.startswith("Partial fit — this posting requires 3 things.")


def test_verdict_words_with_parts_carry_no_grade_percent_id_or_kind():
    # W-3
    import re

    from loomcv.review.words import verdict_words

    payload = _payload(
        lines=[
            _line("Plain thing", "must", "exact"),
            _line("Build Go and Node services", "must", "exact", parts=[("Node", "none")]),
            _line("Kube ops", "must", "adjacent"),
            _line("Side thing", "nice", "exact", parts=[("Rust", "none")]),
        ]
    )
    words = verdict_words(payload, [_job_row("j1", name="n1", employer="E1", title="T1")])
    for text in [
        words.fit or "",
        words.headline,
        words.matches or "",
        words.nice or "",
        *words.close,
        *words.uncovered,
    ]:
        assert not re.search(r"\b[A-D]\b", text), text
        assert "%" not in text and "::" not in text, text
        assert not re.search(r"\bj\d\b", text), text
        for kind in ("(must)", "(nice)", "(responsibility)"):
            assert kind not in text, text


def test_verdict_words_without_required_items():
    # W-2
    import re

    from loomcv.review.words import verdict_words

    words = verdict_words(_payload(lines=[_line("Nice one", "nice", "exact")]))
    assert words.fit is None and words.matches is None and words.nice is None
    assert words.close == [] and words.uncovered == []
    assert "couldn't tell" in words.headline
    assert not re.search(r"\d", words.headline)


def test_verdict_words_count_direct_and_close_and_name_the_close_ones():
    # W-3
    from loomcv.review.words import verdict_words

    payload = _payload(
        lines=[
            _line("Alpha line", "must", "exact"),
            _line("Beta line", "must", "adjacent"),
            _line("Side line", "nice", "adjacent"),
            _line("Gamma line", "must", "adjacent"),
            _line("Delta line", "must", "none"),
        ]
    )
    words = verdict_words(payload)
    assert words.headline == (
        "Partial fit — this posting requires 4 things. Your profile clearly shows 1 and shows "
        "something similar for 2. Missing from your profile entirely: 1 thing, listed below."
    )
    assert words.close == ["Beta line", "Gamma line"]
    assert words.matches == (
        "1 clearly shown; 2 only something similar (expect interview questions on those): "
        "Beta line; Gamma line"
    )
    assert "Alpha line" not in words.matches and "Side line" not in words.matches


def test_verdict_words_turn_claims_into_sentences():
    # W-4
    from loomcv.review.words import verdict_words

    rows = [_job_row("j1", name="n1", employer="E1", title="T1")]
    payload = _payload(
        lines=_must(
            [
                ("Kube ops", "none"),
                ("Terraform work", "none"),
                ("Rust work", "none"),
                ("Haskell work", "none"),
            ]
        ),
        claims=[
            _claim("Kube ops", [("Kubernetes", ["skills"])]),
            _claim("Terraform work", [("Terraform", ["j1"])]),
            _claim("Rust work", [("Rust", ["j1", "skills"])]),
        ],
    )
    skills, job, both, plain = verdict_words(payload, rows).uncovered
    assert skills.startswith("Kube ops") and "skills" in skills and "My profile" in skills
    assert "E1, T1" not in skills
    assert job.startswith("Terraform work") and "E1, T1" in job and "skills" not in job
    assert both.startswith("Rust work") and "E1, T1" in both and "skills" in both
    assert plain == "Haskell work — no bullet in your profile backs this up."
    assert all("j1" not in s for s in (skills, job, both, plain))


def test_verdict_words_count_nice_to_haves():
    # W-5
    from loomcv.review.words import verdict_words

    payload = _payload(
        lines=[
            _line("Must one", "must", "exact"),
            _line("Nice one", "nice", "exact"),
            _line("Nice two", "nice", "adjacent"),
            _line("Nice three", "nice", "none"),
            _line("Duty one", "responsibility", "none"),
        ]
    )
    assert verdict_words(payload).nice == "Nice-to-haves your profile shows: 2 of 3."
    assert verdict_words(_payload(lines=_must([("Must one", "exact")]))).nice is None


def test_verdict_words_carry_no_grade_percent_id_or_kind():
    # W-6
    import re

    from loomcv.review.words import verdict_words

    s = _scenario()
    rows = [*s.rows]
    payloads = [
        s.payload,
        _payload(lines=_must([("L1", "exact"), ("L2", "adjacent"), ("L3", "none")])),
        _payload(lines=[_line("Nice one", "nice", "exact")]),
    ]
    for payload in payloads:
        words = verdict_words(payload, rows)
        strings = [
            words.fit or "",
            words.headline,
            words.matches or "",
            words.nice or "",
            *words.close,
            *words.uncovered,
        ]
        for text in strings:
            assert not re.search(r"\b[A-D]\b", text), text
            assert "%" not in text and "::" not in text, text
            assert not re.search(r"\bj\d\b", text), text
            for kind in ("(must)", "(nice)", "(responsibility)"):
                assert kind not in text, text
    assert verdict_words(s.payload, rows).fit == "Partial fit"  # 2 of 3 → 66 → C


def test_report_body_starts_at_section_one():
    # R-1
    from loomcv.draft import render_report
    from loomcv.review.words import report_body

    s = _scenario()
    text = render_report(s.payload, s.rows)
    assert "## 0 Verdict" in text
    body = report_body(text)
    assert body == text[text.index("## 1 Each item in this posting") :]
    assert "## 0" not in body


def test_report_body_without_section_zero_is_unchanged():
    # R-2
    from loomcv.review.words import report_body

    text = "# Report\n\n## 1 Coverage\n\nrow\n\n## 2 Gap Report\n\nnone\n"
    assert report_body(text) == text
    assert report_body("") == ""


def test_report_body_leaves_the_file_alone(tmp_path):
    # R-3
    from loomcv.draft import render_report
    from loomcv.review.words import report_body

    s = _scenario()
    path = tmp_path / "match_report.md"
    path.write_text(render_report(s.payload, s.rows), encoding="utf-8")
    before = path.read_bytes()
    report_body(path.read_text(encoding="utf-8"))
    assert path.read_bytes() == before


def test_covers_words_drop_the_kind():
    # V-1
    from loomcv.review.doc import coverage_of
    from loomcv.review.words import covers_words

    s = _scenario()
    covered = coverage_of(_doc(s), s.payload, s.candidates)
    assert covers_words(covered["j0::c1"]) == [L_CACHING]
    assert covers_words(covered["j0::c2"]) == [L_SEARCH]
    assert covers_words(["Plain line"]) == ["Plain line"]
    assert covers_words(["Ops (nice)", "Run it (responsibility)"]) == ["Ops", "Run it"]


def test_covers_words_cut_a_long_line():
    # V-2
    from loomcv.review.words import COVERS_CUT, covers_words

    assert COVERS_CUT == 60
    exactly = "x" * 60
    longer = "y" * 61
    assert covers_words([f"{exactly} (must)"]) == [exactly]
    assert covers_words([f"{longer} (must)"]) == ["y" * 60 + "…"]
    assert covers_words(["z" * 30 + " (must)"], cut=10) == ["z" * 10 + "…"]


def test_covers_words_keep_order_and_duplicates():
    # V-3
    from loomcv.review.words import covers_words

    labels = ["Second (must)", "First (nice)", "Second (must)"]
    assert covers_words(labels) == ["Second", "First", "Second"]


def test_edited_bullets_empty_for_the_draft():
    # E-1
    from loomcv.review.doc import edited_bullets

    s = _scenario()
    assert edited_bullets(_doc(s), s.candidates) == set()


def test_edited_bullets_name_an_edited_bullet():
    # E-2
    from loomcv.review.doc import edit, edited_bullets

    s = _scenario()
    doc = edit(_doc(s), "j0", 0, "Cut latency by a lot.")
    assert edited_bullets(doc, s.candidates) == {"j0::c1"}
    back = edit(doc, "j0", 0, s.c.c1.text)
    assert edited_bullets(back, s.candidates) == set()


def test_edited_bullets_never_name_a_bullet_without_a_candidate():
    # E-3
    from loomcv.review.doc import edit, edited_bullets

    s = _scenario()
    doc = edit(_doc(s), "j0", 0, "Cut latency by a lot.")
    others = [c for c in s.candidates if c.bullet_id != "j0::c1"]
    assert edited_bullets(doc, others) == set()
    assert edited_bullets(doc, []) == set()


def test_job_rows_print_text_covers_and_edited_in_document_order():
    # J-1
    from loomcv.review.doc import edit, move
    from loomcv.review.words import job_rows

    s = _scenario()
    doc = edit(_doc(s), "j0", 0, "Cut latency by a lot.")
    doc = move(doc, "j0", 0, 1)
    rows = job_rows(doc, "j0", s.payload, s.candidates)
    assert [r.bullet_id for r in rows] == ["j0::c2", "j0::c1"]
    assert [r.text for r in rows] == [s.c.c2.text, "Cut latency by a lot."]
    assert [r.covers for r in rows] == [[L_SEARCH], [L_CACHING]]
    assert [r.edited for r in rows] == [False, True]
    assert job_rows(doc, "j2", s.payload, s.candidates) == []


def test_stage_label_says_where_the_application_stands():
    # L-1
    from loomcv.application import Status
    from loomcv.review.store import stage_label

    def label(status, **kw):
        return stage_label(Status(app_id="a", status=status, **kw))

    assert label("reviewing") == "Draft ready — review it"
    assert label("done", pages=2) == "✓ PDF ready · 2 pages"
    assert label("done", pages=1) == "✓ PDF ready · 1 page"
    assert label("done") == "✓ PDF ready"
    assert label("finalizing") == "PDF not finished — make it again"
    assert label("failed_at_match") == "Couldn't build a draft — start a new application"
    assert label("failed_at_draft") == "Couldn't build a draft — start a new application"
    assert label("unknown") == "(no record)"
    assert label("something_new") == "(no record)"


def test_entry_words_and_the_tally():
    # L-2
    from datetime import datetime

    from loomcv.application import Status
    from loomcv.review.store import ListEntry
    from loomcv.review.words import entry_words, tally_words

    named = ListEntry(
        app_id="fennimore-web-platform",
        title="Senior Web Platform Engineer",
        name="Fennimore web platform",
        started=datetime(2026, 11, 8, 14, 30),
        status=Status(app_id="fennimore-web-platform", status="reviewing"),
    )
    words = entry_words(named)
    assert words.title == "Senior Web Platform Engineer"
    assert words.sub == "Fennimore web platform · started Nov 8"
    assert words.stage == "Draft ready — review it"
    assert words.done is False

    bare = ListEntry(
        app_id="jd",
        title="jd",
        name=None,
        started=None,
        status=Status(app_id="jd", status="done", pages=1),
    )
    words = entry_words(bare)
    assert words.sub == "jd"
    assert words.stage == "✓ PDF ready · 1 page"
    assert words.done is True

    third = ListEntry(
        app_id="x",
        title="x",
        name="x",
        started=datetime(2026, 1, 3),
        status=Status(app_id="x", status="failed_at_match"),
    )
    assert entry_words(third).sub == "x · started Jan 3"
    assert tally_words([named, bare, third]) == "3 so far · 1 PDF ready"
    assert tally_words([]) == "0 so far · 0 PDF ready"


def test_pdf_words_within_the_limit():
    # P-1
    from loomcv.application import Status
    from loomcv.review.words import pdf_words

    one = Status(
        app_id="a",
        status="done",
        final_pdf="/x/final_resume.pdf",
        pages=1,
        page_limit=2,
        over_limit=False,
    )
    words = pdf_words(one)
    assert words.ok
    assert words.line == "PDF ready — 1 page, within the limit."
    assert words.pdf == "/x/final_resume.pdf"
    assert words.notes == []
    two = one.model_copy(update={"pages": 2})
    assert pdf_words(two).line == "PDF ready — 2 pages, within the limit."


def test_pdf_words_over_the_limit():
    # P-2
    from loomcv.application import Status
    from loomcv.review.words import pdf_words

    over = Status(
        app_id="a",
        status="done",
        final_pdf="/x/final_resume.pdf",
        pages=3,
        page_limit=2,
        over_limit=True,
    )
    words = pdf_words(over)
    assert not words.ok
    assert "3 pages" in words.line and "limit is 2" in words.line
    assert words.pdf == "/x/final_resume.pdf"


def test_pdf_words_notes_and_nothing_before_a_pdf():
    # P-3
    from loomcv.application import Status
    from loomcv.review.words import pdf_words

    warned = Status(
        app_id="a",
        status="done",
        final_pdf="/x/final_resume.pdf",
        pages=1,
        page_limit=2,
        over_limit=False,
        typst_warnings=["font fallback", "unknown glyph"],
    )
    words = pdf_words(warned)
    assert len(words.notes) == 2
    assert "font fallback" in words.notes[0] and "unknown glyph" in words.notes[1]
    for text in (words.line, *words.notes):
        assert "pages=" not in text and "status=" not in text and "page_limit" not in text
    assert pdf_words(Status(app_id="a", status="reviewing")) is None
    assert pdf_words(Status(app_id="a", status="finalizing")) is None
    assert pdf_words(Status(app_id="a", status="done")) is None  # no PDF on record


def test_built_words_count_the_picked_bullets():
    # P-4
    from loomcv.application import Status
    from loomcv.review.words import built_words

    def built(counts):
        return built_words(Status(app_id="a", status="reviewing", counts=counts))

    assert built({"jobs": 3, "bullets": 4, "skills": 2}) == "Draft ready — 4 bullets picked"
    assert built({"bullets": 1}) == "Draft ready — 1 bullet picked"
    assert built({}) == "Draft ready"


# ------------------------------------------ the review document's slots (R-4 … R-8)


def test_l3a_vault_fixed_slots_have_no_setter():
    # L3a R-4 — Name / Contact / Education are printed from the vault: no setter, no commit
    # parameter, carried through `commit` unchanged.
    import inspect

    from loomcv.review import doc as review_doc
    from loomcv.review.doc import commit, layout

    for name in ("set_name", "set_contact", "set_education", "set_header", "set_skills"):
        assert not hasattr(review_doc, name), name
    params = set(inspect.signature(commit).parameters)
    assert not params & {"name", "contact", "education", "header", "skills"}

    doc = _doc(_scenario())
    values = [v for j in doc.jobs for b in j.bullets for v in (b.text, b.project or "")]
    changed = commit(doc, layout(doc), "X", "Y", [g.items for g in doc.skill_groups], values)
    assert (changed.name, changed.contact, changed.education) == (
        "Fictional Person",
        ["Example City", "fictional@example.invalid"],
        ["Example University — BSc, 2014"],
    )


def test_l3a_review_doc_json_round_trip_every_field():
    # L3a R-6
    from loomcv.review.doc import ReviewDoc

    doc = _doc(_scenario())
    assert doc.name and doc.headline and doc.contact and doc.summary and doc.education
    assert [g.name for g in doc.skill_groups] == ["", "Languages"]
    assert ReviewDoc.model_validate_json(doc.model_dump_json()) == doc


def test_l3a_old_shape_review_json_is_rebuilt_with_a_notice(app):
    # R-7 — an older-shape `review.json` (header / skills keys) is not migrated:
    # the document comes from the draft, a notice says so, nothing on disk changes.
    import json

    from loomcv.review.doc import render

    app.start()
    folder = app.apps / "jd"
    fresh = _open(app)
    old = {"header": "# Old Person\n", "skills": ["Go"], "jobs": []}
    (folder / "review.json").write_text(json.dumps(old), encoding="utf-8")
    state_before = (folder / "review.json").read_bytes()
    draft_before = (folder / "tailored_resume.md").read_bytes()

    opened = _open(app)

    assert opened.notice
    assert opened.doc == fresh.doc
    assert render(opened.doc, opened.template_path).encode("utf-8") == draft_before
    assert (folder / "review.json").read_bytes() == state_before
    assert (folder / "tailored_resume.md").read_bytes() == draft_before


def test_l3a_save_and_submit_carry_the_five_sections(tmp_path, monkeypatch, capsys):
    # L3a R-8 — the real pandoc / typst chain over a vault with the header block.
    from loomcv import application
    from loomcv.review.doc import render, set_headline

    app = _App(tmp_path, monkeypatch, capsys, header_block=True)
    app.start()
    opened = _open(app)
    doc = set_headline(opened.doc, "Payments Platform Engineer")
    saved = _save(app, doc, opened)
    assert saved.resume_path.read_text(encoding="utf-8") == render(doc, opened.template_path)

    status = application.submit(
        app.runtime(),
        "jd",
        edited_path=saved.resume_path,
        out_root=app.apps,
        checkpoints=app.checkpoints,
    )

    assert status.status == "done"
    assert status.pages == 1
    final = (app.apps / "jd" / "final_resume.md").read_text(encoding="utf-8")
    assert final.startswith("# Fictional Person\n")
    assert "**Payments Platform Engineer**" in final
    for heading in ("## Summary", "## Skills", "## Experience", "## Education"):
        assert heading in final
    assert "src:" not in final


def test_verdict_words_shorten_long_lines_and_group_parts():
    # A requirement quoted in a card sentence is cut
    # at a word boundary with "…" and loses its final period; two missing parts of one line make
    # one sentence ("them"), and the headline counts both parts.
    from loomcv.draft.verdict import CUT
    from loomcv.review.words import verdict_words

    long_line = (
        "Hands-on experience building and deploying production widget systems, including alpha "
        "pipelines, beta tooling and gamma frameworks."
    )
    payload = _payload(
        lines=[
            _line("Plain thing", "must", "exact"),
            _line(
                long_line,
                "must",
                "exact",
                parts=[("alpha pipelines", "none"), ("beta tooling", "none"), ("gamma", "exact")],
            ),
        ]
    )

    words = verdict_words(payload)

    assert words.headline.endswith("Missing from your profile entirely: 2 things, listed below.")
    quoted = words.matches.split(": ", 1)[1]
    assert quoted.endswith("…") and len(quoted) <= CUT + 1 and not quoted.endswith(".…")
    assert long_line.startswith(quoted[:-1])
    assert len(words.uncovered) == 1
    sentence = words.uncovered[0]
    assert sentence.startswith(
        "alpha pipelines and beta tooling — no bullet in your profile backs these up. "
        'The posting asks for them here: "'
    )
    assert sentence.endswith('…"')
    assert long_line not in sentence


def test_card_note_says_ai_guide_edits_and_new_application():
    # The note under the verdict card — AI-made and fallible, a
    # guide; edits make the PDF but never change the rating; a new assessment is a new application.
    # No letter, percentage or kind token, as on the rest of the card.
    import re

    from loomcv.review.words import CARD_NOTE

    for phrase in (
        "generated by AI",
        "may contain mistakes",
        "the PDF is made from your edited version",
        "does not change the fit rating",
        "start a new application with this posting",
    ):
        assert phrase in CARD_NOTE
    assert "%" not in CARD_NOTE
    assert not re.search(r"\b(exact|adjacent|none|must|nice|[A-D])\b", CARD_NOTE)
