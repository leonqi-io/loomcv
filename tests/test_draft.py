"""Tests for the deterministic render behind `pipeline draft`.

Cases: render D-1 .. D-9, report P-1 .. P-8. Everything is synthetic: payloads are built in-test
from the Pydantic models and **round-tripped through JSON**, so every case also exercises
the `pass` key of `SelectedBullet` / `WalkStep`; index shapes come from the
three committed fixtures (read-only) or are built as `JobRow`s. No network, no model call:
the only adapter is an in-process fake embedding adapter used to build an index.

Project imports are inside each test (as `test_cli.py` does with `main`) so every case
fails on its own at red instead of one collection error.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from loomcv import resources
from loomcv.sync.index_assets import index_vault
from loomcv.sync.parse_assets import parse_star_vault

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _generation_yaml(apps: Path, *, template: Path | None = None) -> str:
    """A tmp `generation.yaml`: the four keys, absolute paths to the repo's
    committed template and pandoc defaults so the case does not depend on the working directory."""
    template = template or resources.path("templates", "resume_default.md")
    return (
        f"output_dir: {apps}\n"
        f"resume_template: {template}\n"
        f"pandoc_defaults: {resources.path('pandoc_resume.yaml')}\n"
        "page_limit: 2\n"
    )


@pytest.fixture(autouse=True)
def _no_default_embedding_function(monkeypatch):
    """No code path may ever call Chroma's default (ONNX) embedding fn."""
    import chromadb.utils.embedding_functions as ef

    def _raise(*_args, **_kwargs):
        raise RuntimeError("default Chroma embedding function must never be called")

    for cls_name in ("DefaultEmbeddingFunction", "ONNXMiniLM_L6_V2"):
        cls = getattr(ef, cls_name, None)
        if cls is not None:
            monkeypatch.setattr(cls, "__call__", _raise)


class FakeEmbeddingAdapter:
    """Deterministic, network-free stand-in for `EmbeddingAdapter`."""

    model_id = "fake-model"
    vector_space_version = 1

    def embed(self, texts):
        return [list(hashlib.sha1(t.encode("utf-8")).digest()[:8]) for t in texts]


# ------------------------------------------------------------------ helpers


def _index(tmp_path: Path, fixture: str, *, include_draft: bool = False):
    """Parse + index a fixture under `tmp_path`; returns `(db, parse_result)`."""
    text = (FIXTURES / fixture).read_text(encoding="utf-8")
    pr = parse_star_vault(text, include_draft=include_draft)
    assert pr.errors() == []
    db, chroma = tmp_path / "db.sqlite", tmp_path / "chroma"
    index_vault(pr, db_path=db, chroma_path=chroma, adapter=FakeEmbeddingAdapter())
    return db, pr


def _job_row(
    job_id="j0",
    *,
    name="Acme — Engineer",
    employer="Acme",
    title="Engineer",
    date_from="2020-01",
    date_to="2021-12",
    job_order=0,
):
    from loomcv.retrieve.search import JobRow

    return JobRow(
        job_id=job_id,
        name=name,
        employer=employer,
        title=title,
        date_from=date_from,
        date_to=date_to,
        job_order=job_order,
    )


def _bullet(
    bullet_id,
    text,
    *,
    project=None,
    line="Line one",
    kind="must",
    pass_number=1,
    walk=1,
    overlap=1,
    rerank_rank=1,
    rrf_position=1,
    covers=(),
):
    from loomcv.retrieve.rank import SelectedBullet

    return SelectedBullet(
        bullet_id=bullet_id,
        project=project,
        text=text,
        line=line,
        kind=kind,
        pass_number=pass_number,
        walk_position=walk,
        overlap=overlap,
        rerank_rank=rerank_rank,
        rrf_position=rrf_position,
        covers=list(covers),
    )


def _job_sel(job_id, job, order, bullets, *, cap=None):
    from loomcv.retrieve.rank import JobSelection

    return JobSelection(
        job_id=job_id,
        job=job,
        job_order=order,
        cap=len(bullets) if cap is None else cap,
        placed=len(bullets),
        bullets=list(bullets),
    )


def _line(text, kind="must", grade="exact", mapped=(), parts=()):
    """`parts` = `(term, match)` pairs; empty → the field is not passed at all."""
    from loomcv.retrieve.rank import LineResult

    extra = {}
    if parts:
        from loomcv.state import CoveragePart

        extra["parts"] = [CoveragePart(term=t, match=m) for t, m in parts]
    return LineResult(
        text=text,
        kind=kind,
        grade=grade,
        mapped_keywords=list(mapped),
        ranked=[],
        max_cosine=None,
        max_bm25=None,
        top_rrf_score=None,
        **extra,
    )


def _claim(line, keywords, *, kind="must", match="exact"):
    from loomcv.retrieve.rank import Claim, ClaimLine

    return ClaimLine(
        line=line,
        kind=kind,
        match=match,
        keywords=[Claim(keyword=k, at=list(at)) for k, at in keywords],
    )


def _payload(
    *,
    jobs=(),
    lines=(),
    gaps=(),
    claims=(),
    unplaced=(),
    uncovered=(),
    covered_by=None,
    total_slots=14,
    passes=1,
    title="Fake Platform Engineer",
):
    """A `MatchPayload` built from the models, then dumped to JSON and read back."""
    from loomcv.retrieve.rank import (
        MatchDiagnostics,
        MatchPayload,
        Selection,
        WalkStep,
    )
    from loomcv.state import JDProfile, RetrievalConfig

    steps = sorted(
        ((b.walk_position, j.job_id, b) for j in jobs for b in j.bullets), key=lambda t: t[0]
    )
    walk = [
        WalkStep(
            bullet_id=b.bullet_id,
            job_id=job_id,
            line=b.line,
            kind=b.kind,
            pass_number=b.pass_number,
        )
        for _, job_id, b in steps
    ]
    selection = Selection(
        total_slots=total_slots,
        jobs=list(jobs),
        walk=walk,
        unplaced=list(unplaced),
        covered_by=dict(covered_by or {}),
        uncovered=list(uncovered),
        passes=passes,
    )
    built = MatchPayload(
        profile=JDProfile(
            title=title, seniority="senior", domain="web", must_have=[ln.text for ln in lines]
        ),
        parse_warnings=[],
        requirements=list(lines),
        gaps=list(gaps),
        claims=list(claims),
        selection=selection,
        config=RetrievalConfig(
            top_k=5, rrf_k=60, total_slots=total_slots, recency_shares=[3, 2, 1]
        ),
        diagnostics=MatchDiagnostics(
            vector_missing=0,
            rerank_pairs=0,
            rerank_model="fake-reranker",
            coverage_model="fake-coverage",
            coverage_calls=0,
            claims_model=None,
            claims_calls=0,
        ),
    )
    dumped = json.dumps(built.model_dump(mode="json"))
    return MatchPayload.model_validate_json(dumped)


def _group(name, *items):
    from loomcv.sync.parse_assets import SkillGroup

    return SkillGroup(name=name, items=list(items))


def _payload_json(payload) -> str:
    return json.dumps(payload.model_dump(mode="json"))


def _resume_bullet_lines(resume: str) -> list[str]:
    return [ln for ln in resume.splitlines() if ln.startswith("- ")]


def _src_ids(resume: str) -> list[str]:
    return re.findall(r"<!-- src: (.+?) -->", resume)


def _section(report: str, heading: str) -> str:
    """The text between `heading` and the next `## ` heading (sub-headings stay inside)."""
    start = report.index(heading + "\n") + len(heading)
    rest = report[start:]
    nxt = rest.find("\n## ")
    return rest if nxt == -1 else rest[:nxt]


def _subsection(report: str, heading: str) -> str:
    """The text between a `### ` heading and the next `## ` / `### ` heading."""
    start = report.index(heading + "\n") + len(heading)
    rest = report[start:]
    cuts = [i for i in (rest.find("\n## "), rest.find("\n### ")) if i != -1]
    return rest if not cuts else rest[: min(cuts)]


_COUNTS_TAIL = (
    "Only the required items count toward the rating at the top; these are here for your "
    "information."
)


# ------------------------------------------------------------------- render


def test_render_three_jobs_placed_bullets_verbatim_with_src():
    from loomcv.draft import render_resume

    rows = [
        _job_row("j0", name="n0", employer="E0", title="T0", job_order=0),
        _job_row("j1", name="n1", employer="E1", title="T1", job_order=1),
        _job_row("j2", name="n2", employer="E2", title="T2", job_order=2),
    ]
    sizes = (7, 5, 2)
    walk = 0
    selections, expected = [], []
    for row, size in zip(rows, sizes, strict=True):
        bullets = []
        for i in range(size):
            walk += 1
            text = f"Did fictional thing {walk}."
            if walk == 3:
                text = "Cut  latency by half."  # inner double space survives byte-for-byte
            bullets.append(_bullet(f"{row.job_id}::b{i}", text, walk=walk))
            expected.append(f"- {text} <!-- src: {row.job_id}::b{i} -->")
        selections.append(_job_sel(row.job_id, row.name, row.job_order, bullets))
    payload = _payload(jobs=selections)
    assert '"pass"' in _payload_json(payload)

    resume = render_resume(payload, rows, [])

    headings = [ln for ln in resume.splitlines() if ln.startswith("### ")]
    assert headings == ["### T0 — E0", "### T1 — E1", "### T2 — E2"]
    lines = _resume_bullet_lines(resume)
    assert len(lines) == 14
    assert lines == expected


def test_render_bullets_in_walk_order_not_list_order():
    from loomcv.draft import render_resume

    bullets = [
        _bullet("j0::c", "Third.", walk=3),
        _bullet("j0::a", "First.", walk=1),
        _bullet("j0::d", "Fourth.", walk=4),
        _bullet("j0::b", "Second.", walk=2),
    ]
    payload = _payload(jobs=[_job_sel("j0", "Acme — Engineer", 0, bullets)])

    resume = render_resume(payload, [_job_row("j0")], [])

    assert _src_ids(resume) == ["j0::a", "j0::b", "j0::c", "j0::d"]


def test_render_bulletless_job_heading_and_dates_in_job_order(tmp_path):
    from loomcv.draft import render_resume
    from loomcv.retrieve.search import load_job_rows

    db, pr = _index(tmp_path, "claims_expected_vault.md", include_draft=True)
    harrowgate, pellworth, tallis = pr.jobs
    assert tallis.bullets == []
    payload = _payload(
        jobs=[
            _job_sel(
                harrowgate.job_id,
                harrowgate.name,
                0,
                [_bullet(harrowgate.bullets[0].bullet_id, harrowgate.bullets[0].text, walk=1)],
            ),
            _job_sel(
                pellworth.job_id,
                pellworth.name,
                1,
                [_bullet(pellworth.bullets[0].bullet_id, pellworth.bullets[0].text, walk=2)],
            ),
        ]
    )

    resume = render_resume(payload, load_job_rows(db), [])

    lines = resume.splitlines()
    headings = [ln for ln in lines if ln.startswith("### ")]
    # A job with no placed bullet prints like any other job —
    # heading + date line, no list — in job order.
    assert [h.split(" — ")[1] for h in headings] == [
        "Harrowgate Mutual Insurance",
        "Pellworth General Insurance",
        "Tallis & Reed Loss Adjusting",
    ]
    assert lines[lines.index(headings[2]) + 1] == "*Sep 2011 – Dec 2013*"
    tail = lines[lines.index(headings[2]) :]
    assert not [ln for ln in tail if ln.startswith("- ")]


def test_render_project_prefix():
    from loomcv.draft import render_resume

    bullets = [
        _bullet("j0::a", "With a label.", project="Helios Player Rewrite", walk=1),
        _bullet("j0::b", "Without a label.", project=None, walk=2),
    ]
    payload = _payload(jobs=[_job_sel("j0", "Acme — Engineer", 0, bullets)])

    lines = _resume_bullet_lines(render_resume(payload, [_job_row("j0")], []))

    assert lines == [
        "- **Helios Player Rewrite:** With a label. <!-- src: j0::a -->",
        "- Without a label. <!-- src: j0::b -->",
    ]


def test_render_header_verbatim_or_absent():
    # The header is the vault's fields, printed verbatim into their slots.
    from loomcv.draft import render_resume
    from loomcv.sync.parse_assets import VaultHeader

    payload = _payload(jobs=[_job_sel("j0", "n", 0, [_bullet("j0::a", "Only bullet.", walk=1)])])
    rows = [_job_row("j0")]

    header = VaultHeader(name="Jane Example", contact=["jane@example.invalid"])
    resume = render_resume(payload, rows, [], header)
    before_experience = resume.split("## Experience")[0]
    assert before_experience == "# Jane Example\n\njane@example.invalid\n\n"
    # empty Skills: `## Experience` right after the header's blank line
    assert resume.startswith("# Jane Example\n\njane@example.invalid\n\n## Experience\n")

    assert render_resume(payload, rows, []).startswith("## Experience\n")
    assert render_resume(payload, rows, [], VaultHeader()).startswith("## Experience\n")


def test_render_dates_present_unknown_and_name_fallback():
    from loomcv.draft import render_resume

    rows = [
        _job_row("j0", employer="E0", title="T0", date_from="2023-01", date_to="present"),
        _job_row("j1", employer="E1", title="T1", date_from="0000", date_to="0000", job_order=1),
        _job_row("j2", name="Name Two", employer=None, title="T2", job_order=2),
        _job_row("j3", name="Name Three", employer="E3", title=None, job_order=3),
    ]
    resume = render_resume(_payload(jobs=[]), rows, [])
    lines = resume.splitlines()

    # No placed bullet → the same heading + date line, no list.
    for heading, dates in [
        ("### T0 — E0", "*Jan 2023 – Present*"),
        ("### T1 — E1", "*0000 – 0000*"),
        ("### Name Two", "*Jan 2020 – Dec 2021*"),
        ("### Name Three", "*Jan 2020 – Dec 2021*"),
    ]:
        assert lines[lines.index(heading) + 1] == dates
    assert not [ln for ln in lines if ln.startswith("- ")]

    placed = _payload(jobs=[_job_sel("j0", "n", 0, [_bullet("j0::a", "Only bullet.", walk=1)])])
    with_bullet = render_resume(placed, rows[:1], []).splitlines()
    assert with_bullet[with_bullet.index("### T0 — E0") + 1] == "*Jan 2023 – Present*"


def test_render_skills_line_verbatim_nothing_added_or_removed(tmp_path):
    from loomcv.draft import render_resume
    from loomcv.retrieve.search import load_claims_inputs, load_job_rows, load_skill_groups

    db, pr = _index(tmp_path, "claims_expected_vault.md", include_draft=True)
    expected = pr.skills
    assert len(expected) == 14
    assert "Kubernetes" not in expected
    assert "Xactimate" in expected
    skills = load_claims_inputs(db)[1]
    assert skills == expected

    first = pr.jobs[0]
    payload = _payload(
        jobs=[
            _job_sel(
                first.job_id,
                first.name,
                0,
                [_bullet(first.bullets[0].bullet_id, "A plain fictional sentence.", walk=1)],
            )
        ],
        lines=[_line("Run Kubernetes clusters", mapped=["Kubernetes"])],
        claims=[_claim("Run Kubernetes clusters", [("Kubernetes", ["skills"])])],
    )

    resume = render_resume(payload, load_job_rows(db), load_skill_groups(db))

    # The flat line is one unnamed group, its items joined by "; ".
    section = "## Skills\n\n" + "; ".join(expected) + "\n"
    assert section in resume
    assert section + "\n## Experience\n" in resume
    assert "Kubernetes" not in resume
    assert "Xactimate" in resume

    (tmp_path / "second").mkdir()
    db2, _pr2 = _index(tmp_path / "second", "index_vault.md")
    assert load_claims_inputs(db2)[1] == []
    plain = render_resume(_payload(jobs=[]), load_job_rows(db2), load_skill_groups(db2))
    assert "## Skills" not in plain


def test_stale_payload_raises_and_cli_exits_1(tmp_path, capsys):
    from loomcv.cli import main
    from loomcv.draft import StaleMatchError, check_fresh
    from loomcv.retrieve.search import load_candidates

    db, _pr = _index(tmp_path, "index_vault.md")
    indexed = {c.bullet_id: c.text for c in load_candidates(db)}
    everything = parse_star_vault(
        (FIXTURES / "index_vault.md").read_text(encoding="utf-8"), include_draft=True
    )
    draft_id = next(b.bullet_id for j in everything.jobs for b in j.bullets if b.status == "draft")
    real_id, real_text = next(iter(indexed.items()))

    def selection(bullet_id, text):
        return _payload(jobs=[_job_sel("j0", "n", 0, [_bullet(bullet_id, text, walk=1)])])

    # fresh: no raise
    check_fresh(selection(real_id, real_text), indexed)

    # (a) an id the index does not have
    with pytest.raises(StaleMatchError) as exc_a:
        check_fresh(selection("nojob::zzz999", "Whatever."), indexed)
    assert "nojob::zzz999" in str(exc_a.value)
    assert "re-run match" in str(exc_a.value)

    # (b) a real id, different text
    with pytest.raises(StaleMatchError) as exc_b:
        check_fresh(selection(real_id, real_text + " (edited)"), indexed)
    assert real_id in str(exc_b.value)
    assert "re-run match" in str(exc_b.value)

    # a draft bullet is not among the confirmed ones
    assert draft_id not in indexed
    with pytest.raises(StaleMatchError):
        check_fresh(selection(draft_id, "Drafted a fictional schema redesign proposal."), indexed)

    # the CLI maps it to exit 1 and creates nothing
    match = tmp_path / "match.json"
    match.write_text(_payload_json(selection("nojob::zzz999", "Whatever.")), encoding="utf-8")
    apps = tmp_path / "apps"
    cfg = tmp_path / "generation.yaml"
    cfg.write_text(_generation_yaml(apps), encoding="utf-8")

    rc = main(["draft", str(match), "--db", str(db), "--generation-config", str(cfg)])

    assert rc == 1
    assert "re-run match" in capsys.readouterr().err
    assert not apps.exists()


def test_render_prints_no_keywords_diagnostics_or_percent(tmp_path):
    from loomcv.draft import render_resume
    from loomcv.retrieve.search import load_candidates, load_job_rows

    db, _pr = _index(tmp_path, "metadata_vault.md")
    candidates = load_candidates(db)
    rows = load_job_rows(db)
    by_job: dict[str, list] = {}
    for walk, c in enumerate(candidates, start=1):
        by_job.setdefault(c.job_id, []).append(
            _bullet(c.bullet_id, c.text, project=c.project, walk=walk)
        )
    names = {c.job_id: c.job for c in candidates}
    selections = [
        _job_sel(r.job_id, names[r.job_id], r.job_order, by_job[r.job_id])
        for r in rows
        if r.job_id in by_job
    ]
    payload = _payload(jobs=selections)

    resume = render_resume(payload, rows, [])

    vault_strings = " ".join(
        [c.text for c in candidates]
        + [c.project or "" for c in candidates]
        + [f"{r.name} {r.employer} {r.title} {r.date_from} {r.date_to}" for r in rows]
    ).casefold()
    checked = 0
    for c in candidates:
        for kw in c.keywords:
            if kw.casefold() not in vault_strings:
                checked += 1
                assert kw.casefold() not in resume.casefold()
    assert checked > 0
    for name in (
        "max_cosine",
        "max_bm25",
        "top_rrf_score",
        "rerank_rank",
        "rrf_position",
        "overlap",
    ):
        assert name not in resume
    stripped = resume
    for c in candidates:
        stripped = stripped.replace(c.text, "")
    assert "%" not in stripped
    assert any("%" in c.text for c in candidates)


# ----------------------------------------------------------------- template
# T-1 .. T-5: `render_resume` renders through the Jinja template named by
# `generation.yaml` `resume_template` (default `loomcv/resources/templates/resume_default.md`).


def test_template_default_skills_before_experience_bullets_verbatim():
    from loomcv.draft import render_resume

    rows = [
        _job_row("j0", name="n0", employer="E0", title="T0", job_order=0),
        _job_row("j1", name="n1", employer="E1", title="T1", job_order=1),
        _job_row("j2", name="n2", employer="E2", title="T2", job_order=2),
    ]
    walk = 0
    selections, expected, expected_ids = [], [], []
    for row, size in zip(rows, (7, 5, 2), strict=True):
        bullets = []
        for i in range(size):
            walk += 1
            text = "Cut  latency by half." if walk == 3 else f"Did fictional thing {walk}."
            bullets.append(_bullet(f"{row.job_id}::b{i}", text, walk=walk))
            expected.append(f"- {text} <!-- src: {row.job_id}::b{i} -->")
            expected_ids.append(f"{row.job_id}::b{i}")
        selections.append(_job_sel(row.job_id, row.name, row.job_order, bullets))
    payload = _payload(jobs=selections)

    resume = render_resume(payload, rows, [_group("", "Go", "SQL")])

    assert "## Skills\n\nGo; SQL\n\n## Experience\n" in resume  # "; "
    assert resume.index("## Skills") < resume.index("## Experience")
    lines = _resume_bullet_lines(resume)
    assert lines == expected
    assert all(ln.endswith(" -->") for ln in lines)
    assert _src_ids(resume) == expected_ids


def test_template_header_then_skills_or_experience():
    # The header slots, then Skills (or Experience when there are none).
    from loomcv.draft import render_resume
    from loomcv.sync.parse_assets import VaultHeader

    payload = _payload(jobs=[_job_sel("j0", "n", 0, [_bullet("j0::a", "Only bullet.", walk=1)])])
    rows = [_job_row("j0")]
    top = "# Jane Example\n\njane@example.invalid\n\n**Platform Engineer**\n\n"

    def next_non_blank(text: str, after: str) -> str:
        rest = text[len(after) :]
        return next(ln for ln in rest.splitlines() if ln.strip())

    header = VaultHeader(
        name="Jane Example", contact=["jane@example.invalid"], headline="Platform Engineer"
    )
    with_skills = render_resume(payload, rows, [_group("", "Go")], header)
    assert with_skills.startswith(top)
    assert next_non_blank(with_skills, top) == "## Skills"

    without_skills = render_resume(payload, rows, [], header)
    assert without_skills.startswith(top)
    assert next_non_blank(without_skills, top) == "## Experience"
    assert "## Skills" not in without_skills

    assert render_resume(payload, rows, [_group("", "Go")]).startswith("## Skills\n")
    assert render_resume(payload, rows, []).startswith("## Experience\n")


def test_resume_date_filter():
    from loomcv.draft import resume_date

    assert resume_date("2021-03") == "Mar 2021"
    assert resume_date("2020-12") == "Dec 2020"
    assert resume_date("2020-01") == "Jan 2020"
    assert resume_date("2025") == "2025"
    assert resume_date("present") == "Present"
    assert resume_date("0000") == "0000"


def test_template_undefined_slot_exit_1_and_srcless_template_renders(tmp_path, capsys):
    from loomcv.cli import main
    from loomcv.draft import ResumeTemplateError, render_resume
    from loomcv.retrieve.search import load_candidates

    db, _pr = _index(tmp_path, "index_vault.md")
    first = load_candidates(db)[0]
    payload = _payload(
        jobs=[_job_sel(first.job_id, first.job, 0, [_bullet(first.bullet_id, first.text, walk=1)])]
    )
    rows = [_job_row(first.job_id)]

    bad = tmp_path / "bad_template.md"
    # D-7: `summary` is a slot now; the retired `header` slot is the one the data lacks.
    bad.write_text("{{ header }}\n## Experience\n", encoding="utf-8")
    with pytest.raises(ResumeTemplateError):
        render_resume(payload, rows, [], template_path=bad)
    with pytest.raises(ResumeTemplateError):
        render_resume(payload, rows, [], template_path=tmp_path / "no_such_template.md")

    match = tmp_path / "match.json"
    match.write_text(_payload_json(payload), encoding="utf-8")
    apps = tmp_path / "apps"
    cfg = tmp_path / "generation.yaml"
    cfg.write_text(_generation_yaml(apps, template=bad), encoding="utf-8")
    capsys.readouterr()

    rc = main(["draft", str(match), "--db", str(db), "--generation-config", str(cfg)])

    assert rc == 1
    assert "bad_template.md" in capsys.readouterr().err
    assert not apps.exists()

    # a template without the `src` comment is the user's choice: it renders, nothing is added
    srcless = tmp_path / "srcless_template.md"
    srcless.write_text(
        "## Experience\n{% for job in jobs %}{% for b in job.bullets %}- {{ b.text }}\n"
        "{% endfor %}{% endfor %}",
        encoding="utf-8",
    )
    resume = render_resume(payload, rows, [], template_path=srcless)
    assert first.text in resume
    assert "src:" not in resume


def test_template_custom_old_order_reproduces_d1_shape(tmp_path):
    from loomcv.draft import render_resume

    template = tmp_path / "old_order.md"
    template.write_text(
        "## Experience\n"
        "{%- for job in jobs %}\n"
        "\n"
        "### {{ job.heading }}\n"
        "{{ job.date_from | resume_date }} – {{ job.date_to | resume_date }}\n"
        "{%- if job.bullets %}\n"
        "{% for b in job.bullets %}\n"
        "- {{ b.label }}{{ b.text }} <!-- src: {{ b.bullet_id }} -->\n"
        "{%- endfor %}\n"
        "{%- endif %}\n"
        "{%- endfor %}\n"
        "{% if skills %}\n"
        "## Skills\n"
        "\n"
        '{{ skills | join(", ") }}\n'
        "{% endif -%}\n",
        encoding="utf-8",
    )
    bullets = [
        _bullet("j0::a", "Cut  latency by half.", walk=1),
        _bullet("j0::b", "Shipped it.", project="Helios", walk=2),
    ]
    payload = _payload(jobs=[_job_sel("j0", "n0", 0, bullets)])
    rows = [_job_row("j0", employer="E0", title="T0", date_from="2020-01", date_to="2021-12")]

    resume = render_resume(payload, rows, [_group("", "Go")], template_path=template)

    assert resume == (
        "## Experience\n"
        "\n"
        "### T0 — E0\n"
        "Jan 2020 – Dec 2021\n"
        "\n"
        "- Cut  latency by half. <!-- src: j0::a -->\n"
        "- **Helios:** Shipped it. <!-- src: j0::b -->\n"
        "\n"
        "## Skills\n"
        "\n"
        "Go\n"
    )


# ------------------------------------------------------------------- report


def test_report_coverage_table_one_row_per_line():
    from loomcv.draft import render_report

    payload = _payload(
        jobs=[
            _job_sel(
                "j0",
                "Acme — Engineer",
                0,
                [_bullet("j0::a1", "Text a1.", walk=1), _bullet("j0::a2", "Text a2.", walk=2)],
            )
        ],
        lines=[
            _line("Build web services", "must", "exact", ["Python", "Flask"]),
            _line("Own deployments", "responsibility", "adjacent", ["Docker"]),
            _line("Know A|B tooling", "nice", "none"),
        ],
        covered_by={
            "Build web services": ["j0::a1", "j0::zz9"],
            "Own deployments": ["j0::a2"],
            "Know A|B tooling": [],
        },
    )

    report = render_report(payload)

    assert report.startswith("# Match report — Fake Platform Engineer\n")
    headings = [
        "## 0 Verdict",
        "## 1 Each item in this posting",
        "## 2 Required, not shown",
        "## 3 Listed in your profile, but no bullet describes it",
        "## 4 Something similar — prepare for the question",
        "## 5 Not on this résumé",
        "## 6 For troubleshooting",
    ]
    positions = [report.index(h + "\n") for h in headings]
    assert positions == sorted(positions)
    assert report.index("\n") < report.index("## 0 Verdict\n") < positions[1]

    table = [
        ln
        for ln in _section(report, "## 1 Each item in this posting").splitlines()
        if ln.startswith("|")
    ]
    assert table == [
        "| Type | What the posting asks | Your résumé | Matched skills | Bullets used |",
        "|---|---|---|---|---|",
        "| Required | Build web services | Clearly shows | Python, Flask | 1 |",
        "| Responsibility | Own deployments | Something similar | Docker | 1 |",
        "| Preferred | Know A\\|B tooling | Doesn't show | — | — |",
    ]
    assert "j0::zz9" not in report


def test_report_gap_report_claimed_suffix_only_for_claimed():
    from loomcv.draft import render_report

    claims = [
        _claim("g1", [("Kafka", ["jobA", "skills"]), ("Pulsar", ["jobB", "skills"])]),
        _claim("n1", [("Redis", ["jobA"])], kind="nice", match="adjacent"),
    ]
    payload = _payload(
        lines=[_line("g1"), _line("g2"), _line("n1", "nice")],
        gaps=["g1", "g2"],
        claims=claims,
    )

    section = _section(render_report(payload), "## 2 Required, not shown")
    body = [ln for ln in section.splitlines() if ln.strip()]

    assert body == [
        "- g1 — you list it at jobA; jobB and in your skills",
        "- g2 — no bullet in your profile backs this up",
    ]
    assert "n1" not in section

    empty = _section(render_report(_payload()), "## 2 Required, not shown")
    assert empty.strip() == "none"


def test_report_claims_section_entries_and_neutral_line():
    from loomcv.draft import render_report

    heading = "## 3 Listed in your profile, but no bullet describes it"
    neutral = "  - add a bullet in My profile if you've done it"
    payload = _payload(
        lines=[_line("c1"), _line("c2", "nice")],
        claims=[
            _claim("c1", [("Kafka", ["jobA", "skills"]), ("Pulsar", ["jobB"])]),
            _claim("c2", [("Redis", ["skills"])], kind="nice", match="adjacent"),
        ],
    )

    section = _section(render_report(payload), heading)
    body = [ln for ln in section.splitlines() if ln.strip()]

    assert body == [
        "- c1",
        "  - Kafka — at jobA and in your skills",
        "  - Pulsar — at jobB",
        neutral,
        "- c2",
        "  - Redis — in your skills",
        neutral,
    ]
    assert section.count(neutral.strip()) == 2

    empty = _section(render_report(_payload()), heading)
    assert empty.strip() == "none"


def test_report_adjacent_section_lists_exactly_adjacent_lines():
    from loomcv.draft import render_report

    payload = _payload(
        lines=[
            _line("Exact line", "must", "exact", ["Python"]),
            _line("Own deployments", "responsibility", "adjacent", ["Docker", "Helm"]),
            _line("Bare adjacent", "nice", "adjacent"),
            _line("No match line", "nice", "none"),
            _line(
                "Python and Go",
                "must",
                "exact",
                ["Python"],
                parts=[("Python", "exact"), ("Go", "none")],
            ),
        ]
    )

    section = _section(render_report(payload), "## 4 Something similar — prepare for the question")
    body = [ln for ln in section.splitlines() if ln.strip()]

    # Every line whose bin is "close" (the same rule as section 0): a compound line with a part
    # that is missing is among them.
    assert body == [
        "- Own deployments — closest skills: Docker, Helm",
        "- Bare adjacent — closest skills: —",
        "- Python and Go — closest skills: Python",
    ]


def test_report_not_on_resume_uncovered_and_unplaced_with_reason():
    from loomcv.draft import render_report
    from loomcv.retrieve.rank import UncoveredLine, UnplacedLine

    payload = _payload(
        lines=[_line("u1"), _line("u2", "responsibility"), _line("u3", "nice", "none")],
        uncovered=[UncoveredLine(line="u1", kind="must")],
        unplaced=[
            UnplacedLine(line="u2", kind="responsibility", reason="budget"),
            UnplacedLine(line="u3", kind="nice", reason="no_candidates"),
        ],
    )

    section = _section(render_report(payload), "## 5 Not on this résumé")
    body = [ln for ln in section.splitlines() if ln.strip()]

    assert body == [
        "### No bullet on this résumé covers it",
        "- u1 (required)",
        "### Left off, and why",
        "- u2 (responsibility) — no room left on the page",
        "- u3 (preferred) — nothing in your profile for it",
    ]

    taken = _payload(
        unplaced=[UnplacedLine(line="u4", kind="must", reason="all_taken")],
    )
    assert "- u4 (required) — the bullets for it are already used for other items" in _section(
        render_report(taken), "## 5 Not on this résumé"
    )

    empty = _section(render_report(_payload()), "## 5 Not on this résumé")
    assert [ln for ln in empty.splitlines() if ln.strip()] == [
        "### No bullet on this résumé covers it",
        "none",
        "### Left off, and why",
        "none",
    ]


def test_report_slots_account_without_length_figures():
    from loomcv.draft import render_report

    payload = _payload(
        jobs=[
            _job_sel("j0", "Job Zero", 0, [_bullet("j0::a", "T a.", walk=1)], cap=3),
            _job_sel(
                "j1",
                "Job One",
                1,
                [_bullet("j1::a", "T b.", walk=2), _bullet("j1::b", "T c.", walk=3)],
                cap=2,
            ),
        ],
        total_slots=14,
        passes=3,
    )

    report = render_report(payload)
    section = _subsection(report, "### Slots")
    body = [ln for ln in section.splitlines() if ln.strip()]

    assert body == [
        "- total_slots: 14",
        "- passes: 3",
        "- Job Zero: cap 3, placed 1",
        "- Job One: cap 2, placed 2",
    ]
    assert "char" not in report.casefold()
    assert "length" not in report.casefold()


def test_report_provenance_one_row_per_placed_bullet():
    from loomcv.draft import render_report

    payload = _payload(
        jobs=[
            _job_sel(
                "j0",
                "Job Zero",
                0,
                [
                    _bullet("j0::a", "T a.", line="Line A", walk=2, pass_number=1),
                    _bullet("j0::b", "T b.", line="Line B", kind="nice", walk=4, pass_number=2),
                ],
            ),
            _job_sel(
                "j1",
                "Job One",
                1,
                [
                    _bullet("j1::a", "T c.", line="Line C", walk=1, pass_number=1),
                    _bullet("j1::b", "T d.", line="Line D", walk=3, pass_number=2),
                ],
            ),
        ]
    )

    section = _subsection(render_report(payload), "### Provenance")
    body = [ln for ln in section.splitlines() if ln.strip()]

    assert body == [
        "- j1::a → Job One → Line C → pass 1 → walk 1",
        "- j0::a → Job Zero → Line A → pass 1 → walk 2",
        "- j1::b → Job One → Line D → pass 2 → walk 3",
        "- j0::b → Job Zero → Line B → pass 2 → walk 4",
    ]


def test_report_has_no_score_fields_or_percent():
    from loomcv.draft import render_report
    from loomcv.retrieve.rank import LineResult

    bullet = _bullet(
        "j0::a",
        "Cut cost by 25%.",
        line="Reduce cost",
        overlap=7,
        rerank_rank=2,
        rrf_position=3,
        covers=["Reduce cost"],
    )
    line = LineResult(
        text="Reduce cost",
        kind="must",
        grade="exact",
        mapped_keywords=["Costing"],
        ranked=[],
        max_cosine=0.91,
        max_bm25=12.5,
        top_rrf_score=0.033,
    )
    payload = _payload(
        jobs=[_job_sel("j0", "Job Zero", 0, [bullet])],
        lines=[line],
        covered_by={"Reduce cost": ["j0::a"]},
        title="Cost Engineer 100%",
    )

    report = render_report(payload)

    for name in (
        "max_cosine",
        "max_bm25",
        "top_rrf_score",
        "rerank_rank",
        "rrf_position",
        "overlap",
    ):
        assert name not in report
    assert "score" not in report.casefold()
    section0 = report[report.index("## 0 Verdict\n") : report.index("\n## 1 ")]
    assert section0.count("%") == 0  # no percentage is printed
    stripped = report.replace(section0, "")
    for payload_string in (
        "Cost Engineer 100%",
        "Cut cost by 25%.",
        "Reduce cost",
        "Costing",
        "Job Zero",
    ):
        stripped = stripped.replace(payload_string, "")
    assert "%" not in stripped
    for number in ("0.91", "12.5", "0.033"):
        assert number not in report


def test_report_sections_in_plain_words():
    # R-1: sections 0–5 carry no term of the pipeline; Slots and Provenance
    # sit under "## 6 For troubleshooting" with their bodies unchanged.
    from loomcv.draft import render_report
    from loomcv.retrieve.rank import UncoveredLine, UnplacedLine

    payload = _payload(
        jobs=[
            _job_sel(
                "j0",
                "Acme — Engineer",
                0,
                [
                    _bullet("j0::a1", "Text a1.", line="Build services", walk=1),
                    _bullet("j0::a2", "Text a2.", line="Own deployments", walk=2, pass_number=2),
                ],
                cap=3,
            )
        ],
        lines=[
            _line("Build services", "must", "exact", ["Python"]),
            _line("Run Kafka", "must", "none"),
            _line("Own deployments", "responsibility", "adjacent", ["Docker"]),
            _line("Know Rust", "nice", "none"),
            _line(
                "Python and Go",
                "must",
                "exact",
                ["Python"],
                parts=[("Python", "exact"), ("Go", "none")],
            ),
        ],
        gaps=["Run Kafka"],
        claims=[_claim("Run Kafka", [("Kafka", ["j0", "skills"])])],
        uncovered=[UncoveredLine(line="Know Rust", kind="nice")],
        unplaced=[UnplacedLine(line="Python and Go", kind="must", reason="budget")],
        covered_by={"Build services": ["j0::a1"], "Own deployments": ["j0::a2"]},
        total_slots=14,
        passes=2,
    )
    rows = [_job_row("j0", employer="Acme", title="Engineer")]

    report = render_report(payload, rows)

    plain = report[report.index("## 0 Verdict") : report.index("## 6 For troubleshooting")]
    for heading in ("## 1 ", "## 2 ", "## 3 ", "## 4 ", "## 5 "):
        assert heading in plain  # every section is non-empty, so every body was exercised
    for term in (
        "vault",
        "Gap Report",
        "Claims",
        "Adjacent",
        "mapped",
        "grade",
        "(must)",
        "(nice)",
        "(responsibility)",
        "@",
        "total_slots",
        "pass ",
        "walk",
        "adjacent",
    ):
        assert term not in plain, term

    troubleshooting = report[report.index("## 6 For troubleshooting") :]
    assert troubleshooting.splitlines()[0] == "## 6 For troubleshooting"
    assert (
        "How this résumé was put together — for checking a problem, not for deciding whether "
        "to apply."
    ) in troubleshooting
    assert [ln for ln in _subsection(report, "### Slots").splitlines() if ln.strip()] == [
        "- total_slots: 14",
        "- passes: 2",
        "- Acme — Engineer: cap 3, placed 2",
    ]
    assert [ln for ln in _subsection(report, "### Provenance").splitlines() if ln.strip()] == [
        "- j0::a1 → Acme — Engineer → Build services → pass 1 → walk 1",
        "- j0::a2 → Acme — Engineer → Own deployments → pass 2 → walk 2",
    ]


# ------------------------------------------------------------------ verdict
# V-1 .. V-9.


def _must_payload(*, exact=0, adjacent=0, none=0, extra_lines=(), claims=()):
    """Must lines `Must line N` graded as asked (exact, adjacent, none in that order)."""
    lines = []
    for grade, count in (("exact", exact), ("adjacent", adjacent), ("none", none)):
        lines += [_line(f"Must line {len(lines) + i}", "must", grade) for i in range(count)]
    lines += list(extra_lines)
    gaps = [ln.text for ln in lines if ln.kind == "must" and ln.grade == "none"]
    return _payload(lines=lines, gaps=gaps, claims=list(claims))


def _verdict_body(report: str) -> list[str]:
    return _section(report, "## 0 Verdict").strip("\n").split("\n")


def _headline(letter, total, direct, close, missing):
    """Section 0's headline: the letter, the two counts and, only when
    something is missing (whole lines + named parts), the third sentence
    ("Missing from your résumé entirely: N things, listed below.")."""
    items = "1 required item" if total == 1 else f"{total} required items"
    text = (
        f"**{letter}** — this posting lists {items}. Your résumé clearly shows {direct} and "
        f"shows something similar for {close}."
    )
    if missing:
        things = "thing" if missing == 1 else "things"
        text += f" Missing from your résumé entirely: {missing} {things}, listed below."
    return text


@pytest.mark.parametrize(
    ("grade", "parts", "bin_", "named"),
    [
        ("exact", [], "direct", None),
        ("adjacent", [], "close", None),
        ("none", [], "none", None),
        ("exact", [("Go", "adjacent")], "close", None),
        ("exact", [("Go", "none")], "close", "Go"),
        ("exact", [("Go", "exact"), ("Node", "exact")], "direct", None),
    ],
    ids=["exact", "adjacent", "none", "part-adjacent", "part-none", "parts-exact"],
)
def test_verdict_bins(grade, parts, bin_, named):
    # V-1: direct / close / not covered; a missing part is named on its own.
    from loomcv.draft import compute_verdict, render_report
    from loomcv.draft.verdict import line_bin

    line = _line("Build Go and Node services", "must", grade, ["Go"], parts=parts)
    payload = _payload(lines=[line], gaps=["Build Go and Node services"] if grade == "none" else [])

    verdict = compute_verdict(payload)
    body = _verdict_body(render_report(payload))

    assert line_bin(payload.requirements[0]) == bin_
    assert (verdict.direct, verdict.close, verdict.uncovered) == (
        int(bin_ == "direct"),
        int(bin_ == "close"),
        int(bin_ == "none"),
    )
    assert verdict.close_lines == (["Build Go and Node services"] if bin_ == "close" else [])
    if named is None:
        assert verdict.missing_parts == 0
        assert all(nc.part is None for nc in verdict.not_covered)
    else:
        assert [(nc.text, nc.part) for nc in verdict.not_covered] == [
            ("Build Go and Node services", named)
        ]
        assert verdict.missing_parts == 1
        entry = [ln for ln in body if ln.startswith(f"- {named} — ")]
        assert entry == [f'- {named} — the posting asks for it here: "Build Go and Node services"']
        assert "listed" not in entry[0]  # a part entry never carries a claims sentence


# (must lines, direct, close) → letter; the rest of the lines are not covered. Points are
# 2 * direct + close out of 2 * M, compared as integers at 90 / 80 / 50.
_THRESHOLD_CASES = [
    (10, 9, 0, "A"),
    (9, 8, 0, "B"),  # 88.9
    (10, 8, 1, "B"),
    (10, 8, 0, "B"),  # 80
    (9, 7, 0, "C"),  # 77.7
    (10, 5, 0, "C"),  # 50
    (4, 0, 4, "C"),
    (10, 4, 1, "D"),  # 45
    (1, 1, 0, "A"),
    (1, 0, 1, "C"),
    (1, 0, 0, "D"),
    (8, 5, 3, "B"),
    (8, 2, 5, "C"),
    (4, 2, 2, "C"),
    (4, 3, 1, "B"),
]


@pytest.mark.parametrize(("total", "direct", "close", "letter"), _THRESHOLD_CASES)
def test_verdict_letter_thresholds(total, direct, close, letter):
    # V-2
    from loomcv.draft import compute_verdict, render_report
    from loomcv.draft import verdict as verdict_module

    payload = _must_payload(exact=direct, adjacent=close, none=total - direct - close)

    verdict = compute_verdict(payload)
    report = render_report(payload)

    assert verdict.letter == letter
    assert verdict.points == 2 * direct + close
    assert _verdict_body(report)[0] == _headline(
        letter, total, direct, close, total - direct - close
    )
    assert verdict_module.THRESHOLDS == (90, 80, 50)


@pytest.mark.parametrize(
    ("exact", "adjacent", "letter"),
    [(2, 2, "C"), (0, 4, "C"), (2, 0, "A")],
    ids=["two-and-two", "all-close", "no-close"],
)
def test_verdict_close_counts_half(exact, adjacent, letter):
    from loomcv.draft import compute_verdict, render_report

    payload = _must_payload(exact=exact, adjacent=adjacent)

    verdict = compute_verdict(payload)
    headline = _verdict_body(render_report(payload))[0]

    assert verdict.letter == letter
    assert headline == _headline(letter, exact + adjacent, exact, adjacent, 0)
    assert headline.endswith(".")


def test_verdict_not_covered_lines_with_job_names():
    from loomcv.draft import compute_verdict, render_report

    lines = [
        _line("Ship Go services", "must", "exact"),
        _line("Design REST APIs", "must", "exact"),
        _line("Tune Postgres", "must", "adjacent"),
        _line("Run Kafka | Pulsar", "must", "none"),
        _line("Lead on-call rotations", "must", "none"),
    ]
    payload = _payload(
        lines=lines,
        gaps=["Run Kafka | Pulsar", "Lead on-call rotations"],
        claims=[_claim("Run Kafka | Pulsar", [("Kafka", ["j0", "skills"])])],
    )
    rows = [_job_row("j0", employer="E0", title="T0")]

    verdict = compute_verdict(payload, rows)
    report = render_report(payload, rows)
    body = _verdict_body(report)

    assert (verdict.must_total, verdict.direct, verdict.close, verdict.letter) == (5, 2, 1, "C")
    assert body[4:7] == [
        "Required, but not shown on your résumé:",
        "- Run Kafka | Pulsar — listed under E0, T0 and in your Skills, but no bullet shows it",
        "- Lead on-call rotations",
    ]
    assert [nc.text for nc in verdict.not_covered] == payload.gaps
    assert "- Run Kafka | Pulsar — you list it at E0, T0 and in your skills" in _section(
        report, "## 2 Required, not shown"
    )

    # The separator pin: jobs joined by "; " in claim-source order, Skills last.
    two_jobs = _payload(
        lines=[_line("Run Kafka", "must", "none")],
        gaps=["Run Kafka"],
        claims=[_claim("Run Kafka", [("Kafka", ["j1"]), ("Pulsar", ["j0"])])],
    )
    two_rows = [
        _job_row("j0", employer="E0", title="T0", job_order=0),
        _job_row("j1", employer="E1", title="T1", job_order=1),
    ]
    assert "- Run Kafka — listed under E1, T1; E0, T0, but no bullet shows it" in _verdict_body(
        render_report(two_jobs, two_rows)
    )
    skills_only = _payload(
        lines=[_line("Run Kafka", "must", "none")],
        gaps=["Run Kafka"],
        claims=[_claim("Run Kafka", [("Kafka", ["skills"])])],
    )
    assert "- Run Kafka — listed in your Skills, but no bullet shows it" in _verdict_body(
        render_report(skills_only, two_rows)
    )


def test_verdict_not_covered_none_when_every_must_line_has_evidence():
    from loomcv.draft import render_report

    body = _verdict_body(render_report(_must_payload(exact=3)))

    assert "Required, but not shown on your résumé: none" in body
    assert not [ln for ln in body if ln.startswith("- ")]


def test_verdict_nice_and_responsibility_never_move_letter():
    from loomcv.draft import compute_verdict, render_report

    # (i) nice 0 / 5, responsibility 1 / 6 — the letter stays A.
    extra = [_line(f"Nice {i}", "nice", "none") for i in range(5)]
    extra += [_line("Resp ok", "responsibility", "exact")]
    extra += [_line(f"Resp none {i}", "responsibility", "none") for i in range(5)]
    payload = _must_payload(exact=9, extra_lines=extra)
    body = _verdict_body(render_report(payload))
    verdict = compute_verdict(payload)
    assert verdict.letter == "A"
    assert (
        f"Preferred items: 0 of 5 shown. Responsibilities: 1 of 6 shown. {_COUNTS_TAIL}"
    ) in body

    # (ii) no nice line at all.
    payload = _must_payload(exact=2, extra_lines=[_line("Resp ok", "responsibility", "exact")])
    body = _verdict_body(render_report(payload))
    assert (
        "Preferred items: none read from this posting. Responsibilities: 1 of 1 shown. "
        f"{_COUNTS_TAIL}"
    ) in body

    # (iii) a posting whose only `none` lines are not must lines.
    payload = _must_payload(
        exact=3,
        adjacent=1,
        extra_lines=[
            _line("Resp none", "responsibility", "none"),
            _line("Nice none", "nice", "none"),
        ],
    )
    verdict = compute_verdict(payload)
    body = _verdict_body(render_report(payload))
    assert verdict.letter == "B"  # 7 of 8 points: a close match counts half
    assert "Required, but not shown on your résumé: none" in body


def test_verdict_section_zero_prints_three_counts():
    # V-4: no percentage; the three counts; no pipeline token outside the
    # legend.
    from loomcv.draft import compute_verdict, render_report
    from loomcv.draft.verdict import LEGEND

    payloads = [
        _must_payload(exact=4),
        _must_payload(exact=2, adjacent=1, none=1),
        _must_payload(none=3),
        _payload(
            lines=[
                _line("Build Go and Node services", "must", "exact", parts=[("Go", "none")]),
                _line("Ship it", "must", "exact"),
            ]
        ),
    ]
    token = re.compile(r"\b(exact|adjacent|none|must)\b", re.IGNORECASE)
    for payload in payloads:
        verdict = compute_verdict(payload)
        section0 = _section(render_report(payload), "## 0 Verdict")
        headline = _verdict_body(render_report(payload))[0]

        assert "%" not in section0
        assert headline == _headline(
            verdict.letter,
            verdict.must_total,
            verdict.direct,
            verdict.close,
            verdict.uncovered + verdict.missing_parts,
        )
        # the third sentence appears exactly when something is missing, and never otherwise
        missing = verdict.uncovered + verdict.missing_parts
        assert ("Missing from your résumé entirely:" in headline) == (missing > 0)
        prose = re.sub(r"Must line \d+", "", section0.replace(LEGEND, ""))  # the fixture's lines
        # "…résumé: none" and "none read from this posting" are plain English, not the token.
        prose = prose.replace("résumé: none", "").replace("none read from", "")
        assert token.search(prose) is None, section0


def test_verdict_parts_on_other_lines_never_count():
    # V-5
    from loomcv.draft import compute_verdict

    with_parts = _must_payload(
        exact=2,
        extra_lines=[
            _line("Nice thing", "nice", "exact", parts=[("Go", "none")]),
            _line("Duty thing", "responsibility", "exact", parts=[("Rust", "none")]),
        ],
    )
    without = _must_payload(
        exact=2,
        extra_lines=[
            _line("Nice thing", "nice", "exact"),
            _line("Duty thing", "responsibility", "exact"),
        ],
    )

    verdict = compute_verdict(with_parts)

    assert verdict.not_covered == []
    assert verdict.model_dump() == compute_verdict(without).model_dump()
    assert (verdict.nice_covered, verdict.nice_total, verdict.letter) == (1, 1, "A")


def test_verdict_payload_without_parts_loads():
    # V-6: a `match.json` written before this ticket has no `parts` key on its lines.
    from loomcv.draft import compute_verdict
    from loomcv.retrieve.rank import MatchPayload

    payload = _must_payload(exact=2, adjacent=1, none=1)
    raw = json.loads(_payload_json(payload))
    for line in raw["requirements"]:
        line.pop("parts", None)

    loaded = MatchPayload.model_validate(raw)

    assert all(line.parts == [] for line in loaded.requirements)
    verdict = compute_verdict(loaded)
    assert (verdict.direct, verdict.close, verdict.uncovered, verdict.letter) == (2, 1, 1, "C")


def test_verdict_no_must_line_is_no_grade():
    from loomcv.draft import compute_verdict, render_report
    from loomcv.draft.verdict import (
        LEGEND,
        LETTER_SENTENCES,
        NO_GRADE_HEADLINE,
        NOT_COVERED_HEAD,
    )

    lines = [
        _line("Nice one", "nice", "exact"),
        _line("Nice two", "nice", "none"),
        _line("Resp one", "responsibility", "exact"),
    ]
    payload = _payload(lines=lines)

    verdict = compute_verdict(payload)
    report = render_report(payload)
    body = _verdict_body(report)

    assert verdict.letter is None
    assert body == [
        NO_GRADE_HEADLINE,
        "",
        f"Preferred items: 1 of 2 shown. Responsibilities: 1 of 1 shown. {_COUNTS_TAIL}",
        "",
        LEGEND,
    ]
    assert body[0].startswith(
        "**?? — no grade**: we couldn't find a required item in this posting, "
    )
    assert "%" not in report
    for sentence in LETTER_SENTENCES.values():
        assert sentence not in report
    assert NOT_COVERED_HEAD not in report

    # A claim on a non-must line never reaches section 0.
    claimed = _payload(lines=lines, claims=[_claim("Nice two", [("Kafka", ["j0"])], kind="nice")])
    assert _verdict_body(render_report(claimed)) == body


def _v7_sections_1_to_6(where: str) -> str:
    """Sections 1–6 of the placement payload below; `where` = the job's name as the
    Gap Report / claims lines print it."""
    return f"""## 1 Each item in this posting

| Type | What the posting asks | Your résumé | Matched skills | Bullets used |
|---|---|---|---|---|
| Required | Build web services | Clearly shows | Python | 1 |
| Required | Run Kafka | Doesn't show | — | — |
| Responsibility | Own deployments | Something similar | Docker | — |
| Preferred | Know Rust | Doesn't show | — | — |

## 2 Required, not shown

- Run Kafka — you list it at {where} and in your skills

## 3 Listed in your profile, but no bullet describes it

- Run Kafka
  - Kafka — at {where} and in your skills
  - add a bullet in My profile if you've done it

## 4 Something similar — prepare for the question

- Own deployments — closest skills: Docker

## 5 Not on this résumé

### No bullet on this résumé covers it

none

### Left off, and why

none

## 6 For troubleshooting

How this résumé was put together — for checking a problem, not for deciding whether to apply.

### Slots

- total_slots: 14
- passes: 1
- Acme — Engineer: cap 1, placed 1

### Provenance

- j0::a1 → Acme — Engineer → Build web services → pass 1 → walk 1
"""


def test_verdict_placement_shape_and_sections_unchanged():
    from loomcv.draft import render_report
    from loomcv.draft.verdict import LEGEND

    payload = _payload(
        jobs=[
            _job_sel(
                "j0",
                "Acme — Engineer",
                0,
                [_bullet("j0::a1", "Text a1.", line="Build web services", walk=1)],
            )
        ],
        lines=[
            _line("Build web services", "must", "exact", ["Python"]),
            _line("Run Kafka", "must", "none"),
            _line("Own deployments", "responsibility", "adjacent", ["Docker"]),
            _line("Know Rust", "nice", "none"),
        ],
        gaps=["Run Kafka"],
        claims=[_claim("Run Kafka", [("Kafka", ["j0", "skills"])])],
        covered_by={"Build web services": ["j0::a1"]},
    )
    rows = [_job_row("j0", employer="E0", title="T0")]

    report = render_report(payload, rows)

    h1_end = report.index("\n")
    assert report[h1_end:].index("\n## ") == report[h1_end:].index("\n## 0 Verdict\n")
    assert report.index("## 0 Verdict\n") < report.index("## 1 Each item in this posting\n")

    def expected_body(suffix):
        return "\n".join(
            [
                _headline("C", 2, 1, 0, 1),
                "",
                "Partial match — several required items are only something similar or not shown "
                "at all; a reviewer will notice them.",
                "",
                "Required, but not shown on your résumé:",
                f"- Run Kafka — listed under {suffix}, but no bullet shows it",
                "",
                f"Preferred items: 0 of 1 shown. Responsibilities: 1 of 1 shown. {_COUNTS_TAIL}",
                "",
                LEGEND,
            ]
        )

    assert _section(report, "## 0 Verdict").strip("\n") == expected_body(
        "E0, T0 and in your Skills"
    )
    assert report[report.index("## 1 Each item") :] == _v7_sections_1_to_6("E0, T0")

    bare = render_report(payload)
    assert _section(bare, "## 0 Verdict").strip("\n") == expected_body("j0 and in your Skills")
    assert bare[bare.index("## 1 Each item") :] == _v7_sections_1_to_6("j0")


def test_verdict_section_has_no_forbidden_content():
    from loomcv.draft import render_report
    from loomcv.draft.verdict import (
        LEGEND,
        LETTER_SENTENCES,
        NO_GRADE_HEADLINE,
        NOT_COVERED_HEAD,
    )
    from loomcv.retrieve.rank import LineResult

    def scored(text, kind, grade):
        return LineResult(
            text=text,
            kind=kind,
            grade=grade,
            mapped_keywords=[],
            ranked=[],
            max_cosine=0.91,
            max_bm25=12.5,
            top_rrf_score=0.033,
        )

    payload = _payload(
        lines=[
            scored("Ship Go services", "must", "exact"),
            _line("Tune Postgres", "must", "adjacent"),
            _line("Run Kafka", "must", "none"),
            _line("Lead on-call rotations", "must", "none"),
            _line("Know Rust", "nice", "exact"),
            _line("Own deployments", "responsibility", "none"),
        ],
        gaps=["Run Kafka", "Lead on-call rotations"],
        claims=[_claim("Run Kafka", [("Kafka", ["j0", "j1", "skills"])])],
    )
    rows = [
        _job_row("j0", employer="E0", title="T0", job_order=0),
        _job_row("j1", employer="E1", title="T1", job_order=1),
    ]

    section0 = _section(render_report(payload, rows), "## 0 Verdict")

    for name in (
        "max_cosine",
        "max_bm25",
        "top_rrf_score",
        "rerank_rank",
        "rrf_position",
        "overlap",
    ):
        assert name not in section0
    for number in ("0.91", "12.5", "0.033"):
        assert number not in section0
    assert "score" not in section0.casefold()
    assert not re.search(r"\bj[01]\b", section0)

    forbidden = re.compile(
        r"\b(must|nice|responsibility|exact|adjacent|covered|coverage|claimed|vault)\b",
        re.IGNORECASE,
    )
    assert forbidden.search(section0.replace(LEGEND, "")) is None
    for fixed in (*LETTER_SENTENCES.values(), NO_GRADE_HEADLINE, NOT_COVERED_HEAD):
        assert forbidden.search(fixed) is None
        assert "score" not in fixed.casefold()
    assert "score" not in LEGEND.casefold()


def test_verdict_job_name_fallbacks():
    from loomcv.draft import compute_verdict, render_report

    payload = _payload(
        lines=[_line("Run Kafka", "must", "none"), _line("Ship Go", "must", "exact")],
        gaps=["Run Kafka"],
        claims=[_claim("Run Kafka", [("Kafka", ["j0"])])],
    )
    cases = [
        ([_job_row("j0", employer="E0", title=None)], "E0"),
        ([_job_row("j0", name="N0", employer=None, title=None)], "N0"),
        ([_job_row("j9")], "j0"),
    ]

    seen = []
    for rows, name in cases:
        verdict = compute_verdict(payload, rows)
        body = _verdict_body(render_report(payload, rows))
        assert f"- Run Kafka — listed under {name}, but no bullet shows it" in body
        seen.append((verdict.letter, verdict.direct, verdict.close, verdict.must_total))
    assert seen == [("C", 1, 0, 2)] * 3  # 2 of 4 points


# ------------------------------------------------- the render split (D-1)


@pytest.mark.parametrize("with_header", [True, False], ids=["header", "no-header"])
def test_render_resume_is_resume_context_then_render_context(with_header):
    # D-1: the split changes no byte; the review document renders through it.
    # Parametrized over the vault's header fields.
    from loomcv.draft import render_resume
    from loomcv.draft.render import DEFAULT_RESUME_TEMPLATE, render_context, resume_context

    header = _l3a_header() if with_header else None
    groups = _l3a_groups()

    rows = [
        _job_row("j0", name="n0", employer="E0", title="T0", job_order=0),
        _job_row("j1", name="n1", employer="E1", title="T1", job_order=1),
        _job_row("j2", name="n2", employer="E2", title="T2", job_order=2),
    ]
    payload = _payload(
        jobs=[
            _job_sel(
                "j0",
                "n0",
                0,
                [
                    _bullet("j0::a", "First.", walk=2),
                    _bullet("j0::b", "Second.", project="Aurora", walk=1),
                ],
            ),
            _job_sel("j1", "n1", 1, [_bullet("j1::c", "Third.", walk=3)]),
        ]
    )

    whole = render_resume(payload, rows, groups, header)
    split = render_context(
        resume_context(payload, rows, groups, header),
        template_path=DEFAULT_RESUME_TEMPLATE,
    )
    assert whole == split
    assert "- **Aurora:** Second. <!-- src: j0::b -->" in whole
    assert "\n\n### T2 — E2\n*Jan 2020 – Dec 2021*\n" in whole  # the bullet-less job
    assert whole.startswith("# Mara Voss\n") is with_header


# ------------------------------------- the five slots (D-1 … D-5, D-8, D-9)


def _l3a_header():  # built lazily so a red run fails per test, not at collection
    from loomcv.sync.parse_assets import VaultHeader

    return VaultHeader(
        name="Mara Voss",
        headline="Senior Web Platform Engineer",
        contact=["Lisbon, PT", "mara.voss@example.invalid"],
        summary="Eleven years building fictional web platforms.",
        education=["Fictional University — BSc Computer Science, 2013", "Pretend Cert, 2021"],
    )


def _l3a_groups():
    return [
        _group("", "Go", "SQL"),
        _group("Languages, Frameworks, and Data", "Python", "TypeScript"),
        _group("Platforms and Delivery", "Kubernetes", "Python"),
    ]


# The default template's output for `_l3a_header()` + `_l3a_groups()` + `_l3a_inputs()` (the
# unnamed group prints last; one ` | ` contact line; a trailing `\`).
L3A_BLOCKS = {
    "name": "# Mara Voss\n\n",
    "contact": "Lisbon, PT | mara.voss@example.invalid\n\n",
    "headline": "**Senior Web Platform Engineer**\n\n",
    "summary": "## Summary\n\nEleven years building fictional web platforms.\n\n",
    "skill_groups": (
        "## Skills\n\n"
        "**Languages, Frameworks, and Data:** Python; TypeScript\\\n"
        "**Platforms and Delivery:** Kubernetes; Python\\\n"
        "Go; SQL\n\n"
    ),
    "experience": (
        "## Experience\n\n"
        "### Staff Engineer — Acme\n"
        "*Mar 2021 – Present*\n\n"
        "- **Helios:** Did a fictional thing. <!-- src: acme::a1b2c3 -->\n\n"
        "### Engineer — Old Co\n"  # no placed bullet: same form, no list
        "*Jan 2010 – Dec 2012*\n"
    ),
    "education": (
        "\n## Education\n\n"
        "Fictional University — BSc Computer Science, 2013\\\n"
        "Pretend Cert, 2021\n"
    ),
}


def _l3a_inputs():
    rows = [
        _job_row(
            "acme",
            name="Acme — Staff Engineer",
            employer="Acme",
            title="Staff Engineer",
            date_from="2021-03",
            date_to="present",
        ),
        _job_row(
            "oldco",
            name="Old Co — Engineer",
            employer="Old Co",
            title="Engineer",
            date_from="2010-01",
            date_to="2012-12",
            job_order=1,
        ),
    ]
    payload = _payload(
        jobs=[
            _job_sel(
                "acme",
                "Acme — Staff Engineer",
                0,
                [_bullet("acme::a1b2c3", "Did a fictional thing.", project="Helios", walk=1)],
            )
        ]
    )
    return payload, rows


def test_l3a_resume_context_slots():
    # D-1 — the six new slots and the flat `skills`; the retired `header` is gone.
    from loomcv.draft.render import resume_context

    payload, rows = _l3a_inputs()
    ctx = resume_context(payload, rows, _l3a_groups(), _l3a_header())
    assert set(ctx) == {
        "name",
        "headline",
        "contact",
        "summary",
        "education",
        "skill_groups",
        "skills",
        "jobs",
    }
    assert ctx["skill_groups"] == [{"name": g.name, "items": g.items} for g in _l3a_groups()]
    assert ctx["skills"] == ["Go", "SQL", "Python", "TypeScript", "Kubernetes"]
    assert ctx["contact"] == ["Lisbon, PT", "mara.voss@example.invalid"]
    empty = resume_context(payload, rows)
    assert (empty["name"], empty["headline"], empty["summary"]) == (None, None, None)
    assert (empty["contact"], empty["education"], empty["skill_groups"]) == ([], [], [])


@pytest.mark.parametrize(
    "emptied", [None, "name", "contact", "headline", "summary", "skill_groups", "education"]
)
def test_l3a_default_template_five_sections_in_order(emptied):
    # D-2 — the exact output; an empty slot drops its block (heading included).
    from loomcv.draft import render_resume

    payload, rows = _l3a_inputs()
    header = _l3a_header()
    groups = _l3a_groups()
    if emptied == "skill_groups":
        groups = []
    elif emptied in ("contact", "education"):
        header = header.model_copy(update={emptied: []})
    elif emptied is not None:
        header = header.model_copy(update={emptied: None})

    resume = render_resume(payload, rows, groups, header)

    expected = "".join(text for key, text in L3A_BLOCKS.items() if key != emptied)
    assert resume == expected
    # the headline is a bold line, never a heading
    lines = resume.splitlines()
    assert ("**Senior Web Platform Engineer**" in lines) is (emptied != "headline")
    assert not [ln for ln in lines if ln.startswith("#") and "Senior Web Platform" in ln]


def test_l3a_hard_breaks_and_one_contact_line():
    # D-3 — every Skills / Education line but the last ends in "\";
    # the H1 and the contact are two lines; one contact item has no "|".
    from loomcv.draft import render_resume

    payload, rows = _l3a_inputs()
    resume = render_resume(payload, rows, _l3a_groups(), _l3a_header())
    lines = resume.splitlines()
    assert lines[0] == "# Mara Voss"
    assert lines[2] == "Lisbon, PT | mara.voss@example.invalid"
    skills = resume.split("## Skills\n\n")[1].split("\n\n")[0].splitlines()
    education = resume.split("## Education\n\n")[1].splitlines()
    for block in (skills, education):
        assert len(block) >= 2
        assert all(ln.endswith("\\") for ln in block[:-1])
        assert not block[-1].endswith("\\")
    one = _l3a_header().model_copy(update={"contact": ["mara.voss@example.invalid"]})
    assert "|" not in render_resume(payload, rows, [], one).splitlines()[2]


def test_l3a_grouped_skills_lines():
    # D-4 — named groups in order as "**Name:** a; b", then the unnamed group's bare line.
    from loomcv.draft import render_resume

    payload, rows = _l3a_inputs()
    resume = render_resume(payload, rows, _l3a_groups())
    section = resume.split("## Skills\n\n")[1].split("\n\n")[0]
    assert section.splitlines() == [
        "**Languages, Frameworks, and Data:** Python; TypeScript\\",
        "**Platforms and Delivery:** Kubernetes; Python\\",
        "Go; SQL",
    ]
    only_unnamed = render_resume(payload, rows, [_group("", "Go", "SQL")])
    assert "## Skills\n\nGo; SQL\n\n## Experience\n" in only_unnamed
    only_named = render_resume(payload, rows, [_group("Cloud", "AWS")])
    assert "## Skills\n\n**Cloud:** AWS\n\n## Experience\n" in only_named


def test_l3a_job_without_placed_bullet_prints_like_a_job_with_bullets():
    # D-5 — inside `## Experience`, in job order, the
    # same heading + italic date line as a job with bullets, just no list.
    from loomcv.draft import render_resume

    payload, rows = _l3a_inputs()
    resume = render_resume(payload, rows)
    experience = resume.split("## Experience\n\n")[1]
    assert experience.splitlines()[-2:] == ["### Engineer — Old Co", "*Jan 2010 – Dec 2012*"]
    assert " | Jan 2010" not in resume


def test_l3a_retired_header_slot_raises(tmp_path):
    # D-7 — a template naming `header` is refused, never silently empty.
    from loomcv.draft import ResumeTemplateError, render_resume

    payload, rows = _l3a_inputs()
    template = tmp_path / "old.md"
    template.write_text("{{ header }}\n", encoding="utf-8")
    with pytest.raises(ResumeTemplateError):
        render_resume(payload, rows, _l3a_groups(), _l3a_header(), template_path=template)


def test_l3a_draft_step_reads_header_and_groups_from_the_index(tmp_path):
    # D-8 — no header argument: `draft_step` prints the synced vault's fields.
    import inspect

    from loomcv.steps import Runtime, draft_step

    assert "header" not in inspect.signature(draft_step).parameters
    db, _pr = _index(tmp_path, "header_vault.md")
    apps = tmp_path / "apps"
    cfg = tmp_path / "generation.yaml"
    cfg.write_text(_generation_yaml(apps), encoding="utf-8")
    runtime = Runtime(db=db, chroma=tmp_path / "chroma", generation_config_path=cfg)

    result = draft_step(runtime, _payload(jobs=[]), out_dir=tmp_path / "out")

    resume = result.resume_path.read_text(encoding="utf-8")
    assert resume.startswith("# Mara Voss\n\n")
    assert "**Senior Web Platform Engineer**" in resume
    assert "**Platforms and Delivery:** Kubernetes; Terraform; GitHub Actions; Python\n" in resume
    assert result.skills == 7


def test_l3a_user_template_may_reorder_or_omit_slots(tmp_path):
    # D-9 — Education before Summary, no headline: the slots are independent.
    from loomcv.draft import render_resume

    template = tmp_path / "student.md"
    template.write_text(
        "# {{ name }}\n\n"
        "{% if education %}## Education\n\n{{ education | join('\\n') }}\n\n{% endif %}"
        "{% if summary %}## Summary\n\n{{ summary }}\n\n{% endif %}"
        "## Experience\n"
        "{% for job in jobs %}{% for b in job.bullets %}"
        "- {{ b.label }}{{ b.text }} <!-- src: {{ b.bullet_id }} -->\n"
        "{% endfor %}{% endfor %}",
        encoding="utf-8",
    )
    payload, rows = _l3a_inputs()
    resume = render_resume(payload, rows, _l3a_groups(), _l3a_header(), template_path=template)
    assert resume.index("## Education") < resume.index("## Summary") < resume.index("## Experience")
    assert "Senior Web Platform Engineer" not in resume
    default = render_resume(payload, rows, _l3a_groups(), _l3a_header())
    assert _resume_bullet_lines(resume) == _resume_bullet_lines(default)


def test_report_required_not_shown_lists_missing_parts():
    # Section 2 lists the named parts the card counts as missing, one entry
    # per requirement line, so the report and the card agree.
    from loomcv.draft import render_report

    payload = _payload(
        lines=[
            _line(
                "Build Go and Node and Rust services",
                "must",
                "exact",
                parts=[("Go", "none"), ("Node", "exact"), ("Rust", "none")],
            ),
            _line("Plain thing", "must", "exact"),
        ]
    )
    section = _section(render_report(payload), "## 2 Required, not shown")
    body = [ln for ln in section.splitlines() if ln.strip()]

    assert body == [
        '- Go and Rust — the posting asks for them here: "Build Go and Node and Rust services"'
    ]
