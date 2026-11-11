"""Tests for `loomcv.complete` — the completion loop.

Cases: W-1 … W-8 (the workbook), S-1 … S-5 (the selection call; S-6 lives in `test_index.py`),
U-1 / U-2 (packaging). Everything goes through the module — answers are built as Python objects
(`JobsAnswers` / `BulletsAnswers`, no file format) — and every model reply comes from an
in-process scripted fake; the autouse `_no_network` fixture refuses any HTTP call. The `ui`
dependency group is never needed. `loomcv.complete` is imported **inside** each test so the red
run fails per test on `ModuleNotFoundError`.

The draft vaults are the two shared expected L1 outputs: `ingest_expected_vault.md` (the short
Sam Example résumé: Acme Widgets with `Defaults`, Freelance Developer with unknown fields) and
`claims_expected_vault.md` (the second persona: a vault-level `Skills` line, no `Defaults`, one
bullet-less job). All content is fictional.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from loomcv import resources
from loomcv.sync.parse_assets import parse_star_vault

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
SELECT_PROMPT = resources.path("prompts", "select_keywords.md")
SAM = (FIXTURES / "ingest_expected_vault.md").read_text(encoding="utf-8")
CLAIMS = (FIXTURES / "claims_expected_vault.md").read_text(encoding="utf-8")

TODAY = date(2026, 10, 12)

ACME = "acme-widgets-platform-engineer"
FREE = "freelance-developer"
HARROW = "harrowgate-mutual-insurance-claims-operations-manager"
PELL = "pellworth-general-insurance-senior-claims-adjuster"
TALLIS = "tallis-reed-loss-adjusting-claims-trainee"

# The Sam Example canned selection reply (handles over the draft bullets, in file order).
SEL_SAM = {
    "J1b1": {"defaults": ["Python", "django"], "own": ["task queue", "API performance"]},
    "J1b2": {"defaults": [], "own": ["error reduction"]},
    "J1b3": {"defaults": ["PostgreSQL", "Redis"], "own": ["invoicing", "a; b"]},
    "J2b1": {"defaults": [], "own": ["web development"]},
}


# The checklist's header items of a draft without a header block.
HEADER_ITEMS = [
    ("header_field", None, None),
    ("header_field", None, None),
    ("header_optional", None, None),
    ("header_optional", None, None),
    ("header_optional", None, None),
]


def _sel_claims() -> dict:
    """The claims canned reply: nine bullets in each of the first two jobs, no `Defaults`
    anywhere, so every `defaults` list is empty and each bullet gets its own keywords."""
    return {
        f"J{j}b{k}": {"defaults": [], "own": [f"claims topic {j}.{k}", f"method {j}.{k}"]}
        for j in (1, 2)
        for k in range(1, 10)
    }


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("network call in test")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


class FakeGenerationAdapter:
    """Scripted, network-free stand-in for `GenerationAdapter`."""

    def __init__(self, replies: list[str], *, temperature: float = 0) -> None:
        self._replies = list(replies)
        self._temperature = temperature
        self.calls: list[tuple[str, str, float]] = []

    @property
    def model_id(self) -> str:
        return "fake-model"

    @property
    def temperature(self) -> float:
        return self._temperature

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        self.calls.append((system, user, temperature))
        return self._replies.pop(0)


def _scripted(*replies) -> FakeGenerationAdapter:
    return FakeGenerationAdapter([r if isinstance(r, str) else json.dumps(r) for r in replies])


def _confirmed_jobs(text: str):
    """The workbook with every job's `Defaults` confirmed as they stand (stage A done)."""
    from loomcv.complete import accept_all, apply_jobs, load_workbook

    wb = load_workbook(text)
    return apply_jobs(wb, accept_all(wb)[0])


def _bullet(wb, job_id: str, local_id: str):
    job = next(j for j in wb.jobs if j.job_id == job_id)
    return next(b for b in job.bullets if b.id == local_id)


# ------------------------------------------------------------------ W-1 load


def test_load_workbook_counts_skills_and_unknown_fields():
    # W-1
    from loomcv.complete import CompleteError, load_workbook

    wb = load_workbook(SAM)
    assert [j.job_id for j in wb.jobs] == [ACME, FREE]
    assert sum(len(j.bullets) for j in wb.jobs) == 4
    assert wb.skills == []
    freelance = wb.jobs[1]
    assert freelance.date_from == freelance.date_to == "0000"
    assert freelance.employer is None
    assert wb.jobs[0].defaults == ["Python", "Django", "PostgreSQL"]
    assert all(b.status == "draft" for j in wb.jobs for b in j.bullets)
    assert all(j.defaults_confirmed is False for j in wb.jobs)
    assert wb.jobs[0].bullets[2].project == "Billing Revamp"

    claims = load_workbook(CLAIMS)
    assert [j.job_id for j in claims.jobs] == [HARROW, PELL, TALLIS]
    assert sum(len(j.bullets) for j in claims.jobs) == 18
    skills_line = re.search(r"^\* \*\*Skills\*\*: (.+)$", CLAIMS, flags=re.MULTILINE)
    assert skills_line is not None
    assert claims.skills == skills_line.group(1).split("; ")
    assert len(claims.skills) == 14
    assert claims.jobs[2].bullets == []

    # File order is kept (the parser sorts by date; the workbook undoes that).
    older_first = (
        "# Vault\n\n"
        "## [Job] Old\n* **From**: 2010\n* **To**: 2011\n* **Bullets**:\n"
        "  - Old text. {id: aaa111, status: draft}\n\n"
        "## [Job] New\n* **From**: 2020\n* **To**: present\n* **Bullets**:\n"
        "  - New text. {id: bbb222, status: draft}\n"
    )
    assert [j.name for j in parse_star_vault(older_first, include_draft=True).jobs] == [
        "New",
        "Old",
    ]
    assert [j.name for j in load_workbook(older_first).jobs] == ["Old", "New"]

    # A parse error (a bullet without an id) is a CompleteError, never a partial workbook.
    with pytest.raises(CompleteError):
        load_workbook(SAM.replace(" {id: e8eb3d, status: draft, updated: 2026-10-10}", ""))


# ------------------------------------------------------------------ W-2 checklist


def test_checklist_lists_exactly_fields_defaults_and_drafts():
    # W-2
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        accept_all,
        apply_bullets,
        apply_jobs,
        checklist,
        load_workbook,
    )

    wb = load_workbook(SAM)
    items = checklist(wb)
    # The header items come first — the draft has no header block.
    assert [(i.kind, i.job_id, i.bullet_id) for i in items] == [
        *HEADER_ITEMS,
        ("job_defaults", ACME, None),
        ("bullet_draft", ACME, f"{ACME}::395f62"),
        ("bullet_draft", ACME, f"{ACME}::e8eb3d"),
        ("bullet_draft", ACME, f"{ACME}::b37b54"),
        ("job_field", FREE, None),
        ("job_field", FREE, None),
        ("job_field", FREE, None),
        ("job_defaults", FREE, None),
        ("bullet_draft", FREE, f"{FREE}::174378"),
    ]
    assert Counter(i.kind for i in items) == {
        "header_field": 2,
        "header_optional": 3,
        "job_field": 3,
        "job_defaults": 2,
        "bullet_draft": 4,
    }
    assert all(i.message for i in items)

    claims_items = checklist(load_workbook(CLAIMS))
    assert Counter(i.kind for i in claims_items) == {
        "header_field": 2,
        "header_optional": 3,
        "job_defaults": 3,
        "bullet_draft": 18,
    }

    # A confirmed bullet with an empty Keywords list is `bullet_keywords` (not `bullet_draft`);
    # a confirmed bullet with keywords produces no item; unconfirmed drafts stay `bullet_draft`.
    jobs_answers, _ = accept_all(wb)
    wb2 = apply_jobs(wb, jobs_answers)
    wb3 = apply_bullets(
        wb2,
        BulletsAnswers(
            bullets={
                f"{ACME}::395f62": BulletAnswer(keywords=[], confirm=True),
                f"{ACME}::e8eb3d": BulletAnswer(keywords=["error reduction"], confirm=True),
            }
        ),
    )
    after = [(i.kind, i.bullet_id) for i in checklist(wb3) if i.kind.startswith("bullet_")]
    assert after == [
        ("bullet_keywords", f"{ACME}::395f62"),
        ("bullet_draft", f"{ACME}::b37b54"),
        ("bullet_draft", f"{FREE}::174378"),
    ]
    assert not [i for i in checklist(wb3) if i.kind == "job_defaults"]


# ------------------------------------------------------------------ W-3 / W-4 jobs


def test_apply_jobs_sets_fields_replaces_defaults_and_appends_skills():
    # W-3
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        JobAnswer,
        JobsAnswers,
        apply_bullets,
        apply_jobs,
        load_workbook,
        skills_unassigned,
    )

    wb = load_workbook(CLAIMS)
    snapshot = wb.model_dump()
    wb2 = apply_jobs(
        wb,
        JobsAnswers(
            jobs={
                TALLIS: JobAnswer(
                    employer="Tallis & Reed (Loss Adjusting)",
                    title="Claims Trainee II",
                    date_to="2014-01",
                ),
                HARROW: JobAnswer(
                    defaults=["Claims triage"],
                    assign_skills=["Power BI", "reserving"],
                    defaults_confirmed=True,
                ),
            }
        ),
    )
    assert wb.model_dump() == snapshot  # the input workbook is not mutated

    tallis = next(j for j in wb2.jobs if j.job_id == TALLIS)
    assert (tallis.employer, tallis.title, tallis.date_to) == (
        "Tallis & Reed (Loss Adjusting)",
        "Claims Trainee II",
        "2014-01",
    )
    assert tallis.date_from == "2011-09"  # a field the answer does not name is unchanged
    harrow = next(j for j in wb2.jobs if j.job_id == HARROW)
    assert harrow.defaults == ["Claims triage", "Power BI", "Reserving"]  # the Skills spelling
    assert harrow.defaults_confirmed is True
    assert next(j for j in wb2.jobs if j.job_id == PELL).defaults_confirmed is False
    assert wb2.skills == wb.skills
    assert len(wb2.skills) == 14
    assert len(skills_unassigned(wb)) == 14
    assert len(skills_unassigned(wb2)) == 12
    assert {"Power BI", "Reserving"}.isdisjoint(skills_unassigned(wb2))

    # A default removed from the job leaves every bullet of the job; an added one
    # joins no bullet (it appears in each bullet's unselected group, computed by `groups`).
    sam = load_workbook(SAM)
    sam = apply_bullets(
        sam,
        BulletsAnswers(
            bullets={
                f"{ACME}::395f62": BulletAnswer(keywords=["python", "django", "task queue"]),
                f"{ACME}::e8eb3d": BulletAnswer(keywords=["Django", "error reduction"]),
                f"{FREE}::174378": BulletAnswer(keywords=["Django", "web development"]),
            }
        ),
    )
    trimmed = apply_jobs(
        sam, JobsAnswers(jobs={ACME: JobAnswer(defaults=["Python", "PostgreSQL", "Redis"])})
    )
    acme = trimmed.jobs[0]
    assert acme.defaults == ["Python", "PostgreSQL", "Redis"]
    assert acme.bullets[0].keywords == ["python", "task queue"]
    assert acme.bullets[1].keywords == ["error reduction"]
    assert all("Redis" not in b.keywords for b in acme.bullets)
    # Another job's bullets are untouched.
    assert trimmed.jobs[1].bullets[0].keywords == ["Django", "web development"]


def test_apply_jobs_unknown_job_id_raises():
    # W-4
    from loomcv.complete import CompleteError, JobAnswer, JobsAnswers, apply_jobs, load_workbook

    wb = load_workbook(CLAIMS)
    bad_answers = [
        JobsAnswers(jobs={"no-such-job": JobAnswer(defaults_confirmed=True)}),
        JobsAnswers(jobs={HARROW: JobAnswer(date_from="2021-13")}),
        JobsAnswers(jobs={HARROW: JobAnswer(assign_skills=["Underwriting"])}),
        JobsAnswers(jobs={HARROW: JobAnswer(defaults=["a; b"])}),
    ]
    for answers in bad_answers:
        with pytest.raises(CompleteError):
            apply_jobs(wb, answers)


# ------------------------------------------------------------------ W-5 groups


def test_groups_partition_a_hand_built_bullet():
    # W-5
    from loomcv.complete import Groups, WbBullet, WbJob, groups

    job = WbJob(
        job_id="j",
        name="J",
        date_from="2020",
        date_to="2021",
        defaults=["Python", "Django", "PostgreSQL", "Redis"],
        bullets=[],
    )
    bullet = WbBullet(id="aaa111", text="t", keywords=["django", "task queue", "Python", "latency"])
    assert groups(job, bullet) == Groups(
        selected_defaults=["Python", "Django"],
        unselected_defaults=["PostgreSQL", "Redis"],
        own=["task queue", "latency"],
    )


# ------------------------------------------------------------------ W-6 bullets


def test_apply_bullets_confirm_stamps_today_and_text_edit_keeps_id():
    # W-6
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        CompleteError,
        apply_bullets,
        load_workbook,
        render_workbook,
    )

    wb = load_workbook(SAM)
    original = {
        b.bullet_id: b for j in parse_star_vault(SAM, include_draft=True).jobs for b in j.bullets
    }
    edited_text = "Reduced checkout errors on the public storefront by 41%."
    wb2 = apply_bullets(
        wb,
        BulletsAnswers(
            bullets={
                f"{ACME}::395f62": BulletAnswer(confirm=True),
                f"{ACME}::e8eb3d": BulletAnswer(text=edited_text),
                f"{ACME}::b37b54": BulletAnswer(project=None),
            }
        ),
    )
    rendered = render_workbook(wb2, today=TODAY)
    parsed = parse_star_vault(rendered, include_draft=True)
    assert parsed.errors() == []
    got = {b.bullet_id: b for j in parsed.jobs for b in j.bullets}

    confirmed = got[f"{ACME}::395f62"]
    assert confirmed.status == "confirmed"
    assert confirmed.updated == TODAY
    untouched = got[f"{FREE}::174378"]
    assert untouched.status == "draft"
    assert untouched.updated == date(2026, 10, 10)

    edited = got[f"{ACME}::e8eb3d"]
    assert edited.text == edited_text
    assert edited.content_hash != original[f"{ACME}::e8eb3d"].content_hash  # the id stayed
    assert got[f"{ACME}::b37b54"].project is None  # an explicit None clears the label
    assert original[f"{ACME}::b37b54"].project == "Billing Revamp"

    for answers in (
        BulletsAnswers(bullets={f"{ACME}::nope00": BulletAnswer(confirm=True)}),
        BulletsAnswers(bullets={f"{ACME}::395f62": BulletAnswer(text="Has {braces} in it.")}),
        BulletsAnswers(bullets={f"{ACME}::395f62": BulletAnswer(keywords=["a; b"])}),
    ):
        with pytest.raises(CompleteError):
            apply_bullets(wb, answers)


# ------------------------------------------------------------------ W-7 bullet-less job


def test_bulletless_job_has_no_bullet_items_and_renders_empty_block():
    # W-7
    from loomcv.complete import checklist, load_workbook, render_workbook

    wb = load_workbook(CLAIMS)
    assert [i.kind for i in checklist(wb) if i.job_id == TALLIS] == ["job_defaults"]

    rendered = render_workbook(wb, today=TODAY)
    assert rendered.endswith(
        "## [Job] Tallis & Reed Loss Adjusting — Claims Trainee\n"
        "* **Employer**: Tallis & Reed Loss Adjusting\n"
        "* **Title**: Claims Trainee\n"
        "* **From**: 2011-09\n"
        "* **To**: 2013-12\n"
        "* **Bullets**:\n"
    )
    parsed = parse_star_vault(rendered, include_draft=True)
    assert parsed.errors() == []
    assert [i.severity for i in parsed.issues] == ["warning"]
    assert "empty job" in parsed.issues[0].message


# ------------------------------------------------------------------ W-8 round trip


def test_accept_all_round_trip_parses_clean_with_all_confirmed(tmp_path):
    # W-8
    from loomcv.complete import (
        accept_all,
        apply_bullets,
        apply_jobs,
        load_workbook,
        render_workbook,
        verify_rendered,
        write_workbook,
    )

    for text, jobs, bullets, bulletless in ((SAM, 2, 4, 0), (CLAIMS, 3, 18, 1)):
        wb = load_workbook(text)
        jobs_answers, bullets_answers = accept_all(wb)
        wb2 = apply_jobs(wb, jobs_answers)
        assert all(j.defaults_confirmed for j in wb2.jobs)
        assert [j.defaults for j in wb2.jobs] == [j.defaults for j in wb.jobs]  # nothing assigned
        wb3 = apply_bullets(wb2, bullets_answers)
        assert all(b.status == "confirmed" for j in wb3.jobs for b in j.bullets)
        assert [b.keywords for j in wb3.jobs for b in j.bullets] == [
            b.keywords for j in wb.jobs for b in j.bullets
        ]  # keywords unchanged

        rendered = render_workbook(wb3, today=TODAY)
        problems, confirmed = verify_rendered(rendered, wb3)
        assert problems == []
        assert confirmed == bullets

        before = parse_star_vault(text, include_draft=True)
        after = parse_star_vault(rendered, include_draft=True)
        assert [j.job_id for j in after.jobs] == [j.job_id for j in before.jobs]
        assert [b.bullet_id for j in after.jobs for b in j.bullets] == [
            b.bullet_id for j in before.jobs for b in j.bullets
        ]
        assert after.skills == before.skills == wb.skills
        assert after.errors() == []

        out = tmp_path / "nested" / "vault.md"
        report = write_workbook(wb3, out, today=TODAY)
        assert out.read_text(encoding="utf-8") == rendered
        assert (report.jobs, report.bullets, report.confirmed) == (jobs, bullets, bullets)
        assert report.jobs_without_bullets == bulletless
        assert report.skills_unassigned == len(wb.skills)
        assert report.problems == []
        out.unlink()


# ------------------------------------------------------------------ S selection


def test_select_prompt_carries_defaults_and_every_handle():
    # S-1
    from loomcv.complete import select_keywords

    # The "refused until every
    # job's Defaults are confirmed" block is gone — the tools list is no longer a gate; the
    # unconfirmed case is F3-1 below.
    adapter = _scripted(SEL_SAM)
    wb, warnings = select_keywords(
        _confirmed_jobs(SAM), adapter=adapter, template_path=SELECT_PROMPT
    )
    assert len(adapter.calls) == 1
    system, user, _ = adapter.calls[0]
    assert '("Job defaults")' in system
    assert '"defaults" contains only items copied exactly from that job\'s defaults list.' in system
    assert "Job defaults: Python; Django; PostgreSQL" in user
    assert "Job defaults: (none)" in user
    handles = {
        "J1b1": "Cut the p95 latency",
        "J1b2": "Reduced checkout errors",
        "J1b3": "Rebuilt the invoice pipeline",
        "J2b1": "Built booking websites",
    }
    for handle, start in handles.items():
        assert user.count(f"- {handle}: ") == 1
        assert f"- {handle}: {start}" in user
    assert isinstance(warnings, list)
    assert [b.status for j in wb.jobs for b in j.bullets] == ["draft"] * 4


def test_selection_reply_sets_keywords_defaults_order_then_own():
    # S-2
    from loomcv.complete import select_keywords

    wb, _ = select_keywords(
        _confirmed_jobs(SAM), adapter=_scripted(SEL_SAM), template_path=SELECT_PROMPT
    )
    first = _bullet(wb, ACME, "395f62")
    # The job's spelling and `Defaults` order, even though the reply said "django" second.
    assert first.keywords == ["Python", "Django", "task queue", "API performance"]
    assert first.status == "draft"

    # A second run keeps what a bullet already carries, at the tail (the model only adds).
    again = dict(SEL_SAM)
    again["J1b1"] = {"defaults": ["PostgreSQL"], "own": ["new thing"]}
    wb2, _ = select_keywords(wb, adapter=_scripted(again), template_path=SELECT_PROMPT)
    assert _bullet(wb2, ACME, "395f62").keywords == [
        "PostgreSQL",
        "new thing",
        "Python",
        "Django",
        "task queue",
        "API performance",
    ]

    # Case-insensitive dedup keeps the first spelling.
    dedup = dict(SEL_SAM)
    dedup["J1b1"] = {"defaults": ["django", "PYTHON"], "own": ["Task queue", "task QUEUE"]}
    wb3, _ = select_keywords(
        _confirmed_jobs(SAM), adapter=_scripted(dedup), template_path=SELECT_PROMPT
    )
    assert _bullet(wb3, ACME, "395f62").keywords == ["Python", "Django", "Task queue"]


def test_selection_drops_unknown_default_with_warning():
    # S-3
    from loomcv.complete import select_keywords

    wb, warnings = select_keywords(
        _confirmed_jobs(SAM), adapter=_scripted(SEL_SAM), template_path=SELECT_PROMPT
    )
    assert "dropped default 'Redis' for J1b3 (not in the job's Defaults)" in warnings
    assert "dropped keyword 'a; b' (contains ;)" in warnings
    third = _bullet(wb, ACME, "b37b54")
    assert third.keywords == ["PostgreSQL", "invoicing"]
    assert all("Redis" not in b.keywords for j in wb.jobs for b in j.bullets)


def test_selection_missing_or_repeated_handle_retries_once_then_raises():
    # S-4
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        CompleteError,
        apply_bullets,
        select_keywords,
    )

    valid = json.dumps(SEL_SAM)
    missing = json.dumps({k: v for k, v in SEL_SAM.items() if k != "J2b1"})
    repeated = (
        '{"J1b1": {"defaults": [], "own": []}, "J1b1": {"defaults": [], "own": []},'
        ' "J1b2": {"defaults": [], "own": []}, "J1b3": {"defaults": [], "own": []},'
        ' "J2b1": {"defaults": [], "own": []}}'
    )
    unknown = json.dumps({**SEL_SAM, "J9b9": {"defaults": [], "own": []}})

    for bad in (missing, repeated):
        adapter = _scripted(bad, valid)
        wb, _ = select_keywords(_confirmed_jobs(SAM), adapter=adapter, template_path=SELECT_PROMPT)
        assert len(adapter.calls) == 2
        assert "was not valid" in adapter.calls[1][1]
        assert "was not valid" not in adapter.calls[0][1]
        assert _bullet(wb, FREE, "174378").keywords == ["web development"]

    adapter = _scripted(missing, unknown)
    with pytest.raises(CompleteError) as exc:
        select_keywords(_confirmed_jobs(SAM), adapter=adapter, template_path=SELECT_PROMPT)
    assert exc.value.raws == [missing, unknown]
    assert len(adapter.calls) == 2

    # A confirmed bullet is neither in the prompt nor touched; the handles
    # are numbered over the draft bullets only.
    wb = apply_bullets(
        _confirmed_jobs(SAM),
        BulletsAnswers(
            bullets={f"{ACME}::395f62": BulletAnswer(keywords=["kept by the user"], confirm=True)}
        ),
    )
    three = {
        "J1b1": {"defaults": ["Python"], "own": ["x"]},
        "J1b2": {"defaults": [], "own": []},
        "J2b1": {"defaults": [], "own": ["y"]},
    }
    adapter = _scripted(three)
    wb2, _ = select_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)
    user = adapter.calls[0][1]
    assert "Cut the p95 latency" not in user
    assert user.count("- J1b1: Reduced checkout errors") == 1
    assert "J1b3" not in user
    assert _bullet(wb2, ACME, "395f62").keywords == ["kept by the user"]
    assert _bullet(wb2, ACME, "395f62").status == "confirmed"
    assert _bullet(wb2, ACME, "e8eb3d").keywords == ["Python", "x"]

    # With no draft bullet left there is no call at all. (A confirmed bullet with no keyword
    # is taken too — so each one here has some.)
    all_confirmed = apply_bullets(
        wb2,
        BulletsAnswers(
            bullets={
                f"{j.job_id}::{b.id}": BulletAnswer(confirm=True, keywords=b.keywords or ["k"])
                for j in wb2.jobs
                for b in j.bullets
            }
        ),
    )
    silent = _scripted()
    same, warnings = select_keywords(all_confirmed, adapter=silent, template_path=SELECT_PROMPT)
    assert silent.calls == []
    assert warnings == []
    assert same == all_confirmed


def test_bulletless_job_contributes_no_handle():
    # S-5
    from loomcv.complete import select_keywords

    adapter = _scripted(_sel_claims())
    wb, _ = select_keywords(_confirmed_jobs(CLAIMS), adapter=adapter, template_path=SELECT_PROMPT)
    user = adapter.calls[0][1]
    wanted = [f"J{j}b{k}" for j in (1, 2) for k in range(1, 10)]
    assert re.findall(r"^- (J\d+b\d+): ", user, flags=re.MULTILINE) == wanted
    assert "### Job 3" not in user
    tallis = next(j for j in wb.jobs if j.job_id == TALLIS)
    assert tallis.bullets == []
    assert tallis.defaults == []
    assert _bullet(wb, HARROW, "d0c625").keywords == ["claims topic 1.1", "method 1.1"]


# ------------------------------------------------------------------ U packaging


def test_gradio_imported_only_in_loomcv_ui():
    # U-1
    pattern = re.compile(r"^\s*(?:import gradio|from gradio)", re.MULTILINE)
    hits = sorted(
        path.relative_to(REPO).as_posix()
        for path in (REPO / "loomcv").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    )
    # The files that import gradio: the pages, the frame (`shell.py`) and the shared `common.py`
    assert hits == [
        "loomcv/ui/app.py",
        "loomcv/ui/common.py",
        "loomcv/ui/review.py",
        "loomcv/ui/shell.py",
    ]


def test_complete_and_cli_import_without_gradio():
    # U-2 — run in a fresh interpreter with gradio blocked in `sys.modules`, so the
    # re-imports cannot leave a second copy of `loomcv.cli` behind for the other test modules.
    code = (
        "import sys\n"
        "sys.modules['gradio'] = None\n"
        "import loomcv.complete\n"
        "import loomcv.cli\n"
        "assert 'gradio' not in [m for m, v in sys.modules.items() if v is not None]\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_review_application_and_cli_import_without_gradio():
    # U-3 — same shape as U-2, for the review package and the boundary it calls
    code = (
        "import sys\n"
        "sys.modules['gradio'] = None\n"
        "import loomcv.review\n"
        "import loomcv.review.doc\n"
        "import loomcv.review.store\n"
        "import loomcv.application\n"
        "import loomcv.cli\n"
        "assert 'gradio' not in [m for m, v in sys.modules.items() if v is not None]\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ----------------------------------- stage D: header + Skills groups (W-1 … W-11)

GROUPS_PROMPT = resources.path("prompts") / "propose_groups.md"  # new file: no check at import
SAM_H1 = "# Draft STAR Vault — generated by pipeline ingest, 2026-10-10\n"
# The Sam Example draft with the header block L1 would extract and a flat Skills line.
SAM_HEADED = SAM.replace(
    SAM_H1,
    SAM_H1
    + "* **Name**: Sam Example\n"
    + "* **Contact**: Example City; sam@example.invalid\n"
    + "* **Education**: B.Sc. Computer Science, Example University\n"
    + "* **Skills**: Go; python\n",
)


def _groups(wb) -> list[tuple[str, list[str]]]:
    return [(g.name, g.items) for g in wb.skill_groups]


def test_l3a_apply_header_sets_fields_newline_becomes_space():
    # W-1 — the only normalisation: a newline inside a one-line field → a space.
    from loomcv.complete import CompleteError, HeaderAnswers, apply_header, load_workbook

    wb = load_workbook(SAM_HEADED)
    out = apply_header(
        wb,
        HeaderAnswers(
            name="Sam\nExample",
            headline="Senior\r\nPlatform Engineer ",
            summary="Ten years\nof work.",
            contact=["Example City", "sam@example.invalid", "+1 000"],
        ),
    )
    assert out.header.name == "Sam Example"
    assert out.header.headline == "Senior Platform Engineer"
    assert out.header.summary == "Ten years of work."
    assert out.header.contact == ["Example City", "sam@example.invalid", "+1 000"]
    assert out.header.education == wb.header.education  # not set → untouched
    assert wb.header.headline is None  # pure
    assert out.jobs == wb.jobs and out.skill_groups == wb.skill_groups

    assert apply_header(out, HeaderAnswers(headline="")).header.headline is None
    with pytest.raises(CompleteError):
        apply_header(wb, HeaderAnswers(contact=["a; b"]))


def test_l3a_apply_skill_groups_as_given():
    # W-2 — no code rule reorders or dedups across groups; one group's casefold
    # duplicates are the only dedup; an emptied group is dropped.
    from loomcv.complete import CompleteError, SkillsAnswers, apply_skill_groups, load_workbook
    from loomcv.sync.parse_assets import SkillGroup

    wb = load_workbook(SAM_HEADED)
    assert _groups(wb) == [("", ["Go", "python"])]
    given = [
        SkillGroup(name="", items=["SQL"]),
        SkillGroup(name="Languages", items=["python", "Go", "go"]),
        SkillGroup(name="Data\nand More", items=["python", "PostgreSQL"]),
        SkillGroup(name="Empty", items=[]),
    ]
    out = apply_skill_groups(wb, SkillsAnswers(groups=given))
    assert _groups(out) == [
        ("", ["SQL"]),
        ("Languages", ["python", "Go"]),
        ("Data and More", ["python", "PostgreSQL"]),
    ]
    assert out.skills == ["SQL", "python", "Go", "PostgreSQL"]
    assert _groups(wb) == [("", ["Go", "python"])]  # pure

    for bad in (
        [SkillGroup(name="A*B", items=["x"])],
        [SkillGroup(name="Cloud", items=["x"]), SkillGroup(name="cloud", items=["y"])],
        [SkillGroup(name="Cloud", items=["x"]), SkillGroup(name="", items=["y"])],
        [SkillGroup(name="Cloud", items=["x; y"])],
    ):
        with pytest.raises(CompleteError):
            apply_skill_groups(wb, SkillsAnswers(groups=bad))


def test_l3a_skills_prefill_is_the_three_source_union():
    # W-3 — draft skills first, then confirmed Defaults in job order.
    from loomcv.complete import load_workbook, skills_prefill

    assert skills_prefill(load_workbook(SAM_HEADED)) == ["Go", "python"]  # nothing confirmed
    confirmed = _confirmed_jobs(SAM_HEADED)
    assert skills_prefill(confirmed) == ["Go", "python", "Django", "PostgreSQL"]
    assert skills_prefill(_confirmed_jobs(SAM)) == ["Python", "Django", "PostgreSQL"]


def test_l3a_checklist_header_items():
    # W-4 — Name / Contact required, the other three optional; a full header adds none.
    from loomcv.complete import HeaderAnswers, apply_header, checklist, load_workbook

    items = checklist(load_workbook(SAM))
    head = [(i.kind, i.job_id, i.message) for i in items[:5]]
    assert [k for k, _, _ in head] == [kind for kind, _, _ in HEADER_ITEMS]
    assert all(job_id is None for _, job_id, _ in head)
    assert head[0][2] == "Name is empty" and head[1][2] == "Contact is empty"
    assert [m.split()[0] for _, _, m in head[2:]] == ["Headline", "Summary", "Education"]
    assert all("(optional)" in m for _, _, m in head[2:])
    assert not [i for i in items[5:] if i.kind.startswith("header_")]

    full = apply_header(
        load_workbook(SAM_HEADED),
        HeaderAnswers(headline="Platform Engineer", summary="Ten years of work."),
    )
    assert not [i for i in checklist(full) if i.kind.startswith("header_")]
    partial = [i.message for i in checklist(load_workbook(SAM_HEADED)) if i.job_id is None]
    assert partial == ["Headline is empty (optional)", "Summary is empty (optional)"]


def test_l3a_accept_all_with_prefill_writes_one_unnamed_group():
    # W-5 — the headless path: ingest's header as extracted, the pre-fill as one group.
    from loomcv.complete import (
        accept_all,
        apply_bullets,
        render_workbook,
        skills_prefill,
        with_prefill,
    )

    stage_a = _confirmed_jobs(SAM_HEADED)
    out = with_prefill(stage_a)
    assert _groups(out) == [("", skills_prefill(stage_a))]
    assert out.header == stage_a.header
    assert out.header.name == "Sam Example"

    done = apply_bullets(out, accept_all(out)[1])
    parsed = parse_star_vault(render_workbook(done, today=TODAY), include_draft=True)
    assert parsed.errors() == []
    assert parsed.skills == ["Go", "python", "Django", "PostgreSQL"]
    assert parsed.header.contact == ["Example City", "sam@example.invalid"]


def test_l3a_workbook_render_load_round_trip_is_byte_equal():
    # W-6 — the header and named + unnamed groups survive render → load → render.
    from loomcv.complete import (
        HeaderAnswers,
        SkillsAnswers,
        apply_header,
        apply_skill_groups,
        load_workbook,
        render_workbook,
    )
    from loomcv.sync.parse_assets import SkillGroup

    wb = apply_header(
        load_workbook(SAM_HEADED),
        HeaderAnswers(headline="Platform Engineer", summary="Ten years; mostly APIs."),
    )
    wb = apply_skill_groups(
        wb,
        SkillsAnswers(
            groups=[
                SkillGroup(name="", items=["Go"]),
                SkillGroup(name="Languages, Frameworks, and Data", items=["python", "Django"]),
            ]
        ),
    )
    first = render_workbook(wb, today=TODAY)
    again = load_workbook(first)
    assert again.header == wb.header
    assert _groups(again) == _groups(wb)
    assert render_workbook(again, today=TODAY) == first
    assert "  * **Languages, Frameworks, and Data**: python; Django\n" in first
    assert "* **Skills**: Go\n" in first


def test_l3a_workbook_skills_is_a_read_only_property():
    # W-7
    from pydantic import ValidationError

    from loomcv.complete import Workbook, load_workbook
    from loomcv.sync.parse_assets import flat_skills

    wb = load_workbook(SAM_HEADED)
    assert wb.skills == flat_skills(wb.skill_groups) == ["Go", "python"]
    with pytest.raises((AttributeError, ValueError, TypeError)):
        wb.skills = ["X"]
    with pytest.raises(ValidationError):
        Workbook(skills=["X"], jobs=[])


def test_l3a_l2_draft_through_stage_d_keeps_l2_counts(tmp_path):
    # W-8 — the L2 synthetic draft with the block: jobs / bullets / confirmed as the L2 W-8.
    from loomcv.complete import accept_all, apply_bullets, with_prefill, write_workbook

    wb = with_prefill(_confirmed_jobs(SAM_HEADED))
    wb = apply_bullets(wb, accept_all(wb)[1])
    out = tmp_path / "vault.md"
    report = write_workbook(wb, out, today=TODAY)
    assert (report.jobs, report.bullets, report.confirmed) == (2, 4, 4)
    assert report.problems == []
    text = out.read_text(encoding="utf-8")
    assert "* **Name**: Sam Example\n" in text
    assert parse_star_vault(text).header.name == "Sam Example"


def test_l3a_propose_groups_names_and_sorts():
    # W-9 — the groups come back as named, in reply order; the item set is unchanged.
    from loomcv.complete import SkillsAnswers, apply_skill_groups, propose_groups, with_prefill

    wb = with_prefill(_confirmed_jobs(SAM_HEADED))
    reply = {
        "groups": [
            {"name": "Languages", "items": ["python", "Go"]},
            {"name": "Frameworks and Data", "items": ["Django", "PostgreSQL"]},
        ]
    }
    adapter = _scripted(reply)
    groups, warnings = propose_groups(wb, adapter=adapter, template_path=GROUPS_PROMPT)
    assert len(adapter.calls) == 1
    assert warnings == []
    out = apply_skill_groups(wb, SkillsAnswers(groups=groups))
    assert _groups(out) == [
        ("Languages", ["python", "Go"]),
        ("Frameworks and Data", ["Django", "PostgreSQL"]),
    ]
    assert {s.casefold() for s in out.skills} == {s.casefold() for s in wb.skills}
    user = adapter.calls[0][1]
    for item in wb.skills:
        assert item in user


def test_l3a_propose_groups_conservation_and_errors():
    # W-10 — a left-out item returns to the unnamed group (first); an invented
    # one is dropped; an invalid reply twice is the module's error and changes nothing.
    from loomcv.complete import CompleteError, propose_groups, with_prefill

    wb = with_prefill(_confirmed_jobs(SAM_HEADED))
    before = wb.model_copy(deep=True)
    reply = {
        "groups": [
            {"name": "Languages", "items": ["PYTHON", "Rust"]},
            {"name": "Frameworks", "items": ["Django"]},
        ]
    }
    groups, warnings = propose_groups(wb, adapter=_scripted(reply), template_path=GROUPS_PROMPT)
    assert [(g.name, g.items) for g in groups] == [
        ("", ["Go", "PostgreSQL"]),
        ("Languages", ["python"]),  # the input's spelling, not the reply's
        ("Frameworks", ["Django"]),
    ]
    assert len(warnings) == 3
    assert any("Rust" in w for w in warnings)
    assert any("Go" in w for w in warnings) and any("PostgreSQL" in w for w in warnings)

    for bad in ("not json at all", json.dumps({"nogroups": []})):
        adapter = _scripted(bad, bad)
        with pytest.raises(CompleteError) as exc:
            propose_groups(wb, adapter=adapter, template_path=GROUPS_PROMPT)
        assert len(exc.value.raws) == 2
        assert len(adapter.calls) == 2
    assert wb == before


def test_l3a_accept_all_path_makes_no_grouping_call():
    # W-11 (Forbidden: a grouping call on the --accept-all path) + the prompt's rule.
    from loomcv.complete import accept_all, apply_bullets, apply_jobs, load_workbook, with_prefill

    silent = _scripted()
    wb = load_workbook(SAM_HEADED)
    wb = with_prefill(apply_jobs(wb, accept_all(wb)[0]))
    apply_bullets(wb, accept_all(wb)[1])
    assert silent.calls == []
    text = GROUPS_PROMPT.read_text(encoding="utf-8")
    assert "never add or remove an item" in text


# ------------------------------- the narrative rewrite (B-1 … B-9, A-1 … A-6, N-1 … N-4)

DRAFT_PROMPT = resources.path("prompts") / "draft_bullet.md"  # new file: no check at import
ACME_KEY = f"{ACME}::395f62"
ACME_P95 = (
    "Moved report generation to a task queue, cutting order-API p95 latency from 40 s to 6 s."
)


def _answers(**kwargs):
    from loomcv.complete import BulletDraftAnswers

    return BulletDraftAnswers(**kwargs)


def _draft(wb, adapter, *, job_id=ACME, key=ACME_KEY, answers=None, budget=200, previous=()):
    from loomcv.complete import draft_bullet

    return draft_bullet(
        wb,
        job_id,
        bullet_key=key,
        answers=answers or _answers(),
        adapter=adapter,
        budget=budget,
        template_path=DRAFT_PROMPT,
        previous=previous,
    )


def test_l3b_draft_bullet_returns_the_proposal_one_call_workbook_unchanged():
    # B-1
    from loomcv.complete import load_workbook

    wb = load_workbook(SAM)
    before = wb.model_copy(deep=True)
    adapter = FakeGenerationAdapter([json.dumps({"bullet": ACME_P95})], temperature=0.7)

    proposal, warnings = _draft(wb, adapter)

    assert proposal == ACME_P95
    assert warnings == []
    assert len(adapter.calls) == 1
    assert adapter.calls[0][2] == 0.7  # the node's temperature, not a literal
    assert wb == before


def test_l3b_draft_prompt_carries_job_text_answers_keywords_budget_only():
    # B-2
    from loomcv.complete import BulletAnswer, BulletsAnswers, apply_bullets

    wb = apply_bullets(
        _confirmed_jobs(SAM),
        BulletsAnswers(
            bullets={ACME_KEY: BulletAnswer(keywords=["Python", "task queue"], confirm=True)}
        ),
    )
    adapter = _scripted({"bullet": ACME_P95})
    _draft(
        wb,
        adapter,
        answers=_answers(situation="Reports blocked the API.", action="Added a queue."),
    )

    system, user, _ = adapter.calls[0]
    assert "200" in system
    for needle in (
        "Acme Widgets — Platform Engineer",
        "Employer: Acme Widgets",
        "Title: Platform Engineer",
        "Cut the p95 latency of the order API from 40 s to 6 s",
        "Situation: Reports blocked the API.",
        "Action: Added a queue.",
        "Python",
        "task queue",
    ):
        assert needle in user
    assert "Result:" not in user
    for other in (
        "Reduced checkout errors",
        "Rebuilt the invoice pipeline",
        "Built booking websites",
        "Freelance Developer",
    ):
        assert other not in user


def test_l3b_draft_with_nothing_to_write_from_makes_no_call():
    # B-3
    from loomcv.complete import CompleteError, load_workbook

    wb = load_workbook(SAM)
    adapter = _scripted({"bullet": "x"})
    blank = _answers(situation="  ", action="\n", result=" ")
    with pytest.raises(CompleteError):
        _draft(wb, adapter, key=None, answers=blank)
    with pytest.raises(CompleteError):
        _draft(wb, adapter, job_id="no-such-job", key=None, answers=_answers(action="did it"))
    with pytest.raises(CompleteError):  # a bullet key of another job
        _draft(wb, adapter, key=f"{FREE}::174378", answers=_answers(action="did it"))
    assert adapter.calls == []


def test_l3b_draft_invalid_reply_retries_once_then_raises():
    # B-4
    from loomcv.complete import CompleteError, load_workbook

    wb = load_workbook(SAM)
    answers = _answers(action="moved reports to a queue")
    for bad in ("not json", {"text": "x"}, {"bullet": 7}, {"bullet": "   "}):
        adapter = _scripted(bad, bad)
        with pytest.raises(CompleteError) as exc:
            _draft(wb, adapter, key=None, answers=answers)
        assert len(exc.value.raws) == 2
        assert len(adapter.calls) == 2
        assert "not valid" in adapter.calls[1][1]

    adapter = _scripted("not json", {"bullet": "Moved reports to a queue."})
    proposal, warnings = _draft(wb, adapter, key=None, answers=answers)
    assert proposal == "Moved reports to a queue."
    assert warnings == []
    assert len(adapter.calls) == 2


def test_l3b_draft_invented_figure_retries_then_warns():
    # B-5 — figure conservation: retry once naming it, then show with a warning.
    from loomcv.complete import load_workbook

    wb = load_workbook(SAM)
    answers = _answers(
        action="moved report generation to a queue", result="p95 latency fell from 900 ms to 300 ms"
    )
    invented = "Moved report generation to a queue, cutting p95 latency by 70 % to 300 ms."
    adapter = _scripted({"bullet": invented}, {"bullet": invented})
    proposal, warnings = _draft(wb, adapter, key=None, answers=answers)
    assert proposal == invented
    assert warnings == ["contains a figure not in your input: 70 %"]
    assert len(adapter.calls) == 2
    assert "70 %" in adapter.calls[1][1]

    clean = "Moved report generation to a queue, cutting p95 latency from 900 ms to 300 ms."
    adapter = _scripted({"bullet": invented}, {"bullet": clean})
    proposal, warnings = _draft(wb, adapter, key=None, answers=answers)
    assert proposal == clean
    assert warnings == []
    assert len(adapter.calls) == 2


def test_l3b_draft_figures_present_in_input_never_warn():
    # B-6 — spacing / case differences are not misses; adding a sign the input lacks is.
    from loomcv.complete import load_workbook

    wb = load_workbook(SAM)
    answers = _answers(
        result="saved $1,200 a month for 10K users, 3x faster; 30 万 orders; 40+ sites"
    )
    reply = "Cut 40s to 6s, saving $1200 a month for 10k users, 3X faster, 30万 orders, 40 sites."
    adapter = _scripted({"bullet": reply})
    proposal, warnings = _draft(wb, adapter, answers=answers)
    assert proposal == reply
    assert warnings == []
    assert len(adapter.calls) == 1

    plus = "Cut 40+ seconds to 6 seconds."  # the input says `40 s`, never `40+`
    adapter = _scripted({"bullet": plus}, {"bullet": plus})
    _, warnings = _draft(wb, adapter)
    assert warnings == ["contains a figure not in your input: 40+"]


def test_l3b_draft_over_budget_warns_never_cuts():
    # B-7
    from loomcv.complete import load_workbook

    wb = load_workbook(SAM)
    long_reply = "a" * 250
    adapter = _scripted({"bullet": long_reply})
    proposal, warnings = _draft(wb, adapter, key=None, answers=_answers(action="did it"))
    assert proposal == long_reply
    assert warnings == ["over the budget (250 > 200)"]
    assert len(adapter.calls) == 1


def test_l3b_draft_previous_proposals_travel_in_the_prompt():
    # B-8
    from loomcv.complete import load_workbook

    wb = load_workbook(SAM)
    adapter = _scripted({"bullet": "Another sentence."}, {"bullet": "Another sentence."})
    _draft(wb, adapter, previous=["P1 sentence."])
    assert "Already proposed" in adapter.calls[0][1]
    assert "P1 sentence." in adapter.calls[0][1]
    _draft(wb, adapter, previous=())
    assert "Already proposed" not in adapter.calls[1][1]


def test_l3b_draft_keywords_only_the_bullets_confirmed_list():
    # B-9 — a draft bullet's keywords are stage B's proposal, not the
    # user's confirmation; a job default not on the bullet is never given.
    from loomcv.complete import BulletAnswer, BulletsAnswers, apply_bullets

    wb = apply_bullets(
        _confirmed_jobs(SAM),
        BulletsAnswers(
            bullets={
                ACME_KEY: BulletAnswer(keywords=["Python"], confirm=True),
                f"{ACME}::e8eb3d": BulletAnswer(keywords=["Storefront CMS"]),
            }
        ),
    )
    adapter = _scripted({"bullet": ACME_P95}, {"bullet": "Reduced errors."})
    _draft(wb, adapter)
    user = adapter.calls[0][1]
    assert "Keywords: Python" in user
    assert "Django" not in user and "PostgreSQL" not in user

    _draft(wb, adapter, key=f"{ACME}::e8eb3d")  # still a draft
    assert "Storefront CMS" not in adapter.calls[1][1]
    assert "Keywords: (none)" in adapter.calls[1][1]


def test_l3b_add_bullet_appends_a_draft_last_in_the_job():
    # A-1
    from loomcv.complete import add_bullet, load_workbook

    wb = load_workbook(SAM)
    before = wb.model_copy(deep=True)
    out, key = add_bullet(wb, FREE, "  Ran   weekly clinic\nworkshops.  ", project=" Clinic Sites ")

    last = out.jobs[1].bullets[-1]
    assert last.text == "Ran weekly clinic workshops."
    assert last.status == "draft"
    assert last.keywords == []
    assert last.project == "Clinic Sites"
    assert last.updated is None
    assert key == f"{FREE}::{last.id}"
    assert [b.id for b in out.jobs[1].bullets[:-1]] == [b.id for b in before.jobs[1].bullets]
    assert out.jobs[0] == before.jobs[0]
    assert wb == before


def test_l3b_add_bullet_id_is_l1s_rule():
    # A-2
    import hashlib

    from loomcv.complete import add_bullet, load_workbook
    from loomcv.sync.parse_assets import _ID_RE

    out, key = add_bullet(load_workbook(SAM), FREE, "Ran   weekly clinic workshops.")
    local = key.split("::", 1)[1]
    assert local == hashlib.sha1(b"Ran weekly clinic workshops.").hexdigest()[:6]
    assert _ID_RE.match(local)


def test_l3b_add_bullet_collision_takes_the_next_ordinal():
    # A-3
    import hashlib

    from loomcv.complete import add_bullet, load_workbook

    text = "Ran weekly clinic workshops."
    first, k1 = add_bullet(load_workbook(SAM), FREE, text)
    second, k2 = add_bullet(first, FREE, text)
    expected = hashlib.sha1(f"{text}\n2".encode()).hexdigest()[:6]
    assert k2 == f"{FREE}::{expected}"
    assert [b.id for b in second.jobs[1].bullets[:2]] == ["174378", k1.split("::", 1)[1]]
    assert second.jobs[1].bullets[-1].id == expected


def test_l3b_add_bullet_refuses_unknown_job_empty_text_brace():
    # A-4
    from loomcv.complete import CompleteError, add_bullet, load_workbook

    wb = load_workbook(SAM)
    before = wb.model_copy(deep=True)
    for job_id, text in (("no-such-job", "A bullet."), (FREE, "  \n "), (FREE, "Used {braces}.")):
        with pytest.raises(CompleteError):
            add_bullet(wb, job_id, text)
    assert wb == before


def test_l3b_add_bullet_round_trips_through_render_and_the_strict_parser():
    # A-5
    from loomcv.complete import add_bullet, load_workbook, render_workbook

    wb, key = add_bullet(
        load_workbook(SAM), FREE, "Ran weekly clinic workshops.", project="Clinics"
    )
    rendered = render_workbook(wb, today=TODAY)
    again = load_workbook(rendered)
    assert render_workbook(again, today=TODAY) == rendered
    assert parse_star_vault(rendered, include_draft=True).errors() == []
    assert f"{{id: {key.split('::', 1)[1]}, status: draft" in rendered
    assert _bullet(again, FREE, key.split("::", 1)[1]).project == "Clinics"


def test_l3b_add_bullet_checklist_draft_then_keywords():
    # A-6
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        add_bullet,
        apply_bullets,
        checklist,
        load_workbook,
    )

    wb, key = add_bullet(load_workbook(SAM), FREE, "Ran weekly clinic workshops.")
    assert ("bullet_draft", key) in [(i.kind, i.bullet_id) for i in checklist(wb)]
    wb = apply_bullets(wb, BulletsAnswers(bullets={key: BulletAnswer(confirm=True)}))
    items = [(i.kind, i.bullet_id) for i in checklist(wb)]
    assert ("bullet_keywords", key) in items
    assert ("bullet_draft", key) not in items


def test_l3b_selection_after_add_covers_only_the_new_draft():
    # N-1 — stage B is unchanged: one call over the drafts, the confirmed bullets absent.
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        add_bullet,
        apply_bullets,
        select_keywords,
    )

    wb = _confirmed_jobs(SAM)
    wb = apply_bullets(
        wb,
        BulletsAnswers(
            bullets={
                f"{j.job_id}::{b.id}": BulletAnswer(keywords=["kept"], confirm=True)
                for j in wb.jobs
                for b in j.bullets
            }
        ),
    )
    wb, key = add_bullet(wb, FREE, "Ran weekly clinic workshops.")
    adapter = _scripted({"J1b1": {"defaults": [], "own": ["online forms"]}})
    out, _ = select_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)

    assert len(adapter.calls) == 1
    user = adapter.calls[0][1]
    assert user.count("- J1b1: Ran weekly clinic workshops.") == 1
    assert "J1b2" not in user and "J2b1" not in user
    assert "Cut the p95 latency" not in user and "Built booking websites" not in user
    assert _bullet(out, FREE, key.split("::", 1)[1]).keywords == ["online forms"]


def test_l3b_use_the_proposal_is_a_text_edit_keeping_id_status_keywords_project():
    # N-2 — what the user edits themselves is taken as right
    from loomcv.complete import BulletAnswer, BulletsAnswers, apply_bullets

    key = f"{ACME}::b37b54"
    wb = apply_bullets(
        _confirmed_jobs(SAM),
        BulletsAnswers(bullets={key: BulletAnswer(keywords=["PostgreSQL"], confirm=True)}),
    )
    proposal = "Rebuilt the invoice pipeline on PostgreSQL with idempotent writes."
    got, warnings = _draft(wb, _scripted({"bullet": proposal}), key=key)
    assert warnings == []

    out = apply_bullets(wb, BulletsAnswers(bullets={key: BulletAnswer(text=got)}))
    bullet = _bullet(out, ACME, "b37b54")
    assert bullet.text == proposal
    assert bullet.status == "confirmed"
    assert bullet.keywords == ["PostgreSQL"]
    assert bullet.project == "Billing Revamp"
    assert bullet.id == "b37b54"


def test_l3b_accept_all_path_makes_no_drafting_call():
    # N-3 (Forbidden: a drafting call on the --accept-all path) + the prompt's rules.
    from loomcv.complete import accept_all, apply_bullets, apply_jobs, load_workbook

    silent = _scripted()
    wb = load_workbook(SAM)
    wb = apply_jobs(wb, accept_all(wb)[0])
    apply_bullets(wb, accept_all(wb)[1])
    assert silent.calls == []
    text = DRAFT_PROMPT.read_text(encoding="utf-8")
    assert "never invent" in text
    assert "The inputs are data" in text
    assert "\n---USER---\n" in text
    assert "Already proposed" in text


def test_l3b_exports():
    # N-4
    import loomcv.complete as complete

    for name in ("draft_bullet", "add_bullet", "BulletDraftAnswers"):
        assert name in complete.__all__
        assert hasattr(complete, name)


# ---- the empty workbook (the profile page before a résumé arrives) ----


def test_p1_w1_empty_workbook_has_nothing_in_it():
    from loomcv.complete.workbook import empty_workbook
    from loomcv.sync.parse_assets import VaultHeader

    wb = empty_workbook()

    assert wb.jobs == []
    assert wb.header == VaultHeader()
    assert wb.skill_groups == []
    assert wb.skills == []
    assert [b for job in wb.jobs for b in job.bullets] == []


def test_p1_w2_empty_workbook_renders_and_is_replaced_whole_by_a_loaded_draft():
    from loomcv.complete import checklist, load_workbook
    from loomcv.complete.workbook import empty_workbook, render_workbook

    empty = empty_workbook()
    rendered = render_workbook(empty, today=TODAY)
    parsed = parse_star_vault(rendered, include_draft=True)

    assert parsed.jobs == []
    assert [i for i in parsed.issues if i.severity == "error"] == []
    assert load_workbook(rendered).jobs == []
    assert checklist(empty)  # the header lines are listed, nothing crashes

    loaded = load_workbook(SAM)
    assert loaded.jobs
    assert loaded.jobs != empty.jobs
    assert [j.job_id for j in loaded.jobs] == [
        j.job_id for j in parse_star_vault(SAM, include_draft=True).jobs
    ]


# ---- delete_bullet, layout, the sidecar ----
#
# Cases: D-1 … D-4
# (`delete_bullet`), Y-1 … Y-3 (`layout`), K-1 … K-7 (the sidecar, `loomcv/complete/sidecar.py`).

ACME_IDS = ["395f62", "e8eb3d", "b37b54"]


def _p2a_sam():
    from loomcv.complete import load_workbook

    return load_workbook(SAM)


def test_p2a_d1_delete_bullet_removes_only_the_named_bullet():
    from loomcv.complete.workbook import delete_bullet

    wb = _p2a_sam()
    before = wb.model_copy(deep=True)

    out = delete_bullet(wb, ACME, ACME_IDS[1])

    acme = next(j for j in out.jobs if j.job_id == ACME)
    assert [b.id for b in acme.bullets] == [ACME_IDS[0], ACME_IDS[2]]
    old_acme = next(j for j in before.jobs if j.job_id == ACME)
    kept = [b for b in old_acme.bullets if b.id != ACME_IDS[1]]
    assert [b.model_dump() for b in acme.bullets] == [b.model_dump() for b in kept]
    assert [j.model_dump() for j in out.jobs if j.job_id != ACME] == [
        j.model_dump() for j in before.jobs if j.job_id != ACME
    ]
    assert out.header == before.header
    assert out.skill_groups == before.skill_groups
    assert wb == before  # pure: the input is untouched


@pytest.mark.parametrize(
    ("job_id", "bullet_id"), [("ghost-job", ACME_IDS[0]), (ACME, "nope00")], ids=["job", "bullet"]
)
def test_p2a_d2_delete_bullet_unknown_job_or_bullet_is_refused(job_id, bullet_id):
    from loomcv.complete import CompleteError
    from loomcv.complete.workbook import delete_bullet

    wb = _p2a_sam()
    before = wb.model_copy(deep=True)

    with pytest.raises(CompleteError):
        delete_bullet(wb, job_id, bullet_id)
    assert wb == before


def test_p2a_d3_deleting_the_last_bullet_keeps_the_job():
    from loomcv.complete.workbook import delete_bullet, render_workbook

    out = delete_bullet(_p2a_sam(), FREE, "174378")

    free = next(j for j in out.jobs if j.job_id == FREE)
    assert free.bullets == []
    parsed = parse_star_vault(render_workbook(out, today=TODAY), include_draft=True)
    assert FREE in [j.job_id for j in parsed.jobs]
    assert parsed.errors() == []


def test_p2a_d4_a_deleted_state_round_trips_through_render_and_load():
    from loomcv.complete import load_workbook
    from loomcv.complete.workbook import delete_bullet, render_workbook

    out = delete_bullet(_p2a_sam(), ACME, ACME_IDS[0])
    back = load_workbook(render_workbook(out, today=TODAY))

    def shape(w):
        return [(j.job_id, [(b.id, b.text) for b in j.bullets]) for j in w.jobs]

    assert shape(back) == shape(out)


def test_p2a_y1_layout_lists_job_ids_and_bullet_keys_in_order():
    from loomcv.complete import load_workbook
    from loomcv.complete.workbook import layout

    wb = load_workbook(CLAIMS)
    result = layout(wb)

    assert [job_id for job_id, _ in result] == [j.job_id for j in wb.jobs]
    assert [job_id for job_id, _ in result] == [HARROW, PELL, TALLIS]
    for (job_id, keys), job in zip(result, wb.jobs, strict=True):
        assert keys == tuple(f"{job_id}::{b.id}" for b in job.bullets)
    assert result[-1][1] == ()


def test_p2a_y2_layout_ignores_text_status_keywords_and_confirmation():
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        JobAnswer,
        JobsAnswers,
        apply_bullets,
        apply_jobs,
    )
    from loomcv.complete.workbook import layout

    wb = _p2a_sam()
    key = f"{ACME}::{ACME_IDS[0]}"
    changed = apply_jobs(
        wb, JobsAnswers(jobs={ACME: JobAnswer(employer="Other Name", defaults_confirmed=True)})
    )
    changed = apply_bullets(
        changed,
        BulletsAnswers(bullets={key: BulletAnswer(text="New text.", keywords=["x"], confirm=True)}),
    )

    assert layout(changed) == layout(wb)


def test_p2a_y3_layout_differs_after_an_add_and_after_a_delete():
    from loomcv.complete import add_bullet
    from loomcv.complete.workbook import delete_bullet, layout

    wb = _p2a_sam()
    added, _key = add_bullet(wb, ACME, "A brand new accomplishment sentence.")
    deleted = delete_bullet(wb, ACME, ACME_IDS[0])

    assert layout(added) != layout(wb)
    assert layout(deleted) != layout(wb)


def test_p2a_k1_write_then_read_round_trips_job_id_to_bool(tmp_path):
    from loomcv.complete import JobAnswer, JobsAnswers, apply_jobs
    from loomcv.complete.sidecar import read_sidecar, write_sidecar

    wb = apply_jobs(_p2a_sam(), JobsAnswers(jobs={ACME: JobAnswer(defaults_confirmed=True)}))
    path = tmp_path / "02_database" / "draft_vault.state.json"

    write_sidecar(wb, path)
    confirmed, warnings = read_sidecar(path)

    assert confirmed == {ACME: True, FREE: False}
    assert warnings == []
    assert json.loads(path.read_text(encoding="utf-8")) == {ACME: True, FREE: False}


def test_p2a_k2_with_confirmed_sets_each_job_and_defaults_the_rest_to_false():
    from loomcv.complete.sidecar import with_confirmed

    out = with_confirmed(_p2a_sam(), {ACME: True})

    assert {j.job_id: j.defaults_confirmed for j in out.jobs} == {ACME: True, FREE: False}


def test_p2a_k3_with_confirmed_ignores_a_job_that_is_not_in_the_workbook():
    from loomcv.complete.sidecar import with_confirmed

    out = with_confirmed(_p2a_sam(), {ACME: True, "ghost": True})

    assert {j.job_id: j.defaults_confirmed for j in out.jobs} == {ACME: True, FREE: False}


@pytest.mark.parametrize("content", ["not json", "[true]", '{"x": "yes"}'])
def test_p2a_k4_a_malformed_sidecar_reads_as_absent_with_one_warning(tmp_path, content):
    from loomcv.complete.sidecar import read_sidecar

    path = tmp_path / "state.json"
    path.write_text(content, encoding="utf-8")

    confirmed, warnings = read_sidecar(path)

    assert confirmed == {}
    assert len(warnings) == 1
    assert isinstance(warnings[0], str)
    assert read_sidecar(tmp_path / "missing.json") == ({}, [])


def test_p2a_k5_load_profile_decides_defaults_confirmed_by_which_file_it_reads(tmp_path):
    from loomcv.complete.sidecar import load_profile, sidecar_for

    db = tmp_path / "02_database"
    db.mkdir()
    working, saved = db / "draft_vault.md", db / "vault.md"
    working.write_text(SAM, encoding="utf-8")
    saved.write_text(SAM, encoding="utf-8")
    sidecar = sidecar_for(working)

    wb_saved, warnings = load_profile(saved, working_copy=working, sidecar=sidecar)
    assert warnings == []
    assert [j.defaults_confirmed for j in wb_saved.jobs] == [True, True]

    wb_work, _ = load_profile(working, working_copy=working, sidecar=sidecar)
    assert [j.defaults_confirmed for j in wb_work.jobs] == [False, False]

    sidecar.write_text(json.dumps({ACME: True}), encoding="utf-8")
    wb_work, _ = load_profile(working, working_copy=working, sidecar=sidecar)
    assert {j.job_id: j.defaults_confirmed for j in wb_work.jobs} == {ACME: True, FREE: False}
    wb_saved, _ = load_profile(saved, working_copy=working, sidecar=sidecar)
    assert [j.defaults_confirmed for j in wb_saved.jobs] == [True, True]


def test_p2a_k6_save_working_copy_writes_both_files_and_keeps_the_sidecar_out_of_the_vault(
    tmp_path,
):
    from loomcv.complete import JobAnswer, JobsAnswers, apply_jobs
    from loomcv.complete.sidecar import load_profile, save_working_copy, sidecar_for
    from loomcv.complete.workbook import render_workbook

    working = tmp_path / "02_database" / "draft_vault.md"
    sidecar = sidecar_for(working)
    wb = apply_jobs(_p2a_sam(), JobsAnswers(jobs={ACME: JobAnswer(defaults_confirmed=True)}))

    report = save_working_copy(wb, working, sidecar=sidecar, today=TODAY)

    assert working.is_file()
    assert sidecar.is_file()
    assert report.problems == []
    back, _ = load_profile(working, working_copy=working, sidecar=sidecar)
    assert [j.defaults_confirmed for j in back.jobs] == [True, False]
    assert working.read_text(encoding="utf-8") == render_workbook(wb, today=TODAY)
    assert "defaults_confirmed" not in working.read_text(encoding="utf-8")


def test_p2a_k7_sidecar_for_sits_beside_the_working_copy():
    from loomcv.complete.sidecar import sidecar_for

    assert sidecar_for(Path("02_database/draft_vault.md")) == Path(
        "02_database/draft_vault.state.json"
    )


# ---- open_counts, suggestion_due ----
#
# Cases: N-1 … N-5
# (`open_counts`, `loomcv/complete/checklist.py`), U-1 … U-5 (`suggestion_due`,
# `loomcv/complete/select.py`).


def _p2b_complete():
    """The Sam workbook with nothing left to check: pure data updates, no `apply_*` semantics."""
    from loomcv.complete import load_workbook
    from loomcv.sync.parse_assets import SkillGroup, VaultHeader

    wb = load_workbook(SAM)
    jobs = []
    for job in wb.jobs:
        update: dict = {
            "defaults_confirmed": True,
            "bullets": [
                b.model_copy(update={"status": "confirmed", "keywords": ["Python"]})
                for b in job.bullets
            ],
        }
        if job.job_id == FREE:
            update.update(employer="Self-employed", date_from="2019-01", date_to="2020-12")
        jobs.append(job.model_copy(update=update))
    return wb.model_copy(
        update={
            "header": VaultHeader(
                name="Sam Doe",
                contact=["sam@example.com"],
                headline="Platform engineer",
                summary="Builds platforms.",
                education=["BSc, Example University"],
            ),
            "skill_groups": [SkillGroup(name="", items=["Python"])],
            "jobs": jobs,
        }
    )


def test_p2b_n1_open_counts_of_a_complete_workbook_is_all_zero():
    from loomcv.complete.checklist import open_counts

    counts = open_counts(_p2b_complete())

    assert counts.header == 0
    assert counts.jobs == {ACME: 0, FREE: 0}
    assert list(counts.jobs) == [ACME, FREE]  # workbook order
    assert counts.skills == 0


def test_p2b_n2_a_jobs_count_is_its_checklist_items():
    from loomcv.complete import checklist
    from loomcv.complete.checklist import open_counts

    wb = _p2b_complete()
    acme = wb.jobs[0]
    bullets = [
        acme.bullets[0].model_copy(update={"status": "draft"}),
        acme.bullets[1].model_copy(update={"status": "draft"}),
        acme.bullets[2].model_copy(update={"status": "confirmed", "keywords": []}),
    ]
    acme = acme.model_copy(
        update={"employer": None, "defaults_confirmed": False, "bullets": bullets}
    )
    wb = wb.model_copy(update={"jobs": [acme, wb.jobs[1]]})

    counts = open_counts(wb)

    assert counts.jobs[ACME] == 5  # employer + tools line + two drafts + one keyword-less bullet
    assert counts.jobs[FREE] == 0
    assert sum(counts.jobs.values()) == len([i for i in checklist(wb) if i.job_id])


def test_p2b_n3_only_name_and_contact_count_for_the_header():
    from loomcv.complete.checklist import open_counts
    from loomcv.sync.parse_assets import VaultHeader

    wb = _p2b_complete().model_copy(update={"header": VaultHeader()})

    counts = open_counts(wb)

    assert counts.header == 2  # Name, Contact; Headline / Summary / Education never count
    assert counts.jobs == {ACME: 0, FREE: 0}
    assert counts.skills == 0


def test_p2b_n4_skills_count_is_one_only_when_the_list_is_empty():
    from loomcv.complete.checklist import open_counts
    from loomcv.sync.parse_assets import SkillGroup

    empty = _p2b_complete().model_copy(update={"skill_groups": []})
    filled = _p2b_complete().model_copy(
        update={"skill_groups": [SkillGroup(name="", items=["Python"])]}
    )

    assert open_counts(empty).skills == 1
    assert open_counts(filled).skills == 0


def test_p2b_n5_a_text_edit_does_not_change_the_counts():
    from loomcv.complete import BulletAnswer, BulletsAnswers, apply_bullets, load_workbook
    from loomcv.complete.checklist import open_counts

    wb = load_workbook(SAM)
    edited = apply_bullets(
        wb, BulletsAnswers(bullets={ACME_KEY: BulletAnswer(text="Cut the order API latency.")})
    )

    assert _bullet(edited, ACME, "395f62").text != _bullet(wb, ACME, "395f62").text
    assert open_counts(edited) == open_counts(wb)


def test_p2b_u1_suggestion_is_due_when_every_job_is_confirmed_and_nothing_has_keywords():
    from loomcv.complete.select import suggestion_due

    assert suggestion_due(_confirmed_jobs(SAM)) is True


def test_p2b_u2_suggestion_is_not_due_while_a_tools_line_is_unconfirmed():
    from loomcv.complete.select import suggestion_due

    wb = _confirmed_jobs(SAM)
    jobs = [wb.jobs[0], wb.jobs[1].model_copy(update={"defaults_confirmed": False})]

    assert suggestion_due(wb.model_copy(update={"jobs": jobs})) is False


def test_p2b_u3_suggestion_is_not_due_once_any_bullet_has_a_keyword():
    from loomcv.complete.select import suggestion_due

    wb = _confirmed_jobs(SAM)
    acme = wb.jobs[0]
    first = acme.bullets[0].model_copy(update={"keywords": ["Python"]})
    acme = acme.model_copy(update={"bullets": [first, *acme.bullets[1:]]})

    assert suggestion_due(wb.model_copy(update={"jobs": [acme, wb.jobs[1]]})) is False


def test_p2b_u4_the_guard_is_on_keywords_not_on_bullet_status():
    from loomcv.complete.select import suggestion_due

    wb = _confirmed_jobs(SAM)
    jobs = [
        job.model_copy(
            update={
                "bullets": [
                    b.model_copy(update={"status": "confirmed", "keywords": []})
                    for b in job.bullets
                ]
            }
        )
        for job in wb.jobs
    ]

    assert suggestion_due(wb.model_copy(update={"jobs": jobs})) is True


def test_p2b_u5_a_workbook_without_jobs_is_never_due():
    from loomcv.complete import Workbook
    from loomcv.complete.select import suggestion_due

    assert suggestion_due(Workbook(jobs=[])) is False


# ---- confirmed_counts, maybe_also, unchanged_since_entry, confirm_unedited
#
# Cases: C-1 … C-4
# (`confirmed_counts`, `loomcv/complete/checklist.py`), M-1 … M-3 (`maybe_also`), X-1 … X-5
# (`unchanged_since_entry` / `confirm_unedited`, `loomcv/complete/workbook.py`).


def _p2c_job(job_id: str, statuses: list[str], *, keywords=None, defaults=None):
    from loomcv.complete import WbBullet, WbJob

    return WbJob(
        job_id=job_id,
        name=job_id.title(),
        date_from="2020",
        date_to="2021",
        defaults=list(defaults or []),
        bullets=[
            WbBullet(
                id=f"{job_id}{i}",
                text=f"Did thing {i} at {job_id}.",
                keywords=list(keywords or []),
                status=status,
            )
            for i, status in enumerate(statuses)
        ],
    )


def test_p2c_c1_confirmed_counts_per_job_in_workbook_order():
    from loomcv.complete import Workbook, confirmed_counts

    wb = Workbook(
        jobs=[
            _p2c_job("a", ["confirmed", "draft", "confirmed", "draft", "confirmed"]),
            _p2c_job("b", ["draft", "draft"]),
        ]
    )

    counts = confirmed_counts(wb)

    assert counts == {"a": (3, 5), "b": (0, 2)}
    assert list(counts) == ["a", "b"]


def test_p2c_c2_a_bulletless_job_counts_zero_of_zero():
    from loomcv.complete import Workbook, confirmed_counts

    wb = Workbook(jobs=[_p2c_job("a", ["draft"]), _p2c_job("empty", [])])

    assert confirmed_counts(wb)["empty"] == (0, 0)


def test_p2c_c3_a_confirmed_bullet_without_keywords_counts_as_confirmed():
    from loomcv.complete import Workbook, confirmed_counts

    wb = Workbook(jobs=[_p2c_job("a", ["confirmed", "draft"], keywords=[])])

    assert confirmed_counts(wb) == {"a": (1, 2)}


def test_p2c_c4_a_text_edit_keeps_the_counts_a_confirm_adds_one():
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        apply_bullets,
        confirmed_counts,
        load_workbook,
    )

    wb = load_workbook(SAM)
    before = confirmed_counts(wb)
    edited = apply_bullets(
        wb, BulletsAnswers(bullets={ACME_KEY: BulletAnswer(text="Cut the order API latency.")})
    )
    confirmed = apply_bullets(wb, BulletsAnswers(bullets={ACME_KEY: BulletAnswer(confirm=True)}))

    assert confirmed_counts(edited) == before
    after = confirmed_counts(confirmed)
    assert after[ACME] == (before[ACME][0] + 1, before[ACME][1])
    assert after[FREE] == before[FREE]


def test_p2c_m1_maybe_also_is_the_defaults_the_bullet_lacks():
    from loomcv.complete import maybe_also

    job = _p2c_job("a", ["draft"], keywords=["django", "Docker"])
    job = job.model_copy(update={"defaults": ["Python", "Django", "PostgreSQL"]})

    assert maybe_also(job, job.bullets[0]) == ["Python", "PostgreSQL"]


def test_p2c_m2_maybe_also_is_empty_when_every_default_is_carried():
    from loomcv.complete import maybe_also

    job = _p2c_job(
        "a",
        ["draft"],
        keywords=["POSTGRESQL", "python", "django", "Docker"],
        defaults=["Python", "Django", "PostgreSQL"],
    )

    assert maybe_also(job, job.bullets[0]) == []


def test_p2c_m3_maybe_also_is_empty_without_defaults():
    from loomcv.complete import maybe_also

    job = _p2c_job("a", ["draft"], keywords=["Docker"], defaults=[])

    assert maybe_also(job, job.bullets[0]) == []


def test_p2c_x1_a_bullet_as_it_entered_is_unchanged():
    from loomcv.complete import load_workbook, unchanged_since_entry
    from loomcv.ingest.extract import _sha6
    from loomcv.ingest.keywords import normalise_ws

    bullet = _bullet(load_workbook(SAM), ACME, "395f62")

    assert bullet.id == _sha6(normalise_ws(bullet.text))
    assert unchanged_since_entry(bullet) is True


def test_p2c_x2_a_text_edit_makes_it_changed():
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        apply_bullets,
        load_workbook,
        unchanged_since_entry,
    )

    wb = apply_bullets(
        load_workbook(SAM),
        BulletsAnswers(bullets={ACME_KEY: BulletAnswer(text="Cut the order API latency.")}),
    )

    assert unchanged_since_entry(_bullet(wb, ACME, "395f62")) is False


def test_p2c_x3_edited_back_to_the_original_text_is_unchanged_again():
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        apply_bullets,
        load_workbook,
        unchanged_since_entry,
    )

    wb = load_workbook(SAM)
    original = _bullet(wb, ACME, "395f62").text
    edited = apply_bullets(
        wb, BulletsAnswers(bullets={ACME_KEY: BulletAnswer(text="Cut the order API latency.")})
    )
    back = apply_bullets(edited, BulletsAnswers(bullets={ACME_KEY: BulletAnswer(text=original)}))

    assert unchanged_since_entry(_bullet(back, ACME, "395f62")) is True


def test_p2c_x4_an_ordinal_id_reads_as_changed():
    from loomcv.complete import load_workbook, unchanged_since_entry
    from loomcv.ingest.extract import _sha6
    from loomcv.ingest.keywords import normalise_ws

    bullet = _bullet(load_workbook(SAM), ACME, "395f62")
    ordinal = bullet.model_copy(update={"id": _sha6(f"{normalise_ws(bullet.text)}\n2")})

    assert unchanged_since_entry(ordinal) is False


def test_p2c_x5_confirm_unedited_names_only_the_unconfirmed_unchanged_bullets():
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        apply_bullets,
        confirm_unedited,
    )

    wb = _confirmed_jobs(SAM)
    wb = apply_bullets(
        wb,
        BulletsAnswers(
            bullets={
                f"{ACME}::395f62": BulletAnswer(confirm=True),  # confirmed, unchanged
                f"{ACME}::e8eb3d": BulletAnswer(text="Reduced storefront errors."),  # edited draft
            }
        ),
    )
    flags = {j.job_id: j.defaults_confirmed for j in wb.jobs}

    answers = confirm_unedited(wb, ACME)

    assert isinstance(answers, BulletsAnswers)
    assert list(answers.bullets) == [f"{ACME}::b37b54"]
    assert answers.bullets[f"{ACME}::b37b54"] == BulletAnswer(confirm=True)
    assert answers.bullets[f"{ACME}::b37b54"].model_fields_set == {"confirm"}

    applied = apply_bullets(wb, answers)

    assert [b.status for b in applied.jobs[0].bullets] == ["confirmed", "draft", "confirmed"]
    assert applied.jobs[1] == wb.jobs[1]
    assert {j.job_id: j.defaults_confirmed for j in applied.jobs} == flags
    assert confirm_unedited(applied, ACME) == BulletsAnswers()
    assert confirm_unedited(applied, ACME).bullets == {}


# ---- the upload fills everything
# once — the structure call, then `select_keywords` over every bullet with the extracted,
# unconfirmed Defaults; the tools list is no longer a gate; a failed keyword call never refuses
# the upload (`first_keywords`).


class _BrokenAdapter(FakeGenerationAdapter):
    """The transport fails on every call (an `OSError`, as a refused connection is)."""

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        self.calls.append((system, user, temperature))
        raise OSError("connection refused")


def test_f3_1_select_keywords_runs_with_unconfirmed_defaults():
    from loomcv.complete import load_workbook, select_keywords

    wb = load_workbook(SAM)
    assert not any(job.defaults_confirmed for job in wb.jobs)
    adapter = _scripted(SEL_SAM)

    out, warnings = select_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)

    assert len(adapter.calls) == 1
    _, user, _ = adapter.calls[0]
    assert "Job defaults: Python; Django; PostgreSQL" in user  # the extracted list is offered
    assert _bullet(out, ACME, "395f62").keywords == [
        "Python",
        "Django",
        "task queue",
        "API performance",
    ]
    assert not any(job.defaults_confirmed for job in out.jobs)  # nothing confirmed by the call
    assert isinstance(warnings, list)


def test_f3_2_first_keywords_fills_the_bullets_and_reports_no_failure():
    from loomcv.complete import first_keywords, load_workbook

    wb = load_workbook(SAM)
    adapter = _scripted(SEL_SAM)

    out, warnings, failure = first_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)

    assert failure is None
    assert len(adapter.calls) == 1
    assert _bullet(out, ACME, "395f62").keywords == [
        "Python",
        "Django",
        "task queue",
        "API performance",
    ]
    assert all(b.status == "draft" for job in out.jobs for b in job.bullets)
    assert [j.defaults_confirmed for j in out.jobs] == [j.defaults_confirmed for j in wb.jobs]
    assert any("a; b" in w for w in warnings)  # the module's own warnings pass through


def test_f3_3_first_keywords_transport_failure_keeps_the_workbook():
    from loomcv.complete import first_keywords, load_workbook

    wb = load_workbook(SAM)
    adapter = _BrokenAdapter([])

    out, warnings, failure = first_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)

    assert out is wb
    assert warnings == []
    assert isinstance(failure, str) and "connection refused" in failure
    assert len(adapter.calls) == 1  # no retry on transport


def test_f3_4_first_keywords_invalid_reply_twice_keeps_the_workbook():
    from loomcv.complete import first_keywords, load_workbook

    wb = load_workbook(SAM)
    adapter = _scripted("not json", "still not json")

    out, warnings, failure = first_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)

    assert out is wb
    assert warnings == []
    assert isinstance(failure, str) and failure
    assert len(adapter.calls) == 2  # one retry, then the upload goes on without keywords


def test_f3_5_first_keywords_without_a_draft_bullet_makes_no_call():
    from loomcv.complete import (
        BulletAnswer,
        BulletsAnswers,
        apply_bullets,
        first_keywords,
        load_workbook,
    )

    wb = load_workbook(SAM)
    # Confirmed *with* keywords (20:49: a confirmed bullet without any is taken by the call).
    every = {
        f"{job.job_id}::{b.id}": BulletAnswer(confirm=True, keywords=["k"])
        for job in wb.jobs
        for b in job.bullets
    }
    wb = apply_bullets(wb, BulletsAnswers(bullets=every))
    adapter = _scripted()

    out, warnings, failure = first_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)

    assert (out, warnings, failure) == (wb, [], None)
    assert adapter.calls == []


# ---- a confirmed bullet with no keyword
# is taken too — else a failed upload call + an early Confirm leaves it without. A confirmed bullet
# that has keywords is still neither in the prompt nor touched.


def test_f3_6_select_keywords_takes_a_confirmed_bullet_without_keywords():
    from loomcv.complete import BulletAnswer, BulletsAnswers, apply_bullets, select_keywords

    wb = apply_bullets(
        load_workbook_sam(),
        BulletsAnswers(
            bullets={
                f"{ACME}::395f62": BulletAnswer(confirm=True),  # confirmed, no keyword
                f"{ACME}::e8eb3d": BulletAnswer(confirm=True, keywords=["error budgets"]),
            }
        ),
    )
    reply = {
        "J1b1": {"defaults": ["Python"], "own": ["task queue"]},
        "J1b2": {"defaults": ["PostgreSQL"], "own": ["invoicing"]},
        "J2b1": {"defaults": [], "own": ["web development"]},
    }
    adapter = _scripted(reply)

    out, _ = select_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)

    assert len(adapter.calls) == 1
    _, user, _ = adapter.calls[0]
    assert "- J1b1: Cut the p95 latency" in user  # the confirmed, keyword-less bullet
    assert "Reduced checkout errors" not in user  # confirmed with keywords: left alone
    assert "- J1b2: Rebuilt the invoice pipeline" in user
    first = _bullet(out, ACME, "395f62")
    assert first.keywords == ["Python", "task queue"]
    assert first.status == "confirmed"  # the call never changes a status
    assert _bullet(out, ACME, "e8eb3d") == _bullet(wb, ACME, "e8eb3d")
    assert _bullet(out, ACME, "b37b54").status == "draft"


def test_f3_7_no_call_when_every_bullet_is_confirmed_with_keywords():
    from loomcv.complete import BulletAnswer, BulletsAnswers, apply_bullets, select_keywords

    wb = load_workbook_sam()
    every = {
        f"{job.job_id}::{b.id}": BulletAnswer(confirm=True, keywords=["k"])
        for job in wb.jobs
        for b in job.bullets
    }
    wb = apply_bullets(wb, BulletsAnswers(bullets=every))
    adapter = _scripted()

    out, warnings = select_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT)

    assert (out, warnings) == (wb, [])
    assert adapter.calls == []


def load_workbook_sam():
    from loomcv.complete import load_workbook

    return load_workbook(SAM)


# ---- a bullet's "suggest" asks for that bullet only — one short
# call, never a profile-wide run; "Suggest search keywords again" (no `only`) stays as it was.


def test_f5_1_select_keywords_only_one_bullet():
    from loomcv.complete import select_keywords

    wb = load_workbook_sam()  # every bullet a keyword-less draft
    adapter = _scripted({"J1b1": {"defaults": ["Python"], "own": ["task queue"]}})

    out, _ = select_keywords(
        wb, adapter=adapter, template_path=SELECT_PROMPT, only={f"{ACME}::395f62"}
    )

    assert len(adapter.calls) == 1
    _, user, _ = adapter.calls[0]
    assert "- J1b1: Cut the p95 latency" in user
    assert "Rebuilt the invoice pipeline" not in user  # another draft of the same job
    assert "Reduced checkout errors" not in user
    assert "J1b2" not in user and "J2b1" not in user  # no other bullet, no other job
    assert _bullet(out, ACME, "395f62").keywords == ["Python", "task queue"]
    for job in wb.jobs:
        for b in job.bullets:
            if (job.job_id, b.id) != (ACME, "395f62"):
                assert _bullet(out, job.job_id, b.id) == b


def test_f5_2_only_a_bullet_the_call_does_not_take_makes_no_call():
    from loomcv.complete import BulletAnswer, BulletsAnswers, apply_bullets, select_keywords

    key = f"{ACME}::395f62"
    wb = apply_bullets(
        load_workbook_sam(),
        BulletsAnswers(bullets={key: BulletAnswer(confirm=True, keywords=["k"])}),
    )
    adapter = _scripted()

    for only in ({key}, {f"{ACME}::nosuch"}, set()):
        out, warnings = select_keywords(wb, adapter=adapter, template_path=SELECT_PROMPT, only=only)
        assert (out, warnings) == (wb, [])
    assert adapter.calls == []
