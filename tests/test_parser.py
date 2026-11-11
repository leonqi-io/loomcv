"""Tests for the v3 job-grammar STAR-vault parser.

Behaviour exercised here:
- v3 grammar: `## [Job]` blocks with Employer / Title /
  From / To / Defaults header lines; bullets carry an explicit `{id: ...}`,
  optional leading `[Tag]` tokens and optional `Project` / `Keywords`
  attribute sub-lines; job order is by date (To desc, From desc, file order).
- ``bullet_id = <job_slug>::<id>``; ``content_hash`` covers
  the whitespace-normalised text only.
- Strict mode — structural defects are collected as errors with line
  numbers; the older `## [Project]` grammar is an error, never migrated.
- The trailing status block (now also carrying `id`).

All fixtures are synthetic — no real career data (M1 protocol).
"""

import inspect
import json
import re
from datetime import date
from pathlib import Path

import pytest

from loomcv.sync import parse_assets
from loomcv.sync.parse_assets import content_hash, parse_star_vault, slugify

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent

HEADER = "* **From**: 2020\n* **To**: 2021\n"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _line_of(doc: str, fragment: str) -> int:
    """1-based number of the first line of ``doc`` containing ``fragment``."""
    for number, line in enumerate(doc.splitlines(), start=1):
        if fragment in line:
            return number
    raise AssertionError(f"{fragment!r} not found in document")


def _vault(bullets: str, *, name: str = "Alpha", header: str = HEADER) -> str:
    return f"## [Job] {name}\n{header}* **Bullets**:\n{bullets}"


def _job(name: str, date_from: str, date_to: str, bullet_id: str = "aaa111") -> str:
    return _vault(
        f"  - Text of {name}. {{id: {bullet_id}}}\n",
        name=name,
        header=f"* **From**: {date_from}\n* **To**: {date_to}\n",
    )


def _messages(result, fragment: str):
    return [i for i in result.errors() if fragment in i.message]


@pytest.fixture()
def basic() -> str:
    return _read("basic_vault.md")


@pytest.fixture()
def malformed() -> str:
    return _read("malformed_vault.md")


@pytest.fixture()
def metadata_vault() -> str:
    return _read("metadata_vault.md")


@pytest.fixture()
def status_block() -> str:
    return _read("status_block_vault.md")


# ---------------------------------------------------------------- normal path


def test_01_job_fields_parsed(basic):
    result = parse_star_vault(basic)
    orbit = next(j for j in result.jobs if j.name == "Orbit Retail — Checkout Engineer")
    assert orbit.job_id == "orbit-retail-checkout-engineer"
    assert orbit.employer == "Orbit Retail"
    assert orbit.title == "Checkout Engineer"
    assert orbit.date_from == "2023-02"
    assert orbit.date_to == "2025-01"
    assert orbit.defaults == ["TypeScript", "Redis"]  # split, stripped, deduped

    by_id = {b.bullet_id: b for b in orbit.bullets}
    orb1 = by_id["orbit-retail-checkout-engineer::orb001"]
    assert orb1.project == "Orbit Checkout Revamp"
    assert orb1.keywords == ["Caching", "TypeScript", "Price Lookups"]
    orb2 = by_id["orbit-retail-checkout-engineer::orb002"]
    assert orb2.project is None
    assert orb2.keywords == []

    # the job's `defaults` are never merged into any bullet
    assert all("Redis" not in b.keywords for j in result.jobs for b in j.bullets)
    assert "TypeScript" not in orb2.keywords


def test_02_jobs_segmented(basic):
    result = parse_star_vault(basic)
    assert [j.name for j in result.jobs] == [
        "Orbit Retail — Checkout Engineer",
        "Nimbus Cloud — Platform Engineer",
    ]
    assert [len(j.bullets) for j in result.jobs] == [2, 3]
    assert result.errors() == []


def test_02b_header_order_independent_and_minimal_job():
    one = (
        "## [Job] Order Independent\n"
        "* **Employer**: Acme\n"
        "* **Title**: Dev\n"
        "* **From**: 2020-01\n"
        "* **To**: 2021-06\n"
        "* **Defaults**: Foo; Bar\n"
        "* **Bullets**:\n"
        "  - Some text. {id: aaa111}\n"
    )
    two = (
        "## [Job] Order Independent\n"
        "* **Defaults**: Foo; Bar\n"
        "* **To**: 2021-06\n"
        "* **Title**: Dev\n"
        "* **From**: 2020-01\n"
        "* **Employer**: Acme\n"
        "* **Bullets**:\n"
        "  - Some text. {id: aaa111}\n"
    )
    r1, r2 = parse_star_vault(one), parse_star_vault(two)
    assert r1.errors() == r2.errors() == []
    assert r1.jobs == r2.jobs
    assert r1.jobs[0].defaults == ["Foo", "Bar"]

    minimal = parse_star_vault(_vault("  - Some text. {id: aaa111}\n"))
    assert minimal.issues == []
    job = minimal.jobs[0]
    assert job.employer is None
    assert job.title is None
    assert job.defaults == []


def test_02c_job_order_by_date_not_file_position():
    doc = "\n".join(
        [
            _job("Oldest", "2015", "2017"),
            _job("Current", "2021", "present"),
            _job("Middle", "2018", "2021"),
        ]
    )
    result = parse_star_vault(doc)
    assert [j.name for j in result.jobs] == ["Current", "Middle", "Oldest"]
    assert [j.order for j in result.jobs] == [0, 1, 2]

    # equal To -> From descending
    tie_to = parse_star_vault("\n".join([_job("A", "2018", "2020"), _job("B", "2019", "2020")]))
    assert [j.name for j in tie_to.jobs] == ["B", "A"]

    # a bare YYYY sorts as YYYY-00: the 2020-01 job comes first
    bare = parse_star_vault(
        "\n".join([_job("Bare", "2018", "2020"), _job("Month", "2018", "2020-01")])
    )
    assert [j.name for j in bare.jobs] == ["Month", "Bare"]

    # fully equal dates keep file order
    equal = parse_star_vault(
        "\n".join([_job("First", "2019", "2020"), _job("Second", "2019", "2020")])
    )
    assert [j.name for j in equal.jobs] == ["First", "Second"]
    assert [j.order for j in equal.jobs] == [0, 1]


def test_03_leading_tags_optional(basic):
    result = parse_star_vault(basic)
    orbit = result.jobs[0]
    first, second = orbit.bullets
    assert first.tags == ["Performance", "Backend"]
    assert first.text.startswith("Cut synthetic checkout latency")
    assert second.tags == ["Reliability"]
    assert not second.text.startswith("[")
    assert second.status == "confirmed"
    assert str(second.updated) == "2026-08-20"

    nimbus = result.jobs[1]
    untagged = next(b for b in nimbus.bullets if b.bullet_id.endswith("::nim003"))
    assert untagged.tags == []
    assert untagged.text == "Wrote a runbook for a pretend on-call rotation."


def test_04_preamble_ignored_no_ghost_job(basic):
    result = parse_star_vault(basic)
    names = [j.name for j in result.jobs]
    assert all("fixture" not in n.lower() and "synthetic" not in n.lower() for n in names)
    assert len(result.jobs) == 2
    # the handle-map-shaped preamble line is not a bullet
    assert not any("→" in b.text for j in result.jobs for b in j.bullets)


def test_05_reserve_is_an_ordinary_keyword(basic):
    result = parse_star_vault(basic)
    nimbus = next(j for j in result.jobs if j.job_id == "nimbus-cloud-platform-engineer")
    nim2 = next(b for b in nimbus.bullets if b.bullet_id.endswith("::nim002"))
    assert "reserve" in nim2.keywords
    assert nim2.keywords == ["reserve", "Feature Flags"]
    assert not hasattr(parse_assets, "RESERVE_TAG")
    with pytest.raises(TypeError):
        parse_star_vault(basic, include_reserve=True)


def test_06_parsing_stops_before_verification_notes(basic):
    result = parse_star_vault(basic)
    all_texts = [b.text for j in result.jobs for b in j.bullets]
    assert not any("Trap" in t or "booby" in t.lower() for t in all_texts)
    # the trap line must not surface as a bullet of the last parsed job
    assert len(result.jobs[-1].bullets) == 3


# ------------------------------------------------------------ malformed input


@pytest.mark.parametrize("key", ["Tags", "Context", "Domain", "Year", "Tech", "Dates", "Keywords"])
def test_07_retired_job_key_is_error_with_line_no(key):
    doc = (
        "## [Job] Alpha\n"
        "* **From**: 2020\n"
        "* **To**: 2021\n"
        f"* **{key}**: whatever\n"
        "* **Bullets**:\n"
        "  - Some text. {id: aaa111}\n"
    )
    result = parse_star_vault(doc)
    errs = result.errors()
    assert len(errs) == 1
    assert errs[0].line_no == 4
    assert f"'{key}' line is not part of the v3 job header" in errs[0].message


def test_07_pre_r1_project_block_is_error_with_line_no():
    doc = (
        "# Title\n"
        "\n"
        "## [Project] Old Style\n"
        "* **Tags**: #a\n"
        "* **Context**: Ctx\n"
        "* **Bullets**:\n"
        "  - [Backend] Old bullet.\n"
    )
    result = parse_star_vault(doc)
    errs = result.errors()
    assert len(errs) == 1
    assert errs[0].line_no == 3
    assert "'## [Project]' is the pre-R1 vault grammar (archived, not migrated)" in errs[0].message
    assert result.jobs == []


@pytest.mark.parametrize(
    ("header", "fragment", "line_of"),
    [
        ("* **To**: 2021\n", "From", "## [Job]"),
        ("* **From**: 2020\n", "To", "## [Job]"),
        ("* **From**: Sept 2023\n* **To**: 2024\n", "From", "Sept 2023"),
        ("* **From**: 2020\n* **To**: 2026-13\n", "To", "2026-13"),
        ("* **From**: present\n* **To**: 2024\n", "From", "present"),
        ("* **From**: 2020\n* **To**: 2021\n* **To**: 2022\n", "duplicate", "To**: 2022"),
        (
            "* **From**: 2020\n* **To**: 2021\n* **Defaults**: A\n* **Defaults**: B\n",
            "duplicate",
            "Defaults**: B",
        ),
    ],
)
def test_07b_from_to_errors(header, fragment, line_of):
    doc = _vault("  - Some text. {id: aaa111}\n", header=header)
    result = parse_star_vault(doc)
    matching = _messages(result, fragment)
    assert [i.line_no for i in matching] == [_line_of(doc, line_of)]


def test_07b_missing_from_or_to_drops_the_job():
    result = parse_star_vault(_vault("  - Some text. {id: aaa111}\n", header="* **To**: 2021\n"))
    assert result.jobs == []
    assert len(result.errors()) == 1


def test_07c_attribute_line_errors():
    before_first = (
        "## [Job] Alpha\n"
        + HEADER
        + "* **Bullets**:\n"
        + "    * **Project**: Orphan\n"
        + "  - Some text. {id: aaa111}\n"
    )
    result = parse_star_vault(before_first)
    errs = _messages(result, "does not follow a bullet")
    assert [i.line_no for i in errs] == [_line_of(before_first, "Orphan")]

    unknown = _vault("  - Some text. {id: aaa111}\n    * **Impact**: big\n")
    result = parse_star_vault(unknown)
    errs = _messages(result, "Impact")
    assert [i.line_no for i in errs] == [_line_of(unknown, "Impact")]

    duplicate = _vault(
        "  - Some text. {id: aaa111}\n    * **Project**: One\n    * **Project**: Two\n"
    )
    result = parse_star_vault(duplicate)
    errs = _messages(result, "duplicate")
    assert [i.line_no for i in errs] == [_line_of(duplicate, "Project**: Two")]

    # attribute lines under a REJECTED bullet add no second error
    rejected = _vault(
        "  - A valid bullet. {id: good01}\n"
        "  - Missing its id block.\n"
        "    * **Project**: P\n"
        "    * **Keywords**: a; b\n"
    )
    result = parse_star_vault(rejected)
    assert len(result.errors()) == 1
    assert result.errors()[0].line_no == _line_of(rejected, "Missing its id")


@pytest.mark.parametrize(
    ("bullet_line", "fragment"),
    [
        ("  - Some text.", "id"),
        ("  - Some text. {id: ABC-1}", "invalid id"),
        ("  - Some text. {id: abc}", "invalid id"),
        ("  - Some text. {id: 123456789}", "invalid id"),
        ("  - Some text. {id: a1b2, id: c3d4}", "duplicate key 'id'"),
    ],
)
def test_08_id_errors(bullet_line, fragment):
    doc = _vault("  - A valid bullet. {id: good01}\n" + bullet_line + "\n")
    result = parse_star_vault(doc)
    errs = result.errors()
    assert len(errs) == 1
    assert errs[0].line_no == _line_of(doc, "Some text.")
    assert fragment in errs[0].message
    assert [b.text for b in result.jobs[0].bullets] == ["A valid bullet."]  # rejected bullet absent


def test_08_same_id_twice_in_one_job_reports_the_second():
    doc = _vault("  - First text. {id: aaa111}\n  - Second text. {id: aaa111}\n")
    result = parse_star_vault(doc)
    errs = _messages(result, "duplicate id")
    assert [i.line_no for i in errs] == [_line_of(doc, "Second text.")]
    assert [b.text for b in result.jobs[0].bullets] == ["First text."]


def test_09_empty_job_is_warning_and_job_returned(malformed):
    # P-1: was an error; a job without bullets is a job in the vault.
    result = parse_star_vault(malformed)
    assert [i for i in result.errors() if i.job == "Empty Job"] == []
    warns = [i for i in result.warnings() if i.job == "Empty Job"]
    assert [i.line_no for i in warns] == [_line_of(malformed, "## [Job] Empty Job")]
    assert any("empty job" in i.message.lower() for i in warns)
    empty = [j for j in result.jobs if j.name == "Empty Job"]
    assert len(empty) == 1
    assert empty[0].bullets == []


def test_10_duplicate_job_name_is_error(malformed):
    result = parse_star_vault(malformed)
    errs = [i for i in result.errors() if "Twin Job" in (i.job or "")]
    assert any("duplicate" in i.message.lower() for i in errs)
    assert [j.name for j in result.jobs].count("Twin Job") == 1


def test_10b_kept_error_cases_reuse_fixtures(malformed, status_block):
    result = parse_star_vault(malformed)
    by_line = {i.line_no: i for i in result.errors()}
    expected = {
        _line_of(malformed, "## [Project] Old Grammar Project"): "pre-R1",
        _line_of(malformed, "## [Job] Missing From Job"): "From",
        _line_of(malformed, "2026-13"): "To",
        _line_of(malformed, "This bullet line has no id"): "id",
    }
    for line_no, fragment in expected.items():
        assert fragment in by_line[line_no].message, (line_no, by_line[line_no].message)
    assert len(result.errors()) == 5  # the four above + the duplicate twin job

    sb = parse_star_vault(status_block)
    banned_lines = {
        _line_of(status_block, "[user_confirmed]"),
        _line_of(status_block, "[status] This"),
        _line_of(status_block, "[updated] This"),
        _line_of(status_block, "[2026-08-20]"),
    }
    assert banned_lines <= {i.line_no for i in sb.errors()}
    malformed_block = _line_of(status_block, "{id: unk001, new_key: value}")
    assert malformed_block in {i.line_no for i in sb.errors()}


def test_11_unknown_section_warns_and_parsing_continues():
    result = parse_star_vault(_read("unknown_section_vault.md"))
    assert [j.name for j in result.jobs] == [
        "Solo Fictional Employer — Engineer",
        "Job After Unknown Section",
    ]
    warnings = [i for i in result.issues if i.severity == "warning"]
    assert len(warnings) == 1
    assert "Random Future Section" in warnings[0].message
    all_texts = [b.text for j in result.jobs for b in j.bullets]
    assert not any("Sneaky" in t for t in all_texts)


# ------------------------------------- vault-level Skills line (P-2 … P-4)


def test_vault_level_skills_line_before_first_job():
    # P-2 — read directly under the H1, no issue. An
    # unknown preamble key is no longer silently ignored — it is exactly one warning.
    doc = "# Vault\n* **Skills**: A; b; a\n* **Tags**: x\n\n" + _job("Alpha", "2020", "2021")
    result = parse_star_vault(doc)
    assert result.skills == ["A", "b"]
    assert result.errors() == []
    assert [w.line_no for w in result.warnings()] == [_line_of(doc, "* **Tags**: x")]
    assert "Tags" in result.warnings()[0].message
    assert [j.name for j in result.jobs] == ["Alpha"]

    # No Skills line → the empty list; the line may also follow blank lines / prose.
    assert parse_star_vault(_job("Alpha", "2020", "2021")).skills == []
    spaced = "# Vault\n\nSome prose.\n\n* **Skills**: Reserving ; Power BI\n\n"
    assert parse_star_vault(spaced + _job("Alpha", "2020", "2021")).skills == [
        "Reserving",
        "Power BI",
    ]


def test_second_or_in_job_skills_line_is_error():
    # P-3
    doc = "# Vault\n* **Skills**: A\n* **Skills**: B\n\n" + _vault(
        "  - Text. {id: aaa111}\n", header=HEADER + "* **Skills**: C\n"
    )
    result = parse_star_vault(doc)
    assert result.skills == ["A"]
    duplicate = _messages(result, "duplicate **Skills** line")
    assert [i.line_no for i in duplicate] == [_line_of(doc, "* **Skills**: B")]
    inside = _messages(result, "**Skills** is a vault-level line")
    assert [i.line_no for i in inside] == [_line_of(doc, "* **Skills**: C")]
    assert len(result.errors()) == 2

    # After a job, even in a skipped section, a Skills line is an error.
    late = _job("Alpha", "2020", "2021") + "\n## Notes\n* **Skills**: D\n"
    late_result = parse_star_vault(late)
    assert late_result.skills == []
    assert [i.line_no for i in _messages(late_result, "vault-level line")] == [
        _line_of(late, "* **Skills**: D")
    ]


def test_pre_r1_grammar_still_error_with_skills_line():
    # P-4 — the Skills line is read; the old grammar stays an error at its own line.
    doc = (
        "# Vault\n* **Skills**: A; B\n\n"
        "## [Project] Old\n* **Tags**: #x\n* **Bullets**:\n  - Old bullet.\n"
    )
    result = parse_star_vault(doc)
    assert result.skills == ["A", "B"]
    errs = result.errors()
    assert [i.line_no for i in errs] == [_line_of(doc, "## [Project] Old")]
    assert "pre-R1" in errs[0].message


# ----------------------------------------------------- stable IDs and hashing


def test_12_edit_text_keeps_id_changes_hash(basic):
    before = parse_star_vault(basic)
    edited = basic.replace("from 900ms to 300ms", "from 900ms to 250ms")
    after = parse_star_vault(edited)

    b_before = before.jobs[0].bullets[0]
    b_after = after.jobs[0].bullets[0]
    assert b_before.bullet_id == b_after.bullet_id
    assert b_before.content_hash != b_after.content_hash

    # every other bullet's hash is untouched (incremental-sync unit mirror)
    others_before = {b.bullet_id: b.content_hash for j in before.jobs for b in j.bullets}
    others_after = {b.bullet_id: b.content_hash for j in after.jobs for b in j.bullets}
    del others_before[b_before.bullet_id], others_after[b_after.bullet_id]
    assert others_before == others_after


def test_13_bullet_id_is_job_slug_and_explicit_id():
    doc = "\n".join(
        [
            _vault(
                "  - [Tag] First. {id: aaa111}\n"
                "    * **Project**: P\n"
                "    * **Keywords**: x; y\n"
                "  - Second untagged. {id: bbb222}\n",
                name="Job One",
            ),
            _vault("  - Same id in another job. {id: aaa111}\n", name="Job Two"),
        ]
    )
    result = parse_star_vault(doc)
    assert result.errors() == []
    ids = [b.bullet_id for j in result.jobs for b in j.bullets]
    assert ids == ["job-one::aaa111", "job-one::bbb222", "job-two::aaa111"]
    assert len(set(ids)) == 3

    # reorder bullets, edit text / tags / keywords / label -> same bullet_ids
    reordered = _vault(
        "  - Second untagged EDITED. {id: bbb222}\n"
        "  - [Other][Tags] First EDITED. {id: aaa111}\n"
        "    * **Project**: Different label\n"
        "    * **Keywords**: z\n",
        name="Job One",
    )
    after = parse_star_vault(reordered)
    assert after.errors() == []
    assert {b.bullet_id for b in after.jobs[0].bullets} == {"job-one::aaa111", "job-one::bbb222"}


def test_14_hash_is_text_only_and_ignores_whitespace():
    a = content_hash("Cut latency from 900ms to 300ms.")
    b = content_hash("Cut latency  from 900ms to 300ms.  \n")
    assert a == b
    assert a != content_hash("Cut latency from 900ms to 250ms.")
    assert len(inspect.signature(content_hash).parameters) == 1


_BASE = (
    "## [Job] Alpha\n"
    "* **Employer**: Alpha Inc\n"
    "* **Title**: Dev\n"
    "* **From**: 2020\n"
    "* **To**: 2021\n"
    "* **Defaults**: Foo\n"
    "* **Bullets**:\n"
    "  - [Backend] Same text. {id: aaa111, status: confirmed, updated: 2026-08-20}\n"
    "    * **Project**: Proj\n"
    "    * **Keywords**: One; Two\n"
)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("[Backend] ", "[Backend][Extra] "),
        ("[Backend] ", ""),
        ("[Backend]", "[Frontend]"),
        ("    * **Project**: Proj\n", ""),
        ("Proj\n", "Other\n"),
        ("    * **Keywords**: One; Two\n", ""),
        ("One; Two", "Three"),
        ("status: confirmed", "status: draft"),
        ("2026-08-20", "2026-09-01"),
        ("id: aaa111", "id: zzz999"),
        ("Alpha Inc", "Beta Inc"),
        ("Dev\n", "Senior Dev\n"),
        ("From**: 2020", "From**: 2019"),
        ("To**: 2021", "To**: 2022"),
        ("Foo", "Bar; Baz"),
        ("## [Job] Alpha", "## [Job] Alpha Renamed"),
    ],
)
def test_14b_hash_unchanged_by_non_text_edits(old, new):
    assert old in _BASE
    base = parse_star_vault(_BASE, include_draft=True)
    edited = parse_star_vault(_BASE.replace(old, new, 1), include_draft=True)
    assert base.errors() == [] and edited.errors() == []
    assert base.jobs[0].bullets[0].content_hash == edited.jobs[0].bullets[0].content_hash


def test_14b_text_edit_changes_hash():
    base = parse_star_vault(_BASE).jobs[0].bullets[0]
    edited = parse_star_vault(_BASE.replace("Same text.", "Different text.")).jobs[0].bullets[0]
    assert base.content_hash != edited.content_hash
    assert base.bullet_id == edited.bullet_id


# ------------------------------------------------ delimiter case-sensitivity


def test_15_lowercase_delimiter_not_treated_as_job():
    """`## [job]` (lowercase) must NOT match the job delimiter: unknown section."""
    doc = (
        "## [job] Lowercase Fake Job\n"
        "* **From**: 2020\n"
        "* **To**: 2021\n"
        "* **Bullets**:\n"
        "    - A bullet that must not appear. {id: aaa111}\n"
    )
    result = parse_star_vault(doc)
    assert len(result.jobs) == 0
    warnings = [i for i in result.issues if i.severity == "warning"]
    assert any("[job]" in w.message for w in warnings)


def test_16_uppercase_delimiter_not_treated_as_job():
    doc = (
        "## [JOB] Uppercase Fake Job\n"
        "* **From**: 2020\n"
        "* **To**: 2021\n"
        "* **Bullets**:\n"
        "    - A bullet that must not appear. {id: aaa111}\n"
    )
    result = parse_star_vault(doc)
    assert len(result.jobs) == 0
    warnings = [i for i in result.issues if i.severity == "warning"]
    assert any("[JOB]" in w.message for w in warnings)


def test_17_empty_document_returns_zero_jobs_and_issues():
    result = parse_star_vault("")
    assert len(result.jobs) == 0
    assert len(result.issues) == 0


# ---------------------------------------------------------------- status block


def test_18_default_status_no_block():
    """Bullet with an id-only block defaults to confirmed / None."""
    b = parse_star_vault(_vault("    - [Backend] Some text. {id: aaa111}\n")).jobs[0].bullets[0]
    assert b.status == "confirmed"
    assert b.updated is None


def test_19_status_block_invisible_to_id_and_hash():
    """Adding/removing status/updated must not change bullet_id or content_hash."""
    without = _vault("    - [Backend] Some text. {id: aaa111}\n")
    with_ = _vault(
        "    - [Backend] Some text. {id: aaa111, status: confirmed, updated: 2026-08-20}\n"
    )
    b_without = parse_star_vault(without).jobs[0].bullets[0]
    b_with = parse_star_vault(with_).jobs[0].bullets[0]
    assert b_without.bullet_id == b_with.bullet_id
    assert b_without.content_hash == b_with.content_hash
    # the id key itself is invisible to the hash: same text under another id
    other = parse_star_vault(_vault("    - [Backend] Some text. {id: zzz999}\n")).jobs[0].bullets[0]
    assert other.content_hash == b_without.content_hash


def test_20_full_status_block_parse():
    doc = _vault(
        "    - [Backend] Some text. {id: aaa111, status: confirmed, updated: 2026-08-20}\n"
    )
    b = parse_star_vault(doc).jobs[0].bullets[0]
    assert b.status == "confirmed"
    assert b.updated == date(2026, 8, 20)
    assert "{status:" not in b.text and "{id:" not in b.text
    assert "Some text" in b.text


def test_21_date_only_block():
    b = (
        parse_star_vault(_vault("    - [Backend] Text. {id: aaa111, updated: 2026-08-20}\n"))
        .jobs[0]
        .bullets[0]
    )
    assert b.status == "confirmed"
    assert b.updated == date(2026, 8, 20)


def test_22_whitespace_variants_status_block():
    compact = _vault("    - [Backend] Text. {id:aaa111,status:draft,updated:2026-08-20}\n")
    spaced = _vault("    - [Backend] Text. { updated:2026-08-20 , id:aaa111 , status:draft }\n")
    b_c = parse_star_vault(compact, include_draft=True).jobs[0].bullets[0]
    b_s = parse_star_vault(spaced, include_draft=True).jobs[0].bullets[0]
    assert b_c.status == b_s.status == "draft"
    assert b_c.updated == b_s.updated == date(2026, 8, 20)
    assert b_c.bullet_id == b_s.bullet_id == "alpha::aaa111"


def test_23_draft_exclusion():
    doc = _vault(
        "    - [Backend] Confirmed bullet. {id: aaa111}\n"
        "    - [Frontend] Draft bullet. {id: bbb222, status: draft}\n"
        "    - [API] Another confirmed. {id: ccc333}\n"
    )
    default = parse_star_vault(doc)
    assert len(default.jobs[0].bullets) == 2
    assert all(b.status != "draft" for b in default.jobs[0].bullets)

    with_draft = parse_star_vault(doc, include_draft=True)
    assert len(with_draft.jobs[0].bullets) == 3

    # a draft's id still blocks a duplicate, even when drafts are filtered out
    clash = _vault(
        "    - Draft bullet. {id: bbb222, status: draft}\n"
        "    - Confirmed bullet reusing the id. {id: bbb222}\n"
    )
    errs = _messages(parse_star_vault(clash), "duplicate id")
    assert [i.line_no for i in errs] == [_line_of(clash, "reusing the id")]


def _job_of(result, name):
    return next(j for j in result.jobs if j.name == name)


def test_25_malformed_block_unknown_key(status_block):
    result = parse_star_vault(status_block)
    job = _job_of(result, "Malformed Block Unknown Key")
    assert len(job.bullets) == 2  # valid bullets before/after survive
    errs = [i for i in result.errors() if i.job == job.name]
    assert len(errs) >= 1
    assert all(i.line_no > 0 for i in errs)


def test_26_bad_status_value(status_block):
    result = parse_star_vault(status_block)
    job = _job_of(result, "Bad Status Value")
    assert len(job.bullets) == 2
    errs = [i for i in result.errors() if i.job == job.name]
    assert len(errs) >= 1
    assert any("status" in i.message.lower() for i in errs)


def test_27_bad_date_format(status_block):
    result = parse_star_vault(status_block)
    job = _job_of(result, "Bad Date Format")
    assert len(job.bullets) == 2
    errs = [i for i in result.errors() if i.job == job.name]
    assert len(errs) >= 1
    assert any("date" in i.message.lower() for i in errs)


def test_28_block_not_at_line_end(status_block):
    result = parse_star_vault(status_block)
    job = _job_of(result, "Block Not At Line End")
    assert len(job.bullets) == 2
    errs = [i for i in result.errors() if i.job == job.name]
    assert len(errs) >= 1


def test_29_empty_text_with_block(status_block):
    result = parse_star_vault(status_block)
    job = _job_of(result, "Empty Text With Block")
    assert len(job.bullets) == 2  # valid bullets before/after survive
    errs = [i for i in result.errors() if i.job == job.name]
    assert len(errs) >= 1


def test_30_duplicate_key(status_block):
    result = parse_star_vault(status_block)
    job = _job_of(result, "Duplicate Key")
    assert len(job.bullets) == 2
    errs = [i for i in result.errors() if i.job == job.name]
    assert len(errs) >= 1
    assert any("duplicate" in i.message.lower() for i in errs)


def test_31_retired_leading_tags(status_block):
    result = parse_star_vault(status_block)
    job = _job_of(result, "Retired Leading Tags")
    assert len(job.bullets) == 2  # valid bullets before/after survive
    errs = [i for i in result.errors() if i.job == job.name]
    assert len(errs) >= 4  # four retired tags
    assert all(i.line_no > 0 for i in errs)


def test_32_provenance_prose_preserved(status_block):
    result = parse_star_vault(status_block)
    job = _job_of(result, "Provenance Prose Preserved")
    b = job.bullets[0]
    assert "(CORRECTION, supersedes prior wording)" in b.text
    assert "{updated:" not in b.text
    # provenance prose participates in content_hash
    assert b.content_hash == content_hash(
        "Text with provenance (CORRECTION, supersedes prior wording)"
    )


# ------------------------------------------------ keywords and defaults


def test_33_keyword_split_rules():
    split_list = parse_assets.split_list
    assert split_list("a; ;b") == ["a", "b"]
    assert split_list("TypeScript; typescript") == ["TypeScript"]
    assert split_list("REST APIs; rest") == ["REST APIs", "rest"]  # no stemming, no splitting
    assert split_list("  Foo Bar ; Baz.js;  CI/CD ;") == ["Foo Bar", "Baz.js", "CI/CD"]
    assert split_list("") == []


def test_34_defaults_never_merged(metadata_vault):
    result = parse_star_vault(metadata_vault)
    assert result.errors() == []
    assert [j.name for j in result.jobs] == [
        "Helios Streaming — Senior Player Engineer",
        "Zephyr Analytics — Data Engineer",
    ]
    helios, zephyr = result.jobs
    assert helios.defaults == ["TypeScript", "Shaka Player"]
    assert (helios.date_from, helios.date_to) == ("2023-07", "present")
    assert (helios.employer, helios.title) == ("Helios Streaming", "Senior Player Engineer")
    assert zephyr.defaults == []

    by_tail = {b.bullet_id.split("::")[1]: b for j in result.jobs for b in j.bullets}
    assert by_tail["hel001"].keywords == ["Shaka Player", "Buffer Tuning"]  # no TypeScript
    assert by_tail["tun001"].keywords == ["Alerting", "Dedupe Rules", "Observability"]
    assert by_tail["zep001"].keywords == []
    assert by_tail["qua001"].keywords == ["Foo Bar", "Baz.js", "CI/CD"]
    assert by_tail["hel001"].project == "Helios Player Rewrite"
    assert result.issues == []


# ------------------------------------------------ the committed synthetic vault


def test_35_synthetic_vault_parses_with_t1_keywords():
    vault = (REPO / "benchmarks_templates/synthetic_webplatform/vault_mara_voss.md").read_text(
        encoding="utf-8"
    )
    confirmed = json.loads(
        (REPO / "dev_flow/eval/t1/keywords_confirmed.json").read_text(encoding="utf-8")
    )
    result = parse_star_vault(vault)
    assert result.issues == []
    assert [j.name for j in result.jobs] == [
        "Brindlewick Media — Senior Web Engineer",
        "Tallowmere Outfitters — Software Engineer, Storefront",
        "Quillhaven Learning — Full-Stack Engineer",
    ]
    bullets = {b.bullet_id: b for j in result.jobs for b in j.bullets}
    assert len(bullets) == 42
    assert sum(len(j.bullets) for j in result.jobs) == 42
    assert all(b.project and b.keywords for b in bullets.values())

    handle_map = re.findall(r"^- (P\d+b\d+) → (\S+)$", vault, flags=re.MULTILINE)
    assert len(handle_map) == 42
    for handle, bullet_id in handle_map:
        assert bullets[bullet_id].keywords == confirmed[handle], handle


# -------------------------------------------------------------------- helpers


def test_slugify_basics():
    assert slugify("Orbit Checkout Revamp") == "orbit-checkout-revamp"
    assert (
        slugify("Bitmovin→Shaka Migration & Optimization")
        == "bitmovin-shaka-migration-optimization"
    )
    assert slugify("Ops & SRE") == "ops-sre"


# ------------------------------- vault-level header block + grouped Skills (P-1 … P-10)

L3A_HEADER = (
    "# Vault\n"
    "* **Name**: Mara Voss\n"
    "* **Headline**: Senior Web Platform Engineer\n"
    "* **Contact**: Lisbon, PT ; mara@example.invalid; +351 000\n"
    "* **Summary**: Eleven years building fictional platforms; mostly web.\n"
    "* **Education**: Fictional University — BSc, 2013; Pretend Cert, 2021\n"
    "\n"
)
L3A_KEYS = ("Name", "Headline", "Contact", "Summary", "Education")


def _header_line(key: str) -> str:
    return next(ln for ln in L3A_HEADER.splitlines() if ln.startswith(f"* **{key}**:"))


def test_l3a_header_keys_before_first_job():
    # P-1 — the five keys as given; Contact / Education split on ";", trimmed.
    result = parse_star_vault(L3A_HEADER + _job("Alpha", "2020", "2021"))
    assert result.issues == []
    header = result.header
    assert header.name == "Mara Voss"
    assert header.headline == "Senior Web Platform Engineer"
    assert header.contact == ["Lisbon, PT", "mara@example.invalid", "+351 000"]
    assert header.summary == "Eleven years building fictional platforms; mostly web."
    assert header.education == ["Fictional University — BSc, 2013", "Pretend Cert, 2021"]
    assert [j.name for j in result.jobs] == ["Alpha"]


@pytest.mark.parametrize("key", L3A_KEYS)
def test_l3a_header_key_absent_is_empty(key):
    # P-2 — an absent key is None (one-line fields) or [] (list fields); nothing else moves.
    doc = L3A_HEADER.replace(_header_line(key) + "\n", "") + _job("Alpha", "2020", "2021")
    result = parse_star_vault(doc)
    assert result.issues == []
    attr = key.lower()
    assert getattr(result.header, attr) == ([] if key in ("Contact", "Education") else None)
    full = parse_star_vault(L3A_HEADER + _job("Alpha", "2020", "2021")).header
    for other in L3A_KEYS:
        if other != key:
            assert getattr(result.header, other.lower()) == getattr(full, other.lower())
    assert parse_star_vault(_job("Alpha", "2020", "2021")).header.model_dump() == {
        "name": None,
        "headline": None,
        "contact": [],
        "summary": None,
        "education": [],
    }


@pytest.mark.parametrize("key", L3A_KEYS)
def test_l3a_header_key_after_first_job_is_error(key):
    # P-3 — inside a job, and in a skipped section after one: an error naming the key.
    line = _header_line(key)
    inside = _vault("  - Text. {id: aaa111}\n", header=HEADER + line + "\n")
    result = parse_star_vault(inside)
    errs = result.errors()
    assert [i.line_no for i in errs] == [_line_of(inside, line)]
    assert f"**{key}**" in errs[0].message and "vault-level line" in errs[0].message
    alpha = result.jobs[0]
    assert (alpha.date_from, alpha.date_to, alpha.defaults) == ("2020", "2021", [])
    assert result.header.model_dump()[key.lower()] in (None, [])

    late = _job("Alpha", "2020", "2021") + "\n## Notes\n" + line + "\n"
    late_result = parse_star_vault(late)
    assert [i.line_no for i in late_result.errors()] == [_line_of(late, line)]


def test_l3a_duplicate_header_key_is_error():
    # P-4
    doc = L3A_HEADER + "* **Name**: Someone Else\n\n" + _job("Alpha", "2020", "2021")
    result = parse_star_vault(doc)
    dup = _messages(result, "duplicate **Name** line")
    assert [i.line_no for i in dup] == [_line_of(doc, "Someone Else")]
    assert len(result.errors()) == 1
    assert result.header.name == "Mara Voss"


def test_l3a_unknown_preamble_key_is_one_warning():
    # P-5 — only in the true preamble (before any `##` heading); nothing else changes.
    doc = L3A_HEADER.replace("* **Name**:", "* **Nmae**:") + _job("Alpha", "2020", "2021")
    result = parse_star_vault(doc)
    assert result.errors() == []
    warnings = result.warnings()
    assert [w.line_no for w in warnings] == [_line_of(doc, "* **Nmae**:")]
    assert "Nmae" in warnings[0].message
    assert result.header.name is None
    assert result.header.headline == "Senior Web Platform Engineer"
    # a key line in a skipped section body stays silent, before the first job too
    skipped = "# Vault\n\n## Notes\n* **Whatever**: x\n\n" + _job("Alpha", "2020", "2021")
    assert [w.message for w in parse_star_vault(skipped).warnings()] == [
        "unknown section '## Notes' skipped"
    ]


def test_l3a_grouped_skills_in_file_order(basic):
    # P-6 — group lines under a Skills line; `skills` = the flat union,
    # casefold-deduped; the jobs parse exactly as in the ungrouped vault.
    grouped = parse_star_vault(_read("basic_grouped_vault.md"))
    assert grouped.errors() == []
    assert [(g.name, g.items) for g in grouped.skill_groups] == [
        ("", ["Code Review", "git"]),
        ("Languages, Frameworks, and Data", ["TypeScript", "Redis", "Python"]),
        ("Platforms and Delivery", ["Feature Flags", "python", "Dashboards"]),
    ]
    assert grouped.skills == [
        "Code Review",
        "git",
        "TypeScript",
        "Redis",
        "Python",
        "Feature Flags",
        "Dashboards",
    ]
    plain = parse_star_vault(basic)

    def shape(result):
        return [
            (
                j.model_dump(exclude={"line_no", "bullets"}),
                [b.model_dump(exclude={"line_no"}) for b in j.bullets],
            )
            for j in result.jobs
        ]

    assert shape(grouped) == shape(plain)


def test_l3a_flat_skills_line_is_one_unnamed_group():
    # P-7 — the flat line parses as one group named ""; group lines may follow it (P3: first).
    flat = parse_star_vault("# Vault\n* **Skills**: a; b\n\n" + _job("Alpha", "2020", "2021"))
    assert [(g.name, g.items) for g in flat.skill_groups] == [("", ["a", "b"])]
    assert flat.skills == ["a", "b"]
    mixed = parse_star_vault(
        "# Vault\n* **Skills**: a; b\n  * **Cloud**: c; A\n\n" + _job("Alpha", "2020", "2021")
    )
    assert mixed.issues == []
    assert [(g.name, g.items) for g in mixed.skill_groups] == [
        ("", ["a", "b"]),
        ("Cloud", ["c", "A"]),
    ]
    assert mixed.skills == ["a", "b", "c"]
    bare = parse_star_vault("# Vault\n* **Skills**:\n\n" + _job("Alpha", "2020", "2021"))
    assert (bare.skill_groups, bare.skills, bare.issues) == ([], [], [])


@pytest.mark.parametrize(
    "block, bad",
    [
        ("* **Skills**:\n  * **Cloud**:\n", "  * **Cloud**:"),
        ("* **Skills**:\n  * **Cl*oud**: a\n", "Cl*oud"),
        ("* **Name**: X\n  * **Cloud**: a\n", "  * **Cloud**: a"),
        ("* **Skills**:\n  * **Cloud**: a\n  * **cloud**: b\n", "**cloud**: b"),
    ],
    ids=["no-items", "star-in-name", "not-under-skills", "duplicate-name"],
)
def test_l3a_bad_group_line_is_error(block, bad):
    # P-8 — each defect is one error at its own line.
    doc = "# Vault\n" + block + "\n" + _job("Alpha", "2020", "2021")
    result = parse_star_vault(doc)
    assert [i.line_no for i in result.errors()] == [_line_of(doc, bad)]


def test_l3a_grouped_skills_after_first_job_is_one_error():
    # P-9 — the Skills line is the error; its group lines are skipped without a second one.
    doc = _job("Alpha", "2020", "2021") + "\n## Notes\n* **Skills**:\n  * **Cloud**: a; b\n"
    result = parse_star_vault(doc)
    errs = result.errors()
    assert [i.line_no for i in errs] == [_line_of(doc, "* **Skills**:")]
    assert "vault-level line" in errs[0].message
    assert result.skill_groups == [] and result.skills == []


def test_l3a_header_vault_fixture_end_to_end():
    # P-10
    result = parse_star_vault(_read("header_vault.md"))
    assert result.issues == []
    assert result.header.name == "Mara Voss"
    assert len(result.header.contact) == 4
    assert len(result.header.education) == 2
    assert [g.name for g in result.skill_groups] == [
        "Languages, Frameworks, and Data",
        "Platforms and Delivery",
    ]
    assert len(result.skills) == 7
    assert [len(j.bullets) for j in result.jobs] == [3]
