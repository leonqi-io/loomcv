"""Tests for `loomcv/coverage.py` (one mapping call per JD over the keyword list).

The prompt template is the real `loomcv/resources/prompts/coverage.md`; every model reply is a
literal JSON string from a scripted `FakeGenerationAdapter`; lines and keywords are in-test
fictional strings (the one exception, `test_render_coverage_prompt_t1_reproduction`, reads the
committed synthetic T1 prompt file). No test reaches the network (autouse fixture, as
`test_parse_jd.py`). `loomcv.coverage` and the new `rank` / `state` symbols are imported inside
each test so every case fails on its own at red.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from loomcv import resources

ROOT = Path(__file__).resolve().parents[1]
T1_PROMPT = ROOT / "dev_flow" / "eval" / "t1" / "keyword_mapping_prompt.md"
COVERAGE_PROMPT = resources.path("prompts", "coverage.md")

KEYWORDS = ["Kafka", "Message Brokers", "Python"]
MUST = "Kafka experience"
RESP = "Run production services"
NICE = "Experience with Rust"
VALID = {
    MUST: {"match": "exact", "keywords": ["Kafka"]},
    RESP: {"match": "none", "keywords": []},
    NICE: {"match": "adjacent", "keywords": ["Python"]},
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("network call in test")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


class FakeGenerationAdapter:
    """Scripted, network-free stand-in for `GenerationAdapter`."""

    def __init__(self, replies, *, temperature: float = 0, model_id: str = "fake-coverage") -> None:
        self._replies = list(replies)
        self._temperature = temperature
        self._model_id = model_id
        self.calls: list[tuple[str, str, float]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def temperature(self) -> float:
        return self._temperature

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        self.calls.append((system, user, temperature))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _cov():
    import loomcv.coverage as coverage_module

    return coverage_module


def _lines():
    from loomcv.retrieve.rank import Requirement

    return [
        Requirement(text=MUST, kind="must"),
        Requirement(text=RESP, kind="responsibility"),
        Requirement(text=NICE, kind="nice"),
    ]


def _candidate(bullet_id, keywords):
    from loomcv.retrieve.rank import Candidate

    return Candidate(
        bullet_id=bullet_id, text="t", job_id="j", job="J", project=None, keywords=keywords
    )


def _stored(mapping=None):
    from loomcv.state import Coverage

    return {k: Coverage.model_validate(v) for k, v in (mapping or VALID).items()}


# ------------------------------------------------------------------ TR9


def test_vault_keywords_union_dedup_sort():
    candidates = [
        _candidate("a", ["Kafka", "alerting"]),
        _candidate("b", ["Alerting", "reserve", "Zeta"]),
        _candidate("c", []),
    ]

    assert _cov().vault_keywords(candidates) == ["alerting", "Kafka", "reserve", "Zeta"]


# ------------------------------------------------------------------ TR10


def _without_rule_3_and_output(lines: list[str]) -> list[str]:
    """The rule text minus rule 3 (up to rule 4) and the Output paragraph (to the end)."""
    kept: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith("3. ") or line == "Output":
            skipping = True
        elif line.startswith("4. "):
            skipping = False
        if not skipping:
            kept.append(line)
    return kept


def test_coverage_prompt_rule_text_byte_identical():
    # Rule 3 and the Output paragraph differ from the reference prompt; every other rule line is
    # still T1's, byte for byte (the T1 file is not edited).
    t1_lines = T1_PROMPT.read_text(encoding="utf-8").splitlines()
    t1_rules = t1_lines[: t1_lines.index("Requirements")]
    product = COVERAGE_PROMPT.read_text(encoding="utf-8").splitlines()
    product_rules = product[: product.index("---USER---")]

    assert "3. " in "\n".join(product_rules) and "Output" in product_rules
    assert _without_rule_3_and_output(product_rules) == _without_rule_3_and_output(t1_rules)


def test_render_coverage_prompt_user_part():
    from loomcv.retrieve.rank import Requirement

    lines = [Requirement(text="L1", kind="must"), Requirement(text="L2", kind="nice")]

    system, user = _cov().render_coverage_prompt(lines, ["a", "b", "c"])

    assert user == "Requirements\n- L1\n- L2\n\nKeywords (3)\n- a\n- b\n- c"
    assert system.endswith("Every requirement must appear once.")
    assert "---USER---" not in system
    assert "---USER---" not in user


def test_render_coverage_prompt_t1_reproduction():
    # The template reproduces the tested artefact: T1's lists in, T1's prompt text out.
    from loomcv.retrieve.rank import Requirement

    text = T1_PROMPT.read_text(encoding="utf-8")
    raw_lines = text.splitlines()
    start = raw_lines.index("Requirements")
    kw_start = next(i for i, ln in enumerate(raw_lines) if ln.startswith("Keywords ("))
    requirements = [ln[2:] for ln in raw_lines[start + 1 : kw_start] if ln.startswith("- ")]
    keywords = [ln[2:] for ln in raw_lines[kw_start + 1 :] if ln.startswith("- ")]
    assert len(requirements) == 11
    assert len(keywords) == 182

    system, user = _cov().render_coverage_prompt(
        [Requirement(text=r, kind="must") for r in requirements], keywords
    )

    # The user part still reproduces T1 byte for byte (the system part is T1's except
    # rule 3 and the Output paragraph — the test above).
    assert user == "\n".join(raw_lines[start:]).strip()
    assert system.endswith("Every requirement must appear once.")


# ------------------------------------------------------------------ TR11 support, TR12


def test_map_coverage_returns_lines_in_order_and_one_call():
    reply = json.dumps(dict(reversed(list(VALID.items()))))
    adapter = FakeGenerationAdapter([reply])

    coverage, calls = _cov().map_coverage(_lines(), KEYWORDS, adapter=adapter)

    assert calls == 1
    assert list(coverage) == [MUST, RESP, NICE]
    assert coverage[MUST].match == "exact"
    assert coverage[NICE].keywords == ["Python"]
    assert adapter.calls[0][2] == 0


def test_map_coverage_accepts_fenced_reply():
    adapter = FakeGenerationAdapter(["```json\n" + json.dumps(VALID) + "\n```"])

    coverage, calls = _cov().map_coverage(_lines(), KEYWORDS, adapter=adapter)

    assert calls == 1
    assert list(coverage) == [MUST, RESP, NICE]


def _missing_line():
    return {k: v for k, v in VALID.items() if k != RESP}, RESP


def _extra_key():
    return {**VALID, "Something else": {"match": "none", "keywords": []}}, "Something else"


def _bad_match():
    return {**VALID, MUST: {"match": "maybe", "keywords": ["Kafka"]}}, "match"


@pytest.mark.parametrize("make", [_missing_line, _extra_key, _bad_match])
def test_map_coverage_invalid_reply_retries_once_then_errors(make, capsys):
    bad, named = make()
    adapter = FakeGenerationAdapter([json.dumps(bad), json.dumps(bad)])

    with pytest.raises(_cov().CoverageError) as excinfo:
        _cov().map_coverage(_lines(), KEYWORDS, adapter=adapter)

    assert len(adapter.calls) == 2
    first_user, second_user = adapter.calls[0][1], adapter.calls[1][1]
    assert second_user.startswith(first_user)
    assert "Your previous answer was not valid" in second_user
    assert "retrying once" in capsys.readouterr().err
    assert "after one retry" in str(excinfo.value)
    assert len(excinfo.value.raws) == 2
    if make is not _bad_match:
        assert named in str(excinfo.value)


def test_map_coverage_non_json_then_valid(capsys):
    adapter = FakeGenerationAdapter(["Sure! Here you go.", json.dumps(VALID)])

    coverage, calls = _cov().map_coverage(_lines(), KEYWORDS, adapter=adapter)

    assert calls == 2
    assert len(adapter.calls) == 2
    assert list(coverage) == [MUST, RESP, NICE]
    assert capsys.readouterr().err.count("retrying once") == 1


def test_map_coverage_unknown_keyword_dropped_and_respelled(capsys):
    reply = {
        MUST: {"match": "exact", "keywords": ["kafka", "Nope"]},
        RESP: {"match": "adjacent", "keywords": ["Unknown Thing"]},
        NICE: {"match": "none", "keywords": []},
    }
    adapter = FakeGenerationAdapter([json.dumps(reply)])

    coverage, calls = _cov().map_coverage(_lines(), KEYWORDS, adapter=adapter)

    assert calls == 1
    assert coverage[MUST].keywords == ["Kafka"]
    # Kept as the model said: graded adjacent, no keywords left.
    assert coverage[RESP].match == "adjacent"
    assert coverage[RESP].keywords == []
    err = capsys.readouterr().err
    assert "dropped keyword 'Nope'" in err
    assert "dropped keyword 'Unknown Thing'" in err
    assert "has no keywords left" in err


def test_map_coverage_transport_error_no_retry():
    adapter = FakeGenerationAdapter([urllib.error.URLError("connection refused")])

    with pytest.raises(_cov().CoverageError):
        _cov().map_coverage(_lines(), KEYWORDS, adapter=adapter)

    assert len(adapter.calls) == 1


# ------------------------------------------------------------------ parts (C-1 .. C-7)

COMPOUND = "Kafka and Python experience"


def _compound_lines():
    from loomcv.retrieve.rank import Requirement

    return [Requirement(text=COMPOUND, kind="must")]


def _compound_reply(parts, *, match="exact", keywords=("Kafka", "Python")):
    return json.dumps({COMPOUND: {"match": match, "keywords": list(keywords), "parts": parts}})


def test_map_coverage_stores_valid_parts():
    # C-1: parts are kept in order; a term differing only by case / inner whitespace is kept as
    # the model spelled it.
    reply = _compound_reply(
        [{"term": "Kafka", "match": "exact"}, {"term": "python", "match": "adjacent"}]
    )
    adapter = FakeGenerationAdapter([reply])

    coverage, calls = _cov().map_coverage(_compound_lines(), KEYWORDS, adapter=adapter)

    assert calls == 1
    assert [(p.term, p.match) for p in coverage[COMPOUND].parts] == [
        ("Kafka", "exact"),
        ("python", "adjacent"),
    ]
    assert coverage[COMPOUND].match == "exact"
    assert coverage[COMPOUND].keywords == ["Kafka", "Python"]


@pytest.mark.parametrize("term", ["Rust", "   "], ids=["not-in-line", "empty"])
def test_map_coverage_drops_part_not_in_line(term, capsys):
    # C-2: drop-only validation; the line and its other parts stay.
    reply = _compound_reply([{"term": "Kafka", "match": "exact"}, {"term": term, "match": "none"}])
    adapter = FakeGenerationAdapter([reply])

    coverage, calls = _cov().map_coverage(_compound_lines(), KEYWORDS, adapter=adapter)

    assert calls == 1
    assert [p.term for p in coverage[COMPOUND].parts] == ["Kafka"]
    assert coverage[COMPOUND].match == "exact"
    err = capsys.readouterr().err
    assert f"coverage: dropped part {term!r} for {COMPOUND!r} (not named in the line)" in err


def test_map_coverage_keeps_first_six_parts(capsys):
    # C-3
    from loomcv.retrieve.rank import Requirement

    text = "A1 and A2 and A3 and A4 and A5 and A6 and A7 and A8"
    parts = [{"term": f"A{i}", "match": "exact"} for i in range(1, 9)]
    reply = json.dumps({text: {"match": "exact", "keywords": ["Kafka"], "parts": parts}})
    adapter = FakeGenerationAdapter([reply])

    coverage, _calls = _cov().map_coverage(
        [Requirement(text=text, kind="must")], KEYWORDS, adapter=adapter
    )

    assert [p.term for p in coverage[text].parts] == ["A1", "A2", "A3", "A4", "A5", "A6"]
    err = capsys.readouterr().err
    assert err.count("kept the first 6") == 1
    assert _cov().MAX_PARTS == 6


def test_map_coverage_none_line_clears_parts():
    # C-4: a line graded none is wholly not covered — its parts are cleared
    reply = _compound_reply(
        [{"term": "Kafka", "match": "none"}, {"term": "Python", "match": "none"}],
        match="none",
        keywords=(),
    )
    adapter = FakeGenerationAdapter([reply])

    coverage, _calls = _cov().map_coverage(_compound_lines(), KEYWORDS, adapter=adapter)

    assert coverage[COMPOUND].match == "none"
    assert coverage[COMPOUND].parts == []


def test_map_coverage_reply_without_parts_validates():
    # C-5: today's reply shape — one call, `parts == []` everywhere
    adapter = FakeGenerationAdapter([json.dumps(VALID)])

    coverage, calls = _cov().map_coverage(_lines(), KEYWORDS, adapter=adapter)

    assert calls == 1
    assert [c.parts for c in coverage.values()] == [[], [], []]


def test_stored_profile_without_parts_loads_and_checks():
    # C-6: a profile saved before this ticket has no `parts`; it loads and replays unchanged
    from loomcv.state import JDProfile

    profile = JDProfile.model_validate(
        {
            "title": "T",
            "seniority": "senior",
            "domain": "d",
            "must_have": [MUST],
            "coverage": VALID,
        }
    )

    assert all(c.parts == [] for c in profile.coverage.values())
    _cov().check_coverage(profile.coverage, _lines(), KEYWORDS)


def test_coverage_prompt_compound_rules():
    # C-7: rule 3 grades the parts of an all-required line; alternatives stay one thing. Checked
    # on the whitespace-flattened text — the old sentence wrapped over two lines.
    system, _user = _cov().render_coverage_prompt(_lines(), KEYWORDS)
    flat = " ".join(system.split())

    assert "that are all required" in flat
    assert '"or", "such as"' in flat
    assert "at least one is met exactly" not in flat
    assert '"parts"' in flat[flat.index("Output") :]
    assert flat.endswith("Every requirement must appear once.")


# ------------------------------------------------------------------ TR13


def test_check_coverage_accepts_complete():
    _cov().check_coverage(_stored(), _lines(), KEYWORDS)


def test_check_coverage_rejects_unknown_key():
    stored = _stored({**VALID, "Not a line": {"match": "none", "keywords": []}})

    with pytest.raises(_cov().StoredCoverageError, match="not a line of this profile"):
        _cov().check_coverage(stored, _lines(), KEYWORDS)


@pytest.mark.parametrize("missing", [MUST, RESP, NICE])
def test_check_coverage_rejects_missing_line_each_kind(missing):
    stored = _stored({k: v for k, v in VALID.items() if k != missing})

    with pytest.raises(_cov().StoredCoverageError, match="no entry for") as excinfo:
        _cov().check_coverage(stored, _lines(), KEYWORDS)

    assert missing in str(excinfo.value)


def test_check_coverage_rejects_keyword_not_in_index():
    stored = _stored({**VALID, NICE: {"match": "adjacent", "keywords": ["Fortran"]}})

    with pytest.raises(_cov().StoredCoverageError, match="not in the index"):
        _cov().check_coverage(stored, _lines(), KEYWORDS)


def test_check_coverage_keyword_lookup_is_case_insensitive():
    stored = _stored({**VALID, MUST: {"match": "exact", "keywords": ["kafka"]}})

    _cov().check_coverage(stored, _lines(), KEYWORDS)


# ------------------------------------------------------- X-5: the claims set


def test_x5_claims_of_set_difference_and_sources():
    # (every job's Defaults + the Skills line) minus the candidates' keywords;
    # casefold, first spelling kept, first-appearance order, one source per job, then "skills".
    from loomcv.retrieve.rank import Claim

    jobs = [("j1", ["Kafka", "Terraform", "terraform"]), ("j2", ["TERRAFORM", "Rust"])]
    skills = ["Terraform", "Go", "kafka"]
    candidates = [_candidate("a", ["Kafka"]), _candidate("b", ["Go"])]

    claims = _cov().claims_of(jobs, skills, candidates)

    assert claims == [
        Claim(keyword="Terraform", at=["j1", "j2", "skills"]),
        Claim(keyword="Rust", at=["j2"]),
    ]
    assert _cov().claims_of([], [], candidates) == []
