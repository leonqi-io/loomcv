"""Tests for `loomcv/parse_jd.py`.

The template is the real `loomcv/resources/prompts/parse_jd.md`; every model
reply is a literal JSON string produced by a scripted `FakeGenerationAdapter`. No test
reaches the network (autouse fixture, same as `test_llm.py`). `loomcv.parse_jd` is
imported inside each test so every case fails on its own at red. The
domain / tech vocabularies and their prompt blocks are gone.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from loomcv import resources

TEMPLATE = resources.path("prompts", "parse_jd.md")
JD_TEXT = "Fictional Corp seeks a senior player engineer. Must know TypeScript."

_BASE = {
    "title": "Fictional Player Engineer",
    "seniority": "senior",
    "domain": "fictional streaming",
    "must_have": ["TypeScript experience"],
    "nice_to_have": ["Mentoring"],
    "ats_keywords": ["TypeScript"],
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("network call in test")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


class FakeGenerationAdapter:
    """Scripted, network-free stand-in for `GenerationAdapter`."""

    def __init__(
        self, replies: list[str], *, temperature: float = 0, model_id: str = "fake-model"
    ) -> None:
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
        return self._replies.pop(0)


def _expected(**overrides) -> dict:
    data = dict(_BASE)
    data.update(overrides)
    return data


def _reply(**overrides) -> str:
    return json.dumps(_expected(**overrides))


def _parse(adapter: FakeGenerationAdapter):
    from loomcv.parse_jd import parse_jd

    return parse_jd(JD_TEXT, adapter=adapter, template_path=TEMPLATE)


# ------------------------------------------------------------------ test 4


def test_render_prompt(tmp_path):
    import jinja2

    from loomcv.parse_jd import render_prompt

    system, user = render_prompt(JD_TEXT, template_path=TEMPLATE)
    # No vocabulary blocks; the schema line ends with `ats_keywords`.
    assert '"domains"' not in system
    assert '"tech": [string]' not in system
    assert "zero or more entries chosen ONLY" not in system
    assert "domain slugs" not in system
    assert "technology names" not in system
    assert system.endswith('"ats_keywords": [string]}')
    assert JD_TEXT in user
    assert "---USER---" not in system
    assert "---USER---" not in user

    undefined = tmp_path / "undefined.md"
    undefined.write_text("{{ nope }}\n---USER---\n{{ jd_text }}\n", encoding="utf-8")
    with pytest.raises(jinja2.exceptions.UndefinedError):
        render_prompt(JD_TEXT, template_path=undefined)

    no_marker = tmp_path / "no_marker.md"
    no_marker.write_text("system only\n{{ jd_text }}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        render_prompt(JD_TEXT, template_path=no_marker)


# ------------------------------------------------------------------ test 5


def test_parse_valid_reply():
    from loomcv.state import JDProfile

    reply = _reply()
    adapter = FakeGenerationAdapter([reply], temperature=0.2)

    result = _parse(adapter)

    assert result.profile == JDProfile(**_expected())
    assert result.warnings == []
    assert result.raw == reply
    assert len(adapter.calls) == 1
    assert adapter.calls[0][2] == 0.2


# ------------------------------------------------------------------ test 6


def test_parse_fenced_think_and_retry():
    from loomcv.state import JDProfile

    body = _reply()

    fenced = f"```json\n{body}\n```"
    adapter = FakeGenerationAdapter([fenced])
    assert _parse(adapter).profile == JDProfile(**_expected())
    assert len(adapter.calls) == 1

    thinking = f"<think>\nlooking at the JD…\n</think>\n{body}"
    adapter = FakeGenerationAdapter([thinking])
    result = _parse(adapter)
    assert result.profile == JDProfile(**_expected())
    assert result.raw == thinking
    assert len(adapter.calls) == 1

    adapter = FakeGenerationAdapter([f"Sure! {body}", body])
    result = _parse(adapter)
    assert result.profile == JDProfile(**_expected())
    assert len(adapter.calls) == 2
    first_system, first_user, _ = adapter.calls[0]
    second_system, second_user, _ = adapter.calls[1]
    assert second_system == first_system
    assert second_user.startswith(first_user)
    assert "not valid" in second_user


# ------------------------------------------------------------------ test 7


def test_parse_two_invalid_replies_raise():
    from loomcv.parse_jd import JDParseError

    adapter = FakeGenerationAdapter(["nope", "{}"])

    with pytest.raises(JDParseError) as exc:
        _parse(adapter)

    assert exc.value.raws == ["nope", "{}"]
    assert len(adapter.calls) == 2


# ------------------------------------------------------------------ test 8


def test_post_validation_ignores_retired_keys_and_retries():
    # A reply from an old prompt still carrying "domains" / "tech" parses; the keys
    # are ignored (pydantic default), with no vocabulary warning and no such profile fields.
    adapter = FakeGenerationAdapter(
        [_reply(domains=["streaming-media", "fintech"], tech=["shaka player", "Rust"])]
    )
    result = _parse(adapter)
    assert result.warnings == []
    dumped = result.profile.model_dump()
    assert "domains" not in dumped
    assert "tech" not in dumped

    adapter = FakeGenerationAdapter([_reply(seniority="principal"), _reply(seniority="staff")])
    result = _parse(adapter)
    assert len(adapter.calls) == 2
    assert result.profile.seniority == "staff"


# ----------------------------------------------------------------- test 8b


def test_must_nice_dedup():
    adapter = FakeGenerationAdapter(
        [_reply(must_have=["Build REST APIs"], nice_to_have=["build rest APIs", "Mentoring"])]
    )
    result = _parse(adapter)
    assert result.profile.must_have == ["Build REST APIs"]
    assert result.profile.nice_to_have == ["Mentoring"]
    assert len(result.warnings) == 1
    assert "duplicate of must_have" in result.warnings[0]

    adapter = FakeGenerationAdapter([_reply(must_have=["X", "X"])])
    result = _parse(adapter)
    assert result.profile.must_have == ["X"]
    assert len(result.warnings) == 1


# ------------------------------------------------------------------ test 9


def test_min_year_ignored_in_reply():
    # PJ-1: a reply that still returns "min_year" parses; the key is dropped silently.
    result = _parse(FakeGenerationAdapter([_reply(min_year=2024)]))
    assert result.warnings == []
    assert "min_year" not in result.profile.model_dump()


def test_render_prompt_has_no_year_rule():
    # PJ-2: neither prompt half mentions min_year or the current year.
    from loomcv.parse_jd import render_prompt

    system, user = render_prompt(JD_TEXT, template_path=TEMPLATE)
    for text in (system, user):
        assert "min_year" not in text.lower()
        assert "current year" not in text.lower()


# ------------------------------------------- prompt v2 (P-1 … P-7)


def _item(text: str, *skills: str) -> dict:
    return {"text": text, "skills": list(skills)}


# ------------------------------------------------------------------ P-1


def test_parse_v2_reply():
    reply = _reply(
        must_have=[
            _item("Operate Kafka clusters", "Kafka"),
            _item("Write TypeScript", "TypeScript"),
        ],
        nice_to_have=[_item("Terraform modules", "Terraform", "IaC")],
        responsibilities=[_item("Own the player release pipeline", "CI/CD")],
        constraints=["Authorized to work in Canada"],
        traits=["Communicate clearly with stakeholders"],
    )

    profile = _parse(FakeGenerationAdapter([reply])).profile

    assert profile.must_have == ["Operate Kafka clusters", "Write TypeScript"]
    assert profile.nice_to_have == ["Terraform modules"]
    assert profile.responsibilities == ["Own the player release pipeline"]
    assert profile.constraints == ["Authorized to work in Canada"]
    assert profile.traits == ["Communicate clearly with stakeholders"]
    assert profile.skills == {
        "Operate Kafka clusters": ["Kafka"],
        "Write TypeScript": ["TypeScript"],
        "Terraform modules": ["Terraform", "IaC"],
        "Own the player release pipeline": ["CI/CD"],
    }

    v1 = _parse(FakeGenerationAdapter([_reply()])).profile
    assert v1.must_have == ["TypeScript experience"]
    assert v1.responsibilities == []
    assert v1.constraints == []
    assert v1.traits == []
    assert v1.skills == {}


# ------------------------------------------------------------------ P-2


def test_parse_mixed_bare_and_object_items():
    reply = _reply(
        must_have=["Bare requirement", _item("Object requirement", "Redis")],
        nice_to_have=[_item("Object nice", "GraphQL"), "Bare nice"],
    )

    result = _parse(FakeGenerationAdapter([reply]))

    assert result.profile.must_have == ["Bare requirement", "Object requirement"]
    assert result.profile.nice_to_have == ["Object nice", "Bare nice"]
    assert result.profile.skills.get("Bare requirement", []) == []
    assert result.profile.skills.get("Bare nice", []) == []
    assert result.profile.skills["Object requirement"] == ["Redis"]
    assert result.profile.skills["Object nice"] == ["GraphQL"]


# ------------------------------------------------------------------ P-3


def test_post_validate_new_lists_and_skills_first_wins():
    reply = _reply(
        must_have=[_item("Build REST APIs", "REST API")],
        nice_to_have=[_item("build rest apis", "GraphQL")],
        responsibilities=[
            _item("Build REST APIs!", "REST API"),
            _item("Run on-call rotations", "incident response"),
        ],
        constraints=["Onsite in Toronto", "Onsite in Toronto"],
        traits=["Stay calm", "Stay calm"],
    )

    result = _parse(FakeGenerationAdapter([reply]))
    profile, warnings = result.profile, result.warnings

    assert profile.must_have == ["Build REST APIs"]
    assert profile.nice_to_have == []
    assert profile.responsibilities == ["Run on-call rotations"]
    assert "dropped responsibilities 'Build REST APIs!' (duplicate of must_have)" in warnings
    assert "dropped nice_to_have 'build rest apis' (duplicate of must_have)" in warnings
    assert "Build REST APIs!" not in profile.skills
    assert profile.skills["Build REST APIs"] == ["REST API"]
    assert profile.skills["Run on-call rotations"] == ["incident response"]
    assert profile.constraints == ["Onsite in Toronto"]
    assert profile.traits == ["Stay calm"]
    assert "dropped constraints 'Onsite in Toronto' (duplicate within constraints)" in warnings
    assert "dropped traits 'Stay calm' (duplicate within traits)" in warnings


# ------------------------------------------------------------------ P-4


def test_generic_skills_kept():
    reply = _reply(
        must_have=[
            _item("Design services", "systems", "REST API", "tools"),
            _item("Use LLMs", "LLM", "llm"),
            _item("Ship safe models", "AI", "AI safety"),
        ],
    )

    result = _parse(FakeGenerationAdapter([reply]))

    assert result.profile.skills["Design services"] == ["systems", "REST API", "tools"]
    assert result.profile.skills["Use LLMs"] == ["LLM"]
    assert result.profile.skills["Ship safe models"] == ["AI", "AI safety"]
    assert not [w for w in result.warnings if "(generic)" in w]


# ------------------------------------------------------------------ P-9


def test_skills_dedup_case_insensitive_nothing_else():
    reply = _reply(
        must_have=[_item("Build AI platform", "AI systems", "ai systems", "APIs", "Python")],
    )

    result = _parse(FakeGenerationAdapter([reply]))

    assert result.profile.skills["Build AI platform"] == ["AI systems", "APIs", "Python"]
    assert not [w for w in result.warnings if "skill" in w or "(generic)" in w]


# ------------------------------------------------------------------ P-5


def test_b1_stderr_line_on_retry(capsys):
    adapter = FakeGenerationAdapter(["not json", _reply()])
    result = _parse(adapter)

    assert result.profile.must_have == ["TypeScript experience"]
    assert len(adapter.calls) == 2
    err = capsys.readouterr().err
    assert err.count("retrying once") == 1
    assert err.startswith("parse_jd: first reply invalid (")

    adapter = FakeGenerationAdapter([_reply()])
    _parse(adapter)
    assert len(adapter.calls) == 1
    assert capsys.readouterr().err == ""


# ------------------------------------------------------------------ P-6


def test_b2_single_line_fence():
    from loomcv.state import JDProfile

    body = _reply()
    for wrapped in (f"```json {body} ```", f"```{body}```", f"```json\n{body}```"):
        adapter = FakeGenerationAdapter([wrapped])
        assert _parse(adapter).profile == JDProfile(**_expected())
        assert len(adapter.calls) == 1


# ------------------------------------------------------------------ P-7


def test_render_prompt_v2_rules():
    from loomcv.parse_jd import render_prompt

    system, _user = render_prompt(JD_TEXT, template_path=TEMPLATE)

    assert '"responsibilities"' in system
    assert '"constraints"' in system
    assert '"traits"' in system
    assert '"skills": [string]' in system
    flat = " ".join(system.split())
    assert "If you cannot name one skill term, the item is not a requirement" in flat


def test_render_prompt_section_rule():
    # P-1: the section decides required vs preferred, and the model
    # judges the section by its meaning; checked on the whitespace-flattened system part.
    from loomcv.parse_jd import render_prompt

    system, _user = render_prompt(JD_TEXT, template_path=TEMPLATE)
    flat = " ".join(system.split())

    # (a) any heading that states what the candidate must have; the examples are not a list
    assert "whatever its own wording" in flat
    assert "What you bring" in flat
    assert "examples, not a complete list" in flat
    # (b) a separate preferred section makes the other qualification section required
    for marker in ("Preferred", "Bonus", "Nice to have", "Plus", "Desired"):
        assert marker in flat
    # (c) an explicit optional marker on the item wins over its section
    assert "An explicit optional marker on the item wins over its section" in flat
    # (d) no preferred section and an unclear heading: the item's own wording decides
    assert "the item's own wording decides" in flat
    # (e) an experience line without a technology is a requirement; qualities stay traits
    assert "experience running cross-team design reviews" in flat
    for quality in ("curious", "calm under pressure", "a patient colleague"):
        assert quality in flat
    # the stale clause: the template renders no "tech" list
    assert '"tech" list' not in flat


# ------------------------------------------------------------------ PH

_YEARS = "5–8+ years of software engineering experience"


def test_exclusive_lists_drop_must_have_in_constraints():
    # PH-1
    reply = _reply(
        must_have=[
            _item(_YEARS, "software engineering"),
            _item("TypeScript experience", "TypeScript"),
        ],
        constraints=[_YEARS],
    )

    result = _parse(FakeGenerationAdapter([reply]))

    assert result.profile.must_have == ["TypeScript experience"]
    assert result.profile.constraints == [_YEARS]
    warning = f"dropped must_have '{_YEARS}' (duplicate of constraints)"
    assert result.warnings.count(warning) == 1
    assert _YEARS not in result.profile.skills


def test_exclusive_lists_drop_nice_to_have_in_traits():
    # PH-2
    reply = _reply(
        nice_to_have=[_item("communicate CLEARLY!", "communication"), _item("Mentoring")],
        traits=["Communicate clearly"],
    )

    result = _parse(FakeGenerationAdapter([reply]))

    assert result.profile.nice_to_have == ["Mentoring"]
    assert result.profile.traits == ["Communicate clearly"]
    assert "dropped nice_to_have 'communicate CLEARLY!' (duplicate of traits)" in result.warnings


def test_exclusive_lists_leave_responsibilities():
    # PH-3: responsibilities are not checked against constraints
    reply = _reply(
        responsibilities=[_item("Onsite in Toronto", "onsite")],
        constraints=["Onsite in Toronto"],
    )

    result = _parse(FakeGenerationAdapter([reply]))

    assert result.profile.responsibilities == ["Onsite in Toronto"]
    assert result.profile.constraints == ["Onsite in Toronto"]
    assert not any("Onsite in Toronto" in warning for warning in result.warnings)


def test_exclusive_lists_drop_before_must_dedup():
    # PH-4: the drop runs before must dedup, so nice is not blamed on must_have
    reply = _reply(
        must_have=[_item(_YEARS, "software engineering")],
        nice_to_have=[_item(_YEARS, "software engineering")],
        constraints=[_YEARS],
    )

    result = _parse(FakeGenerationAdapter([reply]))

    assert result.profile.must_have == []
    assert result.profile.nice_to_have == []
    assert len(result.warnings) == 2
    assert all(warning.endswith("(duplicate of constraints)") for warning in result.warnings)
    assert not any("duplicate of must_have" in warning for warning in result.warnings)


# ------------------------------------------------------------------ P-8


def test_nice_resp_byte_equal_skills_first_wins():
    # P-8: covers existing behaviour — the rank side keeps the text once
    # as a responsibility; the parser keeps it in both lists and nice's skills win.
    text = "Mentor junior engineers"
    reply = _reply(
        nice_to_have=[_item(text, "mentoring")],
        responsibilities=[_item(text, "coaching")],
    )

    profile = _parse(FakeGenerationAdapter([reply])).profile

    assert profile.nice_to_have == [text]
    assert profile.responsibilities == [text]
    assert profile.skills[text] == ["mentoring"]
