"""JD text → `JDProfile` via one generation node.

The prompt is a Jinja2 template (`loomcv/resources/prompts/parse_jd.md`) rendered whole and
split on the `---USER---` line into system / user messages. The model's reply is
cleaned (one leading `<think>` block, one surrounding ``` fence, whitespace) and
validated as `JDProfile`; one retry on failure, then `JDParseError`. A `nice_to_have`
item that duplicates a `must_have` item is dropped. There is no
domain / tech vocabulary any more: a reply that still carries
`domains` / `tech` is accepted and the keys are ignored.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import jinja2
from pydantic import BaseModel, BeforeValidator, ValidationError

from loomcv.llm import GenerationAdapter
from loomcv.state import JDProfile
from loomcv.sync.parse_assets import slugify

_USER_MARKER = "---USER---"
_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)
# One ``` fence, whether the JSON starts on the line after the opening ticks
# (```json\n{...}\n```) or shares the line with them and/or the closing ticks
# (```json {...} ```).
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class JDParseError(Exception):
    """The model's output was not a valid `JDProfile` after one retry."""

    def __init__(self, message: str, raws: list[str]) -> None:
        super().__init__(message)
        self.raws = raws


class JDParseResult(BaseModel):
    profile: JDProfile
    warnings: list[str]
    raw: str


class _Item(BaseModel):
    text: str
    skills: list[str] = []


def _coerce_item(value: Any) -> Any:
    """A bare string (the v1 reply shape) becomes an item with no skills."""
    return {"text": value, "skills": []} if isinstance(value, str) else value


_ItemOrStr = Annotated[_Item, BeforeValidator(_coerce_item)]


class _ReplyV2(BaseModel):
    """The model's reply shape; flattened to `JDProfile` after dedup."""

    title: str
    seniority: Literal["junior", "mid", "senior", "staff"]
    domain: str
    must_have: list[_ItemOrStr]
    nice_to_have: list[_ItemOrStr] = []
    responsibilities: list[_ItemOrStr] = []
    constraints: list[str] = []
    traits: list[str] = []
    ats_keywords: list[str] = []


def render_prompt(jd_text: str, *, template_path: Path) -> tuple[str, str]:
    """Render the template whole, split on the first `---USER---` line → (system, user)."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    template = env.from_string(Path(template_path).read_text(encoding="utf-8"))
    rendered = template.render(jd_text=jd_text)
    lines = rendered.splitlines()
    for i, line in enumerate(lines):
        if line == _USER_MARKER:
            system = "\n".join(lines[:i]).strip()
            user = "\n".join(lines[i + 1 :]).strip()
            return system, user
    raise ValueError(f"prompt template {template_path} has no '{_USER_MARKER}' line")


def parse_jd(jd_text: str, *, adapter: GenerationAdapter, template_path: Path) -> JDParseResult:
    system, user = render_prompt(jd_text, template_path=template_path)

    first = adapter.complete(system, user, temperature=adapter.temperature)
    try:
        profile, raw = _validate(first), first
    except ValidationError as exc:
        print(
            f"parse_jd: first reply invalid ({str(exc).splitlines()[0]}); retrying once",
            file=sys.stderr,
        )
        retry_user = (
            f"{user}\n\nYour previous answer was not valid: {exc}. Return only the JSON object."
        )
        second = adapter.complete(system, retry_user, temperature=adapter.temperature)
        try:
            profile, raw = _validate(second), second
        except ValidationError as exc2:
            raise JDParseError(
                f"model output was not a valid JDProfile after one retry: {exc2}",
                raws=[first, second],
            ) from exc2

    profile, warnings = _post_validate(profile)
    return JDParseResult(profile=profile, warnings=warnings, raw=raw)


def _clean_reply(raw: str) -> str:
    """(a) one leading <think> block, (b) one ``` fence, (c) whitespace."""
    text = _THINK_RE.sub("", raw, count=1).strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    return text.strip()


def _validate(raw: str) -> _ReplyV2:
    return _ReplyV2.model_validate_json(_clean_reply(raw))


def _dedup_exact(values: list[str], field: str) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    warnings: list[str] = []
    for value in values:
        if value in kept:
            warnings.append(f"dropped {field} '{value}' (duplicate within {field})")
        else:
            kept.append(value)
    return kept, warnings


def _clean_skills(requirement: str, skills: list[str]) -> tuple[list[str], list[str]]:
    """Strip and case-insensitive dedup (silent, first spelling wins)."""
    kept: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        key = skill.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(skill.strip())
    return kept, warnings


def _dedup_items(items: list[_Item], field: str) -> tuple[list[_Item], list[str]]:
    texts, warnings = _dedup_exact([item.text for item in items], field)
    first = {}
    for item in items:
        first.setdefault(item.text, item)
    return [first[text] for text in texts], warnings


def _post_validate(reply: _ReplyV2) -> tuple[JDProfile, list[str]]:
    """must/nice/responsibilities dedup;
    skills built from the kept lists (deduplicated, not filtered);
    must_have / nice_to_have items slug-equal to a kept constraints / traits entry
    are dropped with a warning."""
    warnings: list[str] = []

    constraints, constraint_warnings = _dedup_exact(reply.constraints, "constraints")
    traits, trait_warnings = _dedup_exact(reply.traits, "traits")
    warnings += constraint_warnings + trait_warnings

    # constraints win when a slug is in both lists.
    exclusive_slugs = {slugify(entry): "traits" for entry in traits}
    exclusive_slugs.update({slugify(entry): "constraints" for entry in constraints})

    def _without_exclusive(items: list[_Item], field: str) -> list[_Item]:
        candidates: list[_Item] = []
        for item in items:
            owner = exclusive_slugs.get(slugify(item.text))
            if owner is None:
                candidates.append(item)
            else:
                warnings.append(f"dropped {field} '{item.text}' (duplicate of {owner})")
        return candidates

    must_items, must_warnings = _dedup_items(
        _without_exclusive(reply.must_have, "must_have"), "must_have"
    )
    warnings += must_warnings

    must_slugs = {slugify(item.text) for item in must_items}

    def _without_must_duplicates(items: list[_Item], field: str) -> list[_Item]:
        candidates: list[_Item] = []
        for item in items:
            if slugify(item.text) in must_slugs:
                warnings.append(f"dropped {field} '{item.text}' (duplicate of must_have)")
            else:
                candidates.append(item)
        return candidates

    nice_items, nice_warnings = _dedup_items(
        _without_must_duplicates(
            _without_exclusive(reply.nice_to_have, "nice_to_have"), "nice_to_have"
        ),
        "nice_to_have",
    )
    warnings += nice_warnings
    resp_items, resp_warnings = _dedup_items(
        _without_must_duplicates(reply.responsibilities, "responsibilities"), "responsibilities"
    )
    warnings += resp_warnings

    # Built from the kept lists only, must -> nice -> responsibilities, first text wins;
    # a requirement whose cleaned list is empty gets no key.
    skills: dict[str, list[str]] = {}
    seen_texts: set[str] = set()
    for item in [*must_items, *nice_items, *resp_items]:
        if item.text in seen_texts:
            continue
        seen_texts.add(item.text)
        kept, skill_warnings = _clean_skills(item.text, item.skills)
        warnings += skill_warnings
        if kept:
            skills[item.text] = kept

    profile = JDProfile(
        title=reply.title,
        seniority=reply.seniority,
        domain=reply.domain,
        must_have=[item.text for item in must_items],
        nice_to_have=[item.text for item in nice_items],
        responsibilities=[item.text for item in resp_items],
        constraints=constraints,
        traits=traits,
        ats_keywords=reply.ats_keywords,
        skills=skills,
    )
    return profile, warnings
