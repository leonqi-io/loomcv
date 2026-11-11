"""The coverage mapping: one model call per JD over the index's keyword list.

The prompt (`loomcv/resources/prompts/coverage.md`, Jinja2, split on the `---USER---` line like
`parse_jd`) lists every requirement line (must, responsibilities, nice) and the union of
the confirmed bullet keywords, and asks for one grade per line — `exact` / `adjacent` /
`none` — with the keywords the grade rests on. The model never sees a bullet, only the
keyword list. The answer lives on the `JDProfile` (`coverage`, `coverage_model`) and is
replayed, never re-asked.

A reply is complete or it fails: its keys must be
exactly the mapped lines and every value `{"match": …, "keywords": [...]}`; otherwise it is
retried once, then `CoverageError`. No partial mapping is ever returned, so a missing line can
never become a gap. A keyword that is not in the index is dropped with a warning
and the rest are re-spelled to the index's spelling. Nothing is re-graded by code:
no threshold, no count rule. `check_coverage` checks a stored mapping against the
current run before replay.

The same call, asked a second time on the lines graded `none`, over the vault's *claims*
(`claims_of`: the keywords stated in a job's `Defaults` or on the `Skills` line that no bullet
carries) — `check_claims` is its replay check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jinja2
from pydantic import ValidationError

from loomcv import resources
from loomcv.llm import GenerationAdapter
from loomcv.parse_jd import _clean_reply
from loomcv.retrieve.rank import Candidate, Claim, Requirement
from loomcv.state import Coverage, CoveragePart

# The prompt ships with the package, so its path is package-relative.
DEFAULT_COVERAGE_PROMPT = resources.path("prompts", "coverage.md")

_USER_MARKER = "---USER---"

# A validation bound like the prompt's "At most 6 keywords", not config.
MAX_PARTS = 6


class CoverageError(Exception):
    """The coverage call failed: transport error, or no valid reply after one retry."""

    def __init__(self, message: str, raws: list[str]) -> None:
        super().__init__(message)
        self.raws = raws


class StoredCoverageError(Exception):
    """A stored `profile.coverage` does not fit this run."""


class _InvalidReply(Exception):
    pass


def vault_keywords(candidates: list[Candidate]) -> list[str]:
    """The union of the candidates' keywords: case-insensitive dedup keeping the first spelling,
    sorted case-insensitively (the shape T1's prompt listed; no other transformation)."""
    first: dict[str, str] = {}
    for c in candidates:
        for keyword in c.keywords:
            first.setdefault(keyword.casefold(), keyword)
    return sorted(first.values(), key=str.casefold)


def render_coverage_prompt(
    lines: list[Requirement],
    keywords: list[str],
    *,
    template_path: Path = DEFAULT_COVERAGE_PROMPT,
) -> tuple[str, str]:
    """Render the coverage prompt → (system, user)."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    template = env.from_string(Path(template_path).read_text(encoding="utf-8"))
    rendered = template.render(lines=[r.text for r in lines], keywords=keywords)
    rendered_lines = rendered.splitlines()
    for i, line in enumerate(rendered_lines):
        if line == _USER_MARKER:
            system = "\n".join(rendered_lines[:i]).strip()
            user = "\n".join(rendered_lines[i + 1 :]).strip()
            return system, user
    raise ValueError(f"prompt template {template_path} has no '{_USER_MARKER}' line")


def _keep_parts(text: str, answer: Coverage, warnings: list[str]) -> list[CoveragePart]:
    """The answer's parts, drop-only: a line graded `none` has none; a part
    whose term is not (whitespace-normalised, case-insensitive) in the requirement is dropped with
    a warning; at most `MAX_PARTS` are kept."""
    if answer.match == "none":
        return []
    haystack = " ".join(text.split()).casefold()
    kept: list[CoveragePart] = []
    for part in answer.parts:
        needle = " ".join(part.term.split()).casefold()
        if not needle or needle not in haystack:
            warnings.append(
                f"coverage: dropped part {part.term!r} for {text!r} (not named in the line)"
            )
        else:
            kept.append(part)
    if len(kept) > MAX_PARTS:
        warnings.append(f"coverage: {text!r} has {len(kept)} parts; kept the first {MAX_PARTS}")
        kept = kept[:MAX_PARTS]
    return kept


def _validate(
    raw: str, line_texts: list[str], keywords: list[str]
) -> tuple[dict[str, Coverage], list[str]]:
    try:
        reply = json.loads(_clean_reply(raw))
    except json.JSONDecodeError as exc:
        raise _InvalidReply(f"not valid JSON ({exc.msg})") from exc
    if not isinstance(reply, dict):
        raise _InvalidReply("the reply is not a JSON object")

    for text in line_texts:
        if text not in reply:
            raise _InvalidReply(f"missing line {text!r}")
    wanted = set(line_texts)
    for key in reply:
        if key not in wanted:
            raise _InvalidReply(f"unexpected key {key!r}")

    index = {k.casefold(): k for k in keywords}
    warnings: list[str] = []
    coverage: dict[str, Coverage] = {}
    for text in line_texts:
        try:
            answer = Coverage.model_validate(reply[text])
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise _InvalidReply(f"invalid answer for {text!r} ({detail})") from exc
        kept: list[str] = []
        for keyword in answer.keywords:
            spelled = index.get(keyword.casefold())
            if spelled is None:
                warnings.append(
                    f"coverage: dropped keyword {keyword!r} for {text!r} (not in index)"
                )
            elif spelled not in kept:
                kept.append(spelled)
        if answer.match != "none" and not kept:
            warnings.append(f"coverage: {text!r} graded {answer.match} has no keywords left")
        coverage[text] = Coverage(
            match=answer.match, keywords=kept, parts=_keep_parts(text, answer, warnings)
        )
    return coverage, warnings


def _complete(adapter: GenerationAdapter, system: str, user: str, raws: list[str]) -> str:
    try:
        raw = adapter.complete(system, user, temperature=adapter.temperature)
    except (OSError, RuntimeError) as exc:  # URLError is an OSError: no retry
        raise CoverageError(f"coverage call failed: {exc}", raws=raws) from exc
    raws.append(raw)
    return raw


def map_coverage(
    lines: list[Requirement],
    keywords: list[str],
    *,
    adapter: GenerationAdapter,
    template_path: Path = DEFAULT_COVERAGE_PROMPT,
) -> tuple[dict[str, Coverage], int]:
    """One adapter call (two after the single retry) → `(coverage in line order, calls made)`.

    No lines → no call (`({}, 0)`).
    """
    if not lines:
        return {}, 0
    system, user = render_coverage_prompt(lines, keywords, template_path=template_path)
    line_texts = [r.text for r in lines]

    raws: list[str] = []
    first = _complete(adapter, system, user, raws)
    try:
        coverage, warnings = _validate(first, line_texts, keywords)
    except _InvalidReply as exc:
        print(f"coverage: first reply invalid ({exc}); retrying once", file=sys.stderr)
        retry_user = (
            f"{user}\n\nYour previous answer was not valid: {exc}. Return only the JSON object."
        )
        second = _complete(adapter, system, retry_user, raws)
        try:
            coverage, warnings = _validate(second, line_texts, keywords)
        except _InvalidReply as exc2:
            raise CoverageError(
                f"coverage reply invalid after one retry: {exc2}", raws=raws
            ) from exc2

    for warning in warnings:
        print(warning, file=sys.stderr)
    return coverage, len(raws)


def claims_of(
    jobs: list[tuple[str, list[str]]], skills: list[str], candidates: list[Candidate]
) -> list[Claim]:
    """The vault's claims: every keyword stated in a job's `Defaults` or on the
    vault-level `Skills` line that no loaded bullet carries.

    `jobs` is `(job_id, defaults)` in job order. Case-insensitive; the first spelling is kept;
    claims come in first-appearance order (jobs, then `Skills`); each claim lists the `job_id`s
    whose `Defaults` carry it, once each, then `"skills"`. A word on any candidate is evidence
    (the coverage call already saw it), wherever else it is also written.
    """
    evidence = {k.casefold() for c in candidates for k in c.keywords}
    spelled: dict[str, str] = {}
    sources: dict[str, list[str]] = {}

    def add(keyword: str, source: str) -> None:
        folded = keyword.casefold()
        if folded in evidence:
            return
        spelled.setdefault(folded, keyword)
        at = sources.setdefault(folded, [])
        if source not in at:
            at.append(source)

    for job_id, defaults in jobs:
        for keyword in defaults:
            add(keyword, job_id)
    for keyword in skills:
        add(keyword, "skills")
    return [Claim(keyword=spelled[folded], at=at) for folded, at in sources.items()]


def _check_mapping(
    mapping: dict[str, Coverage],
    lines: list[Requirement],
    keywords: list[str],
    *,
    what: str,
    source: str,
) -> None:
    """Raise `StoredCoverageError` at the first difference between a stored mapping and this run.

    Order: a current line without an entry, an entry that is not a current line, a stored
    keyword that is not in `keywords` (case-insensitive).
    """
    for r in lines:
        if r.text not in mapping:
            raise StoredCoverageError(f"stored {what} has no entry for {r.kind} line {r.text!r}")
    current = {r.text for r in lines}
    for key in mapping:
        if key not in current:
            raise StoredCoverageError(
                f"stored {what} has an entry for {key!r}, which is not a line of this profile"
            )
    index = {k.casefold() for k in keywords}
    for key, answer in mapping.items():
        for keyword in answer.keywords:
            if keyword.casefold() not in index:
                raise StoredCoverageError(
                    f"stored {what} keyword {keyword!r} for {key!r} is not in {source}"
                )


def check_coverage(
    coverage: dict[str, Coverage], lines: list[Requirement], keywords: list[str]
) -> None:
    """Check a stored coverage mapping against this run's lines and the index's keywords."""
    _check_mapping(coverage, lines, keywords, what="coverage", source="the index")


def check_claims(
    claims: dict[str, Coverage], lines: list[Requirement], keywords: list[str]
) -> None:
    """Check a stored claims mapping against the lines graded `none` and the current claims:
    the `check_coverage` rule over the asked lines and the claims list."""
    _check_mapping(claims, lines, keywords, what="claims", source="the vault's claims")
