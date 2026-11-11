"""Typed schemas at system boundaries and retrieval-config loading.

`JDProfile` (the per-line coverage mapping lives on the
profile), `Coverage` and `RetrievalConfig`, whose values live in `config/retrieval.yaml`
and are never hardcoded here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from loomcv import resources

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "retrieval.yaml"


CoverageMatch = Literal["exact", "adjacent", "none"]


class CoveragePart(BaseModel):
    """One named thing of an all-required compound line."""

    term: str  # as the requirement spells it
    match: CoverageMatch


class Coverage(BaseModel):
    """The coverage model's answer for one requirement line."""

    match: CoverageMatch  # the line: the strongest part's level
    keywords: list[str]
    parts: list[CoveragePart] = []  # [] = one thing, alternatives, or a line graded none


class JDProfile(BaseModel):
    title: str
    seniority: Literal["junior", "mid", "senior", "staff"]
    domain: str
    must_have: list[str]
    nice_to_have: list[str] = []
    responsibilities: list[str] = []
    constraints: list[str] = []
    traits: list[str] = []
    ats_keywords: list[str] = []
    # requirement string (must / nice / responsibility, byte-equal) -> canonical skill terms
    skills: dict[str, list[str]] = {}
    # requirement string (must / responsibility / nice, byte-equal) -> the coverage model's answer;
    # empty until `match` has run the coverage node, then stored and replayed
    coverage: dict[str, Coverage] = {}
    coverage_model: str | None = None
    # requirement string of a line graded `none` -> the claims call's answer over the vault's
    # claims. `None` = never asked; otherwise exactly what the call returned
    # (`none` answers included), `{}` only when no call was needed. Stored and replayed like
    # `coverage`.
    claims: dict[str, Coverage] | None = None
    claims_model: str | None = None


class RetrievalConfig(BaseModel):
    # A stale yaml still carrying a retired key (hybrid_alpha, min_score, gap_cosine_floor,
    # gap_bm25_floor, judge_candidates, per_project_cap, final_bullets, nice_to_have_weight)
    # fails loudly instead of loading silently.
    model_config = ConfigDict(extra="forbid")

    top_k: int
    rrf_k: int
    # The merge's budget: the résumé's length in bullets, and the per-job
    # shares (by `job_order`, the last value repeating) that divide it. Both required, no default.
    total_slots: int = Field(ge=1)
    recency_shares: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)


def load_retrieval_config(config_path: Path | None = None) -> RetrievalConfig:
    path = config_path or _DEFAULT_CONFIG_PATH
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RetrievalConfig.model_validate(raw)


_DEFAULT_GENERATION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "generation.yaml"


class GenerationConfig(BaseModel):
    # Five keys, five readers; an unknown key
    # fails loudly instead of loading silently. Kept apart from `retrieval.yaml`, which `match`
    # echoes into every payload. `bullet_max_chars` has a default so an older four-key file loads.
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)  # pipeline draft (default output directory)
    resume_template: str = Field(min_length=1)  # pipeline draft -> render_resume
    pandoc_defaults: str = Field(min_length=1)  # pipeline finalize -> export_typ
    page_limit: int = Field(gt=0, strict=True)  # pipeline finalize (page check)
    # Over budget stays a warning.
    bullet_max_chars: int = Field(default=250, gt=0, strict=True)  # pipeline ui -> draft_bullet


def resolve_config_file(value: str) -> Path:
    """`resume_template` / `pandoc_defaults`: an
    absolute path is used as is; a relative one names a shipped resource (`loomcv/resources/`)
    when that file exists, else a file relative to the working directory. A value with a `..`
    part never enters the package lookup."""
    path = Path(value)
    if path.is_absolute():
        return path
    if ".." not in path.parts:
        try:
            return resources.path(*path.parts)
        except FileNotFoundError:
            pass
    return Path.cwd() / path


def load_generation_config(config_path: Path | None = None) -> GenerationConfig:
    path = config_path or _DEFAULT_GENERATION_CONFIG_PATH
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return GenerationConfig.model_validate(raw)
