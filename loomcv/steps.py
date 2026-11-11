"""The three pipeline stages as functions: `match_step`, `draft_step`,
`finalize_step` — the bodies of `pipeline match` / `draft` / `finalize` without argparse, `print`
or exit codes. The CLI and the graph (`loomcv/graph.py`) call these; behaviour is the CLI's,
byte for byte (the match / draft / finalize CLI tests are the proof).

Each step raises the typed errors the CLI maps to exit codes and prints nothing. This module may
import every LoomCV package and imports neither LangGraph nor LangChain (a test pins that):
the graph is orchestration only, the model calls stay in `loomcv/llm.py` over `config/models.yaml`.

`Runtime` is what the steps need from outside — paths, config files, adapter builders. It is lazy:
an adapter or config is built on first use and kept, so a replay that needs no `coverage` node
never resolves one (as the CLI never did). Tests inject fakes into `Runtime.adapters`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import URLError

import yaml
from pydantic import BaseModel, ValidationError

from loomcv import resources
from loomcv.complete.workbook import Workbook, WriteReport, write_workbook
from loomcv.coverage import CoverageError
from loomcv.draft import (
    REPORT_NAME,
    RESUME_NAME,
    check_fresh,
    render_report,
    render_resume,
)
from loomcv.finalize import export as finalize_export
from loomcv.llm import build_generation_adapter, load_node_config
from loomcv.parse_jd import JDParseError, parse_jd
from loomcv.retrieve.rank import MatchPayload, MatchResult
from loomcv.retrieve.rerank import build_rerank_adapter, load_rerank_config
from loomcv.retrieve.search import (
    load_candidates,
    load_claims_inputs,
    load_job_rows,
    load_skill_groups,
    load_vault_header,
    match,
)
from loomcv.state import (
    GenerationConfig,
    JDProfile,
    RetrievalConfig,
    load_generation_config,
    load_retrieval_config,
    resolve_config_file,
)
from loomcv.sync.embedding import build_embedding_adapter, load_embedding_config
from loomcv.sync.index_assets import SyncReport, sync_vault

_DEFAULT_PROMPT = resources.path("prompts", "parse_jd.md")


class InvalidInput(Exception):
    """'Your input is wrong' (CLI exit 1): the message is the exact line the CLI prints."""


class OutputExists(InvalidInput):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        super().__init__(
            f"{', '.join(str(path) for path in paths)} already exists; use --force to overwrite"
        )


class NodeCallError(Exception):
    """A generation node (`parse_jd` / `coverage`) could not be built or its call failed (exit 4);
    `str(exc)` is `<node>: <cause>`, the line the CLI prints."""

    def __init__(self, node: str, cause: Exception) -> None:
        self.node = node
        self.cause = cause
        super().__init__(f"{node}: {cause}")


@dataclass(frozen=True)
class Builders:
    """Config → adapter functions. The CLI passes its own module's names, so a test that patches
    `loomcv.cli.build_*` reaches the steps unchanged."""

    generation: Callable[..., Any] = build_generation_adapter
    embedding: Callable[..., Any] = build_embedding_adapter
    rerank: Callable[..., Any] = build_rerank_adapter


@dataclass
class Runtime:
    """What the steps need from outside; each adapter / config is built on first use and kept."""

    db: Path
    chroma: Path
    prompt: Path = _DEFAULT_PROMPT
    models_config: Path | None = None  # --config
    retrieval_config_path: Path | None = None  # --retrieval-config
    generation_config_path: Path | None = None  # --generation-config
    builders: Builders = field(default_factory=Builders)
    # "parse_jd" | "coverage" | "embedding" | "reranker"; tests inject fakes here.
    adapters: dict[str, Any] = field(default_factory=dict)
    _retrieval: RetrievalConfig | None = field(default=None, init=False, repr=False)
    _generation: GenerationConfig | None = field(default=None, init=False, repr=False)

    def adapter(self, name: str) -> Any:
        if name in self.adapters:
            return self.adapters[name]
        if name in ("parse_jd", "coverage"):
            node_cfg, provider_cfg = load_node_config(name, self.models_config)
            built = self.builders.generation(node_cfg, provider_cfg)
        elif name == "embedding":
            built = self.builders.embedding(load_embedding_config(self.models_config))
        elif name == "reranker":
            built = self.builders.rerank(load_rerank_config(self.models_config))
        else:
            raise KeyError(f"unknown adapter {name!r}")
        self.adapters[name] = built
        return built

    def retrieval(self) -> RetrievalConfig:
        if self._retrieval is None:
            self._retrieval = load_retrieval_config(self.retrieval_config_path)
        return self._retrieval

    def generation(self) -> GenerationConfig:
        if self._generation is None:
            try:
                self._generation = load_generation_config(self.generation_config_path)
            except (OSError, ValidationError, yaml.YAMLError) as exc:
                raise InvalidInput(f"invalid generation config: {exc}") from exc
        return self._generation


def build_runtime(
    *,
    db: Path,
    chroma: Path,
    prompt: Path = _DEFAULT_PROMPT,
    config: Path | None = None,
    retrieval_config: Path | None = None,
    generation_config: Path | None = None,
    builders: Builders | None = None,
) -> Runtime:
    """The entry the UI uses; lazy, so it builds nothing."""
    return Runtime(
        db=db,
        chroma=chroma,
        prompt=prompt,
        models_config=config,
        retrieval_config_path=retrieval_config,
        generation_config_path=generation_config,
        builders=builders or Builders(),
    )


@dataclass(frozen=True)
class ParsedJD:
    profile: JDProfile
    warnings: list[str]


@dataclass(frozen=True)
class MatchStepResult:
    profile: JDProfile  # = result.profile (coverage + claims filled)
    result: MatchResult
    parse_warnings: list[str]
    payload: dict  # match_payload(result, runtime.retrieval(), parse_warnings)


@dataclass(frozen=True)
class DraftResult:
    resume_path: Path
    report_path: Path
    jobs: int
    bullets: int
    skills: int


@dataclass(frozen=True)
class FinalizeResult:
    md: Path
    typ: Path
    pdf: Path
    pages: int
    page_limit: int  # resolved: the argument or generation.yaml's
    warnings: list[str]


def parse_step(runtime: Runtime, jd_text: str) -> ParsedJD:
    """JD text → `parse_jd` node → post-validated profile. The index is checked before the
    parse call is spent: the coverage keyword list needs it, and `IndexNotReadyError` /
    `SchemaVersionError` propagate unwrapped (CLI exit 3); anything else on the generation path is
    `NodeCallError("parse_jd")` (exit 4)."""
    try:
        adapter = runtime.adapter("parse_jd")
    except (KeyError, ValueError, RuntimeError, NotImplementedError) as exc:
        raise NodeCallError("parse_jd", exc) from exc
    load_candidates(runtime.db)  # exit-3 check only; `match` reads the keywords itself
    try:
        result = parse_jd(jd_text, adapter=adapter, template_path=runtime.prompt)
    except (
        KeyError,
        ValueError,
        RuntimeError,
        NotImplementedError,
        URLError,
        JDParseError,
    ) as exc:
        raise NodeCallError("parse_jd", exc) from exc
    return ParsedJD(profile=result.profile, warnings=result.warnings)


def match_payload(result: MatchResult, config: RetrievalConfig, parse_warnings: list[str]) -> dict:
    """Top-level keys in this order: `profile` (with its coverage filled),
    `parse_warnings` (always present), `requirements` (one per line: grade, mapped keywords and
    the reranked list), `gaps` (the must lines graded none), `claims` (the lines graded
    none that the vault's `Defaults` / `Skills` claims cover — an annotation, `gaps` is
    unchanged), `selection` (the merge: placed bullets by job, the walk, the coverage lookups),
    `config` (echoes the config used) and `diagnostics`. Nothing in it is a score across lines."""
    return {
        "profile": result.profile.model_dump(mode="json"),
        "parse_warnings": parse_warnings,
        "requirements": [line.model_dump(mode="json") for line in result.requirements],
        "gaps": result.gaps,
        "claims": [c.model_dump(mode="json") for c in result.claims],
        "selection": result.selection.model_dump(mode="json"),
        "config": config.model_dump(),
        "diagnostics": result.diagnostics.model_dump(mode="json"),
    }


def match_step(
    runtime: Runtime,
    *,
    jd_text: str | None = None,
    profile: JDProfile | None = None,
    parse_warnings: Sequence[str] = (),
    save_profile: Path | None = None,
) -> MatchStepResult:
    """Rank the index against a profile (or a JD text, parsed first). `save_profile` is written
    before the coverage node is resolved and rewritten after a successful run with the coverage
    the run used, so the saved profile replays with zero generation calls."""
    if (jd_text is None) == (profile is None):
        raise ValueError("match_step takes exactly one of jd_text= and profile=")
    warnings = list(parse_warnings)
    if jd_text is not None:
        parsed = parse_step(runtime, jd_text)
        profile, warnings = parsed.profile, parsed.warnings
    assert profile is not None
    if save_profile is not None:
        save_profile.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
    # The coverage node is resolved before any retrieval work; no silent path without it. A
    # profile that already carries its coverage and its claims replays with no node at all.
    coverage = None
    if not (profile.coverage and profile.claims is not None):
        try:
            coverage = runtime.adapter("coverage")
        except (KeyError, ValueError, RuntimeError, NotImplementedError) as exc:
            raise NodeCallError("coverage", exc) from exc
    config = runtime.retrieval()
    adapter = runtime.adapter("embedding")
    reranker = runtime.adapter("reranker")
    try:
        result = match(
            profile,
            db_path=runtime.db,
            chroma_path=runtime.chroma,
            adapter=adapter,
            reranker=reranker,
            config=config,
            coverage=coverage,
        )
    except CoverageError as exc:
        raise NodeCallError("coverage", exc) from exc
    if save_profile is not None:
        save_profile.write_text(result.profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return MatchStepResult(
        profile=result.profile,
        result=result,
        parse_warnings=warnings,
        payload=match_payload(result, config, warnings),
    )


def draft_step(
    runtime: Runtime,
    payload: MatchPayload,
    *,
    out_dir: Path,
    force: bool = False,
) -> DraftResult:
    """`tailored_resume.md` + `match_report.md` from a payload: nothing is written before
    every check has passed. The header and the Skills groups are the vault's, read from the
    index. Raises `IndexNotReadyError` / `SchemaVersionError` (exit 3),
    `StaleMatchError` / `ResumeTemplateError` / `InvalidInput` / `OutputExists` (exit 1)."""
    generation = runtime.generation()
    indexed = {c.bullet_id: c.text for c in load_candidates(runtime.db)}
    rows = load_job_rows(runtime.db)
    skills = load_claims_inputs(runtime.db)[1]
    check_fresh(payload, indexed)
    resume = render_resume(
        payload,
        rows,
        load_skill_groups(runtime.db),
        load_vault_header(runtime.db),
        template_path=resolve_config_file(generation.resume_template),
    )
    targets = {
        RESUME_NAME: resume,
        REPORT_NAME: render_report(payload, rows),
    }
    existing = [out_dir / name for name in targets if (out_dir / name).exists()]
    if existing and not force:
        raise OutputExists(existing)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in targets.items():
            (out_dir / name).write_text(text, encoding="utf-8", newline="")
    except OSError as exc:
        raise InvalidInput(f"cannot write {out_dir}: {exc}") from exc
    placed = sum(len(job.bullets) for job in payload.selection.jobs)
    return DraftResult(
        resume_path=out_dir / RESUME_NAME,
        report_path=out_dir / REPORT_NAME,
        jobs=len(rows),
        bullets=placed,
        skills=len(skills),
    )


def finalize_step(
    runtime: Runtime,
    resume_md: Path,
    *,
    out_dir: Path | None = None,
    page_limit: int | None = None,
    force: bool = False,
) -> FinalizeResult:
    """Strip the audit comments and export `final_resume.md` / `.typ` / `.pdf`: nothing is
    written before every input check has passed. `InvalidInput` / `OutputExists` (exit 1);
    `PandocFailed` / `TypstFailed` propagate (exit 6). Over the page limit is a finding, not an
    error: the caller compares `pages` with `page_limit`."""
    generation = runtime.generation()
    defaults = resolve_config_file(generation.pandoc_defaults)
    if not defaults.is_file():
        raise InvalidInput(f"pandoc defaults file {defaults} does not exist")
    try:
        # Bytes, so no newline translation: the output differs from the input by the comments only.
        text = resume_md.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InvalidInput(f"invalid résumé {resume_md}: {exc}") from exc
    target = out_dir if out_dir is not None else resume_md.parent
    final_md, final_typ, final_pdf = (
        target / name
        for name in (
            finalize_export.FINAL_MD,
            finalize_export.FINAL_TYP,
            finalize_export.FINAL_PDF,
        )
    )
    existing = [path for path in (final_md, final_typ, final_pdf) if path.exists()]
    if existing and not force:
        raise OutputExists(existing)
    try:
        target.mkdir(parents=True, exist_ok=True)
        # A stale .typ / .pdf must not survive beside a new .md if a converter fails below.
        final_typ.unlink(missing_ok=True)
        final_pdf.unlink(missing_ok=True)
        final_md.write_text(finalize_export.strip_src(text), encoding="utf-8", newline="")
    except OSError as exc:
        raise InvalidInput(f"cannot write {target}: {exc}") from exc
    # Through the module attribute: the tests patch `loomcv.finalize.export.export_*`.
    finalize_export.export_typ(final_md, final_typ, defaults)
    warnings = finalize_export.export_pdf(final_typ, final_pdf)
    pages = finalize_export.count_pages(final_pdf)
    return FinalizeResult(
        md=final_md,
        typ=final_typ,
        pdf=final_pdf,
        pages=pages,
        page_limit=page_limit or generation.page_limit,
        warnings=list(warnings),
    )


# ------------------------------------------------------------------ save_profile


class SaveReport(BaseModel):
    """What Save profile did."""

    written: WriteReport  # the vault file as written and verified by the strict parser
    sync: SyncReport | None = None  # None: the index was not built (or not asked)
    index_error: str | None = None  # why it was not built; None when built

    @property
    def ready(self) -> bool:
        """The profile is usable for applications: the index was built from this vault."""
        return self.sync is not None


def save_profile(
    wb: Workbook,
    saved: Path,
    *,
    db: Path,
    chroma: Path,
    adapter: Callable[[], Any],
    today: date,
) -> SaveReport:
    """Save profile = write the vault, then build the index from it.
    `write_workbook(wb, saved)` first — an `OSError` from the write
    propagates. A vault the strict parser reports problems in is written and **not** indexed
    (`index_error` = the problems, as `pipeline sync` would refuse it). Otherwise `adapter()`
    (a zero-argument factory, so a missing embedding config is reported like a failed index)
    and `sync_vault`, which writes the `vault_sha256` row last; any exception from the index
    side → `index_error`, never raised: the index is derived and the vault is already on
    disk."""
    written = write_workbook(wb, saved, today=today)
    if written.problems:
        return SaveReport(written=written, index_error="; ".join(written.problems))
    try:
        report = sync_vault(saved, db_path=db, chroma_path=chroma, adapter=adapter())
    except Exception as exc:  # the index is derived; the vault stays written
        return SaveReport(written=written, index_error=str(exc) or type(exc).__name__)
    return SaveReport(written=written, sync=report)
