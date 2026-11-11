"""`pipeline sync` command-line entry point.

A thin argparse wrapper over the `sync_vault` engine. It owns only the
things the engine does not: argv parsing, adapter construction from config,
exception → exit-code translation, and one-line-per-counter stdout output. No
behaviour is added to the engine here.

Stdlib only — no typer/click/rich. The `sync` sub-parser uses a sub-parser
layout so later sub-commands (like `run`) slot in without
restructuring the parser. Exit codes: 0 success, 1 VaultParseError,
3 the three store errors, 2 argparse usage (its own default, not caught here);
no catch-all, so a genuine bug still surfaces as a traceback.

`pipeline ingest`: 0 draft written and strict-clean (a `0000` date
is not an error); 1 résumé unreadable, output unwritable, no jobs extracted (nothing
written), or strict-parser errors on the written draft (the file stays for reading);
2 argparse; 4 anything on the `ingest` generation path.

`pipeline complete`: 0 success; 1 draft unreadable or with parser
errors (nothing written), output unwritable, or strict-parser problems on the written vault (the
file stays for reading); 2 argparse (no mode flag); 4 anything on the `ingest` generation
path. `pipeline ui`: 1 draft missing or the optional `ui` group not installed.

`pipeline draft` (no model call): 0 both files written; 1 payload unreadable
or invalid, stale (a placed bullet the index no longer has as confirmed, or with other text),
the résumé template unreadable or naming a slot the data lacks (the header is the vault's),
`generation.yaml` invalid, or an output file exists without `--force` (nothing written);
2 argparse; 3 index not ready (incl. `--db` at no file) / schema mismatch.

`pipeline finalize` (no model call, no index): 0 all three files written and
within `page_limit`; 1 input unreadable / not UTF-8, `generation.yaml` invalid, the pandoc
defaults file missing, an output exists without `--force`, or an output unwritable (nothing
written in the first four cases); 2 argparse; 5 page limit exceeded (all files kept); 6 the
converter failed (pandoc non-zero or typst compile error; `final_resume.md` stays, the tool's
message on stderr). 3 and 4 are not used.

`pipeline run` (the three stages as one graph, the only model calls are the
ones `match` makes): `--jd` runs match → draft and stops before the review, stderr ends
`app_id=<id> status=reviewing resume=<path>`; `--resume-edits FILE --app-id ID` accepts the file as
given and runs finalize, stderr ends `pages=N page_limit=L status=done`. 0 success; 1 input wrong
(unreadable file, invalid or existing or unknown id, nothing to submit, stale payload, bad config);
2 argparse (both / neither of `--jd` and `--resume-edits`, `--resume-edits` without `--app-id`);
3 index not ready / schema or vector-space mismatch; 4 the `parse_jd` or `coverage` path failed;
5 page limit exceeded (a finding: the application is `done`, all files kept, a second submit is
allowed); 6 the converter failed (re-run the same `--resume-edits` command to retry).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from urllib.error import URLError

from pydantic import ValidationError

from loomcv import resources
from loomcv.complete import (
    CompleteError,
    accept_all,
    apply_bullets,
    apply_jobs,
    checklist,
    load_workbook,
    select_keywords,
    sidecar_for,
    with_prefill,
    write_workbook,
)
from loomcv.coverage import StoredCoverageError
from loomcv.draft import ResumeTemplateError, StaleMatchError
from loomcv.finalize import export as finalize_export
from loomcv.ingest import IngestError, ResumeReadError, ingest_resume, resume_text
from loomcv.llm import CallRecord, build_generation_adapter, load_node_config
from loomcv.retrieve.rank import MatchPayload
from loomcv.retrieve.rerank import build_rerank_adapter
from loomcv.retrieve.search import (
    IndexNotReadyError,
    match,  # noqa: F401  (the steps call it now; `test_match_without_coverage_node_exit_4` patches this name)
)
from loomcv.state import JDProfile
from loomcv.steps import (
    Builders,
    InvalidInput,
    NodeCallError,
    Runtime,
    draft_step,
    finalize_step,
    match_step,
    parse_step,
)
from loomcv.sync.embedding import build_embedding_adapter, load_embedding_config
from loomcv.sync.index_assets import (
    SCHEMA_VERSION,
    SchemaVersionError,
    StoreInconsistentError,
    SyncReport,
    VaultParseError,
    VectorSpaceMismatchError,
    sync_vault,
)

# Repo-standard default paths; resolved relative to the CWD (run from repo root).
_DEFAULT_DB = "02_database/loomcv.sqlite"
_DEFAULT_CHROMA = "02_database/vector_cache"
# LangGraph's checkpoints live beside the index but never in it: `index_assets.py` owns
# `loomcv.sqlite`'s schema and its migrations drop tables.
_DEFAULT_CHECKPOINTS = "02_database/checkpoints.sqlite"
# The confirmed vault: written by `pipeline complete`, read by `pipeline sync` (one checkout =
# one user's data directory, like the two paths above).
_DEFAULT_VAULT = "02_database/vault.md"
# The working copy: what the app's profile page loads when no saved
# vault exists; the upload and the page's writes target it.
_DEFAULT_WORKING_COPY = "02_database/draft_vault.md"
# The working copy's sidecar: `{job_id: defaults_confirmed}` beside it.
_DEFAULT_SIDECAR = str(sidecar_for(Path(_DEFAULT_WORKING_COPY)))
# The prompt ships with the package, so its default is package-relative.
_DEFAULT_PROMPT = resources.path("prompts", "parse_jd.md")
_DEFAULT_STRUCTURE_PROMPT = resources.path("prompts", "ingest_structure.md")


# Output order is fixed: the eight counters first (from
# `report.counts()`), then `embedded` (an int attribute on the
# report, NOT a `counts()` key), then the two store paths.
_COUNTER_KEYS = (
    "added",
    "updated",
    "metadata_only",
    "deleted",
    "unchanged",
    "jobs_written",
    "jobs_deleted",
    "skills",
)


def _positive_int(value: str) -> int:
    """argparse type for `--page-limit`: a positive integer (anything else is a usage error)."""
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer")
    return number


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with a `sync` sub-parser (sub-parser layout)."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="LoomCV evidence pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sync_p = sub.add_parser("sync", help="Index the STAR vault into SQLite + Chroma.")
    sync_p.add_argument(
        "--vault",
        default=None,
        help="STAR vault Markdown path (v3 job grammar; default: the confirmed vault "
        f"{_DEFAULT_VAULT}, written by `pipeline complete`)",
    )
    sync_p.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"SQLite source-of-record path (default: {_DEFAULT_DB})",
    )
    sync_p.add_argument(
        "--chroma",
        default=_DEFAULT_CHROMA,
        help=f"Chroma persistent store directory (default: {_DEFAULT_CHROMA})",
    )
    sync_p.add_argument(
        "--config",
        default=None,
        help="Embedding config path (default: load_embedding_config's own default, "
        "config/models.yaml)",
    )
    sync_p.add_argument("--rebuild", action="store_true", help="Full re-embed from scratch.")
    sync_p.add_argument(
        "--include-draft",
        action="store_true",
        help="Index draft-status bullets too.",
    )
    sync_p.epilog = (
        "All paths default to the repo-standard locations and are resolved relative "
        "to the current working directory; run `pipeline sync` from the repo root."
    )

    match_p = sub.add_parser(
        "match",
        help="Rank indexed evidence against a JDProfile JSON or a raw JD (prints JSON).",
    )
    source = match_p.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", help="JDProfile JSON file path.")
    source.add_argument(
        "--jd", help="Raw job-description text file; parsed by the parse_jd node first."
    )
    match_p.add_argument(
        "--save-profile",
        default=None,
        help="Write the (post-validated) JDProfile JSON used for matching to this path; after "
        "a successful match it is rewritten with the coverage mapping, so it replays with "
        "zero generation calls.",
    )
    match_p.add_argument(
        "--prompt",
        default=str(_DEFAULT_PROMPT),
        help="parse_jd prompt template (default: the packaged prompt)",
    )
    match_p.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"SQLite source-of-record path (default: {_DEFAULT_DB})",
    )
    match_p.add_argument(
        "--chroma",
        default=_DEFAULT_CHROMA,
        help=f"Chroma persistent store directory (default: {_DEFAULT_CHROMA})",
    )
    match_p.add_argument(
        "--config",
        default=None,
        help="Model config path: its `embedding:` and `reranker:` sections, and its "
        "`providers:`/`nodes:` for `coverage` (unless the profile already carries one) and "
        "`parse_jd` (with --jd) (default: config/models.yaml)",
    )
    match_p.add_argument(
        "--retrieval-config",
        default=None,
        help="Retrieval config path (default: load_retrieval_config's own default, "
        "config/retrieval.yaml)",
    )

    draft_p = sub.add_parser(
        "draft",
        help="Render tailored_resume.md and match_report.md from a match payload (no model call).",
    )
    draft_p.add_argument(
        "match_json", metavar="MATCH_JSON", help="The JSON `pipeline match` printed."
    )
    draft_p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <output_dir>/<stem of MATCH_JSON>/, output_dir from "
        "config/generation.yaml)",
    )
    draft_p.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"SQLite source-of-record path (default: {_DEFAULT_DB})",
    )
    draft_p.add_argument(
        "--generation-config",
        default=None,
        help="Generation config path (default: config/generation.yaml)",
    )
    draft_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files (default: refuse, exit 1).",
    )

    finalize_p = sub.add_parser(
        "finalize",
        help="Strip the audit comments from a tailored résumé and export final_resume.md / .typ / "
        ".pdf with a page check (no model call).",
    )
    finalize_p.add_argument(
        "resume_md",
        metavar="RESUME_MD",
        help="The `tailored_resume.md` `pipeline draft` wrote, or the copy you edited.",
    )
    finalize_p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: the directory of RESUME_MD)",
    )
    finalize_p.add_argument(
        "--page-limit",
        type=_positive_int,
        default=None,
        help="Pages allowed for this run (default: page_limit from config/generation.yaml)",
    )
    finalize_p.add_argument(
        "--generation-config",
        default=None,
        help="Generation config path (default: config/generation.yaml)",
    )
    finalize_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files (default: refuse, exit 1).",
    )

    run_p = sub.add_parser(
        "run",
        help="Start an application (match → draft, stops before the review) or submit your "
        "edited résumé (accepted as given → finalize).",
    )
    mode = run_p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--jd", help="Start: the raw job-description text file; match and draft, then stop."
    )
    mode.add_argument(
        "--resume-edits",
        metavar="FILE",
        help="Submit: the reviewed résumé, accepted as given (passing it is your confirmation); "
        "needs --app-id. Allowed again on a finished application.",
    )
    run_p.add_argument(
        "--app-id",
        default=None,
        help="The application id (letters, digits, . _ -): with --jd the default is the stem of "
        "the JD file; a start never overwrites an existing id; required with --resume-edits.",
    )
    run_p.add_argument(
        "--out-dir",
        default=None,
        help="Applications root; each application lives in <out-dir>/<app-id>/ "
        "(default: output_dir from config/generation.yaml)",
    )
    run_p.add_argument(
        "--page-limit",
        type=_positive_int,
        default=None,
        help="Submit: pages allowed for this run (default: page_limit from config/generation.yaml)",
    )
    run_p.add_argument(
        "--checkpoints",
        default=_DEFAULT_CHECKPOINTS,
        help=f"LangGraph checkpoint database (default: {_DEFAULT_CHECKPOINTS}); "
        "kept apart from the index database",
    )
    run_p.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"Start: SQLite source-of-record path (default: {_DEFAULT_DB})",
    )
    run_p.add_argument(
        "--chroma",
        default=_DEFAULT_CHROMA,
        help=f"Start: Chroma persistent store directory (default: {_DEFAULT_CHROMA})",
    )
    run_p.add_argument(
        "--config",
        default=None,
        help="Start: model config path, as for `pipeline match` (default: config/models.yaml)",
    )
    run_p.add_argument(
        "--retrieval-config",
        default=None,
        help="Start: retrieval config path (default: config/retrieval.yaml)",
    )
    run_p.add_argument(
        "--generation-config",
        default=None,
        help="Generation config path (default: config/generation.yaml)",
    )

    ingest_p = sub.add_parser(
        "ingest",
        help="Turn an old résumé (PDF, Word, Markdown or plain text) into a v3 draft vault.",
    )
    ingest_p.add_argument("resume", help="Résumé file: PDF, Word, Markdown or plain text (UTF-8).")
    ingest_p.add_argument(
        "--out",
        default=None,
        help="Draft vault path to write (default: 02_database/ingest/<resume stem>_draft_vault.md, "
        "relative to the current working directory; 02_database/ is gitignored)",
    )
    ingest_p.add_argument(
        "--config",
        default=None,
        help="Model config path: its `providers:`/`nodes:` for `ingest` "
        "(default: config/models.yaml)",
    )
    ingest_p.add_argument(
        "--prompt-structure",
        default=str(_DEFAULT_STRUCTURE_PROMPT),
        help="Structure prompt template (default: the packaged prompt)",
    )

    complete_p = sub.add_parser(
        "complete",
        help="Headless pass over a draft vault: list what is open, or confirm everything as "
        "proposed (the UI, `pipeline ui`, is the way to work on a draft).",
        description="Reads a draft vault (from `pipeline ingest`); never edits it. Only the "
        "vault's own lines are carried into the output: a hand-edited draft's preamble prose "
        "and unknown sections are not.",
    )
    complete_p.add_argument("draft", help="Draft vault Markdown path (v3 job grammar).")
    mode = complete_p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--checklist",
        action="store_true",
        help="Print the open items as JSON lines (job fields, unconfirmed Defaults, draft "
        "bullets); write nothing.",
    )
    mode.add_argument(
        "--accept-all",
        action="store_true",
        help="Confirm every job's Defaults as they are, run the selection call, confirm every "
        "bullet with the proposed keywords, write the vault (for the acceptance run).",
    )
    complete_p.add_argument(
        "--out",
        default=_DEFAULT_VAULT,
        help=f"Confirmed vault path to write with --accept-all (default: {_DEFAULT_VAULT})",
    )
    complete_p.add_argument(
        "--config",
        default=None,
        help="Model config path: its `providers:`/`nodes:` for `ingest` (default: "
        "config/models.yaml)",
    )

    ui_p = sub.add_parser(
        "ui",
        help="Open the completion loop in a browser (needs the optional `ui` dependency group).",
    )
    ui_p.add_argument(
        "--draft",
        default=None,
        help="Draft vault to work on (default: the --out path, so a bare `pipeline ui` resumes "
        "work on the confirmed vault)",
    )
    ui_p.add_argument(
        "--out",
        default=_DEFAULT_VAULT,
        help=f"Vault path the UI writes (default: {_DEFAULT_VAULT})",
    )
    ui_p.add_argument(
        "--config",
        default=None,
        help="Model config path: its `providers:`/`nodes:` for `ingest` (default: "
        "config/models.yaml)",
    )

    review_p = sub.add_parser(
        "review",
        help="Open the application review page in a browser: start an application, edit the "
        "tailored résumé, export the PDF (needs the optional `ui` dependency group).",
    )
    review_p.add_argument(
        "--out-dir",
        default=None,
        help="Applications root the page lists and starts applications in "
        "(default: output_dir from config/generation.yaml)",
    )
    review_p.add_argument(
        "--checkpoints",
        default=_DEFAULT_CHECKPOINTS,
        help=f"LangGraph checkpoint database (default: {_DEFAULT_CHECKPOINTS}); "
        "kept apart from the index database",
    )
    review_p.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"SQLite source-of-record path (default: {_DEFAULT_DB})",
    )
    review_p.add_argument(
        "--chroma",
        default=_DEFAULT_CHROMA,
        help=f"Chroma persistent store directory (default: {_DEFAULT_CHROMA})",
    )
    review_p.add_argument(
        "--config",
        default=None,
        help="Model config path, as for `pipeline match` (default: config/models.yaml)",
    )
    review_p.add_argument(
        "--retrieval-config",
        default=None,
        help="Retrieval config path (default: config/retrieval.yaml)",
    )
    review_p.add_argument(
        "--generation-config",
        default=None,
        help="Generation config path (default: config/generation.yaml)",
    )

    # The union of `ui`'s and `review`'s flags minus `--draft` /
    # `--out` (the two profile files have fixed paths) — a copied list, not a parent parser, so
    # `review --help` stays byte-equal.
    app_p = sub.add_parser(
        "app",
        help="Open LoomCV in a browser: your profile, applying to a posting, your applications "
        "(needs the optional `ui` dependency group).",
    )
    app_p.add_argument(
        "--out-dir",
        default=None,
        help="Applications root the page lists and starts applications in "
        "(default: output_dir from config/generation.yaml)",
    )
    app_p.add_argument(
        "--checkpoints",
        default=_DEFAULT_CHECKPOINTS,
        help=f"LangGraph checkpoint database (default: {_DEFAULT_CHECKPOINTS}); "
        "kept apart from the index database",
    )
    app_p.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"SQLite source-of-record path (default: {_DEFAULT_DB})",
    )
    app_p.add_argument(
        "--chroma",
        default=_DEFAULT_CHROMA,
        help=f"Chroma persistent store directory (default: {_DEFAULT_CHROMA})",
    )
    app_p.add_argument(
        "--config",
        default=None,
        help="Model config path: providers/nodes for ingest, rewrite and match "
        "(default: config/models.yaml)",
    )
    app_p.add_argument(
        "--retrieval-config",
        default=None,
        help="Retrieval config path (default: config/retrieval.yaml)",
    )
    app_p.add_argument(
        "--generation-config",
        default=None,
        help="Generation config path (default: config/generation.yaml)",
    )
    return parser


def _run_ingest(args: argparse.Namespace) -> int:
    """Résumé → `ingest_resume` → the draft vault file + seven counters on stdout.

    Exit codes: 0 written and strict-clean; 1 unreadable résumé,
    unwritable output, no jobs extracted (nothing written) or strict-parser errors on the
    written draft (counters still printed); 4 anything on the `ingest` generation path.
    Warnings and the parser errors go to stderr.
    """
    resume = Path(args.resume)
    try:
        text = resume_text(resume)
    except ResumeReadError as exc:
        print(f"invalid resume {args.resume}: {exc}", file=sys.stderr)
        return 1
    adapter = None
    try:
        node_cfg, provider_cfg = load_node_config(
            "ingest", Path(args.config) if args.config else None
        )
        adapter = build_generation_adapter(node_cfg, provider_cfg)
        result = ingest_resume(
            text,
            adapter=adapter,
            structure_template=Path(args.prompt_structure),
        )
    except (KeyError, ValueError, RuntimeError, NotImplementedError, URLError, IngestError) as exc:
        print(f"ingest: {exc}", file=sys.stderr)
        return 4
    finally:
        _print_call_records("ingest", adapter)
    for warning in result.warnings:
        print(warning, file=sys.stderr)
    if not result.draft.jobs:
        print("ingest: no jobs extracted", file=sys.stderr)
        return 1

    out = (
        Path(args.out)
        if args.out
        else Path("02_database") / "ingest" / f"{resume.stem}_draft_vault.md"
    )
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.rendered, encoding="utf-8")
    except OSError as exc:
        print(f"ingest: cannot write {out}: {exc}", file=sys.stderr)
        return 1
    errors = [i for i in result.parse_issues if i.severity == "error"]
    for issue in errors:
        print(f"ingest: line {issue.line_no}: {issue.message}", file=sys.stderr)

    bullets = [b for job in result.draft.jobs for b in job.bullets]
    # `dropped` = the warnings that report a removal: bullets, jobs, `;` keywords.
    print(f"jobs={len(result.draft.jobs)}")
    print(f"bullets={len(bullets)}")
    print(f"jobs_without_bullets={sum(1 for j in result.draft.jobs if not j.bullets)}")
    print(f"dropped={sum(1 for w in result.warnings if w.startswith('dropped '))}")
    print(
        "jobs_missing_dates="
        f"{sum(1 for j in result.draft.jobs if '0000' in (j.date_from, j.date_to))}"
    )
    print(f"parser_warnings={sum(1 for i in result.parse_issues if i.severity == 'warning')}")
    print(f"written={out}")
    return 1 if errors else 0


def _run_complete(args: argparse.Namespace) -> int:
    """Draft vault → `--checklist` (JSON lines, nothing written) or `--accept-all` (confirm every
    job's `Defaults` as they are, select keywords, confirm every bullet, write the vault, run
    the strict parser twice). The header lines are
    kept as the draft has them and the Skills pre-fill (the draft's skills ∪ the confirmed
    `Defaults`) joins the unnamed group; the grouping call is never made here.

    Exit codes: 0 success; 1 draft unreadable or with parser errors (nothing written, either
    mode), output unwritable, or strict-parser problems on the written vault (the file stays);
    4 anything on the `ingest` generation path. The draft file is never opened for writing.
    """
    draft = Path(args.draft)
    try:
        text = draft.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"invalid draft {args.draft}: {exc}", file=sys.stderr)
        return 1
    try:
        wb = load_workbook(text)
    except CompleteError as exc:
        print(f"complete: {exc}", file=sys.stderr)
        return 1

    if args.checklist:
        for item in checklist(wb):
            print(json.dumps(item.model_dump(), ensure_ascii=False))
        return 0

    adapter = None
    try:
        node_cfg, provider_cfg = load_node_config(
            "ingest", Path(args.config) if args.config else None
        )
        adapter = build_generation_adapter(node_cfg, provider_cfg)
        wb = apply_jobs(wb, accept_all(wb)[0])
        # The three-source pre-fill as the unnamed group; no grouping call.
        wb = with_prefill(wb)
        wb, warnings = select_keywords(wb, adapter=adapter)
    except (
        KeyError,
        ValueError,
        RuntimeError,
        NotImplementedError,
        URLError,
        OSError,
        CompleteError,
    ) as exc:
        print(f"complete: {exc}", file=sys.stderr)
        return 4
    finally:
        _print_call_records("ingest", adapter)
    for warning in warnings:
        print(warning, file=sys.stderr)
    wb = apply_bullets(wb, accept_all(wb)[1])

    out = Path(args.out)
    try:
        report = write_workbook(wb, out, today=date.today())
    except OSError as exc:
        print(f"complete: cannot write {out}: {exc}", file=sys.stderr)
        return 1
    for problem in report.problems:
        print(f"complete: {problem}", file=sys.stderr)
    print(f"jobs={report.jobs}")
    print(f"jobs_without_bullets={report.jobs_without_bullets}")
    print(f"bullets={report.bullets}")
    print(f"confirmed={report.confirmed}")
    print(f"skills_unassigned={report.skills_unassigned}")
    print(f"written={out}")
    return 1 if report.problems else 0


def _run_ui(args: argparse.Namespace) -> int:
    """Launch the Gradio front end over `loomcv.complete`. `gradio` is an
    optional dependency (the `ui` group) imported only inside `loomcv/ui/`, and `loomcv.ui` only
    inside this branch. Exit 1: the draft is missing, or the `ui` group is not installed."""
    out = Path(args.out)
    draft = Path(args.draft) if args.draft else out
    if not draft.is_file():
        print(f"ui: no draft vault at {draft}", file=sys.stderr)
        return 1
    try:
        from loomcv.ui.app import launch
    except ImportError:
        print("pipeline ui needs the optional ui group: uv sync --group ui", file=sys.stderr)
        return 1
    from loomcv.steps import SaveReport

    launch(
        draft,
        out,
        working_copy=draft,
        sidecar=sidecar_for(draft),
        config=Path(args.config) if args.config else None,
        # Save profile in the dev entry writes `--out` only — no index from it.
        save_profile=lambda wb, *, today: SaveReport(written=write_workbook(wb, out, today=today)),
    )
    return 0


def _run_review(args: argparse.Namespace) -> int:
    """Launch the Gradio review page over the application boundary. `gradio`
    is imported only inside `loomcv/ui/`, and `loomcv.ui.review` only inside this branch. Every
    per-application choice (id, JD, page limit) is made on the page. Exit 1: the `ui`
    group is not installed, or the generation config is invalid."""
    try:
        from loomcv.ui.review import launch
    except ImportError:
        print("pipeline review needs the optional ui group: uv sync --group ui", file=sys.stderr)
        return 1
    runtime = _runtime(args)
    try:
        generation = runtime.generation()
    except InvalidInput as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out_root = Path(args.out_dir) if args.out_dir else Path(generation.output_dir)
    launch(
        runtime,
        out_root=out_root,
        checkpoints=Path(args.checkpoints),
        page_limit_default=generation.page_limit,
    )
    return 0


def _run_app(args: argparse.Namespace) -> int:
    """`pipeline app`: the three-tab frame over the profile page and the review
    page. `gradio` is imported only inside `loomcv/ui/`, and `loomcv.ui.shell` only inside this
    branch. No refusal on a missing vault (*empty* is a state). Exit 1: the `ui` group is not
    installed, or the generation config is invalid."""
    try:
        from loomcv.ui.shell import launch
    except ImportError:
        print("pipeline app needs the optional ui group: uv sync --group ui", file=sys.stderr)
        return 1
    runtime = _runtime(args)
    try:
        generation = runtime.generation()
    except InvalidInput as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out_root = Path(args.out_dir) if args.out_dir else Path(generation.output_dir)
    launch(
        runtime,
        working_copy=Path(_DEFAULT_WORKING_COPY),
        saved=Path(_DEFAULT_VAULT),
        sidecar=Path(_DEFAULT_SIDECAR),
        out_root=out_root,
        checkpoints=Path(args.checkpoints),
        page_limit_default=generation.page_limit,
        config=Path(args.config) if args.config else None,
    )
    return 0


def _run_sync(args: argparse.Namespace) -> int:
    """Construct the adapter from config, run one sync, translate errors to exit codes.

    Adapter construction lives here, not inside `sync_vault(adapter=None)`:
    `load_embedding_config` → `build_embedding_adapter` → `sync_vault(adapter=adapter)`.
    Tests monkeypatch `loomcv.cli.build_embedding_adapter`, so no test loads sentence-transformers
    or hits the network.

    `--vault` defaults to the confirmed vault; with the default and no
    file there, the command refuses (exit 1) before any store or adapter is touched.
    """
    vault = Path(args.vault) if args.vault else Path(_DEFAULT_VAULT)
    if args.vault is None and not vault.is_file():
        print(
            f"sync: no vault at {_DEFAULT_VAULT} — run `pipeline complete` first or pass --vault",
            file=sys.stderr,
        )
        return 1
    config = load_embedding_config(args.config)
    adapter = build_embedding_adapter(config)
    try:
        report = sync_vault(
            vault,
            db_path=Path(args.db),
            chroma_path=Path(args.chroma),
            adapter=adapter,
            include_draft=args.include_draft,
            rebuild=args.rebuild,
        )
    except VaultParseError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (VectorSpaceMismatchError, StoreInconsistentError, SchemaVersionError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if report.schema_rebuild_from is not None:
        print(
            f"schema {report.schema_rebuild_from} → {SCHEMA_VERSION}: full rebuild"
            " (every bullet_id changed)",
            file=sys.stderr,
        )
    _print_report(report, args)
    return 0


def _print_report(report: SyncReport, args: argparse.Namespace) -> None:
    """Emit the fixed-order key=value output to stdout."""
    counts = report.counts()
    for key in _COUNTER_KEYS:
        print(f"{key}={counts[key]}")
    print(f"embedded={report.embedded}")
    print(f"db={args.db}")
    print(f"chroma={args.chroma}")


def _runtime(args: argparse.Namespace) -> Runtime:
    """The steps' `Runtime` from a subcommand's flags. The builders are this module's own names,
    looked up at call time, so a test that patches `loomcv.cli.build_*` reaches the steps; a flag
    the subcommand does not define falls back to its default."""
    config = getattr(args, "config", None)
    retrieval_config = getattr(args, "retrieval_config", None)
    generation_config = getattr(args, "generation_config", None)
    return Runtime(
        db=Path(getattr(args, "db", _DEFAULT_DB)),
        chroma=Path(getattr(args, "chroma", _DEFAULT_CHROMA)),
        prompt=Path(getattr(args, "prompt", _DEFAULT_PROMPT)),
        models_config=Path(config) if config else None,
        retrieval_config_path=Path(retrieval_config) if retrieval_config else None,
        generation_config_path=Path(generation_config) if generation_config else None,
        builders=Builders(
            generation=build_generation_adapter,
            embedding=build_embedding_adapter,
            rerank=build_rerank_adapter,
        ),
    )


def _parse_jd_profile(
    args: argparse.Namespace, runtime: Runtime
) -> tuple[JDProfile, list[str]] | int:
    """`--jd` path: JD file → `parse_jd` node → post-validated profile.

    Returns the profile and its warnings, or an exit code: 1 unreadable JD file;
    3 store not ready (the index is needed for the coverage keyword list, so it is
    checked before the parse call is spent); 4 anything on the generation path
    (config, adapter, transport, unparseable model output).
    """
    try:
        jd_text = Path(args.jd).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"invalid jd {args.jd}: {exc}", file=sys.stderr)
        return 1
    try:
        parsed = parse_step(runtime, jd_text)
    except (IndexNotReadyError, SchemaVersionError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except NodeCallError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    finally:
        _print_call_records("parse_jd", runtime.adapters.get("parse_jd"))
    for warning in parsed.warnings:
        print(warning, file=sys.stderr)
    return parsed.profile, parsed.warnings


def _print_call_records(node: str, adapter: object) -> None:
    """One stderr line per generation call the adapter recorded.

    Only `CallRecord` items count: an adapter without `calls`, or whose `calls` holds
    anything else (the test fakes keep argument tuples there), prints nothing."""
    records = [r for r in getattr(adapter, "calls", []) if isinstance(r, CallRecord)]

    def show(value: object) -> str:
        return "n/a" if value is None else str(value)

    for n, record in enumerate(records, start=1):
        print(
            f"{node} call {n}: {record.elapsed_s:.1f} s · prompt {show(record.prompt_tokens)}"
            f" · completion {show(record.completion_tokens)}"
            f" · reasoning {show(record.reasoning_chars)} chars",
            file=sys.stderr,
        )


def _run_match(args: argparse.Namespace) -> int:
    """Load or parse the profile, rank the index against it, print one JSON document.

    Exit codes: 0 success, gaps
    included; 1 unreadable or invalid profile / JD file, or a stored coverage that does not
    fit this run's lines or index (the claims call shares this exit); 3 store not ready /
    vector-space or schema mismatch; 4 the `parse_jd` or the `coverage` generation path failed
    (the claims call shares the `coverage` path).
    """
    runtime = _runtime(args)
    if args.jd is not None:
        parsed = _parse_jd_profile(args, runtime)
        if isinstance(parsed, int):
            return parsed
        profile, parse_warnings = parsed
    else:
        try:
            profile = JDProfile.model_validate_json(Path(args.profile).read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            print(f"invalid profile {args.profile}: {exc}", file=sys.stderr)
            return 1
        parse_warnings = []
    try:
        step = match_step(
            runtime,
            profile=profile,
            parse_warnings=parse_warnings,
            save_profile=Path(args.save_profile) if args.save_profile else None,
        )
    except (IndexNotReadyError, VectorSpaceMismatchError, SchemaVersionError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except StoredCoverageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except NodeCallError as exc:
        # An unbuildable node has no adapter (no records); a failed call shows what it spent.
        _print_call_records("coverage", runtime.adapters.get("coverage"))
        print(str(exc), file=sys.stderr)
        return 4
    _print_call_records("coverage", runtime.adapters.get("coverage"))
    print(json.dumps(step.payload, indent=2, ensure_ascii=False))
    return 0


def _run_draft(args: argparse.Namespace) -> int:
    """`pipeline draft`: nothing is written before every check has passed."""
    runtime = _runtime(args)
    try:
        generation = runtime.generation()
    except InvalidInput as exc:
        print(str(exc), file=sys.stderr)
        return 1
    match_path = Path(args.match_json)
    try:
        payload = MatchPayload.model_validate_json(match_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        print(f"invalid match payload {match_path}: {exc}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else Path(generation.output_dir) / match_path.stem
    try:
        drafted = draft_step(runtime, payload, out_dir=out_dir, force=args.force)
    except (IndexNotReadyError, SchemaVersionError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (StaleMatchError, ResumeTemplateError, InvalidInput) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"jobs={drafted.jobs} bullets={drafted.bullets} skills={drafted.skills}",
        file=sys.stderr,
    )
    return 0


def _run_finalize(args: argparse.Namespace) -> int:
    """`pipeline finalize`: nothing is written before every input check has passed."""
    try:
        finalized = finalize_step(
            _runtime(args),
            Path(args.resume_md),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            page_limit=args.page_limit,
            force=args.force,
        )
    except InvalidInput as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (finalize_export.PandocFailed, finalize_export.TypstFailed) as exc:
        print(str(exc), file=sys.stderr)
        return 6
    for message in finalized.warnings:
        print(f"typst warning: {message}", file=sys.stderr)
    print(f"pages={finalized.pages} page_limit={finalized.page_limit}", file=sys.stderr)
    return 5 if finalized.pages > finalized.page_limit else 0


def _run_run(args: argparse.Namespace) -> int:
    """`pipeline run`: `--jd` starts an application and stops before the review,
    `--resume-edits` submits the reviewed file and finishes it. `loomcv.application` (and with it
    LangGraph) is imported here, so the other subcommands never pay for it."""
    from loomcv import application

    runtime = _runtime(args)
    try:
        out_root = Path(args.out_dir) if args.out_dir else Path(runtime.generation().output_dir)
        checkpoints = Path(args.checkpoints)
        if args.jd is not None:
            state = application.start(
                runtime,
                Path(args.jd),
                app_id=args.app_id,
                out_root=out_root,
                checkpoints=checkpoints,
            )
        else:
            state = application.submit(
                runtime,
                args.app_id,
                edited_path=Path(args.resume_edits),
                page_limit=args.page_limit,
                out_root=out_root,
                checkpoints=checkpoints,
            )
    except (IndexNotReadyError, VectorSpaceMismatchError, SchemaVersionError) as exc:
        failure = (3, str(exc))
    except (InvalidInput, StaleMatchError, ResumeTemplateError, StoredCoverageError) as exc:
        failure = (1, str(exc))
    except NodeCallError as exc:
        failure = (4, str(exc))
    except (finalize_export.PandocFailed, finalize_export.TypstFailed) as exc:
        failure = (6, str(exc))
    else:
        failure = None
    # The generation nodes' call records, whatever the outcome (as `pipeline match` prints them).
    _print_call_records("parse_jd", runtime.adapters.get("parse_jd"))
    _print_call_records("coverage", runtime.adapters.get("coverage"))
    if failure is not None:
        code, message = failure
        print(message, file=sys.stderr)
        return code
    if args.jd is not None:
        counts = state.counts
        print(
            f"jobs={counts.get('jobs')} bullets={counts.get('bullets')} "
            f"skills={counts.get('skills')}",
            file=sys.stderr,
        )
        print(
            f"app_id={state.app_id} status={state.status} resume={state.resume_path}",
            file=sys.stderr,
        )
        return 0
    for message in state.typst_warnings:
        print(f"typst warning: {message}", file=sys.stderr)
    print(
        f"pages={state.pages} page_limit={state.page_limit} status={state.status}",
        file=sys.stderr,
    )
    return 5 if state.over_limit else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.resume_edits is not None and args.app_id is None:
            parser.error("--resume-edits needs --app-id")
        return _run_run(args)
    if args.command == "draft":
        return _run_draft(args)
    if args.command == "finalize":
        return _run_finalize(args)
    if args.command == "ingest":
        return _run_ingest(args)
    if args.command == "complete":
        return _run_complete(args)
    if args.command == "ui":
        return _run_ui(args)
    if args.command == "review":
        return _run_review(args)
    if args.command == "app":
        return _run_app(args)
    if args.command == "match":
        return _run_match(args)
    return _run_sync(args)


if __name__ == "__main__":
    raise SystemExit(main())
