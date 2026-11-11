"""Export of the tailored résumé: strip the audit comments, Markdown -> typst -> PDF, count
the pages.

No model call, no index, no graph: the third-party imports are `pypandoc` (only for
the path of the pandoc binary it bundles), `typst` and `pypdf`, and nothing from the other
LoomCV packages. All three converters are dependencies, so nothing is looked up on `PATH`.
No layout value lives here: they are in the pandoc defaults file the caller names.
The module is print-free; the caller relays the warnings `export_pdf` returns.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pypandoc
import pypdf
import typst

FINAL_MD = "final_resume.md"
FINAL_TYP = "final_resume.typ"
FINAL_PDF = "final_resume.pdf"

# The comment and the single space before it, when present; nothing line-aware.
_SRC_COMMENT = re.compile(r"[ ]?<!--\s*src:[^>]*-->")

_ROSETTA = "softwareupdate --install-rosetta --agree-to-license"


class PandocFailed(RuntimeError):
    """pandoc returned non-zero, or the bundled binary could not start; the message is the
    tool's own."""


class TypstFailed(RuntimeError):
    """typst could not compile the `.typ`; the message is typst's error text."""


def strip_src(text: str) -> str:
    """Every `<!-- src: … -->` comment and the one space before it removed; every other byte
    kept (the user's edits included)."""
    return _SRC_COMMENT.sub("", text)


def export_typ(md_path: Path, typ_path: Path, defaults_path: Path) -> None:
    """`<bundled pandoc> md -o typ --defaults defaults`; non-zero exit -> `PandocFailed(stderr)`."""
    try:
        pandoc = pypandoc.get_pandoc_path()
        proc = subprocess.run(
            [pandoc, str(md_path), "-o", str(typ_path), "--defaults", str(defaults_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        # On Apple silicon the bundled pandoc is x86_64 and needs Rosetta 2.
        raise PandocFailed(
            f"the bundled pandoc cannot run: {exc}. On Apple silicon, install Rosetta 2 once "
            f"with `{_ROSETTA}`."
        ) from exc
    if proc.returncode != 0:
        raise PandocFailed(proc.stderr.strip() or f"pandoc exited {proc.returncode}")


def export_pdf(typ_path: Path, pdf_path: Path) -> list[str]:
    """`typst.compile_with_warnings`; returns each `TypstWarning`'s message (the caller prints
    them). A compile error -> `TypstFailed`."""
    try:
        _, warnings = typst.compile_with_warnings(str(typ_path), output=str(pdf_path))
    except typst.TypstError as exc:
        raise TypstFailed(str(exc)) from exc
    return [w.message for w in warnings]


def count_pages(pdf_path: Path) -> int:
    """`len(pypdf.PdfReader(pdf_path).pages)`."""
    return len(pypdf.PdfReader(str(pdf_path)).pages)
