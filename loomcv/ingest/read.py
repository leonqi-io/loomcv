"""File → résumé text.

`resume_text(path)` is the one function between an uploaded file and `ingest_resume`: it
dispatches on the casefolded suffix (`.md` / `.markdown` / `.txt` as UTF-8, `.pdf` through
`pypdf`, `.docx` through `python-docx`) and raises `ResumeReadError` — one class, the user's
sentence as its message — for anything it cannot turn into text. No layout analysis, no model
call, no UI import.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import docx
import pypdf
from docx.opc.exceptions import PackageNotFoundError
from docx.text.paragraph import Paragraph
from pypdf.errors import DependencyError, PyPdfError

_ACCEPTED = "Upload a PDF, Word, Markdown or plain-text file."
_NOT_SUPPORTED = f"This file type is not supported. {_ACCEPTED}"
_UNREADABLE = f"This file could not be read. {_ACCEPTED}"
_NO_TEXT_LAYER = (
    "This PDF has no text layer (is it a scan?). "
    "Upload a text PDF, a Word file, Markdown or plain text."
)

_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


class ResumeReadError(Exception):
    """A file that cannot become résumé text. The message is the user's sentence."""


def resume_text(path: Path) -> str:
    """The file's text, by casefolded suffix. Anything else, and anything unreadable, raises
    `ResumeReadError`."""
    suffix = path.suffix.casefold()
    if suffix in _TEXT_SUFFIXES:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ResumeReadError(_UNREADABLE) from exc
    if suffix == ".pdf":
        return _pdf_text(path)
    if suffix == ".docx":
        return _docx_text(path)
    raise ResumeReadError(_NOT_SUPPORTED)


def _pdf_text(path: Path) -> str:
    """Every page's text, pages joined by a blank line; no text at all → refused (a scan)."""
    try:
        reader = pypdf.PdfReader(path)
        text = "\n\n".join(page.extract_text() for page in reader.pages)
    except (PyPdfError, DependencyError, OSError, ValueError) as exc:
        raise ResumeReadError(_UNREADABLE) from exc
    if not text.strip():
        raise ResumeReadError(_NO_TEXT_LAYER)
    return text


def _docx_text(path: Path) -> str:
    """Headers, then the body in document order, then footers; blank lines collapse to one.

    A table row is one line (`_row_line`). A section whose header or footer is linked to the
    previous one is skipped: python-docx would repeat the previous section's paragraphs."""
    try:
        document = docx.Document(str(path))
        lines: list[str] = []
        for section in document.sections:
            if not section.header.is_linked_to_previous:
                lines.extend(p.text for p in section.header.paragraphs)
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                lines.append(block.text)
            else:
                lines.extend(_row_line(row) for row in block.rows)
        for section in document.sections:
            if not section.footer.is_linked_to_previous:
                lines.extend(p.text for p in section.footer.paragraphs)
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise ResumeReadError(_UNREADABLE) from exc
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _row_line(row) -> str:
    """A table row as one line: its cells joined by ` | `, a merged cell once, empties dropped."""
    seen: list = []
    parts: list[str] = []
    for cell in row.cells:
        if any(cell._tc is tc for tc in seen):
            continue
        seen.append(cell._tc)
        value = " ".join(cell.text.split())
        if value:
            parts.append(value)
    return " | ".join(parts)
