"""The reader (`loomcv.ingest.read.resume_text`), file -> résumé text.

Every file is generated in `tmp_path` at test time (typst for the PDFs, python-docx for the Word
files); no binary fixture is committed and no model is called. The refusal tests assert the class
and a non-empty message, never the words. `loomcv.ingest.read` is imported
**inside** each test, so the red run is N failures rather than one collection error.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

THREE_LINES = "First line\n\nSecond line\n\n#pagebreak()\n\nThird line\n"


def _pdf(tmp_path: Path, source: str, name: str = "resume.pdf") -> Path:
    import typst

    src = tmp_path / "source.typ"
    src.write_text(source, encoding="utf-8")
    out = tmp_path / name
    typst.compile(str(src), output=str(out))
    return out


def _docx_with_header_and_footer(tmp_path: Path, name: str = "resume.docx") -> Path:
    import docx

    document = docx.Document()
    section = document.sections[0]
    section.header.paragraphs[0].text = "Header line"
    document.add_paragraph("Body one")
    document.add_paragraph("Body two")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "cell1"
    table.rows[0].cells[1].text = "cell2"
    section.footer.paragraphs[0].text = "Footer line"
    path = tmp_path / name
    document.save(str(path))
    return path


@pytest.mark.parametrize("suffix", [".md", ".markdown", ".txt"])
def test_text_files_come_back_as_utf8_text(tmp_path, suffix):
    # R-1
    from loomcv.ingest.read import resume_text

    path = tmp_path / f"resume{suffix}"
    path.write_text("# Zoë Müller\n\nBuilt café 系统 — 40% faster.\n", encoding="utf-8")

    assert resume_text(path) == path.read_text(encoding="utf-8")


def test_pdf_pages_come_back_in_order_with_a_blank_line_between_pages(tmp_path):
    # R-2
    from loomcv.ingest.read import resume_text

    text = resume_text(_pdf(tmp_path, THREE_LINES))

    assert text.index("First line") < text.index("Second line") < text.index("Third line")
    assert "Second line\n\nThird line" in text


@pytest.mark.parametrize(
    "source",
    ["", "#rect(width: 2cm, height: 2cm, fill: black)\n"],
    ids=["empty", "image-only"],
)
def test_pdf_without_text_is_refused(tmp_path, source):
    # R-3
    from loomcv.ingest.read import ResumeReadError, resume_text

    with pytest.raises(ResumeReadError) as exc:
        resume_text(_pdf(tmp_path, source))
    assert str(exc.value)


def test_docx_header_body_table_row_and_footer_in_order(tmp_path):
    # R-4
    from loomcv.ingest.read import resume_text

    text = resume_text(_docx_with_header_and_footer(tmp_path))
    lines = text.split("\n")

    assert lines[0] == "Header line"
    assert lines[-1] == "Footer line"
    assert "cell1 | cell2" in lines
    assert lines.index("Body one") < lines.index("Body two") < lines.index("cell1 | cell2")
    assert "\n\n\n" not in text


def _docx_with_merged_cell_and_two_sections(tmp_path: Path) -> Path:
    import docx

    document = docx.Document()
    first = document.sections[0]
    first.header.paragraphs[0].text = "Header line"
    first.footer.paragraphs[0].text = "Footer line"
    table = document.add_table(rows=1, cols=3)
    cells = table.rows[0].cells
    cells[0].merge(cells[1]).text = "merged"
    cells[2].text = "tail"
    document.add_section()
    document.add_paragraph("Second section body")
    assert document.sections[1].header.is_linked_to_previous
    path = tmp_path / "resume.docx"
    document.save(str(path))
    return path


def test_docx_merged_cell_and_linked_header_are_not_repeated(tmp_path):
    # R-4b
    from loomcv.ingest.read import resume_text

    text = resume_text(_docx_with_merged_cell_and_two_sections(tmp_path))

    assert "merged | tail" in text.split("\n")
    assert text.count("merged") == 1
    assert text.count("Header line") == 1
    assert text.count("Footer line") == 1
    assert "Second section body" in text


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_suffix_is_matched_case_insensitively(tmp_path, kind):
    # R-5
    from loomcv.ingest.read import resume_text

    if kind == "pdf":
        path = _pdf(tmp_path, THREE_LINES).rename(tmp_path / "RESUME.PDF")
        known = "Second line"
    else:
        path = _docx_with_header_and_footer(tmp_path).rename(tmp_path / "Resume.Docx")
        known = "Body one"

    assert known in resume_text(path)


@pytest.mark.parametrize("name", ["resume.rtf", "resume.doc", "resume"])
def test_unsupported_suffix_is_refused(tmp_path, name):
    # R-6
    from loomcv.ingest.read import ResumeReadError, resume_text

    path = tmp_path / name
    path.write_text("Some text\n", encoding="utf-8")

    with pytest.raises(ResumeReadError) as exc:
        resume_text(path)
    assert str(exc.value)


@pytest.mark.parametrize("kind", ["latin1_md", "junk_pdf", "junk_docx", "missing"])
def test_unreadable_files_are_refused_with_one_exception_type(tmp_path, kind):
    # R-7 (the junk .pdf / .docx and the missing file are refused like any unreadable file)
    from loomcv.ingest.read import ResumeReadError, resume_text

    if kind == "latin1_md":
        path = tmp_path / "resume.md"
        path.write_bytes("caf\xe9 r\xe9sum\xe9\n".encode("latin-1"))
    elif kind == "junk_pdf":
        path = tmp_path / "resume.pdf"
        path.write_bytes(b"not a pdf at all")
    elif kind == "junk_docx":
        path = tmp_path / "resume.docx"
        path.write_bytes(b"not a zip at all")
    else:
        path = tmp_path / "no_such_resume.md"

    with pytest.raises(ResumeReadError) as exc:
        resume_text(path)
    assert str(exc.value)


def test_reader_imports_without_gradio():
    # R-8 — a fresh interpreter with gradio blocked (the shape of
    # `test_complete_and_cli_import_without_gradio`)
    code = (
        "import sys\n"
        "sys.modules['gradio'] = None\n"
        "import loomcv.ingest.read\n"
        "assert 'gradio' not in [m for m, v in sys.modules.items() if v is not None]\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
