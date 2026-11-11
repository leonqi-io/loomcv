"""Tests for the export behind `pipeline finalize`.

Cases: the six export cases and the import check I-1. Everything is synthetic. pandoc
(`pypandoc-binary`) and typst (`typst`) are project dependencies, so the last three of them run
the real bundled binaries on a three-line Markdown — no `skipif`, no stub binary. No network,
no model call.

Project imports are inside each test so every case fails on its own at red instead of one
collection error.
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import yaml

from loomcv import resources

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = resources.path("pandoc_resume.yaml")

THREE_LINES = (
    "## Skills\n\nGo, SQL\n\n## Experience\n\n### Role — Org\n\n- Did a fictional thing.\n"
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("network call in test")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


def test_strip_src_removes_comment_and_one_space_only():
    from loomcv.finalize import strip_src

    text = (
        "- **X:** t <!-- src: a::b -->\n"
        "- t<!-- src: a::b -->\n"
        "- plain line\n"
        "- t <!--  src:  a::b  -->  trailing\n"
        "- **Label <!-- src: a::c -->:** bold\n"
        "- A sentence the user rewrote, in no vault anywhere. <!-- src: j0::zz -->\n"
    )
    expected = (
        "- **X:** t\n"
        "- t\n"
        "- plain line\n"
        "- t  trailing\n"
        "- **Label:** bold\n"
        "- A sentence the user rewrote, in no vault anywhere.\n"
    )

    result = strip_src(text)

    assert result == expected
    assert result == re.sub(r"[ ]?<!--\s*src:[^>]*-->", "", text)
    assert "src:" not in result


def test_strip_src_no_comment_is_identity():
    from loomcv.finalize import strip_src

    text = (
        "Jane Example  \r\n"
        "## Skills\n\nGo, SQL  \n\n"
        "- kept <!-- note --> as is\n"
        "### Role — Org\nJan 2020 – Dec 2021\n"
    )

    assert strip_src(text) == text


def test_count_pages_one_and_three(tmp_path):
    from pypdf import PdfWriter

    from loomcv.finalize import count_pages

    for pages in (1, 3):
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=612, height=792)
        path = tmp_path / f"{pages}.pdf"
        with path.open("wb") as handle:
            writer.write(handle)

        assert count_pages(path) == pages


def test_export_typ_bundled_pandoc_carries_defaults(tmp_path):
    from loomcv.finalize import export_typ

    md = tmp_path / "final_resume.md"
    md.write_text(THREE_LINES, encoding="utf-8")
    typ = tmp_path / "final_resume.typ"

    export_typ(md, typ, DEFAULTS)

    assert typ.is_file()
    source = typ.read_text(encoding="utf-8")
    # The custom template is the layout, not pandoc's
    # default (no `conf(`); the defaults file's variables land where the template places them.
    assert source.startswith("// LoomCV résumé layout")
    assert "#let conf(" not in source
    assert "#show: doc => conf(" not in source
    variables = yaml.safe_load(DEFAULTS.read_text(encoding="utf-8"))["variables"]
    assert f"size: {variables['fontsize']}" in source
    assert f'paper: "{variables["papersize"]}"' in source
    assert f"x: {variables['margin']['x']}" in source
    assert "columns" not in source


def test_export_failures_carry_the_tool_message(tmp_path):
    from loomcv.finalize import PandocFailed, TypstFailed, export_pdf, export_typ

    md = tmp_path / "final_resume.md"
    md.write_text(THREE_LINES, encoding="utf-8")
    bad_defaults = tmp_path / "bad.yaml"
    bad_defaults.write_text("to: nosuchformat\nstandalone: true\n", encoding="utf-8")
    typ = tmp_path / "final_resume.typ"

    with pytest.raises(PandocFailed) as pandoc_exc:
        export_typ(md, typ, bad_defaults)
    assert "nosuchformat" in str(pandoc_exc.value)
    assert not typ.exists()

    broken = tmp_path / "broken.typ"
    broken.write_text("#let x = (\n", encoding="utf-8")
    pdf = tmp_path / "broken.pdf"

    with pytest.raises(TypstFailed) as typst_exc:
        export_pdf(broken, pdf)
    assert "unclosed delimiter" in str(typst_exc.value)
    assert not pdf.exists()


def test_real_chain_one_page_text_readable(tmp_path):
    from pypdf import PdfReader

    from loomcv.finalize import count_pages, export_pdf, export_typ

    md = tmp_path / "final_resume.md"
    md.write_text(THREE_LINES, encoding="utf-8")
    typ, pdf = tmp_path / "final_resume.typ", tmp_path / "final_resume.pdf"

    export_typ(md, typ, DEFAULTS)
    warnings = export_pdf(typ, pdf)

    assert isinstance(warnings, list)
    assert count_pages(pdf) == 1
    text = PdfReader(str(pdf)).pages[0].extract_text()
    for word in ("Skills", "Experience", "Role"):
        assert word in text


def test_import_finalize_loads_no_engine_package():
    # I-1 — fresh interpreter, so modules other tests imported cannot leak in.
    code = (
        "import sys\n"
        "import loomcv.finalize\n"
        "loaded = [m for m in sys.modules if m.split('.')[:2] in "
        "(['loomcv', 'llm'], ['loomcv', 'retrieve'], ['loomcv', 'sync'], ['loomcv', 'draft'])]\n"
        "assert not loaded, loaded\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# The default template's five sections as `pipeline finalize` receives them (src stripped).
FIVE_SECTIONS = (
    "# Mara Voss\n\n"
    "Lisbon, PT | mara.voss@example.invalid\n\n"
    "**Senior Web Platform Engineer**\n\n"
    "## Summary\n\nEleven years building fictional web platforms.\n\n"
    "## Skills\n\n"
    "**Languages, Frameworks, and Data:** Python; TypeScript\\\n"
    "Go; SQL\n\n"
    "## Experience\n\n"
    "### Staff Engineer — Acme\n*Mar 2021 – Present*\n\n"
    "- **Helios:** Did a fictional thing.\n\n"
    "Engineer — Old Co | Jan 2010 – Dec 2012\n\n"
    "## Education\n\n"
    "Fictional University — BSc Computer Science, 2013\\\n"
    "Pretend Cert, 2021\n"
)


def test_l3a_five_sections_through_the_custom_layout(tmp_path):
    # The real chain over the custom typst template: one page, no failure,
    # the hard breaks kept as typst `\`.
    from pypdf import PdfReader

    from loomcv.finalize import count_pages, export_pdf, export_typ

    md = tmp_path / "final_resume.md"
    md.write_text(FIVE_SECTIONS, encoding="utf-8")
    typ, pdf = tmp_path / "final_resume.typ", tmp_path / "final_resume.pdf"

    export_typ(md, typ, DEFAULTS)
    warnings = export_pdf(typ, pdf)

    assert isinstance(warnings, list)
    assert count_pages(pdf) == 1
    source = typ.read_text(encoding="utf-8")
    assert source.startswith("// LoomCV résumé layout")  # the custom layout, not pandoc's default
    assert "TypeScript \\\n" in source
    assert "2013 \\\n" in source
    text = PdfReader(str(pdf)).pages[0].extract_text()
    for word in ("Mara Voss", "Summary", "Skills", "Experience", "Education"):
        assert word in text
