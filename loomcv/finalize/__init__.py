"""Headless export of the tailored résumé: strip the audit comments, Markdown -> typst ->
PDF with bundled converters, read the page count back. No model call, no index, no graph."""

from loomcv.finalize.export import (
    FINAL_MD,
    FINAL_PDF,
    FINAL_TYP,
    PandocFailed,
    TypstFailed,
    count_pages,
    export_pdf,
    export_typ,
    strip_src,
)

__all__ = [
    "FINAL_MD",
    "FINAL_PDF",
    "FINAL_TYP",
    "PandocFailed",
    "TypstFailed",
    "count_pages",
    "export_pdf",
    "export_typ",
    "strip_src",
]
