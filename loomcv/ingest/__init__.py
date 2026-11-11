"""`pipeline ingest`: an old résumé → a v3 draft vault.

One structure call. Every bullet is `{status: draft}`, its text verbatim and with no
`Keywords`; each job carries the skills its section declares as `Defaults` (a job with no
bullets is kept, bullet-less); a skills section outside any job becomes the vault-level
`Skills` line. Zero new facts: the extractor may leave a field empty (`0000` for an unknown
date), never guess it. L2 (`loomcv.complete`) confirms the jobs, selects keywords per bullet
and confirms the bullets.
"""

from loomcv.ingest.extract import (
    IngestResult,
    assign_ids,
    ingest_resume,
    render_structure_prompt,
)
from loomcv.ingest.read import ResumeReadError, resume_text
from loomcv.ingest.render import (
    DraftBullet,
    DraftJob,
    DraftVault,
    IngestError,
    check_rendered,
    render_vault,
)

__all__ = [
    "DraftBullet",
    "DraftJob",
    "DraftVault",
    "IngestError",
    "IngestResult",
    "ResumeReadError",
    "assign_ids",
    "check_rendered",
    "ingest_resume",
    "render_structure_prompt",
    "render_vault",
    "resume_text",
]
