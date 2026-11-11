"""The files the code needs to run — prompts, templates, the stop list, pandoc's defaults.

They ship inside the package, so a wheel install has them too; `config/` keeps only the
settings a user edits. `path` is the one way code finds a shipped resource.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def path(*parts: str) -> Path:
    """The absolute path of a shipped resource, e.g. `path("prompts", "parse_jd.md")`.

    Raises `FileNotFoundError` naming the relative name when it is not there — never a
    fallback."""
    resolved = Path(str(files(__name__).joinpath(*parts))).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"no packaged resource {'/'.join(parts)}")
    return resolved
