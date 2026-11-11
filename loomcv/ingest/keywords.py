"""The helpers both generation calls share.

Middle layer of `loomcv.ingest`. The keyword call that used to live here moved to
`loomcv.complete.select` (it runs behind the user's confirmation of each
job's `Defaults`, as the selection call); what stays is what the structure call
(`extract.py`) and the selection call both use — the template render-and-split, the
call-with-one-retry (every reply is complete or retried once, then `IngestError`), item
cleaning (whitespace normalised, empties dropped, an item containing `;` dropped with a
warning, case-insensitive duplicates removed keeping the first spelling — no word list, no
grounding, no similarity rule) and the repeated-key guard for JSON replies keyed by
handle.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import jinja2
from pydantic import ValidationError

from loomcv.ingest.render import IngestError
from loomcv.llm import GenerationAdapter

# One cleaning implementation for every generation reply: this couples to
# the private `parse_jd._clean_reply`, as `coverage` already does.
from loomcv.parse_jd import _clean_reply

_USER_MARKER = "---USER---"
_WS_RE = re.compile(r"\s+")

T = TypeVar("T")


class _InvalidReply(Exception):
    pass


def normalise_ws(value: str) -> str:
    """Internal whitespace runs (newlines included) → one space; stripped."""
    return _WS_RE.sub(" ", value).strip()


def handle(job_index: int, bullet_index: int) -> str:
    """The temporary keyword-call handle, both indexes 1-based."""
    return f"J{job_index}b{bullet_index}"


def render_split(template_path: Path, **variables: Any) -> tuple[str, str]:
    """Render the template whole, split on the first `---USER---` line → (system, user).

    The `parse_jd.render_prompt` contract (same Jinja environment); a local copy because that
    function renders `jd_text` only."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    template = env.from_string(Path(template_path).read_text(encoding="utf-8"))
    lines = template.render(**variables).splitlines()
    for i, line in enumerate(lines):
        if line == _USER_MARKER:
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i + 1 :]).strip()
    raise ValueError(f"prompt template {template_path} has no '{_USER_MARKER}' line")


def _complete(
    adapter: GenerationAdapter, system: str, user: str, raws: list[str], *, label: str
) -> str:
    try:
        raw = adapter.complete(system, user, temperature=adapter.temperature)
    except (OSError, RuntimeError) as exc:  # URLError is an OSError: no retry on transport
        raise IngestError(f"{label} call failed: {exc}", raws=raws) from exc
    raws.append(raw)
    return raw


def complete_validated(
    adapter: GenerationAdapter,
    system: str,
    user: str,
    validate: Callable[[str], T],
    raws: list[str],
    *,
    label: str,
) -> T:
    """One call; on an invalid reply exactly one retry with the `parse_jd` suffix, then
    `IngestError`. Every reply is appended to `raws` in call order."""
    first = _complete(adapter, system, user, raws, label=label)
    try:
        return validate(_clean_reply(first))
    except (ValidationError, _InvalidReply) as exc:
        print(
            f"ingest: first {label} reply invalid ({str(exc).splitlines()[0]}); retrying once",
            file=sys.stderr,
        )
        retry_user = (
            f"{user}\n\nYour previous answer was not valid: {exc}. Return only the JSON object."
        )
        second = _complete(adapter, system, retry_user, raws, label=label)
        try:
            return validate(_clean_reply(second))
        except (ValidationError, _InvalidReply) as exc2:
            raise IngestError(f"{label} reply invalid after one retry: {exc2}", raws=raws) from exc2


def _clean_items(items: list[str]) -> tuple[list[str], list[str]]:
    """`(kept, warnings)`; nothing is lower-cased, stemmed or split."""
    kept: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = normalise_ws(raw)
        if not item:
            continue
        if ";" in item:
            warnings.append(f"dropped keyword '{item}' (contains ;)")
            continue
        if item.casefold() in seen:
            continue
        seen.add(item.casefold())
        kept.append(item)
    return kept, warnings


def _reject_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _InvalidReply(f"bullet id {key!r} appears more than once")
        obj[key] = value
    return obj
