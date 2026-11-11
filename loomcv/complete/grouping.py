"""The grouping call: a proposal of named Skills groups over the vault's skills.

One call over the workbook's current groups (the pre-fill included) on `nodes.ingest` — the
same node as the structure and selection calls. The reply is `{"groups": [{"name", "items"}]}`;
it is complete or retried once (`complete_validated`). The model names groups and sorts the
items; the code keeps the item set fixed (conservation): a reply item is matched
casefold to an input item and written in the input's spelling, an item the input does not
have is dropped, an item repeated across groups keeps its first group, and an input item the
reply leaves out goes back to the unnamed group (first, where the vault holds it). Each of
those is a warning. Pure: the workbook is not changed — the caller applies the proposal with
`apply_skill_groups`, and the user edits it from there. The headless `--accept-all` path never
calls this (Forbidden).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loomcv import resources
from loomcv.complete.workbook import CompleteError, Workbook
from loomcv.ingest.keywords import _InvalidReply, complete_validated, normalise_ws, render_split
from loomcv.ingest.render import IngestError
from loomcv.llm import GenerationAdapter
from loomcv.sync.parse_assets import SkillGroup

# The prompt ships with the package, so its path is package-relative.
DEFAULT_GROUPS_PROMPT = resources.path("prompts", "propose_groups.md")


def _validate_groups(text: str) -> list[tuple[str, list[str]]]:
    try:
        reply: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _InvalidReply(f"not valid JSON ({exc.msg})") from exc
    if not isinstance(reply, dict) or not isinstance(reply.get("groups"), list):
        raise _InvalidReply("the reply is not an object with a 'groups' list")
    parsed: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for group in reply["groups"]:
        if not isinstance(group, dict):
            raise _InvalidReply("a group is not an object with 'name' and 'items'")
        name, items = group.get("name"), group.get("items")
        if not isinstance(name, str) or "*" in name:
            raise _InvalidReply(f"group name {name!r} is not a string without '*'")
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            raise _InvalidReply(f"group {name!r}: 'items' is not a list of strings")
        name = normalise_ws(name)
        if name and name.casefold() in seen:
            raise _InvalidReply(f"two groups are named {name!r}")
        seen.add(name.casefold())
        parsed.append((name, items))
    return parsed


def propose_groups(
    wb: Workbook,
    *,
    adapter: GenerationAdapter,
    template_path: Path = DEFAULT_GROUPS_PROMPT,
) -> tuple[list[SkillGroup], list[str]]:
    """One call → `(the proposed groups, warnings)`; the unnamed group (left-out items) first.

    Raises `CompleteError` when no valid reply arrives after one retry (`raws` carries every
    reply). With no item there is no call and `([], [])` is returned."""
    by_fold = {item.casefold(): item for item in wb.skills}
    if not by_fold:
        return [], []

    system, user = render_split(
        template_path,
        groups=[{"name": g.name, "items": g.items} for g in wb.skill_groups],
    )
    raws: list[str] = []
    try:
        reply = complete_validated(adapter, system, user, _validate_groups, raws, label="grouping")
    except IngestError as exc:
        raise CompleteError(str(exc), raws=exc.raws) from exc

    warnings: list[str] = []
    placed: set[str] = set()
    named: list[SkillGroup] = []
    unnamed: list[str] = []
    for name, raw_items in reply:
        items: list[str] = []
        for raw in raw_items:
            fold = normalise_ws(raw).casefold()
            if fold not in by_fold:
                warnings.append(f"dropped '{normalise_ws(raw)}' (not one of the vault's skills)")
            elif fold in placed:
                warnings.append(f"'{by_fold[fold]}' was proposed twice; kept in its first group")
            else:
                placed.add(fold)
                items.append(by_fold[fold])
        if not name:
            unnamed.extend(items)
        elif items:
            named.append(SkillGroup(name=name, items=items))
    for fold, item in by_fold.items():
        if fold not in placed:
            warnings.append(f"'{item}' was left out of the proposal; kept in the unnamed group")
            unnamed.append(item)
    unnamed_first = [SkillGroup(name="", items=unnamed)] if unnamed else []
    return [*unnamed_first, *named], warnings
