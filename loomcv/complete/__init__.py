"""The completion loop: draft vault → confirmed vault.

Plain, UI-free logic over a `Workbook`. Stage A (`apply_jobs`): confirm each job's fields and
`Defaults`, assign skills from the vault-level `Skills` list. Stage B (`select_keywords`): one
model call proposes, per draft bullet, which of its job's confirmed defaults apply and the
bullet's own keywords. Stage C (`apply_bullets`): the user confirms each bullet's keywords,
text and project. `render_workbook` / `write_workbook` write the vault and run the strict
parser over it. Driven by tests directly, by `pipeline complete` (`--checklist` /
`--accept-all`) and by the Gradio front end in `loomcv.ui`, which holds no logic of its own.

Stage D: the header lines (`apply_header`), the Skills groups
(`apply_skill_groups`), the three-source pre-fill (`skills_prefill` / `with_prefill`) and one
optional model call that proposes the grouping (`propose_groups`, in `grouping.py`; UI only).

The entry-side narrative rewrite: one model call that drafts a STAR-shaped
bullet from a plain one or from guided answers (`draft_bullet`, in `rewrite.py`; UI only, a
proposal the user edits) and `add_bullet` for a bullet that starts from nothing.

Two pure helpers for the page: `open_counts` (what still needs the
user's eyes, per section) and `suggestion_due` (the guard before the self-running keyword
suggestion).

The profile page's views: `confirmed_counts` (the left list's "x of n
confirmed"), `maybe_also` (the job's tools a bullet's keywords lack), `unchanged_since_entry` and
`confirm_unedited` (the answers of "Confirm all N bullets I haven't edited"). `SkillGroup` and
`split_list` are re-exported from the vault grammar so the page reads its `;`-lists the way the
parser does without importing the index package.

Stage B does not wait for confirmed
`Defaults` — `select_keywords` offers them as they stand — and `first_keywords` is the upload's
second call, whose failure never refuses the upload.
"""

from loomcv.complete.checklist import ChecklistItem, checklist, confirmed_counts, open_counts
from loomcv.complete.grouping import propose_groups
from loomcv.complete.rewrite import draft_bullet
from loomcv.complete.select import first_keywords, select_keywords, suggestion_due
from loomcv.complete.sidecar import (
    load_profile,
    read_sidecar,
    save_working_copy,
    sidecar_for,
    with_confirmed,
    write_sidecar,
)
from loomcv.complete.workbook import (
    BulletAnswer,
    BulletDraftAnswers,
    BulletsAnswers,
    CompleteError,
    Groups,
    HeaderAnswers,
    JobAnswer,
    JobsAnswers,
    SkillsAnswers,
    WbBullet,
    WbJob,
    Workbook,
    WriteReport,
    accept_all,
    add_bullet,
    apply_bullets,
    apply_header,
    apply_jobs,
    apply_skill_groups,
    confirm_unedited,
    delete_bullet,
    groups,
    layout,
    load_workbook,
    maybe_also,
    render_workbook,
    skills_prefill,
    skills_unassigned,
    unchanged_since_entry,
    verify_rendered,
    with_prefill,
    write_workbook,
)
from loomcv.sync.parse_assets import SkillGroup, split_list

__all__ = [
    "BulletAnswer",
    "BulletDraftAnswers",
    "BulletsAnswers",
    "ChecklistItem",
    "CompleteError",
    "Groups",
    "HeaderAnswers",
    "JobAnswer",
    "JobsAnswers",
    "SkillGroup",
    "SkillsAnswers",
    "WbBullet",
    "WbJob",
    "Workbook",
    "WriteReport",
    "accept_all",
    "add_bullet",
    "apply_bullets",
    "apply_header",
    "apply_jobs",
    "apply_skill_groups",
    "checklist",
    "confirm_unedited",
    "confirmed_counts",
    "delete_bullet",
    "draft_bullet",
    "first_keywords",
    "groups",
    "layout",
    "load_profile",
    "load_workbook",
    "maybe_also",
    "open_counts",
    "propose_groups",
    "read_sidecar",
    "render_workbook",
    "save_working_copy",
    "select_keywords",
    "sidecar_for",
    "skills_prefill",
    "skills_unassigned",
    "split_list",
    "suggestion_due",
    "unchanged_since_entry",
    "verify_rendered",
    "with_confirmed",
    "with_prefill",
    "write_sidecar",
    "write_workbook",
]
