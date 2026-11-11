"""The review document: the tailored résumé as bullets the user can reorder, edit, delete
and add to, plus the disk / index side that keeps it between sessions.

`doc.py` is pure (no file, no index, no model call); `store.py` is the only module here that
touches the application directory; `words.py` is pure and turns their records
into the sentences the applications page prints. Nothing is imported at package level, so
`import loomcv.review` loads no module and never `gradio` — the page is `loomcv/ui/review.py`
and calls into here.
"""
