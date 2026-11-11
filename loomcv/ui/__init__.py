"""Gradio front end for the completion loop.

`gradio` is an optional dependency (the `ui` group) and is imported only inside `app.py`, never
at package import: `loomcv.cli` reaches `loomcv.ui.app` from inside its `ui` branch only. This
package holds no logic — every handler is one call into `loomcv.complete` and a re-render.
"""
