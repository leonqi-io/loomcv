{#- LoomCV résumé template: layout only, every slot is
inserted as given.

Slots: name, headline, summary (text or none); contact, education (lists); skill_groups (list,
vault order, each with name — "" for the unnamed group — and items; read items as g['items']);
skills (the groups' flat union, for templates written before the groups); jobs (in job order),
each with heading, date_from, date_to and bullets (walk order), each bullet with label
("**project:** " or ""), text and bullet_id. Filter: resume_date (2021-03 -> Mar 2021,
present -> Present). Keep `<!-- src: {{ b.bullet_id }} -->` on the bullet line: `pipeline
finalize` strips it. A trailing backslash is a hard line break (one Skills group or Education
entry per line in the PDF).

The default order is: header (name, contact, headline) -> Summary -> Skills
-> Experience -> Education. The headline is a bold line, never a heading, so no heading level
depends on where it sits. The slots are independent: a template of your own (generation.yaml
`resume_template`) may print them in any order or leave one out. Named
Skills groups print in vault order, then the unnamed group's bare line last. A job with no
placed bullet prints like any other job (heading + date line), just without a list. -#}
{% if name %}# {{ name }}

{% endif %}{% if contact %}{{ contact | join(" | ") }}

{% endif %}{% if headline %}**{{ headline }}**

{% endif %}{% if summary %}## Summary

{{ summary }}

{% endif %}{% if skills %}## Skills

{% set ordered = (skill_groups | selectattr("name") | list) + (skill_groups | rejectattr("name") | list) -%}
{% for g in ordered if g['items'] %}{% if g.name %}**{{ g.name }}:** {% endif %}{{ g['items'] | join("; ") }}{% if not loop.last %}\{% endif %}
{% endfor %}
{% endif %}## Experience
{%- for job in jobs %}

### {{ job.heading }}
*{{ job.date_from | resume_date }} – {{ job.date_to | resume_date }}*
{%- if job.bullets %}
{% for b in job.bullets %}
- {{ b.label }}{{ b.text }} <!-- src: {{ b.bullet_id }} -->
{%- endfor %}
{%- endif %}
{%- endfor %}
{% if education %}
## Education

{% for e in education %}{{ e }}{% if not loop.last %}\{% endif %}
{% endfor %}{% endif -%}
