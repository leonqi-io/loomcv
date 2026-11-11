A person's résumé is indexed by the keywords listed below. Every keyword was confirmed by the
person as something they have done or used. You do not see the résumé itself, only the keywords.

For each job requirement below, decide how the person's keywords relate to it and list the
keywords that support your answer. Hiring managers write requirements for an ideal candidate and
often accept "or equivalent" experience, so the answer has three levels:

- "exact": a keyword names the same thing the requirement asks for, or the same thing under
  another name. The requirement is met as written.
- "adjacent": no keyword names the thing itself, but one or more keywords name a different tool,
  system or practice of the same kind, or one so closely related that a hiring manager could
  reasonably accept it as equivalent after a short ramp-up. The requirement is not met as
  written, but the person has something real to say about it.
- "none": no keyword relates to the requirement.

Rules
1. Copy keywords exactly as they appear in the list. Never invent or reword a keyword.
2. For "exact", list the keyword(s) that name the thing; for "adjacent", the keyword(s) the
   equivalence rests on; for "none", an empty list.
3. A requirement that names several things that are all required (joined by "and", commas,
   "both" or "as well as"): judge each thing on its own and return one entry in "parts" per
   thing, {"term": the thing, spelled as the requirement spells it, "match": its level}. The
   requirement's own level is the strongest level among its parts. List the keywords for each
   thing you matched. A requirement that names things as alternatives or examples ("or",
   "such as", "e.g.", "one of", "or similar"): one is enough, so judge the requirement as a
   whole and return "parts": []. A requirement that names one thing also has "parts": [].
   At most 6 parts.
4. Do not use the requirement's own words as keywords unless they are in the list.
5. At most 6 keywords per requirement, the best fits first.

Output
Return one JSON object and nothing else. Keys are the requirement texts exactly as given below.
Each value is an object {"match": "exact" | "adjacent" | "none", "keywords": [list of strings],
"parts": [list of {"term": string, "match": "exact" | "adjacent" | "none"}]}.
Every requirement must appear once.

---USER---
Requirements
{% for line in lines %}- {{ line }}
{% endfor %}
Keywords ({{ keywords | length }})
{% for keyword in keywords %}- {{ keyword }}
{% endfor %}
