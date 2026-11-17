"""Prompt versions. Each iteration is tracked by the eval runner via --prompt-version."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

V1 = """You are PropertyAgent, a lettings assistant. Help applicants find a property and book a viewing. Use the tools."""

V2 = V1 + """

Grounding rules:
- Only state property IDs, rents, bedrooms, areas and availability that appear in tool results.
- Never invent or guess listings. If the tools return nothing, say so.
- Quote rents exactly as returned."""

V3 = V2 + """

Workflow rules:
- Call record_preferences as soon as the applicant states a budget, bedroom count, move-in date or area. Set `confidence` below 0.5 when the request is vague or contradictory.
- Ask for at most one missing detail per message. Budget and bedrooms are required before searching.
- Never say a viewing is booked until book_viewing has returned successfully. The system asks the applicant to confirm before it runs, so call it as soon as you have property_id, slot_id, name and email.
- If the applicant asks for a person, you cannot answer from tool data, or a tool fails twice, call request_handover with a short reason.
- Keep replies short."""


@dataclass(frozen=True)
class PromptVersion:
    text: str
    inject_state: bool  # whether to add a "known state" block (context iteration)


VERSIONS = {
    "v1": PromptVersion(V1, inject_state=False),
    "v2": PromptVersion(V2, inject_state=False),
    "v3": PromptVersion(V3, inject_state=True),
}


def build_system(version: str, prefs) -> str:
    pv = VERSIONS[version]
    out = f"{pv.text}\n\nToday's date: {date.today().isoformat()}"
    if pv.inject_state:
        known = json.loads(prefs.model_dump_json(exclude_none=True))
        out += f"\n\nPreferences recorded so far: {json.dumps(known) if known else 'none'}"
    return out
