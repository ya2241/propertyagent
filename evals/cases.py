"""Eval conversations: successes, missing information, and failure cases.

expect keys: status (final status), tools_called, tools_not_called, prefs (subset match),
bookings (count), reply_has (substrings in final reply), tool_errors (exact count)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    message: str
    confirm: bool | None = None


@dataclass
class Case:
    name: str
    category: str  # success | missing_info | failure
    turns: list[Turn]
    expect: dict
    inject_failures: dict = field(default_factory=dict)


BOOK = "book P002-S1 for Jane Doe jane@example.com"

CASES = [
    Case("search_with_all_prefs", "success",
         [Turn("I need a 2 bed under £2400 from 2026-12-20")],
         {"status": "ok", "tools_called": ["record_preferences", "search_properties"],
          "prefs": {"budget_max_pcm": 2400, "bedrooms": 2}, "reply_has": ["P0"]}),
    Case("studio_search", "success",
         [Turn("Looking for a studio, budget £1500")],
         {"status": "ok", "tools_called": ["search_properties"], "prefs": {"bedrooms": 0, "budget_max_pcm": 1500}}),
    Case("book_viewing_confirmed", "success",
         [Turn("I need a 2 bed under £2400"), Turn("I'd like to view P002"), Turn(BOOK), Turn("Yes", confirm=True)],
         {"status": "ok", "tools_called": ["get_viewing_slots", "book_viewing"], "bookings": 1, "reply_has": ["Booked"]}),
    Case("book_viewing_declined", "success",
         [Turn("I'd like to view P002"), Turn(BOOK), Turn("No, wrong slot", confirm=False)],
         {"status": "ok", "bookings": 0}),
    Case("write_waits_for_confirmation", "success",
         [Turn("I'd like to view P002"), Turn(BOOK)],
         {"status": "awaiting_confirmation", "bookings": 0}),

    Case("vague_request_asks_question", "missing_info",
         [Turn("Hi, I want somewhere nice")],
         {"status": "ok", "tools_not_called": ["search_properties", "book_viewing"], "reply_has": ["?"]}),
    Case("budget_only_asks_for_bedrooms", "missing_info",
         [Turn("My budget is £2000")],
         {"status": "ok", "tools_not_called": ["search_properties"], "prefs": {"budget_max_pcm": 2000}, "reply_has": ["bedroom"]}),
    Case("still_missing_after_four_turns", "missing_info",
         [Turn("hello"), Turn("not sure"), Turn("just somewhere nice"), Turn("dunno")],
         {"status": "handover"}),
    Case("explicit_human_request", "missing_info",
         [Turn("Can I speak to a person please?")],
         {"status": "handover", "tools_called": ["request_handover"]}),

    Case("unknown_property_id", "failure",
         [Turn("I'd like to view P999")],
         {"status": "ok", "tool_errors": 1, "bookings": 0}),
    Case("transient_search_failure_recovers", "failure",
         [Turn("I need a 1 bed under £2000")],
         {"status": "ok", "tool_errors": 1, "reply_has": ["P0"]},
         inject_failures={"search_properties": 1}),
    Case("persistent_search_failure_hands_over", "failure",
         [Turn("I need a 1 bed under £2000")],
         {"status": "handover"}, inject_failures={"search_properties": 99}),
    Case("ambiguous_low_confidence", "failure",
         [Turn("not sure, something cheap, maybe 2 bed")],
         {"status": "handover"}),
    Case("no_matching_inventory", "failure",
         [Turn("I need a 6 bed under £900")],
         {"status": "handover"}),
]
