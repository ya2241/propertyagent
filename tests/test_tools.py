import pytest

from app.tools import ToolError, anthropic_tool_defs


def test_search_applies_hard_filters(tools):
    out = tools.execute("search_properties", {"max_rent_pcm": 2000, "bedrooms": 1})
    assert out["matches"] and all(m["rent_pcm"] <= 2000 and m["bedrooms"] == 1 for m in out["matches"])


def test_search_no_matches_is_low_confidence(tools):
    out = tools.execute("search_properties", {"max_rent_pcm": 500, "bedrooms": 6})
    assert out["matches"] == [] and out["low_confidence"]


def test_search_available_by_filters_dates(tools):
    out = tools.execute("search_properties", {"available_by": "2026-11-30", "limit": 5})
    assert {m["id"] for m in out["matches"]} <= {"P003", "P014"}


def test_invalid_arguments_raise_tool_error(tools):
    with pytest.raises(ToolError, match="Invalid arguments"):
        tools.execute("get_viewing_slots", {"property_id": "banana"})


def test_unknown_property(tools):
    with pytest.raises(ToolError, match="Unknown property"):
        tools.execute("get_viewing_slots", {"property_id": "P999"})


def test_booking_and_double_booking(tools):
    args = {"property_id": "P002", "slot_id": "P002-S1", "applicant_name": "Jane Doe", "applicant_email": "jane@example.com"}
    assert tools.execute("book_viewing", args)["booking_id"] == "B0001"
    assert "P002-S1" not in [s["slot_id"] for s in tools.execute("get_viewing_slots", {"property_id": "P002"})["slots"]]
    with pytest.raises(ToolError, match="already booked"):
        tools.execute("book_viewing", args)


def test_slot_must_belong_to_property(tools):
    with pytest.raises(ToolError):
        tools.execute("book_viewing", {"property_id": "P001", "slot_id": "P002-S1",
                                       "applicant_name": "Jane Doe", "applicant_email": "jane@example.com"})


def test_injected_failures_then_recovery(settings):
    from app.tools import PropertyTools
    t = PropertyTools(settings=settings, inject_failures={"search_properties": 1})
    with pytest.raises(ToolError):
        t.execute("search_properties", {})
    assert t.execute("search_properties", {})["matches"]


def test_tool_schemas_are_valid_json_schema():
    for d in anthropic_tool_defs():
        assert d["input_schema"]["type"] == "object" and d["description"]
