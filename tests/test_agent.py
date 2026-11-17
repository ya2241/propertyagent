from app.agent import HANDOVER_REPLY, check_grounding
from app.llm import ScriptedLLM, call, say

BOOK = {"property_id": "P002", "slot_id": "P002-S1", "applicant_name": "Jane Doe", "applicant_email": "jane@example.com"}


def last_user_blocks(llm):
    return [b for m in llm.seen[-1] if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"]]


def test_write_action_is_gated_until_confirmed(make_agent, tools):
    book = call("book_viewing", BOOK)
    llm = ScriptedLLM([book, say("All booked for P002.")])
    agent, conv = make_agent(llm)

    r = agent.handle(conv, "Book P002-S1 for Jane Doe jane@example.com")
    assert r.status == "awaiting_confirmation" and "Pet-friendly" in r.reply
    assert tools.bookings == []  # nothing written yet

    r = agent.handle(conv, "Yes please", confirm=True)
    assert len(tools.bookings) == 1 and r.status == "ok"
    block = next(b for b in last_user_blocks(llm) if b["type"] == "tool_result")
    assert block["tool_use_id"] == book.tool_calls[0].id and "B0001" in block["content"]


def test_declined_action_never_executes(make_agent, tools):
    llm = ScriptedLLM([call("book_viewing", BOOK), say("No problem.")])
    agent, conv = make_agent(llm)
    agent.handle(conv, "book it")
    r = agent.handle(conv, "actually no", confirm=False)
    assert tools.bookings == [] and r.status == "ok"
    assert any(b.get("is_error") for b in last_user_blocks(llm))


def test_ignoring_confirmation_cancels_pending(make_agent, tools):
    llm = ScriptedLLM([call("book_viewing", BOOK), say("Sure, anything else?")])
    agent, conv = make_agent(llm)
    agent.handle(conv, "book it")
    agent.handle(conv, "what about parking?")  # confirm omitted
    assert tools.bookings == [] and conv.pending is None


def test_invalid_write_args_do_not_create_pending(make_agent):
    llm = ScriptedLLM([call("book_viewing", {**BOOK, "applicant_email": "nope"}), say("Could you give me a valid email?")])
    agent, conv = make_agent(llm)
    r = agent.handle(conv, "book it")
    assert r.status == "ok" and conv.pending is None and conv.tool_errors == 1


def test_model_requested_handover(make_agent):
    agent, conv = make_agent(ScriptedLLM([call("request_handover", {"reason": "complaint about landlord"})]))
    r = agent.handle(conv, "I want to complain")
    assert r.status == "handover" and r.reply == HANDOVER_REPLY
    assert agent.handle(conv, "hello?").status == "handover"  # later turns stay with the human


def test_handover_after_repeated_tool_errors(make_agent):
    agent, conv = make_agent(ScriptedLLM([call("get_viewing_slots", {"property_id": "P999"}),
                                          call("get_viewing_slots", {"property_id": "P998"})]))
    r = agent.handle(conv, "view P999 or P998")
    assert r.status == "handover" and "repeated_tool_errors" in r.handover_reason


def test_low_confidence_preferences_trigger_handover(make_agent):
    agent, conv = make_agent(ScriptedLLM([call("record_preferences", {"bedrooms": 2, "confidence": 0.3})]))
    assert agent.handle(conv, "umm maybe 2 bed?").status == "handover"


def test_no_matches_triggers_handover(make_agent):
    llm = ScriptedLLM([call("search_properties", {"max_rent_pcm": 500, "bedrooms": 6})])
    agent, conv = make_agent(llm)
    r = agent.handle(conv, "6 bed for £500")
    assert r.status == "handover" and r.handover_reason == "no_matching_properties"


def test_missing_info_handover_after_clarifying_turns(make_agent):
    agent, conv = make_agent(ScriptedLLM([say("What's your budget?")] * 4))
    statuses = [agent.handle(conv, "something nice").status for _ in range(4)]
    assert statuses == ["ok", "ok", "ok", "handover"]


def test_hallucinated_listing_is_caught_and_retried(make_agent):
    llm = ScriptedLLM([
        call("search_properties", {"max_rent_pcm": 2000, "bedrooms": 1}),
        say("P099 is a lovely flat at £999 pcm."),                 # invented
        say("P020 is an eco-friendly flat at £1,900 pcm."),        # grounded
    ])
    agent, conv = make_agent(llm)
    r = agent.handle(conv, "1 bed under £2000")
    assert r.status == "ok" and "P020" in r.reply and r.turn.grounding_issues == 2
    assert "[system]" in llm.seen[-1][-1]["content"]


def test_persistent_hallucination_hands_over(make_agent):
    agent, conv = make_agent(ScriptedLLM([say("P099 at £999."), say("Still P099 at £999.")]))
    r = agent.handle(conv, "show me flats")
    assert r.status == "handover" and r.handover_reason.startswith("ungrounded_reply")


def test_grounding_accepts_user_supplied_numbers(make_agent):
    agent, conv = make_agent(ScriptedLLM([say("Understood, a budget of £2,000.")]))
    assert agent.handle(conv, "my budget is £2000").status == "ok"


def test_llm_failure_becomes_handover(make_agent):
    class Boom:
        def complete(self, *a, **k): raise RuntimeError("down")
    agent, conv = make_agent(Boom())
    r = agent.handle(conv, "hi")
    assert r.status == "handover" and r.handover_reason.startswith("llm_error")


def test_metrics_track_tokens_cost_and_latency(make_agent):
    agent, conv = make_agent(ScriptedLLM([call("search_properties", {}, tokens=(1000, 50)), say("ok", tokens=(1500, 10))]))
    r = agent.handle(conv, "hi")
    assert r.turn.llm_calls == 2 and r.turn.input_tokens == 2500 and r.turn.output_tokens == 60
    assert r.turn.cost_usd > 0 and r.turn.latency_ms >= r.turn.llm_latency_ms >= 0 and r.turn.tool_calls == 1


def test_check_grounding_direct(make_agent):
    _, conv = make_agent(ScriptedLLM([]))
    conv.known_ids, conv.known_amounts = {"P001"}, {1850}
    assert check_grounding("P001 is £1,850 pcm", conv) == []
    assert len(check_grounding("P002 is £1,900", conv)) == 2
