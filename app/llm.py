"""LLM clients: the real Claude API, a scripted test double, and a deterministic rule-based stand-in."""
from __future__ import annotations

import copy
import itertools
import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResult:
    content: list[dict]
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b["type"] == "text").strip()

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in self.content if b["type"] == "tool_use"]


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResult: ...


class AnthropicLLM:
    """Thin wrapper over the Claude Messages API (needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        self.model = model or os.getenv("PROPERTYAGENT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self._client = None

    def complete(self, system, messages, tools):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        resp = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, tools=tools, messages=messages)
        content: list[dict] = []
        for b in resp.content:
            if b.type == "text" and b.text:
                content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        if not content:
            content = [{"type": "text", "text": "Sorry, I couldn't produce a reply."}]
        return LLMResult(content, Usage(resp.usage.input_tokens, resp.usage.output_tokens))


# ---------------------------------------------------------------- test doubles

_ids = itertools.count(1)


def say(text: str, tokens: tuple[int, int] = (100, 20)) -> LLMResult:
    return LLMResult([{"type": "text", "text": text}], Usage(*tokens))


def call(name: str, input: dict, text: str = "", tokens: tuple[int, int] = (100, 20)) -> LLMResult:
    content = [{"type": "text", "text": text}] if text else []
    content.append({"type": "tool_use", "id": f"toolu_{next(_ids):04d}", "name": name, "input": input})
    return LLMResult(content, Usage(*tokens))


class ScriptedLLM:
    """Returns a fixed sequence of results; records the messages it was shown."""

    def __init__(self, script: list[LLMResult]):
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def complete(self, system, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        if not self.script:
            raise AssertionError("ScriptedLLM script exhausted")
        return self.script.pop(0)


HUMAN = re.compile(r"\b(human|real person|speak to (a |someone)|a person|manager)\b", re.I)
HEDGE = re.compile(r"not sure|no idea|whatever|anything|don'?t know", re.I)
BUDGET = re.compile(r"£\s?(\d[\d,]{2,5})")
BEDS = re.compile(r"\b(\d)\s*[- ]?bed", re.I)
DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
PID = re.compile(r"\bP\d{3}\b")
SLOT = re.compile(r"\bP\d{3}-S\d\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
NAME = re.compile(r"for ([A-Z][a-z]+(?: [A-Z][a-z]+)+)")


class RuleBasedLLM:
    """Deterministic stand-in for the model: regex extraction plus a fixed tool policy.

    It exists so the agent harness, API and eval pipeline run offline with no API key.
    Its scores say nothing about Claude's behaviour; use `--mode live` for that.
    """

    def complete(self, system, messages, tools):
        content = self._decide(messages)
        in_tok = len(json.dumps(messages, default=str)) // 4
        out_tok = sum(len(b.get("text", "")) + len(json.dumps(b.get("input", {}))) for b in content) // 4 + 10
        return LLMResult(content, Usage(in_tok, out_tok))

    # -- helpers
    @staticmethod
    def _text(t): return [{"type": "text", "text": t}]

    @staticmethod
    def _tool(name, args): return [{"type": "tool_use", "id": f"toolu_rb_{next(_ids):04d}", "name": name, "input": args}]

    def _decide(self, messages):
        last = messages[-1]["content"]
        if isinstance(last, list):
            results = [b for b in last if b["type"] == "tool_result"]
            if results:
                return self._after_tool(messages, results)
            last = " ".join(b.get("text", "") for b in last)
        return self._on_user_text(messages, last)

    def _known_prefs(self, messages):
        prefs = {}
        for m in messages:
            if m["role"] == "assistant":
                for b in m["content"]:
                    if b["type"] == "tool_use" and b["name"] == "record_preferences":
                        prefs.update({k: v for k, v in b["input"].items() if v is not None})
        return prefs

    def _next_step(self, prefs):
        if "budget_max_pcm" in prefs and "bedrooms" in prefs:
            args = {"max_rent_pcm": prefs["budget_max_pcm"], "bedrooms": prefs["bedrooms"]}
            if prefs.get("move_in_date"):
                args["available_by"] = prefs["move_in_date"]
            return self._tool("search_properties", args)
        missing = "maximum monthly budget" if "budget_max_pcm" not in prefs else "number of bedrooms"
        return self._text(f"Thanks, noted. What's your {missing}?")

    def _after_tool(self, messages, results):
        uses = {b["id"]: b for b in messages[-2]["content"] if b["type"] == "tool_use"}
        r = results[-1]
        name = uses[r["tool_use_id"]]["name"]
        err = r.get("is_error", False)
        body = r.get("content", "")
        if err and "did not confirm" in body:
            return self._text("No problem, I haven't booked anything. Tell me if you'd like another slot.")
        if name == "record_preferences":
            return self._next_step(self._known_prefs(messages))
        if name == "search_properties":
            if err:
                errors = sum(1 for m in messages if m["role"] == "user" and isinstance(m["content"], list)
                             for b in m["content"] if b.get("is_error"))
                if errors < 2:
                    return self._tool("search_properties", uses[r["tool_use_id"]]["input"])
                return self._text("Sorry, search isn't working right now.")
            lines = [f"{m['id']}: {m['title']} in {m['area']}, £{m['rent_pcm']:,} pcm, {m['bedrooms']} bed, "
                     f"available {m['available_from']}" for m in json.loads(body)["matches"]]
            return self._text("Best matches:\n" + "\n".join(lines) + "\nWould you like to view any of these?")
        if name == "get_viewing_slots":
            if err:
                return self._text("Sorry, I couldn't find slots for that property. Could you check the ID?")
            data = json.loads(body)
            lines = [f"{s['slot_id']}: {s['start']}" for s in data["slots"]]
            return self._text(f"Viewings for {data['property_id']}:\n" + "\n".join(lines) +
                              "\nTell me a slot, your name and email and I'll book it.")
        if name == "book_viewing":
            if err:
                return self._text("Sorry, that booking didn't go through.")
            d = json.loads(body)
            return self._text(f"Booked! Reference {d['booking_id']} for {d['property_id']} at {d['start']}.")
        return self._text("Okay.")

    def _on_user_text(self, messages, text):
        if HUMAN.search(text):
            return self._tool("request_handover", {"reason": "Applicant asked for a person"})
        slot, email, nm = SLOT.search(text), EMAIL.search(text), NAME.search(text)
        if slot and email and nm and re.search("book", text, re.I):
            return self._tool("book_viewing", {"property_id": slot.group().split("-")[0], "slot_id": slot.group(),
                                               "applicant_name": nm.group(1), "applicant_email": email.group()})
        pid = PID.search(text)
        if pid and re.search("view|slot|see", text, re.I):
            return self._tool("get_viewing_slots", {"property_id": pid.group()})
        found = {}
        if m := BUDGET.search(text): found["budget_max_pcm"] = int(m.group(1).replace(",", ""))
        if m := BEDS.search(text): found["bedrooms"] = int(m.group(1))
        elif "studio" in text.lower(): found["bedrooms"] = 0
        if m := DATE.search(text): found["move_in_date"] = m.group(1)
        if found:
            conf = 0.3 if HEDGE.search(text) and len(found) < 2 else 0.9
            return self._tool("record_preferences", {**found, "confidence": conf})
        return self._text("Happy to help! What's your maximum monthly budget and how many bedrooms do you need?")
