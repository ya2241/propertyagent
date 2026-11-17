"""Tool-calling agent loop with a write-confirmation gate, grounding check and human handover."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from pydantic import ValidationError

from .config import Settings
from .llm import LLMClient, ToolCall
from .prompts import build_system
from .schemas import ApplicantPreferences, RecordPreferencesArgs
from .telemetry import Metrics, cost_usd, log_turn
from .tools import REGISTRY, PropertyTools, ToolError, anthropic_tool_defs

log = logging.getLogger("propertyagent")

MAX_TOOL_ERRORS = 2        # tool failures per conversation before handover
MIN_PREF_CONFIDENCE = 0.5  # below this, extracted preferences are not trusted
MAX_CLARIFY_TURNS = 4      # user turns without required info (and no search) before handover

HANDOVER_REPLY = "I'm passing you to a member of our lettings team, who will pick this up with you shortly."
GROUNDING_NUDGE = ("[system] Your last reply contained details that no tool returned: {issues}. "
                   "Rewrite it using only facts from tool results, or call a tool to get them.")

PID_RE = re.compile(r"\bP\d{3}\b")
AMOUNT_RE = re.compile(r"£\s?(\d[\d,]*)")


@dataclass
class PendingAction:
    tool_use_id: str
    name: str
    args: dict
    summary: str
    carried_results: list[dict]  # tool_results for sibling calls in the same assistant turn


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict] = field(default_factory=list)
    prefs: ApplicantPreferences = field(default_factory=ApplicantPreferences)
    pending: PendingAction | None = None
    status: str = "active"  # "active" | "handover"
    handover_reason: str | None = None
    known_ids: set[str] = field(default_factory=set)
    known_amounts: set[int] = field(default_factory=set)
    user_turns: int = 0
    searched: bool = False
    tool_errors: int = 0
    totals: Metrics = field(default_factory=Metrics)


class ConversationStore:
    def __init__(self):
        self._items: dict[str, Conversation] = {}

    def get(self, conversation_id: str | None) -> Conversation:
        if conversation_id and conversation_id in self._items:
            return self._items[conversation_id]
        conv = Conversation(id=conversation_id or uuid.uuid4().hex[:12])
        self._items[conv.id] = conv
        return conv

    def peek(self, conversation_id: str) -> Conversation | None:
        return self._items.get(conversation_id)


@dataclass
class AgentResult:
    reply: str
    status: str  # "ok" | "awaiting_confirmation" | "handover"
    pending: PendingAction | None
    handover_reason: str | None
    turn: Metrics
    totals: Metrics


def _amounts(text: str) -> set[int]:
    out = set()
    for raw in AMOUNT_RE.findall(text):
        digits = raw.replace(",", "")
        if digits.isdigit():
            out.add(int(digits))
    return out


def check_grounding(text: str, conv: Conversation) -> list[str]:
    """Flag property IDs and £ amounts in a reply that no tool result (or the user) ever supplied."""
    issues = [f"{pid} was never returned by a tool" for pid in sorted(set(PID_RE.findall(text)) - conv.known_ids)]
    issues += [f"£{a:,} is not in any tool result" for a in sorted(_amounts(text) - conv.known_amounts)]
    return issues


class PropertyAgent:
    def __init__(self, llm: LLMClient, tools: PropertyTools, settings: Settings | None = None):
        self.llm, self.tools = llm, tools
        self.settings = settings or Settings.from_env()
        self.tool_defs = anthropic_tool_defs()

    # ------------------------------------------------------------ public

    def handle(self, conv: Conversation, message: str, confirm: bool | None = None) -> AgentResult:
        t0 = time.perf_counter()
        turn = Metrics()
        if conv.status == "handover":
            return self._finish(conv, turn, t0, HANDOVER_REPLY, "handover")
        conv.user_turns += 1
        self._begin_turn(conv, message, confirm, turn)
        if conv.status == "handover":  # e.g. the confirmed action failed repeatedly
            return self._finish(conv, turn, t0, HANDOVER_REPLY, "handover")

        grounding_retries = 0
        for _ in range(self.settings.max_steps):
            try:
                result = self._call_llm(conv, turn)
            except Exception as exc:  # network, auth, rate limit...
                log.exception("LLM call failed")
                return self._handover(conv, turn, t0, f"llm_error: {type(exc).__name__}")
            conv.messages.append({"role": "assistant", "content": result.content})
            calls = result.tool_calls

            if not calls:
                issues = check_grounding(result.text, conv)
                if issues:
                    turn.grounding_issues += len(issues)
                    if grounding_retries == 0:
                        grounding_retries = 1
                        conv.messages.append({"role": "user", "content": GROUNDING_NUDGE.format(issues="; ".join(issues))})
                        continue
                    return self._handover(conv, turn, t0, "ungrounded_reply: " + "; ".join(issues))
                if conv.user_turns >= MAX_CLARIFY_TURNS and conv.prefs.missing_required() and not conv.searched:
                    return self._handover(conv, turn, t0, "missing_information: " + ", ".join(conv.prefs.missing_required()))
                return self._finish(conv, turn, t0, result.text, "ok")

            results, pending = [], None
            for c in calls:
                spec = REGISTRY.get(c.name)
                if spec and spec.is_write:
                    if pending is not None:
                        results.append(self._error_block(c.id, "Only one write action per turn."))
                        continue
                    try:
                        args = spec.args_model.model_validate(c.input).model_dump()
                    except ValidationError as exc:
                        turn.tool_errors += 1
                        conv.tool_errors += 1
                        results.append(self._error_block(c.id, f"Invalid arguments: {exc.errors()[0]['msg']}"))
                        continue
                    pending = PendingAction(c.id, c.name, args, self._summarise(c.name, args), [])
                    continue
                results.append(self._execute(conv, c, turn))

            if conv.status == "handover":
                return self._finish(conv, turn, t0, result.text or HANDOVER_REPLY, "handover")
            if pending:  # write action: stop and ask the human to confirm first
                pending.carried_results = results
                conv.pending = pending
                reply = (result.text + "\n\n" if result.text else "") + f"Please confirm: {pending.summary}"
                return self._finish(conv, turn, t0, reply, "awaiting_confirmation")
            conv.messages.append({"role": "user", "content": results})
        return self._handover(conv, turn, t0, "step_limit_reached")

    # ------------------------------------------------------------ internals

    def _begin_turn(self, conv, message, confirm, turn):
        conv.known_amounts |= _amounts(message)
        conv.known_ids |= set(PID_RE.findall(message))
        if conv.pending:
            p, conv.pending = conv.pending, None
            if confirm is True:
                block = self._execute(conv, ToolCall(p.tool_use_id, p.name, p.args), turn)
            else:
                block = self._error_block(p.tool_use_id, "Not executed: the applicant did not confirm this action.")
            content = [*p.carried_results, block]
            if message.strip():
                content.append({"type": "text", "text": message})
        else:
            content = message
        conv.messages.append({"role": "user", "content": content})

    def _call_llm(self, conv, turn):
        system = build_system(self.settings.prompt_version, conv.prefs)
        t = time.perf_counter()
        result = self.llm.complete(system, conv.messages, self.tool_defs)
        dt = (time.perf_counter() - t) * 1000
        turn.llm_calls += 1
        turn.llm_latency_ms += dt
        turn.input_tokens += result.usage.input_tokens
        turn.output_tokens += result.usage.output_tokens
        turn.cost_usd += cost_usd(result.usage.input_tokens, result.usage.output_tokens)
        return result

    def _execute(self, conv, call: ToolCall, turn) -> dict:
        turn.tool_calls += 1
        try:
            if call.name == "record_preferences":
                try:
                    args = RecordPreferencesArgs.model_validate(call.input)
                except ValidationError as exc:
                    raise ToolError(f"Invalid arguments: {exc.errors()[0]['loc']}: {exc.errors()[0]['msg']}") from exc
                conv.prefs.merge(args)
                if conv.prefs.budget_max_pcm:
                    conv.known_amounts.add(conv.prefs.budget_max_pcm)
                if args.confidence < MIN_PREF_CONFIDENCE:
                    self._set_handover(conv, f"low_confidence_preferences ({args.confidence:.2f})")
                data = {"recorded": conv.prefs.model_dump(mode="json", exclude_none=True),
                        "still_missing": conv.prefs.missing_required()}
            elif call.name == "request_handover":
                self._set_handover(conv, str(call.input.get("reason", "requested by model")))
                data = {"handover": True}
            else:
                data = self.tools.execute(call.name, call.input)
                self._ingest(conv, call.name, data)
            return {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(data, default=str)}
        except ToolError as exc:
            turn.tool_errors += 1
            conv.tool_errors += 1
            if conv.tool_errors >= MAX_TOOL_ERRORS:
                self._set_handover(conv, f"repeated_tool_errors: {exc}")
            return self._error_block(call.id, str(exc))

    def _ingest(self, conv, name, data):
        if name == "search_properties":
            conv.searched = True
            for m in data["matches"]:
                conv.known_ids.add(m["id"])
                conv.known_amounts.add(m["rent_pcm"])
            if not data["matches"]:
                self._set_handover(conv, "no_matching_properties")
        elif name in ("get_viewing_slots", "book_viewing"):
            conv.known_ids.add(data["property_id"])

    @staticmethod
    def _error_block(tool_use_id, message):
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": message, "is_error": True}

    @staticmethod
    def _set_handover(conv, reason):
        conv.status, conv.handover_reason = "handover", conv.handover_reason or reason

    def _summarise(self, name, args):
        prop = self.tools.store.get(args["property_id"])
        slot = self.tools.store.slot(args["slot_id"])
        title = f"{prop.title} ({prop.id})" if prop else args["property_id"]
        when = slot.start.strftime("%a %d %b, %H:%M") if slot else args["slot_id"]
        return f"book a viewing of {title} on {when} for {args['applicant_name']} <{args['applicant_email']}>"

    def _handover(self, conv, turn, t0, reason):
        self._set_handover(conv, reason)
        return self._finish(conv, turn, t0, HANDOVER_REPLY, "handover")

    def _finish(self, conv, turn, t0, reply, status):
        turn.latency_ms = (time.perf_counter() - t0) * 1000
        conv.totals.add(turn)
        log_turn(conv.id, status, turn, conv.totals, conv.handover_reason)
        return AgentResult(reply, status, conv.pending, conv.handover_reason, turn, conv.totals)
