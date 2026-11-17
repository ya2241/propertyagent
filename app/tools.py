"""Tool registry. Arguments are validated with Pydantic before any handler runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from pydantic import BaseModel, ValidationError

from .config import Settings
from .data import Booking, PropertyStore
from .retrieval import Retriever, build_retriever
from .schemas import BookArgs, HandoverArgs, RecordPreferencesArgs, SearchArgs, SlotsArgs


class ToolError(Exception):
    """A tool failed in a way the model can read and react to."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: Type[BaseModel]
    is_write: bool = False


TOOL_SPECS = [
    ToolSpec("record_preferences",
             "Record the applicant's lettings preferences (budget, bedrooms, move-in date, area) as soon as "
             "they mention any. Only include values the applicant actually stated.", RecordPreferencesArgs),
    ToolSpec("search_properties",
             "Search available properties. Filters are hard constraints; query ranks by description.", SearchArgs),
    ToolSpec("get_viewing_slots", "List open viewing slots for one property.", SlotsArgs),
    ToolSpec("book_viewing",
             "Book a viewing slot. WRITE ACTION: the system asks the applicant to confirm before it runs. "
             "Call only when you have property_id, slot_id, the applicant's name and email.", BookArgs, is_write=True),
    ToolSpec("request_handover",
             "Hand the conversation to a human agent when information is missing, you are unsure, "
             "the applicant asks for a person, or a tool keeps failing.", HandoverArgs),
]
REGISTRY = {t.name: t for t in TOOL_SPECS}


def anthropic_tool_defs() -> list[dict]:
    return [{"name": t.name, "description": t.description, "input_schema": t.args_model.model_json_schema()}
            for t in TOOL_SPECS]


class PropertyTools:
    def __init__(self, store: PropertyStore | None = None, retriever: Retriever | None = None,
                 settings: Settings | None = None, inject_failures: dict[str, int] | None = None):
        self.settings = settings or Settings.from_env()
        self.store = store or PropertyStore()
        self.retriever = retriever or build_retriever(self.settings.retrieval_mode, list(self.store.properties.values()))
        self._failures = dict(inject_failures or {})  # tool name -> number of calls to fail (for evals)
        self._handlers: dict[str, Callable] = {
            "search_properties": self.search_properties,
            "get_viewing_slots": self.get_viewing_slots,
            "book_viewing": self.book_viewing,
        }

    @property
    def bookings(self) -> list[Booking]:
        return self.store.bookings

    def execute(self, name: str, raw_args: dict) -> dict:
        spec = REGISTRY.get(name)
        handler = self._handlers.get(name)
        if spec is None or handler is None:
            raise ToolError(f"Unknown tool '{name}'")
        try:
            args = spec.args_model.model_validate(raw_args)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise ToolError(f"Invalid arguments: {problems}") from exc
        if self._failures.get(name, 0) > 0:
            self._failures[name] -= 1
            raise ToolError(f"{name} is temporarily unavailable")
        return handler(args)

    # ---- handlers ----

    def search_properties(self, a: SearchArgs) -> dict:
        cands = [
            p for p in self.store.properties.values()
            if (a.max_rent_pcm is None or p.rent_pcm <= a.max_rent_pcm)
            and (a.bedrooms is None or p.bedrooms == a.bedrooms)
            and (a.available_by is None or p.available_from <= a.available_by)
        ]
        ids = [p.id for p in cands]
        if a.query.strip():
            ranked = self.retriever.rank(a.query, ids, a.limit)
        else:
            ranked = [(p.id, 1.0) for p in sorted(cands, key=lambda p: p.rent_pcm)[: a.limit]]
        top = ranked[0][1] if ranked else 0.0
        matches = []
        for pid, score in ranked:
            p = self.store.get(pid)
            matches.append({"id": p.id, "title": p.title, "area": p.area, "rent_pcm": p.rent_pcm,
                            "bedrooms": p.bedrooms, "available_from": p.available_from.isoformat(),
                            "summary": p.description, "score": round(score, 3)})
        return {"matches": matches, "candidates_after_filters": len(cands),
                "top_score": round(top, 3),
                "low_confidence": (not matches) or top < self.settings.low_confidence_score}

    def get_viewing_slots(self, a: SlotsArgs) -> dict:
        if self.store.get(a.property_id) is None:
            raise ToolError(f"Unknown property {a.property_id}")
        slots = self.store.free_slots(a.property_id)
        return {"property_id": a.property_id,
                "slots": [{"slot_id": s.slot_id, "start": s.start.strftime("%Y-%m-%d %H:%M")} for s in slots]}

    def book_viewing(self, a: BookArgs) -> dict:
        if self.store.get(a.property_id) is None:
            raise ToolError(f"Unknown property {a.property_id}")
        slot = self.store.slot(a.slot_id)
        if slot is None or slot.property_id != a.property_id:
            raise ToolError(f"Slot {a.slot_id} does not exist for {a.property_id}")
        if slot not in self.store.free_slots(a.property_id):
            raise ToolError(f"Slot {a.slot_id} is already booked")
        booking = Booking(f"B{len(self.store.bookings) + 1:04d}", a.property_id, a.slot_id,
                          a.applicant_name, a.applicant_email)
        self.store.bookings.append(booking)
        return {"booking_id": booking.booking_id, "property_id": a.property_id,
                "slot_id": a.slot_id, "start": slot.start.strftime("%Y-%m-%d %H:%M")}
