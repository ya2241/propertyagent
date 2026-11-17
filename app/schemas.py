"""Pydantic models: structured extraction, tool arguments, and the HTTP API."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

REQUIRED_PREFS = ("budget_max_pcm", "bedrooms")


class ApplicantPreferences(BaseModel):
    budget_max_pcm: Optional[int] = Field(None, ge=300, le=20000, description="Maximum monthly rent in GBP")
    bedrooms: Optional[int] = Field(None, ge=0, le=8, description="Number of bedrooms; 0 means studio")
    move_in_date: Optional[date] = Field(None, description="Desired move-in date, ISO format YYYY-MM-DD")
    area: Optional[str] = Field(None, description="Preferred area or neighbourhood, if stated")

    def missing_required(self) -> list[str]:
        return [f for f in REQUIRED_PREFS if getattr(self, f) is None]

    def merge(self, other: "ApplicantPreferences") -> None:
        for field_name in ApplicantPreferences.model_fields:
            value = getattr(other, field_name)
            if value is not None:
                setattr(self, field_name, value)


# ---- tool argument models (also used to generate the tool JSON schemas) ----

class RecordPreferencesArgs(ApplicantPreferences):
    confidence: float = Field(
        1.0, ge=0, le=1,
        description="How confident you are that these values reflect what the applicant meant (0-1). "
                    "Use below 0.5 if the request was vague or contradictory.")


class SearchArgs(BaseModel):
    query: str = Field("", description="Free-text wishes, e.g. 'quiet, pets allowed, good for working from home'")
    max_rent_pcm: Optional[int] = Field(None, ge=0)
    bedrooms: Optional[int] = Field(None, ge=0, le=8)
    available_by: Optional[date] = Field(None, description="Only properties available on or before this date")
    limit: int = Field(3, ge=1, le=5)


class SlotsArgs(BaseModel):
    property_id: str = Field(pattern=r"^P\d{3}$", description="Property ID, e.g. P001")


class BookArgs(BaseModel):
    property_id: str = Field(pattern=r"^P\d{3}$")
    slot_id: str = Field(pattern=r"^P\d{3}-S\d$", description="Slot ID from get_viewing_slots, e.g. P001-S2")
    applicant_name: str = Field(min_length=2)
    applicant_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class HandoverArgs(BaseModel):
    reason: str = Field(min_length=3, description="Why a human should take over")


# ---- HTTP API ----

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = ""
    confirm: Optional[bool] = Field(
        None, description="Answer to a pending confirmation: true executes the action, false cancels it.")

    @model_validator(mode="after")
    def _needs_something(self):
        if not self.message.strip() and self.confirm is None:
            raise ValueError("message or confirm is required")
        return self


class PendingActionOut(BaseModel):
    tool: str
    summary: str
    args: dict


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    status: Literal["ok", "awaiting_confirmation", "handover"]
    preferences: ApplicantPreferences
    pending_action: Optional[PendingActionOut] = None
    handover_reason: Optional[str] = None
    metrics: dict
