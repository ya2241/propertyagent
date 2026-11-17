"""FastAPI service. Run: uvicorn app.main:app --reload  (PROPERTYAGENT_LLM=mock for no API key)"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from .agent import ConversationStore, PropertyAgent
from .config import Settings
from .llm import AnthropicLLM, RuleBasedLLM
from .schemas import ChatRequest, ChatResponse, PendingActionOut
from .tools import PropertyTools

app = FastAPI(title="PropertyAgent", description="LLM agent for lettings enquiries")


@lru_cache
def get_store() -> ConversationStore:
    return ConversationStore()


@lru_cache
def get_agent() -> PropertyAgent:
    settings = Settings.from_env()
    llm = RuleBasedLLM() if os.getenv("PROPERTYAGENT_LLM") == "mock" else AnthropicLLM(settings.model)
    return PropertyAgent(llm, PropertyTools(settings=settings), settings)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, agent: PropertyAgent = Depends(get_agent), store: ConversationStore = Depends(get_store)):
    conv = store.get(req.conversation_id)
    result = agent.handle(conv, req.message, req.confirm)
    pending = result.pending
    return ChatResponse(
        conversation_id=conv.id,
        reply=result.reply,
        status=result.status,
        preferences=conv.prefs,
        pending_action=PendingActionOut(tool=pending.name, summary=pending.summary, args=pending.args) if pending else None,
        handover_reason=result.handover_reason,
        metrics={"turn": result.turn.to_dict(), "conversation": result.totals.to_dict()},
    )


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, store: ConversationStore = Depends(get_store)):
    conv = store.peek(conversation_id)
    if conv is None:
        raise HTTPException(404, "Unknown conversation")
    return {"id": conv.id, "status": conv.status, "handover_reason": conv.handover_reason,
            "preferences": conv.prefs, "metrics": conv.totals.to_dict()}
