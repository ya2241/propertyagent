"""Per-turn and per-conversation latency / token / cost logging."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("propertyagent.telemetry")


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Prices are per million tokens and configurable; defaults are placeholders -
    set PRICE_INPUT_PER_MTOK / PRICE_OUTPUT_PER_MTOK to your model's real rates."""
    p_in = float(os.getenv("PRICE_INPUT_PER_MTOK", "3"))
    p_out = float(os.getenv("PRICE_OUTPUT_PER_MTOK", "15"))
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


@dataclass
class Metrics:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    cost_usd: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    grounding_issues: int = 0

    def add(self, other: "Metrics") -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["latency_ms"] = round(d["latency_ms"], 1)
        d["llm_latency_ms"] = round(d["llm_latency_ms"], 1)
        d["cost_usd"] = round(d["cost_usd"], 6)
        return d


def log_turn(conversation_id: str, status: str, turn: Metrics, totals: Metrics,
             handover_reason: str | None = None) -> None:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conversation_id": conversation_id,
        "status": status,
        "handover_reason": handover_reason,
        "turn": turn.to_dict(),
        "totals": totals.to_dict(),
    }
    line = json.dumps(record)
    log.info(line)
    path = os.getenv("PROPERTYAGENT_LOG", "logs/conversations.jsonl")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line + "\n")
