"""Runtime settings, read from environment so evals can sweep them."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "claude-sonnet-5-5"
    prompt_version: str = "v3"
    retrieval_mode: str = "vector"  # "vector" | "keyword"
    max_steps: int = 6  # LLM calls per user turn
    low_confidence_score: float = 0.12  # vector/keyword score below this = weak match

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.getenv("PROPERTYAGENT_MODEL", cls.model),
            prompt_version=os.getenv("PROMPT_VERSION", cls.prompt_version),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", cls.retrieval_mode),
        )
