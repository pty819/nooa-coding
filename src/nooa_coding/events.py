"""Durable host-level events streamed by an :class:`AgentSession`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SessionEventKind = Literal[
    "session",
    "agent",
    "message",
    "tool",
    "command",
    "approval",
    "checkpoint",
    "model_failover",
    "thinking",
    "usage",
    "error",
]


class SessionEvent(BaseModel):
    """Stable event envelope used by CLI, replay, and future frontends."""

    sequence: int = Field(ge=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str
    kind: SessionEventKind
    name: str
    data: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Cumulative token usage statistics for one session."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_calls: int = 0

    def add(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cached_tokens += cached_tokens
        self.total_reasoning_tokens += reasoning_tokens
        self.total_cost_usd += cost_usd
        self.llm_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


__all__ = ["SessionEvent", "SessionEventKind", "TokenUsage"]
