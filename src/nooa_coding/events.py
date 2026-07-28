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


__all__ = ["SessionEvent", "SessionEventKind"]
