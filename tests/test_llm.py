from __future__ import annotations

from typing import Any

import pytest
from nooa.unifiedllm import LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel

from nooa_coding.llm import FailoverLLM


class StubLLM(UnifiedLLM):
    def __init__(self, model: str, *, error: Exception | None = None) -> None:
        super().__init__(model)
        self.error = error
        self.calls = 0

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(
            raw_response=None,
            content=self.model,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": self.model},
        )

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.call(messages, tools, output_model, **kwargs)


@pytest.mark.asyncio
async def test_model_failover_switches_to_next_client() -> None:
    primary = StubLLM("primary", error=TimeoutError("down"))
    secondary = StubLLM("secondary")
    transitions = []
    llm = FailoverLLM(
        [primary, secondary],
        on_failover=lambda source, target, error: transitions.append((source, target, error)),
    )

    response = await llm.acall([])

    assert response.content == "secondary"
    assert llm.active is secondary
    assert transitions[0][:2] == ("primary", "secondary")


@pytest.mark.asyncio
async def test_model_failover_reports_aggregate_failure() -> None:
    llm = FailoverLLM(
        [StubLLM("one", error=RuntimeError("one")), StubLLM("two", error=RuntimeError("two"))]
    )
    with pytest.raises(RuntimeError, match="all configured models failed"):
        await llm.acall([])
