"""Ordered model failover without coupling to provider-specific internals."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import litellm
from nooa.unifiedllm import LLMResponse, Tool, UnifiedLLM, get_llm_client
from pydantic import BaseModel

from .config import ModelEndpoint
from .system_prompt import bind_active_model

# Suppress litellm's noisy "Provider List" and debug banners for custom models.
litellm.suppress_debug_info = True

FailoverSink = Callable[[str, str, str], None]


class FailoverLLM(UnifiedLLM):
    """Try configured NOOA clients in order and remember the last healthy one."""

    def __init__(
        self,
        clients: list[UnifiedLLM],
        *,
        on_failover: FailoverSink | None = None,
    ) -> None:
        if not clients:
            raise ValueError("at least one LLM client is required")
        self.clients = clients
        self.active_index = 0
        self._on_failover = on_failover
        super().__init__(model=clients[0].model)

    @property
    def active(self) -> UnifiedLLM:
        return self.clients[self.active_index]

    @property
    def context_window(self) -> int | None:
        return self.active.context_window

    def count_tokens(self, text: str) -> int:
        return self.active.count_tokens(text)

    def get_model_info(self) -> Any:
        return self.active.get_model_info()

    def _ordered_indices(self) -> list[int]:
        return list(range(self.active_index, len(self.clients))) + list(range(0, self.active_index))

    def _failed(self, index: int, exc: Exception) -> None:
        next_index = (index + 1) % len(self.clients)
        if self._on_failover is not None:
            self._on_failover(
                self.clients[index].model,
                self.clients[next_index].model,
                f"{type(exc).__name__}: {exc}",
            )

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        errors: list[Exception] = []
        order = self._ordered_indices()
        for position, index in enumerate(order):
            client = self.clients[index]
            self.active_index = index
            self.model = client.model
            try:
                bound_messages = bind_active_model(messages, client.model)
                response = client.call(bound_messages, tools, output_model, **kwargs)
                return response
            except Exception as exc:
                errors.append(exc)
                if position + 1 < len(order):
                    self._failed(index, exc)
        raise RuntimeError(
            f"all configured models failed: {[str(error) for error in errors]}"
        ) from errors[-1]

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        errors: list[Exception] = []
        order = self._ordered_indices()
        for position, index in enumerate(order):
            client = self.clients[index]
            self.active_index = index
            self.model = client.model
            try:
                bound_messages = bind_active_model(messages, client.model)
                response = await client.acall(bound_messages, tools, output_model, **kwargs)
                return response
            except Exception as exc:
                errors.append(exc)
                if position + 1 < len(order):
                    self._failed(index, exc)
        raise RuntimeError(
            f"all configured models failed: {[str(error) for error in errors]}"
        ) from errors[-1]

    def close(self) -> None:
        for client in self.clients:
            client.close()

    async def aclose(self) -> None:
        for client in self.clients:
            await client.aclose()


def build_llm(
    endpoints: tuple[ModelEndpoint, ...],
    *,
    on_failover: FailoverSink | None = None,
) -> FailoverLLM:
    clients: list[UnifiedLLM] = []
    for endpoint in endpoints:
        kwargs: dict[str, Any] = {}
        if endpoint.api_base:
            kwargs["api_base"] = endpoint.api_base
        if endpoint.api_key:
            kwargs["api_key"] = endpoint.api_key
        elif endpoint.api_key_env:
            api_key = os.environ.get(endpoint.api_key_env)
            if not api_key:
                raise ValueError(
                    f"missing model credential environment variable: {endpoint.api_key_env}"
                )
            kwargs["api_key"] = api_key
        client = get_llm_client(endpoint.name, client_type=endpoint.client_type, **kwargs)
        clients.append(client)
    return FailoverLLM(clients, on_failover=on_failover)


__all__ = ["FailoverLLM", "build_llm"]
