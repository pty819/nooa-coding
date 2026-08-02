"""Ordered model failover without coupling to provider-specific internals."""

from __future__ import annotations

import json
import os
import urllib.request
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

# Fallback context window when nothing else can determine it.
DEFAULT_CONTEXT_WINDOW = 256_000

# Fields providers use to advertise context length in /v1/models payloads.
_CONTEXT_FIELDS = (
    "context_length",
    "context_window",
    "max_input_tokens",
    "max_context_length",
    "max_model_len",
)


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


def _context_from_models_endpoint(model: str, api_base: str | None, api_key: str | None) -> int | None:
    """Best-effort lookup of a model's context length via the provider /models API."""
    if not api_base:
        return None
    base = api_base.rstrip("/")
    candidates = [f"{base}/v1/models", f"{base}/models"]
    headers = {"User-Agent": "nooa-coding/0.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers=headers)  # noqa: S310
            with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            continue
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            continue
        short = model.split("/")[-1]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or entry.get("model") or "")
            if entry_id not in (model, short):
                continue
            for field in _CONTEXT_FIELDS:
                value = entry.get(field)
                if isinstance(value, int) and value > 0:
                    return value
            nested = entry.get("context_length") or entry.get("model_info")
            if isinstance(nested, dict):
                for field in _CONTEXT_FIELDS:
                    value = nested.get(field)
                    if isinstance(value, int) and value > 0:
                        return value
    return None


def resolve_context_window(llm: UnifiedLLM) -> int:
    """Determine the active model's context window size in tokens.

    Resolution order:
    1. The client's own ``context_window`` (config / registry / litellm info).
    2. litellm ``get_model_info`` for known models.
    3. The provider ``/v1/models`` endpoint, if reachable.
    4. ``DEFAULT_CONTEXT_WINDOW`` (256k) as a safe fallback.
    """
    active = getattr(llm, "active", llm)
    declared = getattr(active, "context_window", None)
    if isinstance(declared, int) and declared > 0:
        return declared

    model = str(getattr(active, "model", getattr(llm, "model", "")) or "")
    try:
        info = litellm.get_model_info(model)
        max_tokens = info.get("max_input_tokens") or info.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens
    except Exception:
        pass

    api_base = getattr(active, "api_base", None)
    api_key = getattr(active, "api_key", None)
    from_endpoint = _context_from_models_endpoint(model, api_base, api_key)
    if from_endpoint:
        return from_endpoint

    return DEFAULT_CONTEXT_WINDOW


__all__ = ["DEFAULT_CONTEXT_WINDOW", "FailoverLLM", "build_llm", "resolve_context_window"]
