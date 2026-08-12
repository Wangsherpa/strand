"""Provider dispatch — route an LLM call to the right backend.

OpenAI gets native structured-output mode (``json_schema``).
Anthropic gets prompted JSON output with schema in the system message.
New providers need only a ``_call_<provider>`` function registered
here, returning a ``_RawResponse``.

``call_llm()`` owns two nested concerns on top of the raw dispatch:
transport retry (rate limits, connection errors, 5xx — see
``strand.llm.retry_defaults``) and structured-output repair (a
response that came back but failed to parse or validate). Both count
toward ``LLMResult.attempts``; only the second sets
``validation_repaired``.

``call_llm()`` is a coroutine — openai/anthropic go over
``httpx.AsyncClient``, litellm via ``litellm.acompletion`` — so an
LLM-heavy workflow doesn't block the event loop for every call's
network round-trip (Phase 8; Phase 7 shipped a synchronous version
built on ``requests``, fine for a serial batch job but a regression
for any concurrent/service-shaped consumer).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, ValidationError

from strand.llm.config import LLMConfig, ModelProvider
from strand.llm.result import LLMResult

# Indirection so tests can replace the sleep used for retry backoff
# without monkeypatching the shared, global asyncio.sleep — mirrors
# strand.core.workflow._sleep's own reasoning (Phase 6), now that
# call_llm is itself a coroutine (Phase 8) rather than a blocking
# synchronous call.
_sleep = asyncio.sleep


def _provider_name(provider: Any) -> str:
    """Normalize LLMConfig.provider to a plain string.

    ``ModelProvider`` subclasses ``str``, but ``str(ModelProvider.OPENAI)``
    gives ``"ModelProvider.OPENAI"``, not ``"openai"`` — Python's
    str+Enum mixin does not use the member's value for ``__str__``.
    Real usage (e.g. invoice_extraction/nodes.py) also passes
    ``provider`` as a plain string directly, which this passes through
    unchanged.
    """
    if isinstance(provider, ModelProvider):
        return provider.value
    return str(provider)


@dataclass
class _RawResponse:
    """What a provider actually returned, before JSON parsing."""

    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


def _build_repair_message(
    original_user_message: str, raw_response: str, error: Exception
) -> str:
    """Augment the user message with the failed response and the
    validation error, asking the model to fix it. Provider-agnostic —
    appended to the user message text rather than added as a new
    conversational turn, so it works identically across all three
    dispatch functions without changing their message-shape contracts.
    """
    return (
        f"{original_user_message}\n\n"
        f"---\n"
        f"Your previous response could not be parsed as valid JSON matching "
        f"the required schema.\n\n"
        f"Your previous response was:\n{raw_response}\n\n"
        f"Validation error:\n{error}\n\n"
        f"Return ONLY a single valid JSON object matching the schema. "
        f"Do not include any explanation, markdown formatting, or "
        f"additional text."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def call_llm(
    config: LLMConfig, user_message: str, output_model: type[BaseModel]
) -> LLMResult:
    """Send *user_message* to the configured provider and return an
    ``LLMResult`` wrapping the validated *output_model* instance.

    Retries transient transport failures per ``config.retry`` (or a
    per-provider default when unset), and makes up to
    ``config.max_output_repair_attempts`` attempts to have the model
    fix a response that failed to parse or validate — both count
    toward the returned ``LLMResult.attempts``.

    Args:
        config: Provider, model, temperature, and credentials.
        user_message: The full user prompt (system prompt is in *config*).
        output_model: A Pydantic model subclass describing the expected shape.

    Returns:
        An ``LLMResult`` wrapping a validated *output_model* instance.

    Raises:
        The last transport error, if retries are exhausted without a
        response. The last validation error, if repair attempts are
        exhausted without a parseable, schema-valid response.
    """
    from strand.llm.retry_defaults import default_retry_policy  # avoid a cycle

    provider = _provider_name(config.provider)
    retry_policy = config.retry or default_retry_policy(provider)
    repair_attempts_left = config.max_output_repair_attempts
    current_user_message = user_message
    validation_repaired = False

    started = time.monotonic()
    attempt = 0

    while True:
        attempt += 1
        try:
            if provider == ModelProvider.OPENAI.value:
                raw = await _call_openai(config, current_user_message, output_model)
            elif provider == ModelProvider.ANTHROPIC.value:
                raw = await _call_anthropic(config, current_user_message, output_model)
            elif provider == ModelProvider.LITELLM.value:
                raw = await _call_litellm(config, current_user_message, output_model)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
        except BaseException as exc:
            if retry_policy.should_retry(exc, attempt):
                await _sleep(retry_policy.backoff_seconds(attempt))
                continue
            raise

        try:
            parsed = output_model.model_validate(json.loads(raw.text))
        except (json.JSONDecodeError, ValidationError) as exc:
            if repair_attempts_left > 0:
                repair_attempts_left -= 1
                validation_repaired = True
                current_user_message = _build_repair_message(
                    user_message, raw.text, exc
                )
                continue
            raise

        latency_ms = (time.monotonic() - started) * 1000
        return LLMResult(
            parsed=parsed,
            raw_text=raw.text,
            provider=provider,
            model=config.model,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            latency_ms=latency_ms,
            attempts=attempt,
            finish_reason=raw.finish_reason,
            validation_repaired=validation_repaired,
        )


# ===================================================================
# Shared schema helper
# ===================================================================


def _ensure_strict(schema: dict) -> None:
    """Add ``additionalProperties: False`` to every object node
    (required by OpenAI/LiteLLM strict mode) — including nested models
    under ``$defs``.

    Pydantic v2 emits nested BaseModel fields as ``$ref`` pointers into
    a top-level ``$defs`` section rather than inlining them, so a walk
    that only follows ``properties``/``anyOf`` never reaches those
    definitions. Fixed here by also recursing into ``$defs`` — found
    while building strand's own test suite (Phase 1), not previously
    documented.
    """
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        for prop in schema.get("properties", {}).values():
            _ensure_strict(prop)
    if "anyOf" in schema:
        for sub in schema["anyOf"]:
            _ensure_strict(sub)
    for defn in schema.get("$defs", {}).values():
        _ensure_strict(defn)


# ===================================================================
# OpenAI
# ===================================================================

_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
_OPENAI_CHAT_PATH = "/chat/completions"


def _openai_url(config: LLMConfig) -> str:
    """Build the chat-completions URL from a *base*, per the OpenAI SDK's
    own convention: base_url includes the ``/v1`` segment, and only
    ``/chat/completions`` is appended — matching Ollama, vLLM, LocalAI,
    and every other OpenAI-compatible proxy's expected base_url shape
    (e.g. ``http://localhost:11434/v1``, this README's own Ollama
    example). Appending a hardcoded ``/v1/chat/completions`` instead
    would double up the ``/v1`` for any base_url following that
    convention — caught before shipping by re-checking the README
    example this fix was meant to satisfy, not by a failing test.
    """
    base = (config.base_url or _OPENAI_DEFAULT_BASE).rstrip("/")
    return f"{base}{_OPENAI_CHAT_PATH}"


async def _call_openai(
    config: LLMConfig, user_message: str, output_model: type[BaseModel]
) -> _RawResponse:
    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    json_schema = output_model.model_json_schema()
    _ensure_strict(json_schema)

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": user_message},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "strict": True,
                "schema": json_schema,
            },
        },
        "temperature": config.temperature,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _openai_url(config),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=config.timeout_s,
        )
    resp.raise_for_status()
    body = resp.json()
    choice = body["choices"][0]
    usage = body.get("usage") or {}

    return _RawResponse(
        text=choice["message"]["content"],
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        finish_reason=choice.get("finish_reason"),
    )


# ===================================================================
# Anthropic
# ===================================================================

_ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com"
_ANTHROPIC_PATH = "/v1/messages"


def _anthropic_url(config: LLMConfig) -> str:
    base = (config.base_url or _ANTHROPIC_DEFAULT_BASE).rstrip("/")
    return f"{base}{_ANTHROPIC_PATH}"


async def _call_anthropic(
    config: LLMConfig, user_message: str, output_model: type[BaseModel]
) -> _RawResponse:
    api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
    json_schema = output_model.model_json_schema()

    # Anthropic doesn't have native structured-output mode — we embed
    # the expected JSON schema in the system prompt.
    schema_block = json.dumps(json_schema, indent=2)
    system = (
        f"{config.system_prompt}\n\n"
        f"You MUST respond with a single JSON object matching this schema. "
        f"Do not include any text outside the JSON object.\n\n"
        f"Expected schema:\n```json\n{schema_block}\n```"
    )

    payload: Dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": config.temperature,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _anthropic_url(config),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.timeout_s,
        )
    resp.raise_for_status()
    body = resp.json()
    usage = body.get("usage") or {}

    # Anthropic returns text; strip markdown code fences if present.
    raw = body["content"][0]["text"]
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]

    return _RawResponse(
        text=raw,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        finish_reason=body.get("stop_reason"),
    )


# ===================================================================
# LiteLLM — unified interface to 100+ providers
# ===================================================================


async def _call_litellm(
    config: LLMConfig, user_message: str, output_model: type[BaseModel]
) -> _RawResponse:
    """Route through LiteLLM for any of its 100+ supported providers.

    The model string follows LiteLLM convention — e.g. ``"gpt-4o-mini"``,
    ``"anthropic/claude-sonnet-4-6"``, ``"ollama/llama3.2"``.

    Structured output is handled via ``response_format``, which LiteLLM
    translates per provider automatically. Retry is handled by
    call_llm()'s own loop, not litellm's own num_retries/fallbacks — so
    strand's retry telemetry (attempts, backoff) is consistent across
    every provider, not just the ones called directly over HTTP.

    Uses ``litellm.acompletion`` (not ``litellm.completion``) so this
    path doesn't block the event loop either, matching the direct-HTTP
    providers above (Phase 8).
    """
    import litellm  # optional dependency — only imported when this path is taken

    json_schema = output_model.model_json_schema()
    _ensure_strict(json_schema)

    response = await litellm.acompletion(
        model=config.model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "strict": True,
                "schema": json_schema,
            },
        },
        temperature=config.temperature,
        api_key=config.api_key,
        api_base=config.base_url,
        max_tokens=config.max_tokens,
        timeout=config.timeout_s,
    )

    choice = response.choices[0]
    usage = getattr(response, "usage", None)

    return _RawResponse(
        text=choice.message.content,
        input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        finish_reason=getattr(choice, "finish_reason", None),
    )
