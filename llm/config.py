"""LLM configuration — provider, model, and connection settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from strand.core.retry import RetryPolicy


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"


@dataclass
class LLMConfig:
    """Everything an LLM node needs to call a model.

    Example::

        LLMConfig(model="gpt-4o-mini", provider="openai")
        LLMConfig(model="claude-sonnet-4-6", provider="anthropic", temperature=0.3)

    Attributes:
        model: The model name / ID understood by the provider.
        provider: One of ``ModelProvider`` values.
        system_prompt: System-level instructions for every call.
        temperature: Sampling temperature (0.0 = deterministic).
        api_key: API key. If ``None``, falls back to the standard
            env var for the chosen provider (``OPENAI_API_KEY`` or
            ``ANTHROPIC_API_KEY``).
        base_url: Custom endpoint base (Ollama, Azure, proxies) — a
            *base*, not a full endpoint: the provider's own path
            (e.g. ``/v1/chat/completions``) is appended to it. A value
            that is already a full endpoint is also accepted (the path
            is not doubled). When ``None`` the provider's own default
            base is used.
        timeout_s: Per-request timeout in seconds, replacing the
            previously hardcoded 120.
        retry: Retry policy for transient transport failures (rate
            limits, connection errors, 5xx). When ``None`` (the
            default), a sensible per-provider default applies — see
            ``strand.llm.retry_defaults`` — rather than "no retry":
            LLM calls are I/O-heavy enough that transient failures are
            the expected case, unlike ``NodeConfig.retry``'s
            no-retry-unless-asked default. ``max_attempts`` bounds the
            TOTAL number of provider calls — structured-output repair
            attempts count toward it too.
        max_output_repair_attempts: How many times to ask the model to
            fix a response that failed JSON parsing or schema
            validation, before giving up. ``1`` (the default) means
            one repair attempt; ``0`` disables repair entirely. Repair
            attempts also count toward ``retry.max_attempts``'s total.
    """

    model: str
    provider: ModelProvider = ModelProvider.OPENAI
    system_prompt: str = ""
    temperature: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    timeout_s: float = 120.0
    retry: Optional[RetryPolicy] = None
    max_output_repair_attempts: int = 1

    def __post_init__(self) -> None:
        if self.max_output_repair_attempts < 0:
            raise ValueError(
                f"max_output_repair_attempts must be >= 0, "
                f"got {self.max_output_repair_attempts}"
            )
        if self.retry is not None and not isinstance(self.retry, RetryPolicy):
            raise TypeError(
                f"retry must be a RetryPolicy, got {type(self.retry).__name__}"
            )
