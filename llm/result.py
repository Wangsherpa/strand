"""LLMResult — the outcome of one call_llm() invocation.

A plain dataclass, like LLMConfig and RetryPolicy — no serialization
requirement, so no need for pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel


@dataclass
class LLMResult:
    """Everything worth knowing about one call_llm() call.

    Attributes:
        parsed: The validated output_model instance — what callers
            actually want. LLMNode.process() unwraps this for
            save_output(); call_llm() itself no longer returns a bare
            model (breaking change from before this phase).
        raw_text: The exact text the provider returned, before JSON
            parsing — useful for debugging a validation failure.
        provider: Normalized provider name ("openai"/"anthropic"/
            "litellm"), regardless of whether LLMConfig.provider was
            passed as the ModelProvider enum or a plain string.
        model: The model name from LLMConfig, echoed back for
            convenience (so a listener doesn't need the original
            LLMConfig in hand).
        input_tokens: Prompt/input token count, if the provider
            reported one.
        output_tokens: Completion/output token count, if reported.
        latency_ms: Wall-clock time across every attempt this call
            took, including transport retries and repair attempts —
            not just the final successful one.
        attempts: Total provider calls made, including transport
            retries and structured-output repair attempts. ``1`` means
            it succeeded on the first try.
        finish_reason: The provider's own reason the response ended
            (e.g. "stop", "length", "end_turn"), if reported.
        validation_repaired: True if at least one repair attempt was
            needed — i.e. a response failed JSON/schema validation and
            had to be corrected by asking the model again. Distinct
            from ``attempts > 1`` alone, which could also mean pure
            transport retries with no repair involved.
    """

    parsed: BaseModel
    raw_text: str
    provider: str
    model: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: float
    attempts: int
    finish_reason: Optional[str]
    validation_repaired: bool = False
