"""Default RetryPolicy per provider — the knowledge strand.core can't have.

strand.core's RetryPolicy takes a bare predicate and knows nothing
about what's transient for any given API. That knowledge belongs one
layer up, here, where depending on `httpx` (already a strand.llm
dependency) and, for litellm, on litellm's own exception hierarchy
(imported lazily, mirroring client.py's own lazy import) is fine.

Applied only when LLMConfig.retry is left unset — an explicit
RetryPolicy always wins.
"""

from __future__ import annotations

import httpx

from strand.core.retry import RetryPolicy
from strand.llm.config import ModelProvider

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _http_retry_on(error: BaseException) -> bool:
    """Shared predicate for the openai/anthropic providers — both call
    the API directly over HTTP via `httpx`, so the same transient-
    failure signals apply to both. `httpx.TimeoutException` covers
    connect/read/write/pool timeouts; `httpx.ConnectError` covers a
    refused or unreachable connection; `httpx.ReadError`/
    `httpx.WriteError`/`httpx.RemoteProtocolError` cover connections
    that die mid-transfer."""
    if isinstance(
        error,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        return status in _RETRYABLE_STATUS_CODES
    return False


def _litellm_retry_on(error: BaseException) -> bool:
    import litellm  # noqa: PLC0415 — mirrors client.py's own lazy import

    return isinstance(
        error,
        (
            litellm.RateLimitError,
            litellm.APIConnectionError,
            litellm.Timeout,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
        ),
    )


def default_retry_policy(provider: str) -> RetryPolicy:
    """The default RetryPolicy for *provider* (a normalized string —
    see ``_provider_name`` in client.py), used when LLMConfig.retry is
    left unset. 3 attempts, exponential backoff from 2 seconds —
    matches the retry shape LLM API rate limits generally call for.
    The 3 attempts bound the TOTAL provider calls, repair attempts
    included (see ``RetryPolicy.max_attempts``).
    """
    retry_on = (
        _litellm_retry_on if provider == ModelProvider.LITELLM.value else _http_retry_on
    )
    return RetryPolicy(max_attempts=3, backoff_base=2.0, retry_on=retry_on)
