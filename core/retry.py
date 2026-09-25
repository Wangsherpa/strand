"""RetryPolicy — node-level retry, for any node kind, not just LLM ones.

``strand.core`` depends on nothing beyond pydantic and the stdlib, so
it cannot import ``openai`` (or anything else) to know that, say,
``RateLimitError`` is transient. ``retry_on`` is exactly that
knowledge, supplied from outside: a tuple of exception types, or a
predicate function. ``strand.llm`` ships sensible per-provider
defaults built on top of this same policy — nothing provider-specific
lives here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Tuple, Type, Union

RetryPredicate = Union[
    Type[BaseException],
    Tuple[Type[BaseException], ...],
    Callable[[BaseException], bool],
]


@dataclass
class RetryPolicy:
    """Node-level retry configuration.

    Attributes:
        max_attempts: Total attempts allowed, including the first —
            ``1`` (the default) means "no retry." Every provider call
            counts toward it, including structured-output repair
            attempts (``strand.llm``'s ``max_output_repair_attempts``),
            so the total spend of one policy is always bounded by this
            number.
        backoff_base: Seconds before the second attempt; doubles each
            attempt after that (exponential backoff), capped at
            ``backoff_max``.
        backoff_max: Upper bound on the computed delay, in seconds.
        jitter: If ``True``, the actual delay is drawn uniformly from
            ``[0, computed_delay]`` (AWS's "full jitter") rather than
            being the computed delay exactly — spreads out retries
            from many concurrent failures instead of having them all
            retry in lockstep.
        retry_on: Exception type(s), or a predicate called with the
            raised exception, deciding whether it's worth retrying.
            Defaults to ``(Exception,)`` — broad, since attaching a
            ``RetryPolicy`` at all is already an opt-in decision.
            ``asyncio.TimeoutError``/``TimeoutError`` are ``Exception``
            subclasses, so a timeout is retryable by default without
            special-casing; ``asyncio.CancelledError`` is a
            ``BaseException`` subclass, so it is never retried by
            default regardless of ``max_attempts``.
    """

    max_attempts: int = 1
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter: bool = False
    retry_on: RetryPredicate = field(default=(Exception,))

    def __post_init__(self) -> None:
        """Validate configuration at construction time.

        ``RetryPolicy`` is a plain dataclass, and pydantic does not
        revalidate stdlib-dataclass instances embedded in ``NodeConfig``
        or ``LLMConfig`` — so nothing else catches a mis-typed
        ``retry_on`` (e.g. a list instead of a tuple), which would
        otherwise blow up with a ``TypeError`` from ``isinstance()`` in
        the middle of a workflow run.
        """
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise TypeError(
                f"max_attempts must be an int, got {type(self.max_attempts).__name__}"
            )
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1 (1 means 'no retry'), got {self.max_attempts}"
            )
        if not RetryPolicy._valid_retry_on(self.retry_on):
            raise TypeError(
                "retry_on must be an exception type, a tuple of exception types, "
                f"or a predicate callable — got {self.retry_on!r}"
            )
        if self.backoff_base < 0 or self.backoff_max < 0:
            raise ValueError(
                f"backoff_base/backoff_max must be >= 0, got "
                f"{self.backoff_base}/{self.backoff_max}"
            )

    @staticmethod
    def _valid_retry_on(retry_on: RetryPredicate) -> bool:
        if isinstance(retry_on, type):
            return issubclass(retry_on, BaseException)
        if isinstance(retry_on, tuple):
            return all(isinstance(t, type) and issubclass(t, BaseException) for t in retry_on)
        return callable(retry_on)

    def should_retry(self, error: BaseException, attempt: int) -> bool:
        """Whether *attempt* (the one that just failed, 1-indexed)
        should be followed by another.

        The ``max_attempts`` boundary is checked here, not by the
        caller — callers only need to loop until this returns
        ``False``.
        """
        if attempt >= self.max_attempts:
            return False
        if callable(self.retry_on) and not isinstance(self.retry_on, type):
            return bool(self.retry_on(error))
        return isinstance(error, self.retry_on)

    def backoff_seconds(self, attempt: int) -> float:
        """Delay before the attempt after *attempt* (1-indexed)."""
        delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
        if self.jitter:
            delay = random.uniform(0, delay)
        return delay
