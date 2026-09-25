"""LLMNode — a ``strand.core`` Node that calls an LLM.

Subclass this for any node that needs language-model intelligence.
You only need to implement two methods:

* ``get_llm_config()`` — return an ``LLMConfig``
* ``build_user_message(ctx)`` — return the user-facing prompt string

The base class handles the API call, JSON parsing, and ``save_output()``.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from typing import ClassVar, Optional

from pydantic import BaseModel

from strand.core import Node, TaskContext
from strand.core.retry import RetryPolicy
from strand.llm.client import call_llm
from strand.llm.config import LLMConfig
from strand.llm.listener import notify_llm_call
from strand.llm.result import LLMResult


class LLMNode(Node):
    """Base class for LLM-powered workflow nodes.

    Subclasses must implement:

    * ``get_llm_config() -> LLMConfig``
    * ``build_user_message(ctx: TaskContext) -> str``

    The inner ``OutputType`` class defines the shape the LLM should return.

    Retry interplay: when the node's ``NodeConfig`` has ``retry`` set,
    the engine owns retries for this node and ``call_llm()``'s internal
    transport retry is disabled for the call (the two layers would
    otherwise multiply into ``engine_attempts x transport_attempts``
    provider calls); the repair budget is unaffected. When the
    ``NodeConfig`` has no retry, ``LLMConfig.retry`` (or the provider
    default) applies as documented.

    Example::

        class SpamFilterNode(LLMNode):
            class OutputType(LLMNode.OutputType):
                is_spam: bool
                confidence: str

            def get_llm_config(self) -> LLMConfig:
                return LLMConfig(model="gpt-4o-mini")

            async def build_user_message(self, ctx: TaskContext) -> str:
                return f"Classify this email:\\nSubject: {ctx.event.subject}\\n{ctx.event.body}"
    """

    kind: ClassVar[str] = "llm"

    class OutputType(Node.OutputType):
        """Override in subclasses to define the LLM's structured output shape."""
        pass

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def get_llm_config(self) -> LLMConfig:
        """Return the provider, model, and temperature for this node."""
        ...

    @abstractmethod
    async def build_user_message(self, ctx: TaskContext) -> str:
        """Build the user prompt from the current task context.

        The system prompt is taken from ``get_llm_config().system_prompt``.
        """
        ...

    # ------------------------------------------------------------------
    # Framework
    # ------------------------------------------------------------------

    async def process(self, ctx: TaskContext) -> TaskContext:
        result: Optional[LLMResult] = None
        error: Optional[BaseException] = None
        try:
            config = self.get_llm_config()
            user_msg = await self.build_user_message(ctx)
            result = await call_llm(
                self._without_transport_retry(config, ctx), user_msg, self.OutputType
            )
        except BaseException as exc:
            error = exc
            raise
        finally:
            # fires exactly once per process() call — on success and on
            # every failure path, including get_llm_config()/
            # build_user_message() raising before any provider call.
            await notify_llm_call(ctx._run_context, self.node_name, result, error)

        self.save_output(result.parsed)
        return ctx

    def _without_transport_retry(self, config: LLMConfig, ctx: TaskContext) -> LLMConfig:
        """When the ENGINE retries this node (``NodeConfig.retry``),
        disable call_llm()'s internal transport retry.

        Otherwise the two layers multiply: a node with
        ``RetryPolicy(max_attempts=3)`` whose transport always fails
        would make 3 engine attempts x 3 transport attempts = 9 provider
        calls for what the user configured as one 3-attempt policy. The
        engine-level policy wins; the LLM layer keeps only its repair
        budget (still bounded by the engine policy's ``max_attempts``
        total, which counts every provider call).
        """
        run = ctx._run_context
        if run is None:
            return config
        nc = run.node_configs.get(self.node_name)
        if nc is None or nc.retry is None:
            return config
        return replace(config, retry=RetryPolicy(max_attempts=1))
