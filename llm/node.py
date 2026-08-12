"""LLMNode — a ``strand.core`` Node that calls an LLM.

Subclass this for any node that needs language-model intelligence.
You only need to implement two methods:

* ``get_llm_config()`` — return an ``LLMConfig``
* ``build_user_message(ctx)`` — return the user-facing prompt string

The base class handles the API call, JSON parsing, and ``save_output()``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Optional

from pydantic import BaseModel

from strand.core import Node, TaskContext
from strand.llm.client import call_llm
from strand.llm.config import LLMConfig
from strand.llm.listener import notify_llm_call


class LLMNode(Node):
    """Base class for LLM-powered workflow nodes.

    Subclasses must implement:

    * ``get_llm_config() -> LLMConfig``
    * ``build_user_message(ctx: TaskContext) -> str``

    The inner ``OutputType`` class defines the shape the LLM should return.

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
        config = self.get_llm_config()
        user_msg = await self.build_user_message(ctx)

        result = None
        error: Optional[BaseException] = None
        try:
            result = await call_llm(config, user_msg, self.OutputType)
        except BaseException as exc:
            error = exc
            raise
        finally:
            notify_llm_call(ctx._run_context, self.node_name, result, error)

        self.save_output(result.parsed)
        return ctx
