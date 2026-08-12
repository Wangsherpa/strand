"""strand.llm — Reusable LLM integration layer for ``strand.core`` workflows.

Provides:

* ``LLMConfig`` — dataclass for provider, model, credentials, timeout,
  retry policy, and structured-output repair budget.
* ``ModelProvider`` — enum of supported backends (OpenAI, Anthropic, …).
* ``LLMNode`` — a ``Node`` subclass that handles the LLM call, JSON
  parsing, and ``save_output()`` so you only write the prompt and
  output schema. Dispatches ``on_llm_call`` to registered
  ``LLMListener``s.
* ``LLMResult`` — the outcome of one ``call_llm()`` call: the parsed
  output plus tokens, latency, attempts, and whether repair was needed.
* ``LLMListener`` — LLM-specific observability, composed alongside
  ``strand.core.WorkflowListener`` rather than living on it.
* ``call_llm()`` — low-level async dispatch function (useful outside
  nodes; ``await`` it directly, or run it via ``asyncio.run()``/
  ``asyncio.get_event_loop().run_until_complete()`` from sync code).
"""

from strand.llm.client import call_llm
from strand.llm.config import LLMConfig, ModelProvider
from strand.llm.listener import LLMListener
from strand.llm.node import LLMNode
from strand.llm.result import LLMResult

__all__ = [
    "call_llm",
    "LLMConfig",
    "LLMListener",
    "LLMNode",
    "LLMResult",
    "ModelProvider",
]
