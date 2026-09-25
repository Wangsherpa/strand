"""LLMListener — LLM-specific observability, composed alongside
strand.core's WorkflowListener rather than living on it.

strand.core's WorkflowListener stays purely graph-shaped, with no LLM
vocabulary in its signatures. A consumer that wants both graph-level
and LLM-level tracing implements both protocols on one class — nothing
here requires subclassing either one specifically, since dispatch is
duck-typed and shared with strand.core's own dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from strand.core.listener import notify_listeners

if TYPE_CHECKING:
    from strand.core.run_context import RunContext
    from strand.llm.result import LLMResult


class LLMListener:
    """Base class for observing call_llm() outcomes.

    Override ``on_llm_call``; there's only one hook. Register the same
    way as any WorkflowListener — pass it in
    ``Workflow(listeners=[...])``, alongside or instead of a
    WorkflowListener. LLMNode dispatches to whichever registered
    listeners implement ``on_llm_call``, ignoring the rest — no
    subclassing requirement, same duck-typed dispatch as
    strand.core's.
    """

    def on_llm_call(
        self,
        run: "RunContext",
        node_id: str,
        result: Optional["LLMResult"],
        error: Optional[BaseException],
    ) -> None:
        """Called once per LLMNode.process() call — i.e. once per
        node execution, not once per internal HTTP attempt. *result*
        already aggregates every transport retry and repair attempt
        that call needed (see ``LLMResult.attempts``).

        *result* is ``None`` if the call failed permanently (see
        *error*); *error* is ``None`` if it succeeded.
        """


async def notify_llm_call(
    run: Optional["RunContext"],
    node_id: str,
    result: Optional["LLMResult"],
    error: Optional[BaseException],
) -> None:
    """Dispatch ``on_llm_call`` to every registered listener that
    implements it, defensively.

    Delegates to strand.core's shared ``notify_listeners`` — the one
    place the swallow-and-log guarantee (and sync/async hook support)
    is enforced — while strand.core itself still knows nothing about
    LLMListener.
    """
    await notify_listeners(run, "on_llm_call", node_id, result, error)
