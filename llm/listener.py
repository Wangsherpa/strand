"""LLMListener — LLM-specific observability, composed alongside
strand.core's WorkflowListener rather than living on it.

Decided in Phase 5, before this protocol existed: strand.core's
WorkflowListener stays purely graph-shaped, with no LLM vocabulary in
its signatures. A consumer that wants both graph-level and LLM-level
tracing implements both protocols on one class — nothing here requires
subclassing either one specifically, since dispatch is duck-typed
(matching strand.core's own WorkflowListener dispatch).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from strand.core.run_context import RunContext
    from strand.llm.result import LLMResult

logger = logging.getLogger(__name__)


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


def notify_llm_call(
    run: Optional["RunContext"],
    node_id: str,
    result: Optional["LLMResult"],
    error: Optional[BaseException],
) -> None:
    """Dispatch ``on_llm_call`` to every registered listener that
    implements it, defensively — mirrors
    ``strand.core.Workflow._notify``'s guarantee that a raising
    listener never affects the node's own outcome. Not shared code
    with strand.core by design: strand.core doesn't know LLMListener
    exists, and shouldn't need to.
    """
    if run is None:
        return
    for listener in run.listeners:
        method = getattr(listener, "on_llm_call", None)
        if method is None:
            continue
        try:
            method(run, node_id, result, error)
        except Exception:
            logger.exception(
                "Listener %r raised in on_llm_call() — ignoring.", listener
            )
