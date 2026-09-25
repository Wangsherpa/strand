"""WorkflowListener — the observability extension point for strand.core.

Subclass and override only the hooks you care about; every method
defaults to a no-op. ``Workflow`` dispatches every hook through
:func:`notify_listeners`, which guarantees that an exception raised
from a listener is logged and swallowed, never allowed to affect the
workflow's own outcome or control flow — tracing must never become a
new failure mode. ``asyncio.CancelledError`` is the one exception: it
is re-raised, because a listener must not swallow a cancellation.

Hooks may be plain methods or coroutines — the engine awaits either,
so an ``async def`` hook receives its events instead of having them
silently dropped.

Emission only. No I/O, no persistence, and no redaction policy lives
here or in ``Workflow``'s dispatch — a listener implementation decides
what (if anything) to do with what it's handed. ``strand.trace`` is one
such implementation, provided separately and entirely optional.

LLM-specific telemetry (token usage, latency, provider/model) is
deliberately not part of this protocol — see ``strand.llm``'s own
listener. This protocol only knows about graph execution.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Optional

from strand.core.run_context import RunContext

logger = logging.getLogger(__name__)


async def notify_listeners(
    run: Optional[RunContext], method_name: str, *args: Any
) -> None:
    """Duck-typed, defensive dispatch of *method_name* to every listener
    registered on *run*.

    The single place the "a raising listener never affects the run"
    guarantee is enforced — ``Workflow._notify`` and ``strand.llm``'s
    ``notify_llm_call`` both delegate here. Each hook is called as
    ``method(run, *args)``; if it returns a coroutine, that coroutine is
    awaited. ``asyncio.CancelledError`` raised by a hook is re-raised
    (never swallowed); anything else is logged and swallowed.
    """
    if run is None:
        return
    for listener in run.listeners:
        method = getattr(listener, method_name, None)
        if method is None:
            continue
        try:
            result = method(run, *args)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except BaseException:
            logger.exception(
                "Listener %r raised in %s() — ignoring.", listener, method_name
            )


class WorkflowListener:
    """Base class for observing workflow execution.

    Register instances via ``Workflow(listeners=[...])``. Registered
    listeners are inherited by nested workflows through ``RunContext``,
    so a listener registered on the outermost workflow also observes
    every nested child's node executions, under the same execution_id.

    Every hook may be a plain method or a coroutine — the engine awaits
    either (see :func:`notify_listeners`).
    """

    def on_workflow_start(self, run: RunContext, event: Any) -> None:
        """Called once when a workflow begins traversing its graph.

        Fires for every nested workflow too, not just the outermost —
        each has its own ``workflow_name``/``workflow_version`` on
        *run*, but all of them share one ``execution_id`` for the
        whole composition.
        """

    def on_workflow_end(
        self,
        run: RunContext,
        status: str,
        duration_ms: float,
        error: Optional[BaseException],
    ) -> None:
        """Called once when a workflow's own traversal finishes.

        *status* is ``"completed"`` or ``"error"``. *error* is the
        exception that propagated, if any — ``Workflow`` always
        re-raises it after this fires; nothing here suppresses it.
        """

    def on_node_start(
        self, run: RunContext, node_id: str, node_kind: str, attempt: int
    ) -> None:
        """Called before the node (or router) at *node_id* runs.

        *node_kind* is the registered class's ``Node.kind`` —
        ``"node"`` by default, ``"router"`` for ``BaseRouter``
        subclasses, ``"llm"`` for ``LLMNode`` subclasses
        (``strand.llm``). *attempt* is 1-indexed: ``1`` on the first
        try, incrementing per retry attempt.
        """

    def on_node_end(
        self,
        run: RunContext,
        node_id: str,
        node_kind: str,
        status: str,
        duration_ms: float,
        output: Any,
        attempt: int,
    ) -> None:
        """Called after a node completes without raising.

        *output* is whatever the node stored via ``save_output()``
        (``None`` if it didn't — not itself an error). This is the raw
        output object; ``Workflow`` makes no decision about capturing,
        hashing, or redacting it. That policy belongs to the listener
        implementation, not the engine.

        *attempt* is which attempt succeeded (``1`` if it succeeded on
        the first try) — so a listener doesn't have to correlate
        against ``on_node_start`` calls just to know whether a completed
        node needed retries.
        """

    def on_node_error(
        self,
        run: RunContext,
        node_id: str,
        node_kind: str,
        error: BaseException,
        attempt: int,
        will_retry: bool,
        duration_ms: float,
    ) -> None:
        """Called after a node's (or router's) failed attempt.

        *attempt* is the attempt that failed (1-indexed). *will_retry*
        is ``True`` when the configured retry policy will make another
        attempt; the exception is re-raised or routed via
        ``NodeConfig.on_error`` only when it is ``False``.
        *duration_ms* is the wall-clock time of that attempt.
        """

    def on_route(
        self,
        run: RunContext,
        router_id: str,
        chosen_next: Optional[str],
        matched_rule: Optional[str],
        reason: Optional[str],
    ) -> None:
        """Called after a router decides where to go next.

        *reason* is one of ``"matched_rule"``, ``"fallback"``, or
        ``"no_match_no_fallback"``. *matched_rule* is the matching
        ``RouterNode`` rule's ``node_name`` when
        ``reason == "matched_rule"``, else ``None``.
        """
