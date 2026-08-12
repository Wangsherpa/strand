"""Workflow — the execution engine that walks a DAG of nodes."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC
from contextlib import contextmanager
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from strand.core.context import TaskContext
from strand.core.listener import WorkflowListener
from strand.core.node import Node
from strand.core.router import BaseRouter
from strand.core.run_context import RunContext
from strand.core.schema import NodeConfig, WorkflowSchema
from strand.core.validator import WorkflowValidator

logger = logging.getLogger(__name__)

# Indirection so tests can replace the sleep used for retry backoff
# without monkeypatching the shared, global asyncio.sleep (which could
# affect pytest-asyncio's own scheduling, not just this module).
_sleep = asyncio.sleep


class Workflow(ABC):
    """Abstract base class for defining and executing workflows.

    Subclasses declare their graph structure via the ``workflow_schema``
    class variable.  The engine walks the graph sequentially, passing a
    shared ``TaskContext`` through each node.

    Routes are evaluated via ``BaseRouter.route()`` instead of the
    normal ``Node.process()`` path.

    Nested workflows share the same ``TaskContext`` — call
    ``child_workflow.run(context=parent_context)`` to compose
    workflows without copying payloads.

    Example::

        class MyWorkflow(Workflow):
            workflow_schema = WorkflowSchema(
                start="analyze",
                nodes=[
                    NodeConfig(node="analyze", connections=["route"]),
                    NodeConfig(node="route", connections=["respond", "escalate"],
                               is_router=True),
                    NodeConfig(node="respond"),
                    NodeConfig(node="escalate"),
                ],
                registry={
                    "analyze": AnalyzeNode,
                    "route":   RouterNode,
                    "respond": RespondNode,
                    "escalate": EscalateNode,
                },
            )
    """

    workflow_schema: ClassVar[WorkflowSchema]

    def __init__(self, *, listeners: Optional[List[WorkflowListener]] = None) -> None:
        """
        Args:
            listeners: ``WorkflowListener`` instances observing this
                workflow's execution. Inherited by any nested workflow
                run via ``run_async(context=...)`` from within a node —
                a listener registered on the outermost workflow also
                observes every nested child's node executions.
        """
        self._schema = self.workflow_schema
        self._validator = WorkflowValidator(self._schema)
        self._validator.validate()
        self._registry = self._schema.registry
        self._node_configs: Dict[str, NodeConfig] = self._build_node_configs()
        self._listeners: List[WorkflowListener] = list(listeners) if listeners else []

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def to_mermaid(self) -> str:
        """Return a Mermaid.js flowchart of this workflow's DAG."""
        return self._schema.to_mermaid()

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of this workflow's DAG."""
        return self._schema.to_dot()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        event: Any = None,
        *,
        context: Optional[TaskContext] = None,
    ) -> TaskContext:
        """Execute the workflow synchronously.

        Creates a new event loop.  Use this from scripts or Celery tasks.
        For async contexts (FastAPI, inside another node), use
        :meth:`run_async` instead.

        Args:
            event: Raw event data.  Required unless *context* is provided.
            context: Existing ``TaskContext`` to resume.  When passed,
                *event* is ignored and the workflow reuses the shared
                context (used for nested workflow composition).

        Returns:
            The ``TaskContext`` after workflow execution completes.
        """
        if context is None and event is None:
            raise ValueError("Either event or context must be provided.")
        return asyncio.run(self._execute(event, context))

    async def run_async(
        self,
        event: Any = None,
        *,
        context: Optional[TaskContext] = None,
    ) -> TaskContext:
        """Execute the workflow asynchronously.

        Use this when an event loop is already running (FastAPI handlers,
        inside another node, etc.).

        Args:
            event: Raw event data.  Required unless *context* is provided.
            context: Existing ``TaskContext`` to resume.  When passed,
                *event* is ignored.

        Returns:
            The ``TaskContext`` after workflow execution completes.
        """
        if context is None and event is None:
            raise ValueError("Either event or context must be provided.")
        return await self._execute(event, context)

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    async def _execute(
        self,
        event: Any = None,
        existing_context: Optional[TaskContext] = None,
    ) -> TaskContext:
        """Walk the graph from ``start`` until there is no next node.

        This is the heart of the engine.  It:

        1. Creates (or reuses) a ``TaskContext``.
        2. Parses *event* through ``event_schema`` when starting fresh.
        3. Installs this workflow's ``RunContext`` (inheriting identity
           from the ancestor's when nested), and restores the
           ancestor's afterward.
        4. Delegates each node to :meth:`_run_node`, then asks
           :meth:`_next_key` where to go — following ``process()`` on
           regular nodes and ``route()`` on routers.
        5. Notifies listeners of workflow start/end around the whole
           traversal, regardless of whether it completes or raises.
        """
        # --- context setup ---------------------------------------------------
        if existing_context is not None:
            task_context = existing_context
            task_context.should_stop = False
        else:
            task_context = TaskContext(event=event)
            task_context.event = self._schema.event_schema(**event)

        # Install this workflow's RunContext. When nested (existing_context
        # was passed in and already carries an ancestor's RunContext),
        # inherit its execution_id/started_at/listeners so every nested
        # workflow's node executions land in the same overall trace, while
        # this workflow still traverses its own graph via its own
        # node_configs. Restored to the ancestor's below, on the way out.
        parent_run = task_context._run_context
        task_context._run_context = RunContext.child_of(
            parent_run,
            workflow_name=type(self).__name__,
            workflow_version=self._schema.version,
            node_configs=self._node_configs,
            own_listeners=self._listeners,
        )

        started = time.monotonic()
        self._notify(task_context, "on_workflow_start", task_context.event)

        error: Optional[BaseException] = None
        try:
            current_key: Optional[str] = self._schema.start

            # --- main loop ---------------------------------------------------
            while current_key is not None:
                if task_context.should_stop:
                    logger.info("Workflow stop-signalled — halting execution.")
                    break

                task_context._run_context.traversal.append(current_key)
                task_context, error_route = await self._run_node(current_key, task_context)
                if error_route is not None:
                    current_key = error_route
                else:
                    current_key = await self._next_key(current_key, task_context)
        except BaseException as exc:  # noqa: BLE001 — re-raised unchanged below
            error = exc
            raise
        finally:
            duration_ms = (time.monotonic() - started) * 1000
            self._notify(
                task_context,
                "on_workflow_end",
                "error" if error is not None else "completed",
                duration_ms,
                error,
            )
            # Only restore an ancestor's RunContext — a nested child
            # returning control to its parent's still-running loop (or
            # unwinding through it on error). At the top level
            # (parent_run is None) there is nothing left to run against
            # this context, so this workflow's own RunContext is left
            # installed: the caller can still read
            # execution_id/workflow_version off the result. This
            # finally block also fixes a bug present since Phase 4: the
            # restore previously only ran on the non-exception path, so
            # a node that caught an exception from a nested child and
            # continued would keep running with the child's RunContext
            # (wrong node_configs/workflow_version) installed.
            if parent_run is not None:
                task_context._run_context = parent_run

        return task_context

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------

    async def _run_node(
        self, current_key: str, task_context: TaskContext
    ) -> Tuple[TaskContext, Optional[str]]:
        """Instantiate, run, and clean up the node at *current_key*,
        retrying and enforcing a timeout per ``NodeConfig.retry`` /
        ``NodeConfig.timeout_s`` when either is set.

        Routers are skipped here — they have no ``process()`` to run;
        routing itself is handled separately by :meth:`_next_key` /
        :meth:`_handle_router`, once this method returns. They still
        get on_node_start/on_node_end notifications (``node_kind ==
        "router"`` is the signal that the actual decision is reported
        separately via ``on_route``), for symmetric "which step ran"
        coverage across every key visited. Routers are never retried
        or timed out — ``route()`` is synchronous, deterministic
        rule evaluation, not I/O.

        Returns the (possibly updated) context and, when this node's
        failure was routed via ``NodeConfig.on_error`` instead of
        raised, the registry key execution should jump to next —
        ``None`` in every other case, meaning the caller should
        determine the next key the normal way, via :meth:`_next_key`.

        This is the sole seam through which every node execution
        passes.
        """
        node_class = self._registry[current_key]
        node_kind = node_class.kind
        nc = self._node_configs.get(current_key)
        retry_policy = nc.retry if nc is not None else None
        timeout_s = nc.timeout_s if nc is not None else None

        attempt = 1
        while True:
            node_instance: Optional[Node] = None
            error: Optional[BaseException] = None

            self._notify(task_context, "on_node_start", current_key, node_kind, attempt)
            started = time.monotonic()

            try:
                with self._node_context(current_key):
                    if not issubclass(node_class, BaseRouter):
                        node_instance = node_class(
                            task_context=task_context,
                            node_id=current_key,
                        )
                        coro = node_instance.process(task_context)
                        if timeout_s is not None:
                            task_context = await asyncio.wait_for(coro, timeout_s)
                        else:
                            task_context = await coro
            except BaseException as exc:  # noqa: BLE001 — re-raised unless retrying
                error = exc
            finally:
                duration_ms = (time.monotonic() - started) * 1000
                if error is not None:
                    will_retry = retry_policy is not None and retry_policy.should_retry(
                        error, attempt
                    )
                    self._notify(
                        task_context,
                        "on_node_error",
                        current_key,
                        node_kind,
                        error,
                        attempt,
                        will_retry,
                    )
                else:
                    output = task_context.nodes.get(current_key)
                    self._notify(
                        task_context,
                        "on_node_end",
                        current_key,
                        node_kind,
                        "completed",
                        duration_ms,
                        output,
                        attempt,
                    )
                if node_instance is not None:
                    await node_instance.cleanup()

            if error is None:
                return task_context, None
            if not will_retry:
                if nc is not None and nc.on_error is not None:
                    task_context.errors[current_key] = f"{type(error).__name__}: {error}"
                    return task_context, nc.on_error
                raise error

            await _sleep(retry_policy.backoff_seconds(attempt))
            attempt += 1

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    async def _next_key(
        self, current_key: str, task_context: TaskContext
    ) -> Optional[str]:
        """Return the next node key, or ``None`` if this is a terminal node."""
        nc = self._node_configs.get(current_key)
        if nc is None or not nc.connections:
            return None

        if nc.is_router:
            router_cls = self._registry[current_key]
            router: BaseRouter = router_cls(
                task_context=task_context, node_id=current_key
            )
            return await self._handle_router(router, task_context)

        # Linear flow — follow the first (and only) connection.
        return nc.connections[0]

    async def _handle_router(
        self, router: BaseRouter, task_context: TaskContext
    ) -> Optional[str]:
        """Delegate to the router and return the chosen next key."""
        chosen_next = router.route(task_context)

        if router.last_matched_rule is not None:
            reason = "matched_rule"
        elif chosen_next is not None:
            reason = "fallback"
        else:
            reason = "no_match_no_fallback"

        self._notify(
            task_context,
            "on_route",
            router.node_name,
            chosen_next,
            router.last_matched_rule,
            reason,
        )
        return chosen_next

    # ------------------------------------------------------------------
    # Listener dispatch
    # ------------------------------------------------------------------

    def _notify(self, task_context: TaskContext, method_name: str, *args: Any) -> None:
        """Call *method_name* on every listener registered for this run.

        A listener raising is logged and swallowed here — never
        allowed to affect workflow execution. This is the one place
        that guarantee is enforced; every call site in this module
        relies on it rather than re-implementing it.
        """
        run = task_context._run_context
        if run is None:
            return
        for listener in run.listeners:
            method = getattr(listener, method_name, None)
            if method is None:
                continue
            try:
                method(run, *args)
            except Exception:
                logger.exception(
                    "Listener %r raised in %s() — ignoring.", listener, method_name
                )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_node_configs(self) -> Dict[str, NodeConfig]:
        """Build a dictionary of every node key → ``NodeConfig``.

        Nodes referenced in ``connections`` that do not have an explicit
        ``NodeConfig`` entry get an implicit one with no connections
        (i.e. they become terminal nodes).
        """
        configs: Dict[str, NodeConfig] = {}
        for nc in self._schema.nodes:
            configs[nc.node] = nc
            for connected_key in nc.connections:
                if connected_key not in configs:
                    configs[connected_key] = NodeConfig(node=connected_key)
        return configs

    # ------------------------------------------------------------------
    # Logging context manager
    # ------------------------------------------------------------------

    @contextmanager
    def _node_context(self, key: str):
        """Log entry/exit for *key* and re-raise any exception."""
        logger.info("Starting node: %s", key)
        try:
            yield
        except Exception:
            logger.exception("Error in node: %s", key)
            raise
        else:
            logger.info("Finished node: %s", key)
