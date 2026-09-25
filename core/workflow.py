"""Workflow — the execution engine that walks a DAG of nodes."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from abc import ABC
from contextlib import contextmanager
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from strand.core.context import TaskContext
from strand.core.listener import WorkflowListener, notify_listeners
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
           and listeners from a still-running ancestor's when nested),
           marks it finished on the way out, and restores the
           ancestor's afterwards.
        4. Delegates each node to :meth:`_run_node` and each router to
           :meth:`_handle_router`, then asks :meth:`_next_key` where to
           go — following ``process()`` on regular nodes and
           ``route()`` on routers, and jumping along
           ``NodeConfig.on_error`` when a node fails.
        5. Notifies listeners of workflow start/end around the whole
           traversal, regardless of whether it completes or raises.
        """
        # --- context setup ---------------------------------------------------
        parent_run: Optional[RunContext] = None
        if existing_context is not None:
            task_context = existing_context
            parent_run = task_context._run_context
            if parent_run is not None and parent_run.finished:
                # The caller passed a context left over from a previous,
                # completed run ("resume"). That finished run's identity
                # and listeners must NOT be inherited — a new top-level
                # execution mints its own execution_id (see
                # RunContext.child_of).
                parent_run = None
            if parent_run is None:
                # Fresh or resumed run: a new traversal starts
                # un-stopped, even if the resumed context carries a
                # should_stop from its previous traversal. Nested
                # children must NOT touch this — a parent that signalled
                # stop before running a child still expects the stop to
                # be honoured once the child returns.
                task_context.should_stop = False
        else:
            task_context = TaskContext(event=event)
            task_context.event = self._schema.event_schema(**event)

        # Install this workflow's RunContext. When nested (a still-running
        # ancestor's RunContext is installed), inherit its
        # execution_id/started_at/listeners so every nested workflow's node
        # executions land in the same overall trace, while this workflow
        # still traverses its own graph via its own node_configs. Restored
        # to the ancestor's below, on the way out.
        task_context._run_context = RunContext.child_of(
            parent_run,
            workflow_name=type(self).__name__,
            workflow_version=self._schema.version,
            node_configs=self._node_configs,
            own_listeners=self._listeners,
        )
        this_run = task_context._run_context

        started = time.monotonic()
        await self._notify(task_context, "on_workflow_start", task_context.event)

        error: Optional[BaseException] = None
        try:
            current_key: Optional[str] = self._schema.start

            # --- main loop ---------------------------------------------------
            while current_key is not None:
                if task_context.should_stop:
                    logger.info("Workflow stop-signalled — halting execution.")
                    break

                nc = self._node_configs.get(current_key)
                if nc is not None and nc.is_router:
                    current_key = await self._handle_router(current_key, task_context)
                    continue

                task_context, error_route = await self._run_node(current_key, task_context)
                if error_route is not None:
                    current_key = error_route
                else:
                    current_key = self._next_key(current_key)
        except BaseException as exc:  # noqa: BLE001 — re-raised unchanged below
            error = exc
            raise
        finally:
            duration_ms = (time.monotonic() - started) * 1000
            await self._notify(
                task_context,
                "on_workflow_end",
                "error" if error is not None else "completed",
                duration_ms,
                error,
            )
            this_run.finished = True
            # Only restore an ancestor's RunContext — a nested child
            # returning control to its parent's still-running loop (or
            # unwinding through it on error). At the top level
            # (parent_run is None) this workflow's own RunContext is
            # left installed, marked finished: the caller can still read
            # execution_id/workflow_version off the result, while a
            # later run on the same context starts a fresh execution.
            if parent_run is not None:
                task_context._run_context = parent_run

        return task_context

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------

    async def _run_node(
        self, current_key: str, task_context: TaskContext
    ) -> Tuple[TaskContext, Optional[str]]:
        """Instantiate, run, and clean up the (non-router) node at
        *current_key*, retrying and enforcing a timeout per
        ``NodeConfig.retry`` / ``NodeConfig.timeout_s`` when either is
        set. Routers never reach this method — the main loop routes
        them through :meth:`_handle_router` instead.

        ``asyncio.CancelledError`` is re-raised immediately: never
        retried, never routed to ``on_error`` — cancellation means the
        caller wants the run to stop. (It is a ``BaseException``, so
        ``RetryPolicy``'s default never matches it.)

        ``cleanup()`` always runs exactly once per attempt, BEFORE the
        retry/error decision — a raising teardown can therefore neither
        skip a retry nor skip ``on_error`` routing, and a cleanup
        failure while the node itself already failed is logged without
        masking the node's own error.

        A ``process()`` that returns a *different* ``TaskContext``
        instance (e.g. a rebuilt one) has the engine's ``RunContext``
        carried over to it; returning ``None`` is accepted as "kept the
        context in place"; returning anything else is a ``TypeError``.

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

            await self._notify(task_context, "on_node_start", current_key, node_kind, attempt)
            started = time.monotonic()

            try:
                with self._node_context(current_key):
                    node_instance = self._instantiate(node_class, task_context, current_key)
                    coro = node_instance.process(task_context)
                    if timeout_s is not None:
                        result = await asyncio.wait_for(coro, timeout_s)
                    else:
                        result = await coro
                    if isinstance(result, TaskContext):
                        if result is not task_context:
                            # A node may return a fresh/rebuilt context —
                            # carry the engine's bookkeeping over to it
                            # (execution identity, listeners, node
                            # configs) so the run continues on the
                            # returned object.
                            result._run_context = task_context._run_context
                        task_context = result
                    elif result is not None:
                        error = TypeError(
                            f"Node '{current_key}' process() returned "
                            f"{type(result).__name__}; expected the TaskContext "
                            f"(or None to keep the current one)."
                        )
                    # result is None: the node kept/mutated the context in
                    # place and returned nothing — accepted.
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 — re-raised unless retrying
                error = exc
            finally:
                duration_ms = (time.monotonic() - started) * 1000
                if node_instance is not None:
                    try:
                        await node_instance.cleanup()
                    except asyncio.CancelledError:
                        raise
                    except BaseException as cleanup_exc:
                        if error is None:
                            error = cleanup_exc
                        else:
                            logger.exception(
                                "Node '%s' cleanup() raised after the node itself "
                                "failed — keeping the node's own error.",
                                current_key,
                            )

            if error is not None:
                will_retry = (
                    retry_policy is not None
                    and retry_policy.should_retry(error, attempt)
                )
                await self._notify(
                    task_context,
                    "on_node_error",
                    current_key,
                    node_kind,
                    error,
                    attempt,
                    will_retry,
                    duration_ms,
                )
                if will_retry:
                    await _sleep(retry_policy.backoff_seconds(attempt))
                    attempt += 1
                    continue
                if nc is not None and nc.on_error is not None:
                    task_context.errors[current_key] = f"{type(error).__name__}: {error}"
                    return task_context, nc.on_error
                raise error

            output = task_context.nodes.get(current_key)
            await self._notify(
                task_context,
                "on_node_end",
                current_key,
                node_kind,
                "completed",
                duration_ms,
                output,
                attempt,
            )
            return task_context, None

    # ------------------------------------------------------------------
    # Router execution
    # ------------------------------------------------------------------

    async def _handle_router(
        self, current_key: str, task_context: TaskContext
    ) -> Optional[str]:
        """Instantiate the router at *current_key*, route, and report.

        Routers run outside :meth:`_run_node`: routing is synchronous,
        deterministic rule evaluation — never retried or timed out. A
        raising rule IS reported as ``on_node_error`` (not, as before,
        as a completed node with the error attributed to nothing), and
        ``on_route`` carries the decision either way.
        """
        router_cls = self._registry[current_key]
        router: BaseRouter = self._instantiate(router_cls, task_context, current_key)

        chosen_next: Optional[str] = None
        error: Optional[BaseException] = None
        started = time.monotonic()

        await self._notify(task_context, "on_node_start", current_key, router.kind, 1)
        try:
            chosen_next = router.route(task_context)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 — re-raised after notifying
            error = exc
        finally:
            try:
                await router.cleanup()
            except asyncio.CancelledError:
                raise
            except BaseException as cleanup_exc:
                if error is None:
                    error = cleanup_exc
                else:
                    logger.exception(
                        "Router '%s' cleanup() raised after route() failed — "
                        "keeping the route error.",
                        current_key,
                    )

        duration_ms = (time.monotonic() - started) * 1000
        if error is not None:
            await self._notify(
                task_context,
                "on_node_error",
                current_key,
                router.kind,
                error,
                1,
                False,
                duration_ms,
            )
            raise error

        output = task_context.nodes.get(current_key)
        await self._notify(
            task_context,
            "on_node_end",
            current_key,
            router.kind,
            "completed",
            duration_ms,
            output,
            1,
        )

        matched_rule = getattr(router, "last_matched_rule", None)
        if matched_rule is not None:
            reason = "matched_rule"
        elif chosen_next is not None:
            reason = "fallback"
        else:
            reason = "no_match_no_fallback"

        await self._notify(
            task_context,
            "on_route",
            router.node_name,
            chosen_next,
            matched_rule,
            reason,
        )
        return chosen_next

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def _next_key(self, current_key: str) -> Optional[str]:
        """Return the next node key, or ``None`` if this is a terminal node.

        Only non-router nodes reach this — routers are dispatched from
        the main loop via :meth:`_handle_router`. The validator
        guarantees a non-router node has at most one connection.
        """
        nc = self._node_configs.get(current_key)
        if nc is None or not nc.connections:
            return None
        return nc.connections[0]

    # ------------------------------------------------------------------
    # Listener dispatch
    # ------------------------------------------------------------------

    async def _notify(self, task_context: TaskContext, method_name: str, *args: Any) -> None:
        """Call *method_name* on every listener registered for this run.

        Delegates to ``strand.core.listener.notify_listeners`` — the one
        place the "a raising listener is logged and swallowed, never
        allowed to affect workflow execution" guarantee is enforced.
        Hooks may be sync or async; coroutine hooks are awaited.
        """
        await notify_listeners(task_context._run_context, method_name, *args)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _instantiate(
        self,
        node_class: Type[Node],
        task_context: TaskContext,
        node_id: str,
    ) -> Node:
        """Instantiate *node_class* for *node_id*, tolerating both eras
        of constructor style.

        Nodes written against the current convention accept
        ``task_context=``/``node_id=`` keyword arguments (by inheriting
        ``Node.__init__`` or declaring their own). Subclasses written
        against the earlier engine were called with no arguments and
        managed their own state — for those, instantiate bare and assign
        ``task_context``/``_node_id`` directly, so existing subclasses
        keep working unmodified.
        """
        try:
            sig = inspect.signature(node_class.__init__)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            accepts_kwargs = any(
                param.kind is inspect.Parameter.VAR_KEYWORD
                for param in sig.parameters.values()
            )
            accepts_named = (
                "task_context" in sig.parameters and "node_id" in sig.parameters
            )
            if accepts_kwargs or accepts_named:
                return node_class(task_context=task_context, node_id=node_id)
        instance = node_class()
        instance.task_context = task_context
        instance._node_id = node_id
        return instance

    def _build_node_configs(self) -> Dict[str, NodeConfig]:
        """Build a dictionary of every node key → ``NodeConfig``.

        Nodes referenced in ``connections`` that do not have an explicit
        ``NodeConfig`` entry get an implicit one with no connections
        (i.e. they become terminal nodes). A ``BaseRouter`` referenced
        this way is rejected: routers need an explicit config
        (``is_router=True`` plus their own connections), and an implicit
        one would silently turn routing into a dead end.
        """
        configs: Dict[str, NodeConfig] = {}
        explicit = {nc.node for nc in self._schema.nodes}
        for nc in self._schema.nodes:
            configs[nc.node] = nc
            for connected_key in nc.connections:
                if connected_key in configs or connected_key in explicit:
                    # Already configured — either explicitly (wherever it
                    # appears in the nodes list) or implicitly (created
                    # below by an earlier reference).
                    continue
                node_class = self._schema.registry.get(connected_key)
                if node_class is not None and issubclass(node_class, BaseRouter):
                    raise ValueError(
                        f"Node '{connected_key}' is referenced as a connection "
                        f"but has no explicit NodeConfig, and its registered "
                        f"class is a BaseRouter — routers need an explicit "
                        f"config (is_router=True plus their own connections)."
                    )
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
