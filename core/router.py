"""Router — conditional branching for workflow graphs."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

from pydantic import BaseModel

from strand.core.context import TaskContext
from strand.core.node import Node


class RouterNode(ABC):
    """A single routing rule evaluated during conditional branching.

    RouterNode is **not** a ``Node`` subclass — it represents a
    decision point, not a processing step. Each instance implements
    ``determine_next_node()``, which returns a registry key (``str``)
    when the rule matches, or ``None`` when it does not.

    Subclasses may call ``save_output()`` to record the routing
    decision in the task context for observability.
    """

    def __init__(
        self,
        task_context: TaskContext | None = None,
        node_id: str | None = None,
    ) -> None:
        self.task_context = task_context
        self._node_id = node_id

    @property
    def node_name(self) -> str:
        """The registry key for this router, falling back to the class name."""
        return self._node_id or self.__class__.__name__

    @abstractmethod
    def determine_next_node(self, task_context: TaskContext) -> Optional[str]:
        """Evaluate the routing rule against the current task context.

        Args:
            task_context: The current workflow execution context.

        Returns:
            The registry key of the next node to execute, or ``None``
            if this rule does not match.
        """
        ...

    def save_output(self, output: BaseModel) -> None:
        """Store a routing decision in the shared task context."""
        self.task_context.nodes[self.node_name] = output

    def get_output(self, key: str) -> Optional[object]:
        """Retrieve output stored by another node or router.

        Args:
            key: The registry key of the node whose output to retrieve.
        """
        return self.task_context.nodes.get(key)


class BaseRouter(Node):
    """A workflow node that conditionally routes to one of several branches.

    BaseRouter evaluates a list of ``RouterNode`` instances in order
    and follows the first one that matches. If none match, it falls
    back to ``fallback`` (which may be ``None``, ending the workflow).

    ``process()`` is a deliberate no-op — routing is handled by the
    ``Workflow`` engine, which calls ``route()`` instead of ``process()``
    when it encounters a router node.

    Attributes:
        routes: Ordered list of ``RouterNode`` instances to evaluate.
        fallback: Registry key of the default node when no rule matches.
        last_matched_rule: The ``node_name`` of whichever rule matched
            on the most recent ``route()`` call, or ``None`` if none
            matched (fallback was used, or no fallback was set).
            Read by ``Workflow`` after ``route()`` returns, to report a
            reason alongside the ``on_route`` listener event.
    """

    kind: ClassVar[str] = "router"

    routes: List[RouterNode] = []
    fallback: Optional[str] = None

    def __init__(
        self,
        task_context: TaskContext | None = None,
        node_id: str | None = None,
    ) -> None:
        super().__init__(task_context=task_context, node_id=node_id)
        # `routes` is declared at class level for a convenient, readable
        # API (`routes = [Rule1(), Rule2()]`), which means every
        # instance of this router class would otherwise share the exact
        # same RouterNode objects. route() mutates rule.task_context in
        # place, so without per-instance copies, two BaseRouter
        # instances of the same class (e.g. two concurrent runs, or the
        # same class registered under two registry keys) would stomp on
        # each other's in-flight task_context. Copying each rule here
        # gives every router instance its own independent set.
        #
        # The source list is `self.routes`, NOT `type(self).routes`: a
        # subclass may assign its own routes on the instance before
        # calling super().__init__(), and reading the class attribute
        # would silently discard them. The copy is deep: a rule instance
        # may itself hold mutable state (a seen-set, counters) that a
        # shallow copy would still share across routers and runs.
        self.routes = [copy.deepcopy(rule) for rule in self.routes]
        self.last_matched_rule: Optional[str] = None

    async def process(self, task_context: TaskContext) -> TaskContext:
        """No-op. Routing is handled by the Workflow engine via ``route()``."""
        pass

    def route(self, task_context: TaskContext) -> Optional[str]:
        """Evaluate routing rules and return the next node key.

        Args:
            task_context: The current workflow execution context.

        Returns:
            The registry key of the next node, or ``None`` to end the
            workflow.
        """
        self.last_matched_rule = None
        for route_node in self.routes:
            route_node.task_context = task_context
            next_key = route_node.determine_next_node(task_context)
            if next_key is not None:
                self.last_matched_rule = route_node.node_name
                return next_key
        return self.fallback
