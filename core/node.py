"""Base node — the abstract foundation for all workflow nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from pydantic import BaseModel

from strand.core.context import TaskContext


class Node(ABC):
    """Abstract base class for all workflow nodes.

    Implements the Chain of Responsibility pattern: each node receives
    the shared TaskContext, processes it, and returns it for the next
    node in the graph.

    Subclasses must implement ``process()``. Override ``cleanup()`` to
    release per-instance resources (clients, connections) between runs.

    Attributes:
        task_context: The shared context for the current workflow run.
        node_id: The registry key under which this node was registered.
        kind: An open, uninterpreted label reported to
            ``WorkflowListener`` hooks as ``node_kind`` — ``"node"`` by
            default, overridden to ``"router"`` on ``BaseRouter`` and
            ``"llm"`` on ``LLMNode`` (``strand.llm``). ``strand.core``
            never inspects this value itself; consumers may set their
            own on custom subclasses.
    """

    kind: ClassVar[str] = "node"

    class OutputType(BaseModel):
        """Structured output base for nodes that produce typed results.

        Subclass this inside your node to define the shape of
        ``save_output()`` payloads.
        """

        pass

    def __init__(
        self,
        task_context: TaskContext | None = None,
        node_id: str | None = None,
    ) -> None:
        self.task_context = task_context
        self._node_id = node_id

    @property
    def node_name(self) -> str:
        """The registry key for this node, falling back to the class name."""
        return self._node_id or self.__class__.__name__

    def save_output(self, output: BaseModel) -> None:
        """Store a node's structured output in the shared task context.

        Args:
            output: A pydantic model representing the result of this node.
        """
        self.task_context.nodes[self.node_name] = output

    def get_output(self, key: str) -> Optional[OutputType]:
        """Retrieve the output stored by another node.

        Args:
            key: The registry key of the node whose output to retrieve.

        Returns:
            The stored output if present, otherwise ``None``.
        """
        return self.task_context.nodes.get(key)

    def get_error(self, key: str) -> Optional[str]:
        """Retrieve the string summary of another node's failure.

        Only populated for nodes that raised and were routed here via
        that node's ``NodeConfig.on_error`` — the counterpart to
        ``get_output()`` for the error path.

        Args:
            key: The registry key of the node whose failure to retrieve.

        Returns:
            The stored error summary if present, otherwise ``None``.
        """
        return self.task_context.errors.get(key)

    @abstractmethod
    async def process(self, task_context: TaskContext) -> TaskContext:
        """Process the task context in the responsibility chain.

        Each node in the workflow processes the task and passes it
        to the next node through the workflow orchestrator.

        Args:
            task_context: The shared context object passed through the workflow.

        Returns:
            Updated TaskContext with this node's processing results.
        """
        ...

    async def cleanup(self) -> None:
        """Release any per-instance resources held by the node.

        Called after the node finishes processing, including when an
        exception propagates out. Override this when the node holds
        clients or connections that need explicit teardown between runs
        (for example, when the same node type is re-entered by a child
        workflow during composition).
        """
        pass
