"""Task context — the shared state bus passed between workflow nodes."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr

from strand.core.run_context import RunContext


class TaskContext(BaseModel):
    """Context container for workflow task execution.

    TaskContext maintains the state and results of a workflow's execution,
    tracking the original event, intermediate node results, and additional
    metadata throughout the processing flow.

    Attributes:
        event: The original event that triggered the workflow.
        nodes: Dictionary storing results and state from each node's execution.
        metadata: Dictionary storing workflow-level metadata and configuration.
        should_stop: Boolean flag indicating whether the workflow should stop.
        errors: Stores a string summary of each node's failure, keyed
            by node id — populated only for nodes routed via
            ``NodeConfig.on_error`` (the failing node's own output is
            never written). A string, not the exception object itself,
            so this stays serializable like the rest of this class.

    ``event``, ``nodes``, ``metadata``, ``should_stop``, and ``errors``
    are domain data — free for node authors to read and write, and safe
    to serialize (``model_dump()``). Engine bookkeeping (identity,
    versioning, the resolved node-config map, the traversal log) lives
    separately in a private ``RunContext``, set by ``Workflow`` and
    never included in serialization. Use the ``execution_id`` /
    ``workflow_version`` properties for the one piece of that domain
    code is meant to read.
    """

    event: Any
    nodes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stores results and state from each node's execution",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stores workflow-level metadata and configuration",
    )
    should_stop: bool = Field(
        default=False,
        description="Flag indicating whether the workflow should stop execution",
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Node id -> string summary, for nodes routed via on_error",
    )

    _run_context: Optional[RunContext] = PrivateAttr(default=None)

    def update_node(self, key: str, **kwargs: Any) -> None:
        """Merge keyword arguments into the stored state for a node key.

        Args:
            key: The registry key of the node whose state is being updated.
            **kwargs: Key-value pairs to merge into the node's state dict.
        """
        self.nodes[key] = {**self.nodes.get(key, {}), **kwargs}

    def stop_workflow(self) -> None:
        """Signal that the workflow should stop after the current node completes.

        Can be called from any node to halt further processing.
        Once called, the execution loop will not advance to the next node.
        """
        self.should_stop = True

    @property
    def execution_id(self) -> Optional[str]:
        """The current execution's ID, or ``None`` outside a workflow run.

        Stable across an entire nested composition — a parent workflow
        and every child it runs share the same ``execution_id``. Safe
        to stamp onto domain output rows for lineage/traceability.
        """
        return self._run_context.execution_id if self._run_context else None

    @property
    def workflow_version(self) -> Optional[str]:
        """The ``WorkflowSchema.version`` of the currently-executing
        workflow, or ``None`` outside a run or when unset.

        Unlike ``execution_id``, this reflects whichever workflow
        (parent or nested child) is *currently* traversing its graph.
        """
        return self._run_context.workflow_version if self._run_context else None
