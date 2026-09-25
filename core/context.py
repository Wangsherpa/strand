"""Task context — the shared state bus passed between workflow nodes."""

from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr

from strand.core.run_context import RunContext

if TYPE_CHECKING:
    from strand.core.schema import NodeConfig


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
    versioning, the resolved node-config map, listeners) lives
    separately in a private ``RunContext``, set by ``Workflow`` and
    excluded from pickling/deepcopy (see ``__getstate__``). Use the
    ``execution_id`` / ``workflow_version`` properties and
    ``get_node_config()`` for the pieces of that bookkeeping domain
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

    def get_node_config(self, key: str) -> Optional["NodeConfig"]:
        """The currently-executing workflow's ``NodeConfig`` for *key*.

        ``None`` outside a workflow run, or for keys without a config
        (e.g. implicit terminal nodes). This is the sanctioned read
        surface for domain code that needs graph knowledge — the
        pre-RunContext convention of publishing the whole config map
        under ``metadata["nodes"]`` was engine bookkeeping and is no
        longer written there.
        """
        run = self._run_context
        if run is None:
            return None
        return run.node_configs.get(key)

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude engine bookkeeping from pickling/deepcopy.

        ``RunContext`` (and everything it references — listeners, node
        configs) is run-scoped: it is never serialized with domain data
        and is not re-installed on unpickle. This is what the class
        docstring's "excluded from pickling/deepcopy" promise refers to.
        """
        state = super().__getstate__()
        # pydantic stores private attrs under "__pydantic_private__" (a
        # flat key is tolerated too, in case that layout ever changes).
        # The key is KEPT but its value replaced with None: the RunContext
        # itself — listeners, node configs — never enters the pickle
        # graph, and the unpickled context is cleanly "outside a run"
        # (its execution_id/workflow_version read as None).
        private = state.get("__pydantic_private__")
        if isinstance(private, dict):
            private["_run_context"] = None
        elif "_run_context" in state:
            state["_run_context"] = None
        return state

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> "TaskContext":
        """Deep-copy without the engine's RunContext.

        Mirrors pydantic's own ``__deepcopy__`` (which does not go
        through ``__getstate__`` and would otherwise carry the
        RunContext — listeners, node configs and all — into the copy),
        but replaces ``_run_context`` with ``None``: the copy is cleanly
        "outside a run", exactly like an unpickled context.
        """
        from copy import deepcopy as _deepcopy

        cls = type(self)
        new = cls.__new__(cls)
        if memo is not None:
            memo[id(self)] = new
        object.__setattr__(new, "__dict__", _deepcopy(self.__dict__, memo))
        object.__setattr__(
            new, "__pydantic_extra__",
            _deepcopy(getattr(self, "__pydantic_extra__", None), memo),
        )
        object.__setattr__(new, "__pydantic_fields_set__", set(self.__pydantic_fields_set__))
        private = getattr(self, "__pydantic_private__", None) or {}
        copied_private = _deepcopy(
            {key: value for key, value in private.items() if key != "_run_context"}, memo
        )
        copied_private["_run_context"] = None
        object.__setattr__(new, "__pydantic_private__", copied_private)
        return new

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
