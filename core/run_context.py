"""RunContext — engine-held bookkeeping for one workflow execution.

Unlike ``TaskContext`` (domain data: ``event``, ``nodes``, ``metadata``),
``RunContext`` holds framework internals that domain code should not
see or mutate directly: identity, versioning, timing, the resolved
node-config map used for traversal, and — from later phases — listener
registration and the executed-edge trace.

Never passed to nodes directly. ``TaskContext.execution_id`` /
``TaskContext.workflow_version`` are the sanctioned read-only surface
for domain code that needs to stamp these onto its own output.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RunContext:
    """Bookkeeping for one workflow execution, held by the engine.

    Attributes:
        execution_id: Identifies this execution across an entire nested
            composition — shared by a parent workflow and every child
            it runs, so they land in the same trace.
        started_at: UTC timestamp this execution began, shared with
            nested children the same way as ``execution_id``.
        workflow_name: The ``Workflow`` subclass currently traversing
            its graph. Distinct per nested workflow, unlike
            ``execution_id``.
        workflow_version: That workflow's ``WorkflowSchema.version``.
        node_configs: That workflow's resolved
            ``{registry_key: NodeConfig}`` map — distinct per nested
            workflow, since each traverses its own graph.
        listeners: WorkflowListener instances observing this execution.
            Shared with nested children (plus any they register of
            their own) so a listener registered on the root observes
            every nested workflow's node executions too.
        traversal: Registry keys actually visited, in order, for this
            workflow's own execution (not its children's).
    """

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    workflow_name: str = ""
    workflow_version: Optional[str] = None
    node_configs: Dict[str, Any] = field(default_factory=dict)
    listeners: List[Any] = field(default_factory=list)
    traversal: List[str] = field(default_factory=list)

    @classmethod
    def child_of(
        cls,
        parent: Optional["RunContext"],
        *,
        workflow_name: str,
        workflow_version: Optional[str],
        node_configs: Dict[str, Any],
        own_listeners: Optional[List[Any]] = None,
    ) -> "RunContext":
        """Build the RunContext for one workflow's execution.

        When *parent* is given (nested composition via
        ``child.run_async(context=parent_ctx)``), the new RunContext
        inherits ``execution_id``/``started_at``/``listeners`` from it,
        so the child's node executions land in the same overall trace —
        but gets its own ``workflow_name``/``workflow_version``/
        ``node_configs``, since the child traverses its own graph.
        Any ``own_listeners`` this workflow was itself constructed with
        (``Workflow(listeners=[...])``) are appended after the
        inherited ones — both fire.

        When *parent* is ``None`` (a fresh top-level run), a new
        ``execution_id`` and ``started_at`` are minted, and
        ``listeners`` is just ``own_listeners``.
        """
        own_listeners = list(own_listeners) if own_listeners else []
        if parent is None:
            return cls(
                workflow_name=workflow_name,
                workflow_version=workflow_version,
                node_configs=node_configs,
                listeners=own_listeners,
            )
        return cls(
            execution_id=parent.execution_id,
            started_at=parent.started_at,
            workflow_name=workflow_name,
            workflow_version=workflow_version,
            node_configs=node_configs,
            listeners=parent.listeners + own_listeners,
        )
