"""Workflow schema — declarative graph definition using string keys."""

from __future__ import annotations

from typing import List, Optional, Type

from pydantic import BaseModel, Field

from strand.core.registry import NodeRegistry
from strand.core.retry import RetryPolicy


class NodeConfig(BaseModel):
    """Configuration for a single node in the workflow graph.

    Attributes:
        node: Registry key identifying this node.
        connections: Registry keys of nodes that may follow this one.
        is_router: If ``True``, the Workflow engine calls ``route()``
            instead of ``process()`` on this node.
        description: Optional human-readable description of the node's purpose.
        retry: Retry policy for this node, or ``None`` (the default) to
            attempt it exactly once, exactly as before this field
            existed. Applies to every node kind — deterministic I/O
            nodes need retry just as much as LLM ones do.
        timeout_s: Per-attempt timeout in seconds, or ``None`` (the
            default) for no timeout. Wrapped in ``asyncio.wait_for``;
            a timeout raises like any other exception, so ``retry``'s
            ``retry_on`` decides whether it's worth retrying.
        on_error: Registry key to route to instead of aborting the run,
            once ``retry`` (if any) is exhausted and this node's
            ``process()`` still raises. ``None`` (the default) means
            the exception propagates and the run aborts, exactly as
            before this field existed. The failing node's own output
            is never written; the target node can inspect what failed
            via ``Node.get_error(node_id)``.
    """

    model_config = {"extra": "forbid"}

    node: str
    connections: List[str] = Field(default_factory=list)
    is_router: bool = False
    description: Optional[str] = None
    retry: Optional[RetryPolicy] = None
    timeout_s: Optional[float] = None
    on_error: Optional[str] = None


class WorkflowSchema(BaseModel):
    """Complete declarative definition of a workflow graph.

    Every string in ``start``, ``NodeConfig.node``, and
    ``NodeConfig.connections`` must be a key in ``registry``.

    Attributes:
        description: Optional human-readable description of the workflow.
        version: Optional version string for this workflow definition.
            Stamped onto every execution's ``RunContext`` (surfaced to
            domain code via ``TaskContext.workflow_version``), so
            results can be attributed to the workflow version that
            produced them — most useful once prompts, routing, or
            schemas start changing between runs.
        event_schema: Pydantic model used to validate incoming events.
        start: Registry key of the entry-point node.
        nodes: Ordered list of node configurations that define the graph.
        registry: Mapping from string keys to ``Node`` subclasses.
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    description: Optional[str] = None
    version: Optional[str] = None
    event_schema: Type[BaseModel]
    start: str
    nodes: List[NodeConfig]
    registry: NodeRegistry = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def to_mermaid(self) -> str:
        """Return a Mermaid.js ``flowchart TD`` diagram of the workflow.

        Paste the output into any Markdown viewer that supports Mermaid
        (GitHub, Notion, Obsidian, etc.). ``on_error`` edges render as
        dashed ``-. on_error .->`` arrows.
        """
        lines = ["flowchart TD"]
        if self.version:
            lines.append(f"    %% workflow version: {self.version}")
        for nc in self.nodes:
            label = nc.description or nc.node
            open_b, close_b = _mermaid_brackets(nc)
            lines.append(f"    {nc.node}{open_b}{label}{close_b}")
        for nc in self.nodes:
            for conn in nc.connections:
                arrow = " -- decision --> " if nc.is_router else " --> "
                lines.append(f"    {nc.node}{arrow}{conn}")
        for nc in self.nodes:
            if nc.on_error is not None:
                lines.append(f"    {nc.node} -. on_error .-> {nc.on_error}")
        if self.start:
            lines.append(f"    start([START]) --> {self.start}")
        return "\n".join(lines) + "\n"

    def to_graph_dict(self) -> dict:
        """Return a JSON-serializable snapshot of this graph's structure,
        for diffing two versions of a workflow in code review — e.g.
        ``schema_a.to_graph_dict() != schema_b.to_graph_dict()``, or a
        line-by-line diff of their ``json.dumps(..., indent=2)`` output.

        Deliberately excludes ``registry`` (maps to ``Node`` subclasses
        — code, not data) and ``event_schema`` (a pydantic model type),
        neither of which round-trips through JSON meaningfully. For the
        same reason, ``NodeConfig.retry`` is reported only as
        ``has_retry`` plus its numeric fields — ``retry_on`` may be an
        arbitrary type or callable. This is a one-way structural
        snapshot, not enough to reconstruct a live ``WorkflowSchema``;
        the registry and event_schema must still come from the caller.
        """
        return {
            "description": self.description,
            "version": self.version,
            "start": self.start,
            "nodes": [_node_config_graph_dict(nc) for nc in self.nodes],
        }

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the workflow.

        Render with: ``dot -Tpng workflow.dot -o workflow.png``
        """
        lines = ["digraph workflow {", '    rankdir=TD;', '    node [shape=box, style=rounded];']
        if self.version:
            lines.append(f'    // workflow version: {self.version}')
        for nc in self.nodes:
            label = nc.description or nc.node
            shape = "diamond" if nc.is_router else "box"
            lines.append(f'    {nc.node} [label="{label}", shape={shape}];')
        for nc in self.nodes:
            for conn in nc.connections:
                label = "decision" if nc.is_router else ""
                lines.append(f'    {nc.node} -> {conn} [label="{label}"];')
        for nc in self.nodes:
            if nc.on_error is not None:
                lines.append(
                    f'    {nc.node} -> {nc.on_error} '
                    f'[label="on_error", style=dashed, color=red];'
                )
        if self.start:
            lines.append(f'    start [label="START", shape=oval];')
            lines.append(f"    start -> {self.start};")
        lines.append("}")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# Mermaid shape helpers
# ------------------------------------------------------------------


def _node_config_graph_dict(nc: NodeConfig) -> dict:
    """The JSON-serializable slice of one NodeConfig — see
    ``WorkflowSchema.to_graph_dict``."""
    return {
        "node": nc.node,
        "connections": list(nc.connections),
        "is_router": nc.is_router,
        "description": nc.description,
        "on_error": nc.on_error,
        "has_retry": nc.retry is not None,
        "retry_max_attempts": nc.retry.max_attempts if nc.retry else None,
        "timeout_s": nc.timeout_s,
    }


def _mermaid_brackets(nc: NodeConfig) -> tuple[str, str]:
    """Return (open, close) bracket pair for a Mermaid node shape."""
    if nc.is_router:
        return "{", "}"            # rhombus — decision
    if not nc.connections:
        return "([", "])"          # stadium — terminal
    return "[", "]"                # rectangle — process

