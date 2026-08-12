"""Suggested record shapes for a workflow trace.

These are exactly that — suggested. Consumers with their own storage
schema (a Delta table, an OpenTelemetry span, a database row) are not
expected to adopt these classes; they exist so ``InMemoryCollector``
and ``JsonlWriter`` have somewhere to put what they collect, and as a
starting point for a consumer who doesn't have an opinion yet.

No field here exists because one specific consumer's domain needed it
— see the strand improvement plan's genericity test. Domain-specific
fields (an email path, a skip reason, an entity ID) belong in the
consumer's own record, correlated by ``execution_id``, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional


@dataclass
class NodeSpan:
    """One node's (or router's) lifecycle within an execution."""

    node_id: str
    node_kind: str
    status: str  # "completed" | "error"
    duration_ms: float
    attempt: int = 1
    output: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class RouteDecision:
    """One router's decision within an execution."""

    router_id: str
    chosen_next: Optional[str]
    matched_rule: Optional[str]
    reason: Optional[str]


@dataclass
class ExecutionRecord:
    """One workflow execution — possibly spanning several nested
    workflows, all sharing one ``execution_id``.

    ``workflow_name``/``workflow_version`` reflect the *outermost*
    workflow only. Spans and routes from nested children are folded
    into the same record's ``spans``/``routes`` lists, in the order
    they actually happened, since a trace should read as one execution
    end to end regardless of how many nested workflows contributed.
    """

    execution_id: str
    workflow_name: str
    workflow_version: Optional[str]
    started_at: datetime
    status: str = "running"
    duration_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    spans: List[NodeSpan] = field(default_factory=list)
    routes: List[RouteDecision] = field(default_factory=list)
