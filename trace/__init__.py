"""strand.trace — an optional, reference WorkflowListener implementation.

Depends on ``strand.core`` (it implements ``WorkflowListener``) and the
stdlib only — no new third-party dependency, no persistence opinion.
This exists so a second consumer of ``strand`` doesn't have to
reimplement "collect events into a record" from scratch; it is not the
only valid way to consume ``WorkflowListener``, and applications with
their own storage (Delta tables, OpenTelemetry, a database) should
write their own listener instead of adapting this one.

Public API
----------

.. autosummary::

    ExecutionRecord    One workflow execution, assembled from its events.
    NodeSpan           One node's start-to-end lifecycle.
    RouteDecision      One router's decision.
    InMemoryCollector  Assembles ExecutionRecords in memory — tests, debugging.
    JsonlWriter        Appends one JSON line per raw event to a file.
"""

from strand.trace.collectors import InMemoryCollector, JsonlWriter
from strand.trace.models import ExecutionRecord, NodeSpan, RouteDecision

__all__ = [
    "ExecutionRecord",
    "InMemoryCollector",
    "JsonlWriter",
    "NodeSpan",
    "RouteDecision",
]
