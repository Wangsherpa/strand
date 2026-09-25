"""Shared helpers for the strand test suite."""

import asyncio
import concurrent.futures

from pydantic import BaseModel

from strand.core import (
    Node,
    TaskContext,
    Workflow,
    WorkflowListener,
    WorkflowSchema,
)


class Event(BaseModel):
    value: int = 0


def run_async(coro):
    """Run a coroutine from a sync test, regardless of any running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


class RecordingListener(WorkflowListener):
    """Logs every event the engine dispatches."""

    def __init__(self):
        self.events = []

    def _rec(self, name, **kw):
        self.events.append((name, kw))

    def on_workflow_start(self, run, event):
        self._rec("workflow_start", workflow=run.workflow_name, execution_id=run.execution_id)

    def on_workflow_end(self, run, status, duration_ms, error):
        self._rec(
            "workflow_end", workflow=run.workflow_name, status=status,
            error=type(error).__name__ if error else None,
        )

    def on_node_start(self, run, node_id, node_kind, attempt):
        self._rec("node_start", node_id=node_id, kind=node_kind, attempt=attempt)

    def on_node_end(self, run, node_id, node_kind, status, duration_ms, output, attempt):
        self._rec(
            "node_end", node_id=node_id, kind=node_kind, status=status,
            duration_ms=duration_ms, attempt=attempt,
        )

    def on_node_error(self, run, node_id, node_kind, error, attempt, will_retry, duration_ms):
        self._rec(
            "node_error", node_id=node_id, kind=node_kind,
            error=type(error).__name__, attempt=attempt, will_retry=will_retry,
            duration_ms=duration_ms,
        )

    def on_route(self, run, router_id, chosen_next, matched_rule, reason):
        self._rec("route", router_id=router_id, chosen_next=chosen_next, reason=reason)


class OkNode(Node):
    """A node that does nothing and returns the context."""

    async def process(self, ctx):
        return ctx


class SaveNode(Node):
    """Like OkNode, but records its visit in ctx.nodes via save_output()."""

    async def process(self, ctx):
        self.save_output(self.OutputType())
        return ctx


def make_workflow(*nodes, start, registry, event_schema=Event):
    """Build a Workflow subclass from NodeConfigs (validate on instantiation)."""

    class WF(Workflow):
        workflow_schema = WorkflowSchema(
            event_schema=event_schema, start=start, nodes=list(nodes), registry=registry,
        )

    return WF
