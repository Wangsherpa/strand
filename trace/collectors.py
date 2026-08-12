"""Reference WorkflowListener implementations.

Both collect the same six events; they differ only in what they do
with them. Neither makes a payload-capture decision on your behalf —
JsonlWriter's best-effort ``model_dump()``-or-``repr()`` and
InMemoryCollector's raw object reference are just this module's own
choices as ONE example consumer, not a policy strand.core enforces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from strand.core.listener import WorkflowListener
from strand.core.run_context import RunContext
from strand.trace.models import ExecutionRecord, NodeSpan, RouteDecision


class InMemoryCollector(WorkflowListener):
    """Assembles one ``ExecutionRecord`` per ``execution_id`` in memory.

    Handles nested workflows via a per-execution nesting-depth counter:
    the first ``on_workflow_start`` for a given ``execution_id`` creates
    the record (its ``workflow_name`` becomes the record's, since the
    outermost workflow is always the first to start); the matching
    depth-zero ``on_workflow_end`` finalizes ``status``/``duration_ms``.
    Node spans and route decisions from every nested workflow in
    between land in that same record, in the order they happened.

    Intended for tests and local debugging — for anything that needs
    to survive past the process, write your own listener with your own
    storage.
    """

    def __init__(self) -> None:
        self.records: Dict[str, ExecutionRecord] = {}
        self._depth: Dict[str, int] = {}

    def on_workflow_start(self, run: RunContext, event: Any) -> None:
        depth = self._depth.get(run.execution_id, 0)
        if depth == 0:
            self.records[run.execution_id] = ExecutionRecord(
                execution_id=run.execution_id,
                workflow_name=run.workflow_name,
                workflow_version=run.workflow_version,
                started_at=run.started_at,
            )
        self._depth[run.execution_id] = depth + 1

    def on_workflow_end(
        self,
        run: RunContext,
        status: str,
        duration_ms: float,
        error: Optional[BaseException],
    ) -> None:
        self._depth[run.execution_id] = self._depth.get(run.execution_id, 1) - 1
        if self._depth[run.execution_id] > 0:
            return  # a nested child ending; the record isn't finished yet
        record = self.records.get(run.execution_id)
        if record is None:
            return
        record.status = status
        record.duration_ms = duration_ms
        if error is not None:
            record.error_type = type(error).__name__
            record.error_message = str(error)

    def on_node_end(
        self,
        run: RunContext,
        node_id: str,
        node_kind: str,
        status: str,
        duration_ms: float,
        output: Any,
        attempt: int,
    ) -> None:
        record = self.records.get(run.execution_id)
        if record is None:
            return
        record.spans.append(
            NodeSpan(
                node_id=node_id,
                node_kind=node_kind,
                status=status,
                duration_ms=duration_ms,
                output=output,
                attempt=attempt,
            )
        )

    def on_node_error(
        self,
        run: RunContext,
        node_id: str,
        node_kind: str,
        error: BaseException,
        attempt: int,
        will_retry: bool,
    ) -> None:
        record = self.records.get(run.execution_id)
        if record is None:
            return
        record.spans.append(
            NodeSpan(
                node_id=node_id,
                node_kind=node_kind,
                status="error",
                duration_ms=0.0,
                attempt=attempt,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )

    def on_route(
        self,
        run: RunContext,
        router_id: str,
        chosen_next: Optional[str],
        matched_rule: Optional[str],
        reason: Optional[str],
    ) -> None:
        record = self.records.get(run.execution_id)
        if record is None:
            return
        record.routes.append(
            RouteDecision(
                router_id=router_id,
                chosen_next=chosen_next,
                matched_rule=matched_rule,
                reason=reason,
            )
        )


def _best_effort_payload(value: Any) -> Any:
    """A pydantic model dumps cleanly to JSON; anything else falls
    back to repr(). This is JsonlWriter's own choice as one example
    consumer — not a payload policy strand.core imposes."""
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            pass
    return repr(value)


class JsonlWriter(WorkflowListener):
    """Appends one JSON line per raw event to *path*, as it happens.

    Unlike ``InMemoryCollector`` (which assembles complete records),
    this writes events immediately — suitable for tailing a live run,
    or for a separate downstream process to assemble records from.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)

    def _write(self, event: str, **fields: Any) -> None:
        record = {"event": event, **fields}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def on_workflow_start(self, run: RunContext, event: Any) -> None:
        self._write(
            "workflow_start",
            execution_id=run.execution_id,
            workflow_name=run.workflow_name,
            workflow_version=run.workflow_version,
            started_at=run.started_at,
        )

    def on_workflow_end(
        self,
        run: RunContext,
        status: str,
        duration_ms: float,
        error: Optional[BaseException],
    ) -> None:
        self._write(
            "workflow_end",
            execution_id=run.execution_id,
            workflow_name=run.workflow_name,
            status=status,
            duration_ms=duration_ms,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
        )

    def on_node_start(
        self, run: RunContext, node_id: str, node_kind: str, attempt: int
    ) -> None:
        self._write(
            "node_start",
            execution_id=run.execution_id,
            node_id=node_id,
            node_kind=node_kind,
            attempt=attempt,
        )

    def on_node_end(
        self,
        run: RunContext,
        node_id: str,
        node_kind: str,
        status: str,
        duration_ms: float,
        output: Any,
        attempt: int,
    ) -> None:
        self._write(
            "node_end",
            execution_id=run.execution_id,
            node_id=node_id,
            node_kind=node_kind,
            status=status,
            duration_ms=duration_ms,
            output=_best_effort_payload(output),
            attempt=attempt,
        )

    def on_node_error(
        self,
        run: RunContext,
        node_id: str,
        node_kind: str,
        error: BaseException,
        attempt: int,
        will_retry: bool,
    ) -> None:
        self._write(
            "node_error",
            execution_id=run.execution_id,
            node_id=node_id,
            node_kind=node_kind,
            error_type=type(error).__name__,
            error_message=str(error),
            attempt=attempt,
            will_retry=will_retry,
        )

    def on_route(
        self,
        run: RunContext,
        router_id: str,
        chosen_next: Optional[str],
        matched_rule: Optional[str],
        reason: Optional[str],
    ) -> None:
        self._write(
            "route",
            execution_id=run.execution_id,
            router_id=router_id,
            chosen_next=chosen_next,
            matched_rule=matched_rule,
            reason=reason,
        )
