"""Reference WorkflowListener implementations.

Both consume the same workflow events; they differ in what they do
with them. Neither makes a payload-capture decision on your behalf —
JsonlWriter's best-effort ``model_dump()``-or-``repr()`` and
InMemoryCollector's raw object reference are just this module's own
choices as ONE example consumer, not a policy strand.core enforces.

InMemoryCollector keeps records in memory forever (one per
``execution_id``) until you call ``clear()``/``forget()`` — it is
intended for tests and local debugging. JsonlWriter holds its file
handle open for the writer's lifetime; call ``close()`` (or use it as
a context manager) when you are done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from strand.core.listener import WorkflowListener
from strand.core.run_context import RunContext
from strand.trace.models import ExecutionRecord, NodeSpan, RouteDecision


def _error_fields(error: Optional[BaseException]) -> Dict[str, Optional[str]]:
    """The one place an exception becomes trace fields — every consumer
    of error events (records and JSONL alike) goes through here, so the
    two encodings cannot drift apart."""
    if error is None:
        return {"error_type": None, "error_message": None}
    return {"error_type": type(error).__name__, "error_message": str(error)}


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

    def clear(self) -> None:
        """Drop all records and depth counters."""
        self.records.clear()
        self._depth.clear()

    def forget(self, execution_id: str) -> None:
        """Drop one execution's record (and its depth counter)."""
        self.records.pop(execution_id, None)
        self._depth.pop(execution_id, None)

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
        depth = self._depth.get(run.execution_id, 1) - 1
        if depth > 0:
            self._depth[run.execution_id] = depth
            return  # a nested child ending; the record isn't finished yet
        self._depth.pop(run.execution_id, None)  # execution done — no stale counter
        record = self.records.get(run.execution_id)
        if record is None:
            return
        record.status = status
        record.duration_ms = duration_ms
        fields = _error_fields(error)
        record.error_type = fields["error_type"]
        record.error_message = fields["error_message"]

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
        duration_ms: float,
    ) -> None:
        record = self.records.get(run.execution_id)
        if record is None:
            return
        fields = _error_fields(error)
        record.spans.append(
            NodeSpan(
                node_id=node_id,
                node_kind=node_kind,
                status="error",
                duration_ms=duration_ms,
                attempt=attempt,
                error_type=fields["error_type"],
                error_message=fields["error_message"],
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

    def on_parallel_end(
        self,
        run: RunContext,
        node_id: str,
        status: str,
        duration_ms: float,
        error: Optional[BaseException],
    ) -> None:
        record = self.records.get(run.execution_id)
        if record is None:
            return
        fields = _error_fields(error)
        record.spans.append(
            NodeSpan(
                node_id=node_id,
                node_kind="parallel",
                status=status,
                duration_ms=duration_ms,
                error_type=fields["error_type"],
                error_message=fields["error_message"],
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

    The file handle is opened lazily on the first event and kept for
    the writer's lifetime (opening/closing per event would be wasteful
    and slow on synced volumes); lines are flushed after each write so
    tailing still sees them promptly. Call :meth:`close` when done, or
    use the writer as a context manager.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._fh: Optional[Any] = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying file handle, if any."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write(self, event: str, **fields: Any) -> None:
        record = {"event": event, **fields}
        if self._fh is None:
            self._fh = open(self._path, "a", encoding="utf-8")
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

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
            **_error_fields(error),
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
        duration_ms: float,
    ) -> None:
        self._write(
            "node_error",
            execution_id=run.execution_id,
            node_id=node_id,
            node_kind=node_kind,
            attempt=attempt,
            will_retry=will_retry,
            duration_ms=duration_ms,
            **_error_fields(error),
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

    def on_parallel_start(
        self, run: RunContext, node_id: str, branches: Any
    ) -> None:
        self._write(
            "parallel_start",
            execution_id=run.execution_id,
            node_id=node_id,
            branches=list(branches),
        )

    def on_parallel_end(
        self,
        run: RunContext,
        node_id: str,
        status: str,
        duration_ms: float,
        error: Optional[BaseException],
    ) -> None:
        self._write(
            "parallel_end",
            execution_id=run.execution_id,
            node_id=node_id,
            status=status,
            duration_ms=duration_ms,
            **_error_fields(error),
        )
