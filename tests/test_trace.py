"""strand.trace regression tests — collector behavior, efficiency, and cleanup."""

import asyncio
import json

import pytest

from strand.core import Node, NodeConfig

from strand.trace.collectors import InMemoryCollector, JsonlWriter, _error_fields

from tests.helpers import SaveNode, make_workflow


class BoomNode(Node):
    """Fails after a tiny delay so the measured duration is non-zero."""

    async def process(self, ctx):
        await asyncio.sleep(0.01)
        raise ValueError("nope")


def _run_boom(listeners):
    with pytest.raises(ValueError, match="nope"):
        make_workflow(
            NodeConfig(node="b"), start="b", registry={"b": BoomNode}
        )(listeners=listeners).run({"value": 1})


# ---------------------------------------------------------------------------
# Error spans carry real data (A1)
# ---------------------------------------------------------------------------


def test_error_span_records_real_duration_ms():
    collector = InMemoryCollector()
    _run_boom([collector])
    record = next(iter(collector.records.values()))
    span = record.spans[0]
    assert span.status == "error"
    assert span.duration_ms > 0  # real measurement, not a fabricated 0.0
    assert span.error_type == "ValueError"
    assert span.error_message == "nope"
    assert record.error_type == "ValueError"


def test_error_fields_helper():
    assert _error_fields(None) == {"error_type": None, "error_message": None}
    assert _error_fields(ValueError("boom")) == {
        "error_type": "ValueError",
        "error_message": "boom",
    }


# ---------------------------------------------------------------------------
# Record lifecycle: depth cleanup, forget, clear (H3)
# ---------------------------------------------------------------------------


def test_depth_counter_cleaned_after_end():
    collector = InMemoryCollector()
    _run_boom([collector])
    assert collector._depth == {}


def test_forget_drops_one_record():
    collector = InMemoryCollector()
    _run_boom([collector])
    execution_id = next(iter(collector.records))
    collector.forget(execution_id)
    assert execution_id not in collector.records
    assert execution_id not in collector._depth


def test_clear_drops_everything():
    collector = InMemoryCollector()
    _run_boom([collector])
    collector.clear()
    assert collector.records == {}
    assert collector._depth == {}


# ---------------------------------------------------------------------------
# Nested workflows fold into one record under one execution_id
# ---------------------------------------------------------------------------


def test_nested_workflows_fold_into_one_record():
    class ParentNest(Node):
        async def process(self, ctx):
            await make_workflow(
                NodeConfig(node="c"), start="c", registry={"c": SaveNode}
            )().run_async(context=ctx)
            return ctx

    collector = InMemoryCollector()
    wf = make_workflow(
        NodeConfig(node="p"), start="p", registry={"p": ParentNest}
    )(listeners=[collector])
    ctx = wf.run({"value": 1})

    records = list(collector.records.values())
    assert len(records) == 1
    assert records[0].execution_id == ctx.execution_id
    assert {span.node_id for span in records[0].spans} == {"p", "c"}
    assert records[0].status == "completed"


# ---------------------------------------------------------------------------
# JsonlWriter: one open handle, real durations, context-manager close (H2)
# ---------------------------------------------------------------------------


def test_jsonl_writer_reuses_one_handle(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = JsonlWriter(path)
    with writer:
        _run_boom([writer])
        first_handle = writer._fh
        _run_boom([writer])
        assert writer._fh is first_handle
    assert writer._fh is None  # closed on context exit


def test_jsonl_writer_writes_events_with_real_durations(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = JsonlWriter(path)
    with writer:
        _run_boom([writer])
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    events = [line["event"] for line in lines]
    # one run: workflow_start, node_start, node_error, workflow_end
    assert events == ["workflow_start", "node_start", "node_error", "workflow_end"]
    node_error = lines[2]
    assert node_error["node_id"] == "b"
    assert node_error["duration_ms"] > 0
    assert node_error["error_type"] == "ValueError"
    assert node_error["will_retry"] is False
