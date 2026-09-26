"""Parallel-group execution tests — fail_fast semantics (phase 3a)."""

import asyncio

import pytest

from strand.core import Node, NodeConfig, TaskContext

from strand.trace.collectors import InMemoryCollector, JsonlWriter

from tests.helpers import OkNode, RecordingListener, SaveNode, make_workflow, run_async


def _group(parallel, **overrides):
    kwargs = dict(node="group", parallel=parallel, connections=["join"])
    kwargs.update(overrides)
    return NodeConfig(**kwargs)


# ---------------------------------------------------------------------------
# Happy path: concurrency, join, nested groups
# ---------------------------------------------------------------------------


def test_branches_run_concurrently_and_join_runs():
    # Deterministic proof of concurrency: branch A waits for a gate that
    # only branch B can open. Sequential execution would deadlock (and
    # fail the wait_for timeout); concurrent execution passes.
    gates = {"a_entered": asyncio.Event(), "b_opens": asyncio.Event()}

    class BranchA(Node):
        async def process(self, ctx):
            gates["a_entered"].set()
            await asyncio.wait_for(gates["b_opens"].wait(), 5)
            self.save_output(self.OutputType())
            return ctx

    class BranchB(Node):
        async def process(self, ctx):
            await gates["a_entered"].wait()   # A is running before B opens the gate
            gates["b_opens"].set()
            self.save_output(self.OutputType())
            return ctx

    wf = make_workflow(
        _group(["a", "b"]),
        NodeConfig(node="a"), NodeConfig(node="b"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": BranchA, "b": BranchB, "join": SaveNode},
    )()
    ctx = wf.run({"value": 1})
    assert {"a", "b", "join"} <= set(ctx.nodes)


def test_join_sees_all_branch_outputs():
    class Join(Node):
        async def process(self, ctx):
            assert {"a", "b"} <= set(ctx.nodes), ctx.nodes
            self.save_output(self.OutputType())
            return ctx

    wf = make_workflow(
        _group(["a", "b"]),
        NodeConfig(node="a"), NodeConfig(node="b"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": SaveNode, "b": SaveNode, "join": Join},
    )()
    ctx = wf.run({"value": 1})
    assert "join" in ctx.nodes


def test_nested_groups_run():
    wf = make_workflow(
        _group(["subgroup", "b"]),
        NodeConfig(node="subgroup", parallel=["x", "y"]),   # nested group, terminal
        NodeConfig(node="x"), NodeConfig(node="y"), NodeConfig(node="b"),
        NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "subgroup": OkNode,
                  "x": SaveNode, "y": SaveNode, "b": SaveNode, "join": SaveNode},
    )()
    ctx = wf.run({"value": 1})
    assert {"x", "y", "b", "join"} <= set(ctx.nodes)


# ---------------------------------------------------------------------------
# Fail-fast semantics
# ---------------------------------------------------------------------------


def test_branch_failure_cancels_siblings_and_propagates():
    flags = {"sibling_cancelled": False, "join_ran": False}

    class FailingBranch(Node):
        async def process(self, ctx):
            await asyncio.sleep(0.01)          # let the sibling start first
            raise ValueError("branch failed")

    class SlowBranch(Node):
        async def process(self, ctx):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                flags["sibling_cancelled"] = True
                raise
            return ctx

    class Join(Node):
        async def process(self, ctx):
            flags["join_ran"] = True
            return ctx

    wf = make_workflow(
        _group(["fail", "slow"]),
        NodeConfig(node="fail"), NodeConfig(node="slow"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "fail": FailingBranch,
                  "slow": SlowBranch, "join": Join},
    )()
    with pytest.raises(ValueError, match="branch failed"):
        wf.run({"value": 1})
    assert flags["sibling_cancelled"] is True   # fail-fast cancelled the sibling
    assert flags["join_ran"] is False


def test_group_on_error_routes():
    class FailingBranch(Node):
        async def process(self, ctx):
            raise ValueError("branch failed")

    class Handler(Node):
        async def process(self, ctx):
            self.save_output(self.OutputType())
            return ctx

    wf = make_workflow(
        _group(["fail"], on_error="handler"),
        NodeConfig(node="fail"), NodeConfig(node="join"), NodeConfig(node="handler"),
        start="group",
        registry={"group": OkNode, "fail": FailingBranch,
                  "join": SaveNode, "handler": Handler},
    )()
    ctx = wf.run({"value": 1})
    assert "handler" in ctx.nodes
    assert "join" not in ctx.nodes          # routed away from the join
    assert "ValueError" in ctx.errors["group"]


def test_group_timeout_cancels_branches():
    flags = {"cancelled": False}

    class SlowBranch(Node):
        async def process(self, ctx):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                flags["cancelled"] = True
                raise
            return ctx

    wf = make_workflow(
        _group(["slow"], timeout_s=0.05),
        NodeConfig(node="slow"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "slow": SlowBranch, "join": SaveNode},
    )()
    with pytest.raises(asyncio.TimeoutError):
        wf.run({"value": 1})
    assert flags["cancelled"] is True


def test_external_cancellation_stops_the_whole_group():
    class SlowBranch(Node):
        async def process(self, ctx):
            await asyncio.sleep(30)
            return ctx

    wf = make_workflow(
        _group(["a", "b"]),
        NodeConfig(node="a"), NodeConfig(node="b"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": SlowBranch, "b": SlowBranch, "join": SaveNode},
    )()

    async def scenario():
        task = asyncio.create_task(wf.run_async({"value": 1}))
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_async(scenario())


def test_stop_workflow_in_branch_halts_before_join():
    flags = {"join_ran": False}

    class StoppingBranch(Node):
        async def process(self, ctx):
            ctx.stop_workflow()
            self.save_output(self.OutputType())
            return ctx

    class Join(Node):
        async def process(self, ctx):
            flags["join_ran"] = True
            return ctx

    wf = make_workflow(
        _group(["stop", "other"]),
        NodeConfig(node="stop"), NodeConfig(node="other"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "stop": StoppingBranch,
                  "other": SaveNode, "join": Join},
    )()
    ctx = wf.run({"value": 1})
    assert flags["join_ran"] is False       # the outer walk honoured the stop
    assert "stop" in ctx.nodes


# ---------------------------------------------------------------------------
# Branch contract: shared context only
# ---------------------------------------------------------------------------


def test_branch_replacing_context_rejected():
    class FreshCtxBranch(Node):
        async def process(self, ctx):
            return TaskContext(event=ctx.event)

    wf = make_workflow(
        _group(["fresh"]),
        NodeConfig(node="fresh"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "fresh": FreshCtxBranch, "join": SaveNode},
    )()
    with pytest.raises(RuntimeError, match="parallel branch"):
        wf.run({"value": 1})


# ---------------------------------------------------------------------------
# Observability: parallel events land in listeners and collectors
# ---------------------------------------------------------------------------


def test_parallel_events_reported_to_listener_and_collector():
    listener = RecordingListener()
    collector = InMemoryCollector()
    wf = make_workflow(
        _group(["a", "b"]),
        NodeConfig(node="a"), NodeConfig(node="b"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": SaveNode, "b": SaveNode, "join": SaveNode},
    )(listeners=[listener, collector])
    ctx = wf.run({"value": 1})

    starts = [kw for name, kw in listener.events if name == "parallel_start"]
    ends = [kw for name, kw in listener.events if name == "parallel_end"]
    assert len(starts) == 1
    assert starts[0]["node_id"] == "group"
    assert starts[0]["branches"] == ["a", "b"]
    assert len(ends) == 1
    assert ends[0]["status"] == "completed"
    assert ends[0]["duration_ms"] >= 0

    record = collector.records[ctx.execution_id]
    group_spans = [s for s in record.spans if s.node_kind == "parallel"]
    assert len(group_spans) == 1
    assert group_spans[0].status == "completed"


def test_parallel_error_events_carry_the_failure(tmp_path):
    class FailingBranch(Node):
        async def process(self, ctx):
            raise ValueError("branch failed")

    listener = RecordingListener()
    writer = JsonlWriter(tmp_path / "trace.jsonl")
    with writer:
        wf = make_workflow(
            _group(["fail"]),
            NodeConfig(node="fail"), NodeConfig(node="join"),
            start="group",
            registry={"group": OkNode, "fail": FailingBranch, "join": SaveNode},
        )(listeners=[listener, writer])
        with pytest.raises(ValueError, match="branch failed"):
            wf.run({"value": 1})

    ends = [kw for name, kw in listener.events if name == "parallel_end"]
    assert ends[0]["status"] == "error"
    assert ends[0]["error"] == "ValueError"

    import json as _json
    lines = [_json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    parallel_end = [l for l in lines if l["event"] == "parallel_end"][0]
    assert parallel_end["status"] == "error"
    assert parallel_end["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# collect policy
# ---------------------------------------------------------------------------


def test_collect_runs_every_branch_despite_failures():
    flags = {"slow_completed": False, "join_ran": False}

    class FailingBranch(Node):
        async def process(self, ctx):
            await asyncio.sleep(0.01)
            raise ValueError("branch failed")

    class SlowBranch(Node):
        async def process(self, ctx):
            await asyncio.sleep(0.05)          # would be cancelled under fail_fast
            flags["slow_completed"] = True
            self.save_output(self.OutputType())
            return ctx

    class Join(Node):
        async def process(self, ctx):
            flags["join_ran"] = True
            self.save_output(self.OutputType())
            return ctx

    wf = make_workflow(
        _group(["fail", "slow"], error_policy="collect"),
        NodeConfig(node="fail"), NodeConfig(node="slow"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "fail": FailingBranch,
                  "slow": SlowBranch, "join": Join},
    )()
    ctx = wf.run({"value": 1})                     # does NOT raise
    assert flags["slow_completed"] is True         # sibling was not cancelled
    assert flags["join_ran"] is True               # join still ran
    assert "ValueError" in ctx.errors["fail"]
    assert "slow" in ctx.nodes                     # partial output preserved


def test_collect_records_every_failed_branch():
    class FailingBranch(Node):
        async def process(self, ctx):
            raise ValueError(self.node_name)

    wf = make_workflow(
        _group(["a", "b"], error_policy="collect"),
        NodeConfig(node="a"), NodeConfig(node="b"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": FailingBranch, "b": FailingBranch,
                  "join": SaveNode},
    )()
    ctx = wf.run({"value": 1})
    assert "ValueError" in ctx.errors["a"]
    assert "ValueError" in ctx.errors["b"]
    assert "group" not in ctx.errors                # branch failures are not group failures


def test_collect_join_salvages_partial_success():
    class FailingBranch(Node):
        async def process(self, ctx):
            raise ValueError("no data")

    class SalvageJoin(Node):
        class OutputType(Node.OutputType):
            ok: bool
            missing: list

        async def process(self, ctx):
            missing = [b for b in ("a", "b") if self.get_error(b) is not None]
            self.save_output(self.OutputType(ok=not missing, missing=missing))
            return ctx

    wf = make_workflow(
        _group(["a", "b"], error_policy="collect"),
        NodeConfig(node="a"), NodeConfig(node="b"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": SaveNode, "b": FailingBranch,
                  "join": SalvageJoin},
    )()
    ctx = wf.run({"value": 1})
    assert ctx.nodes["join"].ok is False
    assert ctx.nodes["join"].missing == ["b"]


def test_collect_branch_on_error_recovery_completes_the_group():
    class FailingBranch(Node):
        async def process(self, ctx):
            raise ValueError("primary failed")

    class BranchHandler(Node):
        async def process(self, ctx):
            self.save_output(self.OutputType())
            return ctx

    wf = make_workflow(
        _group(["a"], error_policy="collect"),
        NodeConfig(node="a", on_error="a_handler"),
        NodeConfig(node="a_handler"),
        NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": FailingBranch,
                  "a_handler": BranchHandler, "join": SaveNode},
    )()
    ctx = wf.run({"value": 1})
    assert "a_handler" in ctx.nodes               # the branch recovered on its own
    assert "join" in ctx.nodes                    # the group completed normally
    # the node-level failure summary is still recorded (existing on_error
    # semantics: the failing node's error lands in ctx.errors)
    assert "ValueError" in ctx.errors["a"]


def test_collect_cancellation_still_propagates():
    class SlowBranch(Node):
        async def process(self, ctx):
            await asyncio.sleep(30)
            return ctx

    wf = make_workflow(
        _group(["a"], error_policy="collect"),
        NodeConfig(node="a"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": SlowBranch, "join": SaveNode},
    )()

    async def scenario():
        task = asyncio.create_task(wf.run_async({"value": 1}))
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_async(scenario())


def test_collect_group_timeout_is_group_failure():
    class SlowBranch(Node):
        async def process(self, ctx):
            await asyncio.sleep(30)
            return ctx

    wf = make_workflow(
        _group(["a"], error_policy="collect", timeout_s=0.05),
        NodeConfig(node="a"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "a": SlowBranch, "join": SaveNode},
    )()
    with pytest.raises(asyncio.TimeoutError):
        wf.run({"value": 1})


def test_collect_does_not_collect_non_exception_base():
    class RudeBranch(Node):
        async def process(self, ctx):
            raise KeyboardInterrupt("rude branch")

    wf = make_workflow(
        _group(["rude"], error_policy="collect"),
        NodeConfig(node="rude"), NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "rude": RudeBranch, "join": SaveNode},
    )()
    with pytest.raises(KeyboardInterrupt, match="rude branch"):
        wf.run({"value": 1})
