"""Core-engine regression tests — the review findings, plus basic behavior."""

import asyncio
import copy
import pickle
import socket

import pytest
from pydantic import ValidationError

from strand.core import (
    BaseRouter,
    Node,
    NodeConfig,
    RetryPolicy,
    RouterNode,
    TaskContext,
    WorkflowListener,
)

from tests.helpers import (
    Event,
    OkNode,
    RecordingListener,
    SaveNode,
    make_workflow,
    run_async,
)


# ---------------------------------------------------------------------------
# Basic behavior (pre-existing contracts that must keep working)
# ---------------------------------------------------------------------------


def test_linear_flow_runs_and_stores_outputs():
    wf = make_workflow(
        NodeConfig(node="a", connections=["b"]),
        NodeConfig(node="b"),
        start="a",
        registry={"a": SaveNode, "b": SaveNode},
    )()
    ctx = wf.run({"value": 1})
    assert set(ctx.nodes) == {"a", "b"}


def test_event_is_validated_through_event_schema():
    wf = make_workflow(NodeConfig(node="a"), start="a", registry={"a": OkNode})()
    with pytest.raises(ValidationError):
        wf.run({"value": "not-an-int"})


def test_missing_event_and_context_rejected():
    wf = make_workflow(NodeConfig(node="a"), start="a", registry={"a": OkNode})()
    with pytest.raises(ValueError, match="event or context"):
        run_async(wf.run_async())


def test_nested_child_shares_execution_id_and_listeners():
    class ParentNest(Node):
        async def process(self, ctx):
            await make_workflow(
                NodeConfig(node="c"), start="c", registry={"c": SaveNode}
            )().run_async(context=ctx)
            return ctx

    listener = RecordingListener()
    wf = make_workflow(NodeConfig(node="p"), start="p", registry={"p": ParentNest})(
        listeners=[listener]
    )
    ctx = wf.run({"value": 1})
    assert ctx.errors == {}
    assert ctx.execution_id is not None
    # both the parent and the nested child's events land under one execution_id
    ids = {kw["execution_id"] for name, kw in listener.events if name == "workflow_start"}
    assert len(ids) == 1
    names = {kw["workflow"] for name, kw in listener.events if name == "workflow_start"}
    assert names == {"WF"}  # parent and child both reported their start


# ---------------------------------------------------------------------------
# RetryPolicy validation (#13)
# ---------------------------------------------------------------------------


class TestRetryPolicyValidation:
    def test_list_retry_on_rejected_at_construction(self):
        with pytest.raises(TypeError, match="retry_on"):
            RetryPolicy(max_attempts=3, retry_on=[ValueError])

    def test_non_exception_tuple_rejected(self):
        with pytest.raises(TypeError, match="retry_on"):
            RetryPolicy(retry_on=(ValueError, "not-an-exception"))

    def test_max_attempts_below_one_rejected(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryPolicy(max_attempts=0)

    def test_negative_backoff_rejected(self):
        with pytest.raises(ValueError, match="backoff"):
            RetryPolicy(backoff_base=-1.0)

    def test_valid_forms_accepted(self):
        RetryPolicy(retry_on=ValueError)
        RetryPolicy(retry_on=(ValueError, TypeError))
        RetryPolicy(retry_on=lambda e: isinstance(e, ValueError))

    def test_should_retry_boundary_and_default_predicate(self):
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry(ValueError("x"), attempt=1) is True
        assert policy.should_retry(ValueError("x"), attempt=2) is True
        assert policy.should_retry(ValueError("x"), attempt=3) is False
        # CancelledError is a BaseException — never matched by the default
        assert policy.should_retry(asyncio.CancelledError(), attempt=1) is False

    def test_backoff_seconds_doubles_and_caps(self):
        policy = RetryPolicy(backoff_base=2.0, backoff_max=10.0)
        assert policy.backoff_seconds(1) == 2.0
        assert policy.backoff_seconds(2) == 4.0
        assert policy.backoff_seconds(3) == 8.0
        assert policy.backoff_seconds(4) == 10.0  # capped


# ---------------------------------------------------------------------------
# on_error cycles are rejected at validation (#2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nodes",
    [
        [NodeConfig(node="boom", on_error="boom")],
        [NodeConfig(node="a", on_error="b"), NodeConfig(node="b", on_error="a")],
        [
            NodeConfig(node="a", connections=["b"], on_error="x"),
            NodeConfig(node="b", on_error="a"),
            NodeConfig(node="x"),
        ],
    ],
    ids=["self", "two-node", "mixed-connections"],
)
def test_on_error_cycles_rejected(nodes):
    with pytest.raises(ValueError, match="cycle"):
        make_workflow(
            *nodes, start=nodes[0].node,
            registry={nc.node: OkNode for nc in nodes},
        )()


def test_acyclic_on_error_accepted():
    wf = make_workflow(
        NodeConfig(node="a", connections=["b"], on_error="h"),
        NodeConfig(node="b"),
        NodeConfig(node="h"),
        start="a",
        registry={"a": SaveNode, "b": SaveNode, "h": SaveNode},
    )()
    ctx = wf.run({"value": 1})
    assert set(ctx.nodes) == {"a", "b"}


# ---------------------------------------------------------------------------
# Cancellation is never retried or routed to on_error (#3)
# ---------------------------------------------------------------------------


class SlowNode(Node):
    async def process(self, ctx):
        await asyncio.sleep(30)
        return ctx


class HandlerNode(Node):
    ran = False

    async def process(self, ctx):
        HandlerNode.ran = True
        return ctx


def _slow_workflow(on_error):
    return make_workflow(
        NodeConfig(node="slow", connections=["handler"],
                   on_error="handler" if on_error else None),
        NodeConfig(node="handler"),
        start="slow",
        registry={"slow": SlowNode, "handler": HandlerNode},
    )()


def test_wait_for_timeout_aborts_even_with_on_error():
    async def scenario():
        HandlerNode.ran = False
        try:
            await asyncio.wait_for(_slow_workflow(True).run_async({"value": 1}), timeout=0.2)
        except asyncio.TimeoutError:
            return True
        return False

    assert run_async(scenario()) is True
    assert HandlerNode.ran is False


def test_task_cancel_stops_the_run():
    async def scenario():
        HandlerNode.ran = False
        task = asyncio.create_task(_slow_workflow(True).run_async({"value": 1}))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await asyncio.wait_for(task, 5)
        except asyncio.CancelledError:
            return True
        return False

    assert run_async(scenario()) is True
    assert HandlerNode.ran is False


# ---------------------------------------------------------------------------
# should_stop survives nested children; resume resets it (#4)
# ---------------------------------------------------------------------------


class StopNode(Node):
    run_child = False

    async def process(self, ctx):
        ctx.stop_workflow()
        if type(self).run_child:
            await make_workflow(
                NodeConfig(node="c"), start="c", registry={"c": SaveNode}
            )().run_async(context=ctx)
        return ctx


class AfterNode(Node):
    ran = False

    async def process(self, ctx):
        AfterNode.ran = True
        return ctx


def _parent_workflow():
    return make_workflow(
        NodeConfig(node="stop", connections=["after"]),
        NodeConfig(node="after"),
        start="stop",
        registry={"stop": StopNode, "after": AfterNode},
    )()


@pytest.mark.parametrize("run_child", [False, True], ids=["no-child", "with-child"])
def test_stop_workflow_halts_the_run(run_child):
    StopNode.run_child = run_child
    AfterNode.ran = False
    _parent_workflow().run({"value": 1})
    assert AfterNode.ran is False


def test_resuming_a_stopped_context_starts_unstopped():
    plain = make_workflow(NodeConfig(node="n"), start="n", registry={"n": SaveNode})()
    ctx = TaskContext(event=Event())
    ctx.should_stop = True
    ctx2 = plain.run(context=ctx)
    assert ctx2.should_stop is False
    assert "n" in ctx2.nodes


# ---------------------------------------------------------------------------
# Legacy router/node constructors keep working (#5)
# ---------------------------------------------------------------------------


class ValueRule(RouterNode):
    def determine_next_node(self, ctx):
        return "branch_a" if ctx.event.value >= 10 else None


class LegacyRouter(BaseRouter):
    routes = [ValueRule()]
    fallback = "branch_b"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold


class NoSuperRouter(BaseRouter):
    routes = [ValueRule()]
    fallback = "branch_b"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold  # never calls super().__init__()


@pytest.mark.parametrize(
    "router_cls", [LegacyRouter, NoSuperRouter], ids=["custom-init", "no-super"]
)
def test_legacy_router_constructors_route(router_cls):
    wf = make_workflow(
        NodeConfig(node="router", is_router=True, connections=["branch_a", "branch_b"]),
        NodeConfig(node="branch_a"),
        NodeConfig(node="branch_b"),
        start="router",
        registry={"router": router_cls, "branch_a": SaveNode, "branch_b": SaveNode},
    )()
    ctx = wf.run({"value": 42})
    assert "branch_a" in ctx.nodes


def test_legacy_node_constructor_keeps_working():
    class LegacyNode(Node):
        def __init__(self, flag: bool = True):  # no engine kwargs
            super().__init__()
            self.flag = flag

        async def process(self, ctx):
            self.save_output(self.OutputType())
            return ctx

    wf = make_workflow(NodeConfig(node="n"), start="n", registry={"n": LegacyNode})()
    ctx = wf.run({"value": 1})
    assert "n" in ctx.nodes


# ---------------------------------------------------------------------------
# Reusing a finished TaskContext starts a fresh execution (#6)
# ---------------------------------------------------------------------------


def test_reusing_finished_context_mints_fresh_execution():
    listener = RecordingListener()
    wf = make_workflow(NodeConfig(node="n"), start="n", registry={"n": SaveNode})(
        listeners=[listener]
    )
    ctx1 = wf.run({"value": 1})
    id1 = ctx1.execution_id
    events_after_run1 = len(listener.events)

    ctx2 = wf.run(context=ctx1)
    assert ctx2.execution_id != id1
    # each run produces exactly 4 events (start/start/end/end) — not doubled
    assert len(listener.events) == events_after_run1 + 4


# ---------------------------------------------------------------------------
# process() may return a fresh context or None (#7)
# ---------------------------------------------------------------------------


class FreshCtxNode(Node):
    async def process(self, ctx):
        return TaskContext(event=ctx.event)


class RoundTripNode(Node):
    async def process(self, ctx):
        return TaskContext.model_validate(ctx.model_dump())


class NoneReturnNode(Node):
    cleaned = 0

    async def process(self, ctx):
        return None

    async def cleanup(self):
        NoneReturnNode.cleaned += 1


@pytest.mark.parametrize(
    "node_cls", [FreshCtxNode, RoundTripNode, NoneReturnNode],
    ids=["fresh-context", "round-trip", "none-return"],
)
def test_returned_context_variants_complete(node_cls):
    wf = make_workflow(
        NodeConfig(node="first", connections=["second"]),
        NodeConfig(node="second"),
        start="first",
        registry={"first": node_cls, "second": SaveNode},
    )()
    ctx = wf.run({"value": 1})
    assert "second" in ctx.nodes


def test_none_return_node_is_still_cleaned_up():
    wf = make_workflow(NodeConfig(node="n"), start="n", registry={"n": NoneReturnNode})()
    NoneReturnNode.cleaned = 0
    wf.run({"value": 1})
    assert NoneReturnNode.cleaned == 1


def test_non_context_return_is_a_clear_type_error():
    class BadReturn(Node):
        async def process(self, ctx):
            return "not-a-context"

    wf = make_workflow(NodeConfig(node="b"), start="b", registry={"b": BadReturn})()
    with pytest.raises(TypeError, match="expected the TaskContext"):
        wf.run({"value": 1})


# ---------------------------------------------------------------------------
# cleanup() ordering (#8)
# ---------------------------------------------------------------------------


class BoomNode(Node):
    runs = 0

    async def process(self, ctx):
        BoomNode.runs += 1
        raise ValueError("real error")


class BadCleanupNode(BoomNode):
    async def cleanup(self):
        raise RuntimeError("teardown failed")


def test_cleanup_failure_does_not_mask_node_error(no_backoff):
    listener = RecordingListener()
    wf = make_workflow(
        NodeConfig(node="boom", connections=["handler"], on_error="handler",
                   retry=RetryPolicy(max_attempts=3, backoff_base=1.0)),
        NodeConfig(node="handler"),
        start="boom",
        registry={"boom": BadCleanupNode, "handler": HandlerNode},
    )(listeners=[listener])
    BoomNode.runs = 0
    HandlerNode.ran = False

    ctx = wf.run({"value": 1})

    assert BoomNode.runs == 3                     # retries still happened
    assert HandlerNode.ran is True                # on_error still routed
    assert "ValueError" in ctx.errors["boom"]     # the node's own error recorded
    err_events = [kw for name, kw in listener.events if name == "node_error"]
    assert [e["will_retry"] for e in err_events] == [True, True, False]
    # the retry sleeps went through the seam (2 backoff waits, instantaneous)
    assert len(no_backoff["engine"]) == 2


def test_cleanup_failure_with_no_node_error_propagates():
    class CleanupOnlyFail(Node):
        async def process(self, ctx):
            return ctx

        async def cleanup(self):
            raise RuntimeError("teardown failed")

    wf = make_workflow(NodeConfig(node="c"), start="c", registry={"c": CleanupOnlyFail})()
    with pytest.raises(RuntimeError, match="teardown failed"):
        wf.run({"value": 1})


# ---------------------------------------------------------------------------
# Implicit router targets rejected at construction (#9)
# ---------------------------------------------------------------------------


class AlwaysRule(RouterNode):
    def determine_next_node(self, ctx):
        return "end"


class AlwaysRouter(BaseRouter):
    routes = [AlwaysRule()]


def test_implicit_router_target_rejected():
    with pytest.raises(ValueError, match="explicit"):
        make_workflow(
            NodeConfig(node="start", connections=["route"]),
            start="start",
            registry={"start": OkNode, "route": AlwaysRouter, "end": SaveNode},
        )()


# ---------------------------------------------------------------------------
# Router failure events (A2) and on_node_error duration (A1)
# ---------------------------------------------------------------------------


class ExplodingRule(RouterNode):
    def determine_next_node(self, ctx):
        raise RuntimeError("rule exploded")


class ExplodingRouter(BaseRouter):
    routes = [ExplodingRule()]


def test_router_exception_reported_as_error_event():
    listener = RecordingListener()
    wf = make_workflow(
        NodeConfig(node="r", is_router=True, connections=["a", "b"]),
        NodeConfig(node="a"),
        NodeConfig(node="b"),
        start="r",
        registry={"r": ExplodingRouter, "a": SaveNode, "b": SaveNode},
    )(listeners=[listener])
    with pytest.raises(RuntimeError, match="rule exploded"):
        wf.run({"value": 1})

    errs = [kw for name, kw in listener.events if name == "node_error"]
    assert len(errs) == 1
    assert errs[0]["error"] == "RuntimeError"
    assert errs[0]["duration_ms"] >= 0
    # the router was NOT reported as completed
    ends = [kw for name, kw in listener.events if name == "node_end"]
    assert all(kw["node_id"] != "r" for kw in ends)


# ---------------------------------------------------------------------------
# Listener dispatch: async hooks awaited (A4), raising listeners swallowed (E6)
# ---------------------------------------------------------------------------


def test_async_listener_hooks_are_awaited():
    class AsyncListener(WorkflowListener):
        def __init__(self):
            self.hits = []

        async def on_node_end(self, run, node_id, node_kind, status, duration_ms, output, attempt):
            self.hits.append(node_id)

    al = AsyncListener()
    wf = make_workflow(NodeConfig(node="n"), start="n", registry={"n": SaveNode})(
        listeners=[al]
    )
    wf.run({"value": 1})
    assert al.hits == ["n"]


def test_raising_listener_is_swallowed():
    class RudeListener(WorkflowListener):
        def on_node_end(self, *args):
            raise KeyboardInterrupt("rude listener")

    wf = make_workflow(NodeConfig(node="n"), start="n", registry={"n": SaveNode})(
        listeners=[RudeListener()]
    )
    ctx = wf.run({"value": 1})  # must not raise
    assert "n" in ctx.nodes


# ---------------------------------------------------------------------------
# Serialization excludes the RunContext (#15)
# ---------------------------------------------------------------------------


class SocketListener(WorkflowListener):
    def __init__(self):
        self.sock = socket.socket()


def test_pickle_and_deepcopy_exclude_run_context():
    sl = SocketListener()
    try:
        wf = make_workflow(NodeConfig(node="n"), start="n", registry={"n": SaveNode})(
            listeners=[sl]
        )
        ctx = wf.run({"value": 1})

        restored = pickle.loads(pickle.dumps(ctx))
        assert restored.nodes == ctx.nodes
        assert restored.execution_id is None

        copied = copy.deepcopy(ctx)
        assert copied.nodes == ctx.nodes
        assert copied.execution_id is None
    finally:
        sl.sock.close()


# ---------------------------------------------------------------------------
# BaseRouter routes: instance-assigned rules survive, rule state isolated (A3)
# ---------------------------------------------------------------------------


def test_instance_assigned_routes_are_used():
    class RuleA(RouterNode):
        def determine_next_node(self, ctx):
            return "a"

    class InstanceRoutes(BaseRouter):
        def __init__(self, **kwargs):
            self.routes = [RuleA()]  # assigned before super().__init__()
            super().__init__(**kwargs)

    router = InstanceRoutes(task_context=TaskContext(event=Event(value=1)), node_id="rt")
    assert router.route(TaskContext(event=Event(value=1))) == "a"


def test_rule_state_isolated_between_router_instances():
    class StatefulRule(RouterNode):
        def __init__(self, *args, seen=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen = seen if seen is not None else []

        def determine_next_node(self, ctx):
            self.seen.append(ctx.event.value)
            return "a"

    class StatefulRouter(BaseRouter):
        routes = [StatefulRule()]

    r1 = StatefulRouter(task_context=TaskContext(event=Event(value=1)), node_id="r1")
    r2 = StatefulRouter(task_context=TaskContext(event=Event(value=2)), node_id="r2")
    r1.route(TaskContext(event=Event(value=1)))
    r2.route(TaskContext(event=Event(value=2)))
    assert r1.routes[0].seen == [1]
    assert r2.routes[0].seen == [2]
    # the class-level rule object is untouched
    assert StatefulRouter.routes[0].seen == []


# ---------------------------------------------------------------------------
# Schema: visualization includes on_error (A5), unknown fields rejected (A6)
# ---------------------------------------------------------------------------


def test_visualization_includes_on_error_edges():
    wf = make_workflow(
        NodeConfig(node="boom", connections=["ok"], on_error="handler"),
        NodeConfig(node="ok"),
        NodeConfig(node="handler"),
        start="boom",
        registry={"boom": OkNode, "ok": OkNode, "handler": OkNode},
    )()
    assert "on_error" in wf.to_mermaid()
    assert "on_error" in wf.to_dot()


def test_misspelled_node_config_fields_rejected():
    with pytest.raises(ValidationError):
        NodeConfig(node="a", on_erorr="handler", retrys=RetryPolicy(max_attempts=3))


# ---------------------------------------------------------------------------
# get_node_config accessor (A7)
# ---------------------------------------------------------------------------


def test_get_node_config_returns_the_current_config():
    class ConfigReader(Node):
        saw = None

        async def process(self, ctx):
            ConfigReader.saw = ctx.get_node_config("m")
            return ctx

    wf = make_workflow(
        NodeConfig(node="m", retry=RetryPolicy(max_attempts=2)),
        start="m",
        registry={"m": ConfigReader},
    )()
    wf.run({"value": 1})
    assert ConfigReader.saw is not None
    assert ConfigReader.saw.retry.max_attempts == 2
