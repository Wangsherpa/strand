"""Parallel-group schema/validation tests (execution lands in a later phase).

These lock down the declarative half of the feature: field defaults,
structural validation rules, visualization, and the loud
NotImplementedError guard that runs until execution exists.
"""

import pytest
from pydantic import ValidationError

from strand.core import BaseRouter, NodeConfig, RetryPolicy, RouterNode

from tests.helpers import OkNode, SaveNode, make_workflow


def _group_node(parallel, **overrides):
    kwargs = dict(node="group", parallel=parallel, connections=["join"])
    kwargs.update(overrides)
    return NodeConfig(**kwargs)


def _registry(*branches):
    return {b: SaveNode for b in branches} | {"group": OkNode, "join": SaveNode}


# ---------------------------------------------------------------------------
# Field defaults and pydantic-level validation
# ---------------------------------------------------------------------------


def test_parallel_defaults_to_empty_and_fail_fast():
    nc = NodeConfig(node="a")
    assert nc.parallel == []
    assert nc.error_policy == "fail_fast"


def test_bad_error_policy_value_rejected():
    with pytest.raises(ValidationError):
        NodeConfig(node="a", error_policy="bogus")


# ---------------------------------------------------------------------------
# Structural validation rules (check #6)
# ---------------------------------------------------------------------------


def test_error_policy_without_parallel_rejected():
    with pytest.raises(ValueError, match="error_policy"):
        make_workflow(
            NodeConfig(node="a", error_policy="collect"),
            start="a",
            registry={"a": OkNode},
        )()


def test_parallel_with_is_router_rejected():
    with pytest.raises(ValueError, match="cannot also be a router"):
        make_workflow(
            _group_node(["b"], is_router=True),
            start="group",
            registry=_registry("b"),
        )()


def test_parallel_with_multiple_connections_rejected():
    # the generic multi-connection rule fires first — either message is a
    # correct rejection of the same schema
    with pytest.raises(ValueError, match="multiple connections|single"):
        make_workflow(
            _group_node(["b"], connections=["join1", "join2"]),
            start="group",
            registry=_registry("b") | {"join1": SaveNode, "join2": SaveNode},
        )()


def test_duplicate_branch_rejected():
    with pytest.raises(ValueError, match="more than once"):
        make_workflow(
            _group_node(["b", "b"]),
            start="group",
            registry=_registry("b"),
        )()


def test_self_branch_rejected():
    # cycle detection catches this first ("a group listing itself IS a
    # cycle") — the dedicated "lists itself" message is the fallback
    with pytest.raises(ValueError, match="cycle|itself"):
        make_workflow(
            _group_node(["group"]),
            start="group",
            registry=_registry(),
        )()


def test_unknown_branch_key_rejected():
    with pytest.raises(ValueError, match="unknown connection/parallel"):
        make_workflow(
            _group_node(["missing"]),
            start="group",
            registry=_registry(),
        )()


def test_retry_on_group_rejected():
    with pytest.raises(ValueError, match="retry is not supported on parallel"):
        make_workflow(
            _group_node(["b"], retry=RetryPolicy(max_attempts=3)),
            start="group",
            registry=_registry("b"),
        )()


# ---------------------------------------------------------------------------
# Cycle detection and reachability follow parallel edges
# ---------------------------------------------------------------------------


def test_cycle_through_branches_rejected():
    # group -> branch b -> back to group: infinite recursion at runtime
    with pytest.raises(ValueError, match="cycle"):
        make_workflow(
            NodeConfig(node="group", parallel=["b"], connections=["join"]),
            NodeConfig(node="b", connections=["group"]),
            NodeConfig(node="join"),
            start="group",
            registry={"group": OkNode, "b": SaveNode, "join": SaveNode},
        )()


def test_branch_reachable_only_via_parallel_accepted():
    wf = make_workflow(
        NodeConfig(node="group", parallel=["vendor", "items"], connections=["join"]),
        NodeConfig(node="vendor"),
        NodeConfig(node="items"),
        NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "vendor": SaveNode,
                  "items": SaveNode, "join": SaveNode},
    )()
    assert wf is not None  # construction (and therefore validation) passed


def test_implicit_branch_gets_terminal_config():
    # branch has no explicit NodeConfig -> implicit terminal config
    wf = make_workflow(
        NodeConfig(node="group", parallel=["vendor"], connections=["join"]),
        NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "vendor": SaveNode, "join": SaveNode},
    )()
    assert wf._node_configs["vendor"].connections == []
    assert wf._node_configs["vendor"].parallel == []


def test_implicit_router_branch_rejected():
    class BranchRule(RouterNode):
        def determine_next_node(self, ctx):
            return "join"

    class BranchRouter(BaseRouter):
        routes = [BranchRule()]

    with pytest.raises(ValueError, match="explicit"):
        make_workflow(
            NodeConfig(node="group", parallel=["router_branch"], connections=["join"]),
            NodeConfig(node="join"),
            start="group",
            registry={"group": OkNode, "router_branch": BranchRouter,
                      "join": SaveNode},
        )()


# ---------------------------------------------------------------------------
# Visualization and graph dict
# ---------------------------------------------------------------------------


def test_mermaid_renders_parallel_edges():
    wf = make_workflow(
        NodeConfig(node="group", parallel=["vendor", "items"], connections=["join"]),
        NodeConfig(node="vendor"),
        NodeConfig(node="items"),
        NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "vendor": SaveNode,
                  "items": SaveNode, "join": SaveNode},
    )()
    mermaid = wf.to_mermaid()
    assert "group == parallel ==> vendor" in mermaid
    assert "group == parallel ==> items" in mermaid
    assert "group --> join" in mermaid  # the continuation edge


def test_dot_renders_parallel_edges():
    wf = make_workflow(
        NodeConfig(node="group", parallel=["vendor"], connections=["join"]),
        NodeConfig(node="vendor"),
        NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "vendor": SaveNode, "join": SaveNode},
    )()
    dot = wf.to_dot()
    assert 'label="parallel"' in dot
    assert 'label="on_error"' not in dot


def test_graph_dict_includes_parallel_and_error_policy():
    wf = make_workflow(
        NodeConfig(node="group", parallel=["vendor"], connections=["join"],
                   error_policy="collect"),
        NodeConfig(node="vendor"),
        NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "vendor": SaveNode, "join": SaveNode},
    )()
    graph = wf._schema.to_graph_dict()
    group = graph["nodes"][0]
    assert group["parallel"] == ["vendor"]
    assert group["error_policy"] == "collect"


# ---------------------------------------------------------------------------
# A valid group schema executes (covered in depth by test_parallel_execution)
# ---------------------------------------------------------------------------


def test_parallel_group_runs():
    wf = make_workflow(
        NodeConfig(node="group", parallel=["vendor"], connections=["join"]),
        NodeConfig(node="vendor"),
        NodeConfig(node="join"),
        start="group",
        registry={"group": OkNode, "vendor": SaveNode, "join": SaveNode},
    )()
    ctx = wf.run({"value": 1})
    assert {"vendor", "join"} <= set(ctx.nodes)
