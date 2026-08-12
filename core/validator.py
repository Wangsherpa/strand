"""Workflow validator — ensures a workflow schema is a valid DAG."""

from collections import deque
from typing import Optional, Set

from strand.core.router import BaseRouter
from strand.core.schema import NodeConfig, WorkflowSchema


class WorkflowValidator:
    """Validates that a ``WorkflowSchema`` forms a well-structured DAG.

    Performs five checks:

    1. **Registry integrity** — every string key referenced by the schema
       exists in ``registry``.
    2. **DAG structure** — the graph contains no cycles and all nodes are
       reachable from the start node.
    3. **Connection rules** — only nodes marked ``is_router`` may have
       multiple outgoing connections.
    4. **Router flag consistency** — ``is_router`` must agree with
       whether the registered class actually is a ``BaseRouter``
       subclass, in both directions.
    5. **on_error targets** — ``NodeConfig.on_error``, when set, must
       name a key present in ``registry`` — the same guarantee
       ``connections`` already gets, since ``on_error`` is just
       another way execution can jump to a node.

    Raises ``ValueError`` on the first violation found.
    """

    def __init__(self, workflow_schema: WorkflowSchema) -> None:
        self._schema = workflow_schema

    def validate(self) -> None:
        """Run all validation checks.

        Raises:
            ValueError: If any validation check fails.
        """
        self._validate_registry()
        self._validate_dag()
        self._validate_connections()
        self._validate_router_flag_consistency()
        self._validate_on_error_targets()

    # ------------------------------------------------------------------
    # Registry integrity
    # ------------------------------------------------------------------

    def _validate_registry(self) -> None:
        registry_keys = set(self._schema.registry.keys())

        if self._schema.start not in registry_keys:
            raise ValueError(
                f"Start node '{self._schema.start}' is not in the registry."
            )

        for nc in self._schema.nodes:
            if nc.node not in registry_keys:
                raise ValueError(
                    f"Node '{nc.node}' is not in the registry."
                )
            unknown = set(nc.connections) - registry_keys
            if unknown:
                raise ValueError(
                    f"Node '{nc.node}' references unknown connections: {unknown}"
                )

    # ------------------------------------------------------------------
    # DAG structure
    # ------------------------------------------------------------------

    def _validate_dag(self) -> None:
        if self._has_cycle():
            raise ValueError("Workflow schema contains a cycle.")

        reachable = self._get_reachable_keys()
        all_keys = {nc.node for nc in self._schema.nodes}
        unreachable = all_keys - reachable
        if unreachable:
            raise ValueError(
                f"The following nodes are unreachable: {unreachable}"
            )

    def _has_cycle(self) -> bool:
        """DFS-based cycle detection."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _dfs(key: str) -> bool:
            visited.add(key)
            rec_stack.add(key)

            nc = self._get_config(key)
            if nc is not None:
                for neighbor in nc.connections:
                    if neighbor not in visited:
                        if _dfs(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(key)
            return False

        for nc in self._schema.nodes:
            if nc.node not in visited:
                if _dfs(nc.node):
                    return True
        return False

    def _get_reachable_keys(self) -> Set[str]:
        """BFS from the start node to find all reachable keys.

        Follows ``on_error`` targets alongside ``connections`` — a
        dedicated error-handling node reachable only via ``on_error``
        is still a legitimate part of the graph, not a dead one.
        """
        reachable: Set[str] = set()
        queue = deque([self._schema.start])

        while queue:
            key = queue.popleft()
            if key not in reachable:
                reachable.add(key)
                nc = self._get_config(key)
                if nc is not None:
                    queue.extend(nc.connections)
                    if nc.on_error is not None:
                        queue.append(nc.on_error)

        return reachable

    # ------------------------------------------------------------------
    # Connection rules
    # ------------------------------------------------------------------

    def _validate_connections(self) -> None:
        for nc in self._schema.nodes:
            if len(nc.connections) > 1 and not nc.is_router:
                raise ValueError(
                    f"Node '{nc.node}' has multiple connections but is not "
                    f"marked as a router."
                )

    # ------------------------------------------------------------------
    # Router flag consistency
    # ------------------------------------------------------------------

    def _validate_router_flag_consistency(self) -> None:
        """Reject a mismatch between ``is_router`` and the registered class.

        ``Workflow`` decides how to execute a node purely from
        ``NodeConfig.is_router`` — routers are never instantiated as a
        regular ``Node`` and never get ``process()`` called; everything
        else does. If that flag disagrees with what the registered
        class actually is, the mismatch fails silently at runtime
        instead of loudly at validation time: a ``BaseRouter``
        registered with ``is_router=False`` gets treated as a regular
        node whose ``process()`` is a no-op, and a plain ``Node``
        registered with ``is_router=True`` never gets ``process()``
        called at all.
        """
        for nc in self._schema.nodes:
            node_class = self._schema.registry.get(nc.node)
            if node_class is None:
                continue  # already reported by _validate_registry

            is_router_class = issubclass(node_class, BaseRouter)

            if nc.is_router and not is_router_class:
                raise ValueError(
                    f"Node '{nc.node}' is marked is_router=True but its "
                    f"registered class '{node_class.__name__}' is not a "
                    f"BaseRouter subclass."
                )
            if not nc.is_router and is_router_class:
                raise ValueError(
                    f"Node '{nc.node}' registers a BaseRouter subclass "
                    f"('{node_class.__name__}') but is not marked "
                    f"is_router=True."
                )

    # ------------------------------------------------------------------
    # on_error targets
    # ------------------------------------------------------------------

    def _validate_on_error_targets(self) -> None:
        registry_keys = set(self._schema.registry.keys())
        for nc in self._schema.nodes:
            if nc.on_error is not None and nc.on_error not in registry_keys:
                raise ValueError(
                    f"Node '{nc.node}' has on_error='{nc.on_error}', which "
                    f"is not in the registry."
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_config(self, key: str) -> Optional[NodeConfig]:
        """Return the ``NodeConfig`` for *key*, or ``None``."""
        for nc in self._schema.nodes:
            if nc.node == key:
                return nc
        return None
