"""strand.core — A minimal, dependency-light workflow execution engine.

``strand.core`` knows how to represent, validate, and execute DAG-based
workflows.  It knows **nothing** about LLMs, HTTP, databases, prompts,
or any application service.  Its sole external dependency is ``pydantic``.

Public API
----------

.. autosummary::

    TaskContext        Shared state bus passed between nodes.
    RunContext         Engine-held bookkeeping for one workflow execution.
    WorkflowListener   Observability extension point — override any subset of hooks.
    Node               Abstract base class for all workflow nodes.
    BaseRouter         Node subclass for conditional branching.
    RouterNode         A single routing rule evaluated during branching.
    NodeRegistry       TypedDict alias mapping string keys to ``Node`` subclasses.
    NodeConfig         Configuration for one node in the workflow graph.
    RetryPolicy        Node-level retry configuration (any node kind, not just LLM).
    WorkflowSchema     Complete declarative definition of a workflow graph.
    WorkflowValidator  Validates that a schema is a well-formed DAG.
    Workflow           Abstract base class — the execution engine.
"""

from strand.core.context import TaskContext
from strand.core.listener import WorkflowListener
from strand.core.node import Node
from strand.core.registry import NodeRegistry
from strand.core.retry import RetryPolicy
from strand.core.router import BaseRouter, RouterNode
from strand.core.run_context import RunContext
from strand.core.schema import NodeConfig, WorkflowSchema
from strand.core.validator import WorkflowValidator
from strand.core.workflow import Workflow

__all__ = [
    "BaseRouter",
    "Node",
    "NodeConfig",
    "NodeRegistry",
    "RetryPolicy",
    "RouterNode",
    "RunContext",
    "TaskContext",
    "Workflow",
    "WorkflowListener",
    "WorkflowSchema",
    "WorkflowValidator",
]
