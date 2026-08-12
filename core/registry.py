"""Node registry — maps string identifiers to ``Node`` subclasses.

A registry is a plain dictionary. The keys are human-readable strings
chosen by the workflow author; the values are ``Node`` subclasses.

Example::

    from strand.core.node import Node
    from strand.core.registry import NodeRegistry

    registry: NodeRegistry = {
        "analyze": AnalyzeNode,
        "respond": RespondNode,
        "escalate": EscalateNode,
    }
"""

from __future__ import annotations

from typing import Dict, Type

from strand.core.node import Node

NodeRegistry = Dict[str, Type[Node]]
