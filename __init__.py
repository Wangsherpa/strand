"""Strand — A lightweight workflow execution engine with optional LLM support.

Subpackages
-----------
* ``strand.core`` — DAG execution engine (zero AI deps).
* ``strand.llm``  — reusable LLM node base class and provider dispatch.
"""

from dotenv import load_dotenv

load_dotenv()