# Strand

**A lightweight DAG workflow execution engine with optional LLM support.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`strand` is two packages in one:

| Package | Purpose | Dependencies |
|---|---|---|
| `strand.core` | Represent, validate, and execute DAG workflows | `pydantic` |
| `strand.llm` | Reusable LLM node base class with multi-provider dispatch | `pydantic`, `httpx`, `litellm` (optional) |

Neither package depends on FastAPI, Celery, databases, or any AI SDK.

---

## Installation

```bash
uv sync
```

For development with the optional litellm backend:

```bash
uv sync --extra litellm
```

---

## Quickstart

### 5-minute workflow (no LLM)

```python
from pydantic import BaseModel
from strand.core import Node, TaskContext, NodeConfig, WorkflowSchema, Workflow


class NameEvent(BaseModel):
    name: str


class GreetNode(Node):
    class OutputType(Node.OutputType):
        greeting: str

    async def process(self, ctx: TaskContext) -> TaskContext:
        self.save_output(
            self.OutputType(greeting=f"Hello, {ctx.event.name}!")
        )
        return ctx


class GreetWorkflow(Workflow):
    workflow_schema = WorkflowSchema(
        event_schema=NameEvent,
        start="greet",
        nodes=[NodeConfig(node="greet")],
        registry={"greet": GreetNode},
    )


result = GreetWorkflow().run({"name": "Alice"})
print(result.nodes["greet"].greeting)  # → Hello, Alice!
```

### Adding an LLM node

```python
from strand.llm import LLMNode, LLMConfig


class SummarizeNode(LLMNode):
    class OutputType(LLMNode.OutputType):
        summary: str
        sentiment: str

    def get_llm_config(self) -> LLMConfig:
        return LLMConfig(
            model="gpt-4o-mini",
            system_prompt="Summarize the text. Return the sentiment as positive, negative, or neutral.",
        )

    async def build_user_message(self, ctx: TaskContext) -> str:
        return ctx.event.text
```

### Switching LLM providers

```python
# OpenAI (lightweight — uses httpx, no extra deps)
LLMConfig(model="gpt-4o-mini", provider="openai")

# Anthropic (lightweight — uses httpx, no extra deps)
LLMConfig(model="claude-sonnet-4-6", provider="anthropic")

# LiteLLM (100+ providers, built-in retry/fallback, cost tracking)
LLMConfig(model="gpt-4o-mini", provider="litellm")
LLMConfig(model="anthropic/claude-sonnet-4-6", provider="litellm")
LLMConfig(model="ollama/llama3.2", provider="litellm")
LLMConfig(model="together_ai/mistralai/Mixtral-8x7B", provider="litellm")

# Ollama via OpenAI-compatible base_url (no litellm needed)
LLMConfig(model="llama3.2", provider="openai", base_url="http://localhost:11434/v1")
```

### Provider backends at a glance

| Provider | Backend | Structured output | Retry | Dependencies |
|---|---|---|---|---|
| `openai` | Direct HTTP (`httpx`) | Native `json_schema` | Automatic (default policy) | `httpx` |
| `anthropic` | Direct HTTP (`httpx`) | Prompted JSON | Automatic (default policy) | `httpx` |
| `litellm` | LiteLLM library | Auto-translated per provider | Automatic (default policy) | `litellm` |

**When to use which:**

- **`openai` / `anthropic`** — zero extra dependencies. Good when you only call one provider and want minimal footprint.
- **`litellm`** — one dependency, 100+ providers. Use when you need retry logic, fallback models, cost tracking, or non-OpenAI/Anthropic providers.

Both paths produce identical structured output. Switch between them by changing one line in `get_llm_config()` — your nodes and `OutputType` classes stay the same.

### LiteLLM — retry, fallback, and 100+ providers

When using `provider="litellm"`, the model string follows [LiteLLM's convention](https://docs.litellm.ai/docs/providers) — prefix with the provider name for non-OpenAI models:

```python
# OpenAI models — just the model name
LLMConfig(model="gpt-4o-mini", provider="litellm")

# Anthropic — provider prefix
LLMConfig(model="anthropic/claude-sonnet-4-6", provider="litellm")

# Ollama — provider prefix
LLMConfig(model="ollama/llama3.2", provider="litellm")

# 100+ more: together_ai, groq, bedrock, vertex_ai, cohere, mistral, deepseek...
```

**Built-in retry and fallback** — no extra code:

```python
# Retry on failure (default: 0)
LLMConfig(model="gpt-4o-mini", provider="litellm")

# Fallback model if primary fails — set LITELLM_FALLBACKS env var:
# export LITELLM_FALLBACKS='["anthropic/claude-sonnet-4-6", "gpt-4o"]'
```

**Structured output** — LiteLLM translates `response_format` per provider automatically. OpenAI gets native `json_schema` mode; Anthropic gets tool-use; other providers get prompted JSON. Your `OutputType` Pydantic model stays the same regardless.

**Install litellm** (only if you use `provider="litellm"`):

```bash
uv sync --extra litellm
```

Set your provider's API key as usual (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) — litellm reads from the environment automatically.

### Conditional branching

```python
from strand.core import BaseRouter, RouterNode


class IsPositiveRule(RouterNode):
    def determine_next_node(self, ctx):
        sentiment = ctx.nodes["summarize"].sentiment
        return "positive_handler" if sentiment == "positive" else None


class SentimentRouter(BaseRouter):
    routes = [IsPositiveRule()]
    fallback = "negative_handler"
```

---

## Concepts

### `strand.core`

| Class | Role |
|---|---|
| `Node` | Abstract base — implement `process(ctx)` |
| `TaskContext` | Shared state bus flowing through every node |
| `BaseRouter` | Conditional branching node |
| `RouterNode` | Individual routing rule |
| `NodeConfig` | Declares one node's connections and routing flag |
| `WorkflowSchema` | Full graph definition (nodes + edges + registry) |
| `WorkflowValidator` | DAG validation — cycle detection, reachability, connection rules |
| `Workflow` | Execution engine — `run()` / `run_async()` |

**Key design choice:** All node references are **strings**, not Python classes. This means an LLM can emit a workflow as JSON, and the `registry` dict maps those strings to actual `Node` subclasses.

```python
WorkflowSchema(
    start="classify",                  # ← string
    nodes=[
        NodeConfig(node="classify", connections=["router"]),
        NodeConfig(node="router", connections=["extract", "reject"], is_router=True),
    ],
    registry={
        "classify": ClassifyNode,      # ← maps string → class
        "router":   RouterNode,
        "extract":  ExtractNode,
        "reject":   RejectNode,
    },
)
```

### `strand.llm`

| Class | Role |
|---|---|
| `LLMConfig` | Dataclass: model, provider, temperature, API key, base URL |
| `ModelProvider` | Enum: `openai`, `anthropic`, `litellm` (extensible) |
| `LLMNode(Node)` | Base class — implement `get_llm_config()` and `build_user_message()` |
| `call_llm()` | Low-level async dispatch — returns an `LLMResult` (await it, or wrap in `asyncio.run()`) |

**Key design choice:** `LLMNode` extends `strand.core.Node` from *outside* the core package. `strand.core` knows nothing about LLMs. The LLM integration is a separate concern in `strand.llm`.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              APPLICATION LAYER               │
│                                              │
│  invoice_extraction/                         │
│    ClassifyNode(LLMNode)                     │
│    ExtractNode(LLMNode)                      │
│    NotInvoiceNode(Node)                      │
│    InvoiceRouter(BaseRouter)                 │
│                                              │
├──────────────────────────────────────────────┤
│              RUNTIME LAYER                   │
│                                              │
│  strand/llm/              strand/core/       │
│    LLMNode(Node)             Workflow        │
│    LLMConfig                 Node (ABC)      │
│    ModelProvider             BaseRouter      │
│    call_llm()                TaskContext     │
│    ┌──────────────────┐      WorkflowSchema  │
│    │ provider backends│      Validator       │
│    │                  │                      │
│    │  openai (http)   │                      │
│    │  anthropic (http)│    depends on:       │
│    │  litellm (sdk)   │      pydantic        │
│    └──────────────────┘                      │
│      depends on:                             │
│        strand.core                           │
│        pydantic                              │
│        httpx                                 │
│        litellm (optional)                    │
└──────────────────────────────────────────────┘
```

---

## DAG Visualization

Every workflow gets `to_mermaid()` and `to_dot()` for free:

```python
wf = MyWorkflow()
print(wf.to_mermaid())   # Paste into GitHub/Notion/Obsidian — renders as a diagram
print(wf.to_dot())       # Feed to Graphviz: dot -Tpng out.dot -o out.png
```

Shape conventions:

| Mermaid | Shape | Meaning |
|---|---|---|
| `[label]` | Rectangle | Processing node |
| `{label}` | Rhombus | Router / decision |
| `([label])` | Stadium | Terminal node |

---

## Retry, timeouts & error routing

Every `NodeConfig` accepts:

- `retry` — a `RetryPolicy` (`max_attempts` including the first, exponential
  backoff, jitter, and a `retry_on` exception type / tuple / predicate).
  Defaults to no retry.
- `timeout_s` — per-attempt timeout (a timeout is just another failure:
  `retry_on` decides whether it is retried).
- `on_error` — a registry key to route to when retries are exhausted, instead of
  aborting the run. The failing node's error summary lands in
  `ctx.errors[node]`, readable via `Node.get_error(node)`.

```python
NodeConfig(
    node="classify",
    retry=RetryPolicy(max_attempts=3, backoff_base=1.0, retry_on=(ValueError,)),
    timeout_s=30.0,
    on_error="fallback_handler",
)
```

For LLM nodes, `LLMConfig.retry` (or a per-provider default of 3 attempts) handles
transient transport failures, and `max_output_repair_attempts` asks the model to
fix responses that failed to parse or validate. `RetryPolicy.max_attempts` bounds
the TOTAL provider calls, repair attempts included. When a node's `NodeConfig.retry`
is set, the engine owns retries for that node and the LLM layer's transport retry
is disabled for its calls — the two layers never multiply.

---

## Observability

Pass `WorkflowListener` instances to `Workflow(listeners=[...])` to observe runs:

- `on_workflow_start` / `on_workflow_end` — around each (possibly nested) workflow
- `on_node_start` / `on_node_end` / `on_node_error` — per node attempt, with real
  attempt numbers, retry decisions, and durations
- `on_route` — every router decision

Hooks may be sync or async; a raising listener is logged and never affects the
run. Nested workflows inherit the outermost listeners under one `execution_id`.
`strand.trace` ships two reference listeners: `InMemoryCollector` (assembles
`ExecutionRecord`s for tests and debugging) and `JsonlWriter` (one JSON line per
event, for tailing a live run).

---

## Package Structure

```
strand/
    __init__.py

    core/                    # DAG execution engine
        __init__.py          #   12 public exports
        context.py           #   TaskContext
        node.py              #   Node (ABC)
        registry.py          #   NodeRegistry type alias
        retry.py             #   RetryPolicy
        listener.py          #   WorkflowListener + shared dispatch
        run_context.py       #   RunContext (engine bookkeeping)
        router.py            #   BaseRouter, RouterNode
        schema.py            #   NodeConfig, WorkflowSchema
        validator.py         #   WorkflowValidator
        workflow.py          #   Workflow

    llm/                     # LLM integration layer
        __init__.py          #   6 public exports
        config.py            #   LLMConfig, ModelProvider
        client.py            #   Provider dispatch (openai, anthropic, litellm)
        node.py              #   LLMNode(Node)
        listener.py          #   LLMListener
        result.py            #   LLMResult
        retry_defaults.py    #   per-provider default RetryPolicy

    trace/                   # optional reference listeners
        __init__.py          #   InMemoryCollector, JsonlWriter, ExecutionRecord
        collectors.py        #   the two reference listeners
        models.py            #   ExecutionRecord, NodeSpan, RouteDecision
```

| Module | LOC | Dependencies |
|---|---|---|
| `strand/core/` | 741 | `pydantic` |
| `strand/llm/` | 293 | `pydantic`, `httpx`, `strand.core` |
| **Total** | **1,034** | |

---

## What Strand is NOT

- **Not an HTTP server** — no FastAPI, Flask, or Starlette
- **Not a task queue** — no Celery, RQ, or ARQ
- **Not a database** — no SQLAlchemy, Alembic, or ORM
- **Not an AI SDK** — no required dependency on OpenAI SDK, Anthropic SDK, or pydantic-ai. The `openai` and `anthropic` providers use plain `httpx`. The `litellm` provider is optional.
- **Not a prompt manager** — no Jinja2 templates or prompt loaders
- **Not an observability platform** — no Langfuse/OpenTelemetry integration, but `WorkflowListener` hooks (plus the `strand.trace` reference listeners) give you the events to build one
- **Not a deployment tool** — no Docker, Kubernetes, or cloud configs

These belong in your application layer. Strand is the engine — you build the car around it.

---

## Example Application

See `invoice_extraction/` for a complete working example:

- `schema.py` — `EmailEvent` pydantic model
- `nodes.py` — `ClassifyNode(LLMNode)`, `ExtractNode(LLMNode)`, `NotInvoiceNode(Node)`, `InvoiceRouter(BaseRouter)`
- `workflow.py` — `InvoiceExtractionWorkflow` — classify → router → extract / reject
- `run.py` — Test runner with real invoice and non-invoice email scenarios

Run it (set `OPENAI_API_KEY` first):

```bash
PYTHONPATH=. python invoice_extraction/run.py
```

To switch to LiteLLM, change one line in `nodes.py`:

```python
# Before
LLMConfig(model="gpt-4o-mini", provider="openai")

# After
LLMConfig(model="gpt-4o-mini", provider="litellm")
```

---

## Design Decisions

### Why strings for node references?

Node keys are strings (`"classify"`), not Python classes. This means:

- **LLM-friendly** — an AI can emit JSON workflow definitions without generating Python imports
- **Serializable** — the entire schema can be dumped to YAML/JSON and loaded back
- **Single registry** — one `registry` dict is the source of truth for name → class resolution

### Why Pydantic?

Pydantic is the only hard dependency of `strand.core`. It provides:
- Runtime validation of events and node outputs
- Type coercion (string → datetime, etc.)
- Serialization (`.model_dump()` / `.model_validate()`)
- Industry-standard API — most Python developers already know it

### Why async?

All `Node.process()` methods are `async`. This enables I/O-bound nodes without blocking, future support for concurrent execution, and compatibility with async web frameworks.

### Why separate `strand.core` and `strand.llm`?

`strand.core` is the pure graph engine — it knows nothing about language models. `strand.llm` extends it with LLM capabilities from *outside* the core. This separation means:
- `strand.core` stays dependency-light (just pydantic)
- `strand.llm` can evolve independently (new providers, streaming, tool use)
- Applications that don't need LLMs don't pull in LLM dependencies

---

## License

MIT
