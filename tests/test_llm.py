"""LLM-layer regression tests — the review findings, plus happy-path coverage."""

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from strand.core import NodeConfig, RetryPolicy, Workflow, WorkflowListener, WorkflowSchema
from strand.llm import LLMConfig, LLMNode, ModelProvider
from strand.llm.client import call_llm
from strand.llm.retry_defaults import _http_retry_on, _litellm_retry_on, default_retry_policy

from tests.helpers import Event, run_async


class Out(BaseModel):
    answer: str


@pytest.fixture
def fake_transport(monkeypatch):
    """Swap strand's httpx.AsyncClient for a MockTransport-backed one.

    Tests assign ``state["handler"]`` (a callable taking the request and
    returning an ``httpx.Response`` or raising) and read the call/error/
    instance counters from ``state``.
    """
    import strand.llm.client as client_mod

    real_client = httpx.AsyncClient
    state = {"handler": None, "calls": [], "transport_errors": 0, "instances": 0}

    def recording_handler(request):
        state["calls"].append(request)
        return state["handler"](request)

    class FakeClient(real_client):
        def __init__(self, *args, **kwargs):
            state["instances"] += 1
            super().__init__(*args, transport=httpx.MockTransport(recording_handler), **kwargs)

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", FakeClient)
    return state


def _config(**overrides):
    kwargs = dict(model="gpt-4o-mini", provider=ModelProvider.OPENAI, api_key="test-key")
    kwargs.update(overrides)
    return LLMConfig(**kwargs)


def _ok_response(content):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    })


def _down_handler(request):
    raise httpx.ConnectError("provider down")


# ---------------------------------------------------------------------------
# Happy path + provider normalization
# ---------------------------------------------------------------------------


def test_call_llm_success(fake_transport):
    fake_transport["handler"] = lambda req: _ok_response('{"answer": "hi"}')
    result = run_async(call_llm(_config(), "hi", Out))
    assert isinstance(result.parsed, Out)
    assert result.parsed.answer == "hi"
    assert result.attempts == 1
    assert result.provider == "openai"
    assert result.input_tokens == 5
    assert result.output_tokens == 3
    assert result.validation_repaired is False
    assert len(fake_transport["calls"]) == 1


def test_provider_string_and_enum_are_equivalent(fake_transport):
    fake_transport["handler"] = lambda req: _ok_response('{"answer": "x"}')
    result = run_async(call_llm(_config(provider="openai"), "hi", Out))
    assert result.provider == "openai"


# ---------------------------------------------------------------------------
# Null content engages the repair path instead of escaping as TypeError (#10)
# ---------------------------------------------------------------------------


def test_null_content_engages_repair_then_clean_error(fake_transport):
    fake_transport["handler"] = lambda req: _ok_response(None)
    config = _config(
        retry=RetryPolicy(max_attempts=3, backoff_base=0.001),
        max_output_repair_attempts=3,
    )
    with pytest.raises((json.JSONDecodeError, ValidationError)):
        run_async(call_llm(config, "classify this", Out))
    # repair attempted, total bounded by max_attempts
    assert len(fake_transport["calls"]) == 3


# ---------------------------------------------------------------------------
# max_attempts bounds the TOTAL provider calls, repair included (#11)
# ---------------------------------------------------------------------------


def test_no_transport_retry_makes_exactly_one_call(fake_transport):
    fake_transport["handler"] = lambda req: _ok_response("NOT JSON")
    config = _config(retry=RetryPolicy(max_attempts=1), max_output_repair_attempts=3)
    with pytest.raises((json.JSONDecodeError, ValidationError)):
        run_async(call_llm(config, "hi", Out))
    assert len(fake_transport["calls"]) == 1


def test_repair_calls_bounded_by_max_attempts(fake_transport, no_backoff):
    fake_transport["handler"] = lambda req: _ok_response("NOT JSON")
    config = _config(
        retry=RetryPolicy(max_attempts=3, backoff_base=0.001),
        max_output_repair_attempts=3,
    )
    with pytest.raises((json.JSONDecodeError, ValidationError)):
        run_async(call_llm(config, "hi", Out))
    assert len(fake_transport["calls"]) == 3


# ---------------------------------------------------------------------------
# Engine retry and transport retry never multiply (#12)
# ---------------------------------------------------------------------------


class FailLLMNode(LLMNode):
    class OutputType(LLMNode.OutputType):
        answer: str

    def get_llm_config(self):
        return LLMConfig(
            model="gpt-4o-mini", provider=ModelProvider.OPENAI, api_key="k",
            retry=RetryPolicy(max_attempts=3, backoff_base=2.0),
            max_output_repair_attempts=0,
        )

    async def build_user_message(self, ctx):
        return "hi"


def _run_llm_workflow(node_config):
    class WF(Workflow):
        workflow_schema = WorkflowSchema(
            event_schema=Event, start="llm", nodes=[node_config],
            registry={"llm": FailLLMNode},
        )

    with pytest.raises(httpx.ConnectError):
        WF().run({"value": 1})


def test_node_retry_disables_transport_retry(fake_transport, no_backoff):
    fake_transport["handler"] = _down_handler
    _run_llm_workflow(
        NodeConfig(node="llm", retry=RetryPolicy(max_attempts=3, backoff_base=1.0))
    )
    assert len(fake_transport["calls"]) == 3          # not 3 x 3 = 9
    assert no_backoff["client"] == []                 # no transport-level sleeps
    assert len(no_backoff["engine"]) == 2             # engine-level sleeps only


def test_no_node_retry_uses_the_llm_config_policy(fake_transport, no_backoff):
    fake_transport["handler"] = _down_handler
    _run_llm_workflow(NodeConfig(node="llm"))
    assert len(fake_transport["calls"]) == 3
    assert len(no_backoff["client"]) == 2             # the LLM layer retried
    assert no_backoff["engine"] == []


# ---------------------------------------------------------------------------
# CancelledError is never retried by call_llm (#3b)
# ---------------------------------------------------------------------------


async def _cancel_handler(request):
    raise asyncio.CancelledError()


def test_cancelled_error_propagates_unretried(fake_transport, no_backoff):
    fake_transport["handler"] = _cancel_handler
    config = _config(retry=RetryPolicy(max_attempts=3, backoff_base=0.001))
    with pytest.raises(asyncio.CancelledError):
        run_async(call_llm(config, "hi", Out))
    assert len(fake_transport["calls"]) == 1
    assert no_backoff["client"] == []


def test_cancelled_error_not_retried_even_with_base_exception_policy(fake_transport):
    fake_transport["handler"] = _cancel_handler
    config = _config(
        retry=RetryPolicy(max_attempts=3, backoff_base=0.001, retry_on=BaseException)
    )
    with pytest.raises(asyncio.CancelledError):
        run_async(call_llm(config, "hi", Out))
    assert len(fake_transport["calls"]) == 1


# ---------------------------------------------------------------------------
# base_url accepts bases and full endpoints without doubling (#14)
# ---------------------------------------------------------------------------


def test_full_endpoint_base_url_not_doubled():
    import strand.llm.client as client_mod

    full = "https://gateway.internal/v1/chat/completions"
    assert client_mod._openai_url(_config(base_url=full)) == full
    assert client_mod._anthropic_url(
        _config(base_url="https://api.anthropic.com/v1/messages")
    ) == "https://api.anthropic.com/v1/messages"


def test_base_gets_provider_path_appended():
    import strand.llm.client as client_mod

    assert client_mod._openai_url(_config(base_url="http://localhost:11434/v1")) == (
        "http://localhost:11434/v1/chat/completions"
    )


def test_strip_endpoint_path_for_litellm():
    import strand.llm.client as client_mod

    assert client_mod._strip_endpoint_path(
        "https://gateway.internal/v1/chat/completions"
    ) == "https://gateway.internal/v1"


# ---------------------------------------------------------------------------
# One httpx client reused across attempts (H1)
# ---------------------------------------------------------------------------


def test_single_client_reused_across_attempts(fake_transport, no_backoff):
    fake_transport["handler"] = _down_handler
    config = _config(retry=RetryPolicy(max_attempts=3, backoff_base=0.001))
    with pytest.raises(httpx.ConnectError):
        run_async(call_llm(config, "hi", Out))
    assert fake_transport["instances"] == 1
    assert len(fake_transport["calls"]) == 3


# ---------------------------------------------------------------------------
# on_llm_call fires once per process(), on every outcome (E7)
# ---------------------------------------------------------------------------


class _LLMRecorder(WorkflowListener):
    def __init__(self):
        self.calls = []

    def on_llm_call(self, run, node_id, result, error):
        self.calls.append((node_id, result, error))


def test_on_llm_call_fires_on_pre_call_failures(fake_transport):
    class BrokenPromptNode(LLMNode):
        class OutputType(LLMNode.OutputType):
            answer: str

        def get_llm_config(self):
            return _config()

        async def build_user_message(self, ctx):
            raise KeyError("missing field")

    recorder = _LLMRecorder()

    class WF(Workflow):
        workflow_schema = WorkflowSchema(
            event_schema=Event, start="llm", nodes=[NodeConfig(node="llm")],
            registry={"llm": BrokenPromptNode},
        )

    with pytest.raises(KeyError):
        WF(listeners=[recorder]).run({"value": 1})
    assert len(recorder.calls) == 1
    node_id, result, error = recorder.calls[0]
    assert node_id == "llm"
    assert result is None
    assert isinstance(error, KeyError)


def test_on_llm_call_fires_on_success(fake_transport):
    fake_transport["handler"] = lambda req: _ok_response('{"answer": "ok"}')

    class OkLLMNode(LLMNode):
        class OutputType(LLMNode.OutputType):
            answer: str

        def get_llm_config(self):
            return _config()

        async def build_user_message(self, ctx):
            return "hi"

    recorder = _LLMRecorder()

    class WF(Workflow):
        workflow_schema = WorkflowSchema(
            event_schema=Event, start="llm", nodes=[NodeConfig(node="llm")],
            registry={"llm": OkLLMNode},
        )

    ctx = WF(listeners=[recorder]).run({"value": 1})
    assert len(recorder.calls) == 1
    node_id, result, error = recorder.calls[0]
    assert error is None
    assert result is not None and result.parsed.answer == "ok"
    assert ctx.nodes["llm"].answer == "ok"


# ---------------------------------------------------------------------------
# LLMConfig validation
# ---------------------------------------------------------------------------


def test_negative_repair_budget_rejected():
    with pytest.raises(ValueError, match="max_output_repair_attempts"):
        LLMConfig(model="m", max_output_repair_attempts=-1)


def test_non_retry_policy_retry_rejected():
    with pytest.raises(TypeError, match="RetryPolicy"):
        LLMConfig(model="m", retry="not-a-policy")


# ---------------------------------------------------------------------------
# Default retry predicate coverage
# ---------------------------------------------------------------------------


def _status_error(status):
    return httpx.HTTPStatusError(
        f"{status} from provider",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(status),
    )


@pytest.mark.parametrize(
    "error",
    [
        httpx.TimeoutException("timeout"),
        httpx.ConnectError("refused"),
        httpx.ReadError("mid-transfer"),
        httpx.WriteError("mid-transfer"),
        httpx.RemoteProtocolError("protocol"),
        _status_error(429),
        _status_error(500),
        _status_error(503),
    ],
    ids=["timeout", "connect", "read", "write", "protocol", "429", "500", "503"],
)
def test_transient_errors_are_retryable(error):
    assert _http_retry_on(error) is True


@pytest.mark.parametrize("status", [400, 401, 404, 422], ids=str)
def test_non_transient_statuses_not_retryable(status):
    assert _http_retry_on(_status_error(status)) is False


def test_non_httpx_errors_not_retryable():
    assert _http_retry_on(ValueError("x")) is False


def test_default_policy_uses_provider_specific_predicate():
    assert default_retry_policy(ModelProvider.OPENAI.value).retry_on is _http_retry_on
    assert default_retry_policy(ModelProvider.ANTHROPIC.value).retry_on is _http_retry_on
    assert default_retry_policy(ModelProvider.LITELLM.value).retry_on is _litellm_retry_on
    assert default_retry_policy("litellm").max_attempts == 3


# ---------------------------------------------------------------------------
# Strict schema memoization (H4)
# ---------------------------------------------------------------------------


def test_strict_schema_is_memoized_per_model():
    import strand.llm.client as client_mod

    client_mod._strict_schema.cache_clear()
    first = client_mod._strict_schema(Out)
    second = client_mod._strict_schema(Out)
    assert first is second
    assert client_mod._strict_schema.cache_info().hits == 1
    assert first["additionalProperties"] is False
