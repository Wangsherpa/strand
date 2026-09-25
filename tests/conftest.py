"""Pytest fixtures for the strand test suite."""

import pytest


@pytest.fixture
def no_backoff(monkeypatch):
    """Replace retry-backoff sleeps with an instantaneous, recording seam.

    Returns a dict with "engine"/"client" lists of the (rounded) delays
    that each layer would have slept.
    """
    import strand.core.workflow as workflow_mod
    import strand.llm.client as client_mod

    sleeps = {"engine": [], "client": []}

    async def engine_sleep(delay):
        sleeps["engine"].append(round(delay, 3))

    async def client_sleep(delay):
        sleeps["client"].append(round(delay, 3))

    monkeypatch.setattr(workflow_mod, "_sleep", engine_sleep)
    monkeypatch.setattr(client_mod, "_sleep", client_sleep)
    return sleeps
