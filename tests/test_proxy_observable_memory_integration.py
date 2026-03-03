"""Integration tests for Observable Memory proxy integration.

Tests the full request/response cycle with a real proxy server and real SQLite storage.
Two tiers:
  - Setup tests: no API key needed (health/stats only)
  - Flow tests: require ANTHROPIC_API_KEY (observation stored, memory injected, thread isolation)

Run all:
    ANTHROPIC_API_KEY=... pytest tests/test_proxy_observable_memory_integration.py -v

Run setup-only (no API key):
    pytest tests/test_proxy_observable_memory_integration.py -v -k "Setup"
"""

import os
import sqlite3
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("headroom.observable_memory")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Cheap model for both observer and test requests
OBSERVER_MODEL = "claude-haiku-4-5-20251001"
TEST_MODEL = "claude-haiku-4-5-20251001"

# Very low threshold so observer fires after even a single short message
_LOW_THRESHOLD = 0.0001

# How long to wait for the background observer LLM call to complete
_OBSERVER_WAIT_S = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_om_db(tmp_path):
    db_path = str(tmp_path / "om_test.db")
    yield db_path
    for suffix in ["", "-shm", "-wal"]:
        Path(db_path + suffix).unlink(missing_ok=True)


@pytest.fixture
def om_client(temp_om_db):
    """Proxy with Observable Memory enabled and a very low firing threshold."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        observable_memory_enabled=True,
        observable_memory_observer_model=OBSERVER_MODEL,
        observable_memory_db_path=temp_om_db,
        observable_memory_message_threshold_ratio=_LOW_THRESHOLD,
        observable_memory_observation_threshold_ratio=_LOW_THRESHOLD,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client, temp_om_db


@pytest.fixture
def no_om_client():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        observable_memory_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic_request(client, messages, thread_id, api_key, max_tokens=150):
    return client.post(
        "/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "x-headroom-thread-id": thread_id,
        },
        json={
            "model": TEST_MODEL,
            "max_tokens": max_tokens,
            "messages": messages,
        },
    )


def _get_text(resp_json):
    return "".join(
        b.get("text", "")
        for b in resp_json.get("content", [])
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _load_observations(db_path, thread_id):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT data FROM observations WHERE thread_id = ?", (thread_id,)
    ).fetchall()
    conn.close()
    return rows[0][0] if rows else None


# ---------------------------------------------------------------------------
# Setup tests (no API key needed)
# ---------------------------------------------------------------------------


class TestObservableMemorySetup:
    """Proxy startup and basic endpoint tests — no API key needed."""

    def test_proxy_starts_with_om_enabled(self, om_client):
        client, _ = om_client
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_proxy_starts_with_om_disabled(self, no_om_client):
        resp = no_om_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_stats_endpoint_works_with_om(self, om_client):
        client, _ = om_client
        resp = client.get("/stats")
        assert resp.status_code == 200

    def test_om_handler_initialized(self, om_client):
        """Proxy server should have an observable_memory_handler when OM is enabled."""
        client, _ = om_client
        # Access the underlying app's server instance
        app = client.app
        server = app.state.server if hasattr(app.state, "server") else None
        if server is not None:
            assert server.observable_memory_handler is not None


# ---------------------------------------------------------------------------
# Observer flow tests (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
class TestObservableMemoryObserverFlow:
    """Full observation cycle: request → observer fires → observations stored."""

    def test_request_succeeds_with_om_enabled(self, om_client):
        """Basic sanity: a request through an OM-enabled proxy returns 200."""
        client, _ = om_client
        resp = _anthropic_request(
            client,
            [{"role": "user", "content": "Say hello."}],
            thread_id="test-basic",
            api_key=ANTHROPIC_API_KEY,
        )
        assert resp.status_code == 200

    def test_observation_stored_after_request(self, om_client):
        """After a request, the background observer fires and persists observations."""
        client, db_path = om_client
        thread_id = f"test-store-{int(time.time())}"

        resp = _anthropic_request(
            client,
            [{"role": "user", "content": "I am building a service called Nighthawk using Python 3.12. Acknowledge briefly."}],
            thread_id=thread_id,
            api_key=ANTHROPIC_API_KEY,
        )
        assert resp.status_code == 200

        time.sleep(_OBSERVER_WAIT_S)

        obs = _load_observations(db_path, thread_id)
        assert obs is not None, f"No observations stored for thread_id={thread_id!r}"
        assert obs.strip(), "Stored observations should not be empty"

    def test_memory_injected_in_subsequent_request(self, om_client):
        """The <memory> block from prior turns is injected into the next request's system prompt."""
        client, db_path = om_client
        thread_id = f"test-inject-{int(time.time())}"
        # Unique token that only appears in turn 1, not turn 2
        unique_fact = f"CODEWORD{int(time.time())}"

        # Turn 1 — plant a unique fact using neutral language
        resp1 = _anthropic_request(
            client,
            [{"role": "user", "content": f"My project tracking number is {unique_fact}. Just say 'got it'."}],
            thread_id=thread_id,
            api_key=ANTHROPIC_API_KEY,
        )
        assert resp1.status_code == 200

        # Wait for observer to store the observation
        time.sleep(_OBSERVER_WAIT_S)
        obs = _load_observations(db_path, thread_id)
        assert obs is not None, "Observation must be stored before turn 2"

        # Turn 2 — fresh messages with NO mention of the tracking number
        resp2 = _anthropic_request(
            client,
            [{"role": "user", "content": "What is my project tracking number? Check your memory context."}],
            thread_id=thread_id,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=200,
        )
        assert resp2.status_code == 200

        text = _get_text(resp2.json())
        assert unique_fact in text, (
            f"Expected tracking number {unique_fact!r} in turn-2 response (injected via <memory>), got: {text!r}"
        )

    def test_no_crash_when_thread_id_not_in_header(self, om_client):
        """Requests without an explicit thread ID should still succeed (content-hash fallback)."""
        client, _ = om_client
        resp = client.post(
            "/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                # no x-headroom-thread-id header
            },
            json={
                "model": TEST_MODEL,
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hello, respond briefly."}],
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Thread isolation tests (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
class TestObservableMemoryThreadIsolation:
    """Observations are scoped to thread_id — different threads do not share state."""

    def test_different_threads_have_isolated_observations(self, om_client):
        client, db_path = om_client
        ts = int(time.time())
        thread_a = f"thread-a-{ts}"
        thread_b = f"thread-b-{ts}"
        project_id = f"PROJ{ts}"

        # Thread A — establish a unique project fact
        resp_a = _anthropic_request(
            client,
            [{"role": "user", "content": f"I am working on project {project_id}, a Python microservice. Acknowledge briefly."}],
            thread_id=thread_a,
            api_key=ANTHROPIC_API_KEY,
        )
        assert resp_a.status_code == 200

        # Thread B — completely separate conversation, no mention of project_id
        resp_b = _anthropic_request(
            client,
            [{"role": "user", "content": "I am working on a JavaScript frontend. Acknowledge briefly."}],
            thread_id=thread_b,
            api_key=ANTHROPIC_API_KEY,
        )
        assert resp_b.status_code == 200

        time.sleep(_OBSERVER_WAIT_S)

        # Thread A should have observations containing the project ID
        obs_a = _load_observations(db_path, thread_a)
        assert obs_a is not None, "Thread A should have observations"
        assert project_id in obs_a, (
            f"Thread A observations should mention {project_id!r}: {obs_a!r}"
        )

        # Thread B's observations (if any) must NOT contain thread A's project ID
        obs_b = _load_observations(db_path, thread_b)
        if obs_b:
            assert project_id not in obs_b, (
                f"Thread B should not see thread A's project ID. Got: {obs_b!r}"
            )

    def test_same_thread_id_accumulates_observations(self, om_client):
        """Multiple requests on the same thread ID use the same observation store."""
        client, db_path = om_client
        thread_id = f"thread-acc-{int(time.time())}"

        resp1 = _anthropic_request(
            client,
            [{"role": "user", "content": "I am debugging a memory leak in a Python asyncio service. The leak appears in the connection pool. Acknowledge briefly."}],
            thread_id=thread_id,
            api_key=ANTHROPIC_API_KEY,
        )
        assert resp1.status_code == 200

        time.sleep(_OBSERVER_WAIT_S)

        resp2 = _anthropic_request(
            client,
            [{"role": "user", "content": "We narrowed it down to the aiohttp session not being closed. Acknowledge briefly."}],
            thread_id=thread_id,
            api_key=ANTHROPIC_API_KEY,
        )
        assert resp2.status_code == 200

        time.sleep(_OBSERVER_WAIT_S)

        obs = _load_observations(db_path, thread_id)
        assert obs is not None, "Observations should exist after two turns"
        assert obs.strip(), "Observations should not be empty"
