"""Property-based and unit tests for the SIREN FastAPI server.

Validates: Requirements 8.1–8.8, 13.1–13.4
"""
# Feature: siren-env, Property 14: Observation Round-Trip

import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from fastapi.testclient import TestClient

import server.app as app_module
from server.app import app
from server.schemas import ObservationSchema

_GENESIS_HASH = hashlib.sha256(b"genesis").hexdigest()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> TestClient:
    """Return a TestClient with a fresh global env state."""
    app_module._env = None
    app_module._env_initialized = False
    return TestClient(app)


def _valid_incident_dict(**overrides) -> dict:
    base = {
        "id": "INC-001",
        "severity": 3,
        "type": "phishing",
        "system": "web-01",
        "time_remaining": 5,
        "resolved": False,
    }
    base.update(overrides)
    return base


def _valid_observation_dict(**overrides) -> dict:
    base = {
        "incidents": [_valid_incident_dict()],
        "sla_credits": 100.0,
        "proof_score": 0.5,
        "missing_logs": 0,
        "active_sessions": 1,
        "ledger_entries": 0,
        "ledger_hash": _GENESIS_HASH,
        "current_tick": 0,
        "max_ticks": 15,
        "audit_triggered": False,
        "task_id": "easy",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Hypothesis strategies for Property 14
# ---------------------------------------------------------------------------

incident_st = st.fixed_dictionaries({
    "id": st.from_regex(r"INC-\d{3}", fullmatch=True),
    "severity": st.integers(min_value=1, max_value=5),
    "type": st.sampled_from(["ransomware", "phishing", "ddos", "insider_threat", "data_breach"]),
    "system": st.sampled_from(["db-prod-01", "web-01", "auth-svc", "api-gw", "cache-01"]),
    "time_remaining": st.integers(min_value=-10, max_value=20),
    "resolved": st.booleans(),
})

observation_st = st.fixed_dictionaries({
    "incidents": st.lists(incident_st, min_size=1, max_size=5),
    "sla_credits": st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    "proof_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "missing_logs": st.integers(min_value=0, max_value=50),
    "active_sessions": st.integers(min_value=0, max_value=10),
    "ledger_entries": st.integers(min_value=0, max_value=100),
    "ledger_hash": st.from_regex(r"[0-9a-f]{64}", fullmatch=True),
    "current_tick": st.integers(min_value=0, max_value=100),
    "max_ticks": st.integers(min_value=1, max_value=100),
    "audit_triggered": st.booleans(),
    "task_id": st.sampled_from(["easy", "medium", "hard"]),
})


# ===========================================================================
# Property 14: Observation Round-Trip
# ===========================================================================

@given(obs=observation_st)
@settings(max_examples=50)
def test_property_14_observation_round_trip(obs):
    """Property 14: Observation Round-Trip

    For any valid observation dict, serializing via ObservationSchema and
    deserializing back shall produce an equivalent observation dict, with
    ledger_hash preserved as a 64-character lowercase hex string.

    **Validates: Requirements 13.1–13.4**
    """
    # Serialize via Pydantic schema
    schema = ObservationSchema(**obs)
    round_tripped = schema.model_dump()

    # Top-level scalar fields must be preserved
    assert round_tripped["sla_credits"] == obs["sla_credits"]
    assert round_tripped["proof_score"] == obs["proof_score"]
    assert round_tripped["missing_logs"] == obs["missing_logs"]
    assert round_tripped["active_sessions"] == obs["active_sessions"]
    assert round_tripped["ledger_entries"] == obs["ledger_entries"]
    assert round_tripped["current_tick"] == obs["current_tick"]
    assert round_tripped["max_ticks"] == obs["max_ticks"]
    assert round_tripped["audit_triggered"] == obs["audit_triggered"]
    assert round_tripped["task_id"] == obs["task_id"]

    # ledger_hash must be preserved as 64-char lowercase hex
    assert round_tripped["ledger_hash"] == obs["ledger_hash"]
    assert len(round_tripped["ledger_hash"]) == 64
    assert round_tripped["ledger_hash"] == round_tripped["ledger_hash"].lower()
    assert all(c in "0123456789abcdef" for c in round_tripped["ledger_hash"])

    # Incidents list must be preserved
    assert len(round_tripped["incidents"]) == len(obs["incidents"])
    for rt_inc, orig_inc in zip(round_tripped["incidents"], obs["incidents"]):
        assert rt_inc["id"] == orig_inc["id"]
        assert rt_inc["severity"] == orig_inc["severity"]
        assert rt_inc["type"] == orig_inc["type"]
        assert rt_inc["system"] == orig_inc["system"]
        assert rt_inc["time_remaining"] == orig_inc["time_remaining"]
        assert rt_inc["resolved"] == orig_inc["resolved"]


# ===========================================================================
# Task 7.4: Unit tests for HTTP endpoints
# ===========================================================================

def test_health_returns_200():
    """GET /health returns 200 with {"status": "ok"}."""
    client = _make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_reset_returns_valid_observation():
    """POST /reset returns a valid observation with all required fields."""
    client = _make_client()
    response = client.post("/reset", json={"task_id": "easy"})
    assert response.status_code == 200

    data = response.json()
    assert "incidents" in data
    assert "sla_credits" in data
    assert "proof_score" in data
    assert "missing_logs" in data
    assert "active_sessions" in data
    assert "ledger_entries" in data
    assert "ledger_hash" in data
    assert "current_tick" in data
    assert "max_ticks" in data
    assert "audit_triggered" in data
    assert "task_id" in data

    assert data["current_tick"] == 0
    assert data["missing_logs"] == 0
    assert data["ledger_entries"] == 0
    assert len(data["ledger_hash"]) == 64
    assert data["task_id"] == "easy"


def test_step_before_reset_returns_400():
    """POST /step before POST /reset returns HTTP 400."""
    # Fresh client with no reset called
    client = _make_client()
    response = client.post("/step", json={"action": 0})
    assert response.status_code == 400
    assert "not initialized" in response.json()["detail"].lower() or \
           "reset" in response.json()["detail"].lower()


def test_step_invalid_action_returns_422():
    """POST /step with action=5 (out of range) returns HTTP 422."""
    client = _make_client()
    # First reset
    client.post("/reset", json={"task_id": "easy"})
    # Then step with invalid action
    response = client.post("/step", json={"action": 5})
    assert response.status_code == 422


def test_state_returns_current_observation():
    """GET /state returns current observation without side effects (current_tick doesn't change)."""
    client = _make_client()
    client.post("/reset", json={"task_id": "easy"})

    state1 = client.get("/state").json()
    state2 = client.get("/state").json()

    assert state1["current_tick"] == state2["current_tick"]
    assert state1 == state2
