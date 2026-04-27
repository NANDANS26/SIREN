# Feature: siren-env, Property 8: Reward Formula
"""Property-based tests for the reward formula.

Validates: Requirements 4.1–4.8
"""
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from env.models import Incident
from env.reward import compute_reward


# --- Strategies ---

incident_strategy = st.builds(
    Incident,
    id=st.just("INC-001"),
    severity=st.integers(min_value=1, max_value=5),
    type=st.just("ransomware"),
    system=st.just("db-prod-01"),
    time_remaining=st.integers(min_value=0, max_value=100),
    resolved=st.just(False),
    escalated=st.just(False),
    isolated=st.just(False),
)

resolved_incident_strategy = st.one_of(st.none(), incident_strategy)

unresolved_list_strategy = st.lists(incident_strategy, min_size=0, max_size=5)


@given(
    action=st.integers(min_value=0, max_value=4),
    resolved_incident=resolved_incident_strategy,
    sla_delta=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    old_missing_logs=st.integers(min_value=0, max_value=50),
    missing_log_delta=st.integers(min_value=0, max_value=10),
    audit_triggered=st.booleans(),
    unresolved_incidents=unresolved_list_strategy,
    remaining_sla=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_reward_formula(
    action,
    resolved_incident,
    sla_delta,
    old_missing_logs,
    missing_log_delta,
    audit_triggered,
    unresolved_incidents,
    remaining_sla,
):
    """Property 8: reward returned by compute_reward() equals the expected formula sum.

    **Validates: Requirements 4.1–4.8**
    """
    new_missing_logs = old_missing_logs + missing_log_delta

    # Independently compute expected reward using the same formula
    expected = 0.0
    if resolved_incident is not None:
        expected += 0.5 + resolved_incident.severity * 0.1
        if action == 1:
            expected -= 0.4
            if audit_triggered:
                expected -= 0.6
        if action == 2:
            expected += 0.2
        expected += 0.1 * (remaining_sla / 100.0)
    if action == 0 and any(inc.severity >= 4 for inc in unresolved_incidents):
        expected -= 0.3
    expected -= sla_delta * 0.02
    expected -= (new_missing_logs - old_missing_logs) * 0.03

    actual = compute_reward(
        action=action,
        resolved_incident=resolved_incident,
        sla_delta=sla_delta,
        old_missing_logs=old_missing_logs,
        new_missing_logs=new_missing_logs,
        audit_triggered=audit_triggered,
        unresolved_incidents=unresolved_incidents,
        remaining_sla=remaining_sla,
    )

    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (
        f"Reward mismatch: got {actual}, expected {expected} "
        f"(action={action}, resolved={resolved_incident}, sla_delta={sla_delta}, "
        f"old_missing={old_missing_logs}, new_missing={new_missing_logs}, "
        f"audit={audit_triggered}, remaining_sla={remaining_sla})"
    )
