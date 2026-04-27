"""Task configurations for Easy, Medium, and Hard difficulty levels."""
from __future__ import annotations

import copy
from typing import Dict

from env.models import Incident, PassCriteria, TaskConfig
from env.scenarios import EASY_SCENARIOS, MEDIUM_SCENARIOS, HARD_SCENARIOS


def _easy_incidents() -> list:
    return [
        Incident(
            id="INC-001",
            severity=3,
            type="phishing",
            system="web-server-02",
            time_remaining=10,
            resolved=False,
            escalated=False,
            isolated=False,
            scenario_id="SC-001",
        )
    ]


def _medium_incidents() -> list:
    return [
        Incident(
            id="INC-001",
            severity=4,
            type="ransomware",
            system="db-prod-01",
            time_remaining=6,
            resolved=False,
            escalated=False,
            isolated=False,
            scenario_id="SC-002",
        ),
        Incident(
            id="INC-002",
            severity=3,
            type="data_exfiltration",
            system="auth-service-03",
            time_remaining=8,
            resolved=False,
            escalated=False,
            isolated=False,
            scenario_id="SC-008",
        ),
    ]


def _hard_incidents() -> list:
    return [
        Incident(
            id="INC-001",
            severity=5,
            type="ransomware",
            system="db-prod-01",
            time_remaining=4,
            resolved=False,
            escalated=False,
            isolated=False,
            requires_isolation=True,
            scenario_id="SC-002",
        ),
        Incident(
            id="INC-002",
            severity=4,
            type="ddos",
            system="web-server-02",
            time_remaining=5,
            resolved=False,
            escalated=False,
            isolated=False,
            scenario_id="SC-006",
        ),
        Incident(
            id="INC-003",
            severity=3,
            type="insider_threat",
            system="auth-service-03",
            time_remaining=6,
            resolved=False,
            escalated=False,
            isolated=False,
            scenario_id="SC-005",
        ),
    ]


EASY_TASK = TaskConfig(
    task_id="easy",
    incidents=_easy_incidents(),
    max_ticks=15,
    audit_probability=0.0,
    audit_from_tick_zero=False,
    initial_sla_credits=100.0,
    initial_proof_score=1.0,
    base_drain_rate=1.0,
    pass_criteria=PassCriteria(
        incident_resolved=True,
        missing_logs_zero=True,
        min_episode_score=0.6,
    ),
)

MEDIUM_TASK = TaskConfig(
    task_id="medium",
    incidents=_medium_incidents(),
    max_ticks=8,
    audit_probability=0.3,
    audit_from_tick_zero=False,
    initial_sla_credits=100.0,
    initial_proof_score=1.0,
    base_drain_rate=1.0,
    pass_criteria=PassCriteria(
        high_severity_resolved=True,
        min_proof_score=0.5,
        min_sla_credits=20.0,
    ),
)

HARD_TASK = TaskConfig(
    task_id="hard",
    incidents=_hard_incidents(),
    max_ticks=10,
    audit_probability=1.0,
    audit_from_tick_zero=True,
    initial_sla_credits=100.0,
    initial_proof_score=1.0,
    base_drain_rate=3.0,           # triple SLA pressure — 3 credits per active session per tick
    pass_criteria=PassCriteria(
        severity5_resolved=True,
        no_fast_patch_during_audit=True,
        min_episode_score=0.7,
    ),
)

# Registry of all tasks by task_id
TASKS: Dict[str, TaskConfig] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}
