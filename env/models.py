"""Core data models for the SIREN RL environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Incident:
    """A simulated security incident."""
    id: str                  # e.g. "INC-001"
    severity: int            # 1–5
    type: str                # e.g. "ransomware", "phishing"
    system: str              # e.g. "db-prod-01"
    time_remaining: int      # ticks until SLA breach
    resolved: bool = False
    escalated: bool = False  # set by escalate_human
    isolated: bool = False   # set by isolate_system
    requires_isolation: bool = False  # if True, must be isolated before it can be patched
    scenario_id: str = ""    # links to Scenario in env/scenarios.py for ground truth grading


@dataclass
class PassCriteria:
    """Task-specific pass criteria evaluated by the Grader."""
    # Easy task fields
    incident_resolved: bool = False          # at least one incident resolved
    missing_logs_zero: bool = False          # missing_logs == 0
    min_episode_score: float = 0.0           # minimum grader episode score

    # Medium task fields
    high_severity_resolved: bool = False     # highest-severity incident resolved
    min_proof_score: float = 0.0             # minimum proof_score
    min_sla_credits: float = 0.0             # minimum sla_credits remaining

    # Hard task fields
    severity5_resolved: bool = False         # severity-5 incident resolved
    no_fast_patch_during_audit: bool = False # zero fast_patch actions during audit


@dataclass
class TaskConfig:
    """Configuration for a task difficulty level."""
    task_id: str
    incidents: List[Incident]
    max_ticks: int
    audit_probability: float        # 0.0 = never, 1.0 = always
    audit_from_tick_zero: bool
    initial_sla_credits: float
    initial_proof_score: float
    pass_criteria: PassCriteria
    base_drain_rate: float = 1.0    # SLA credits drained per tick per active session
