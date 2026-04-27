"""
Data models for the SIREN environment.

Follows the OpenEnv typed model pattern using Action/Observation/State base classes.
Mirrors the richness of carla_env/models.py and repl_env/models.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SirenAction(Action):
    """Action for the SIREN environment.

    The agent submits a single integer (0–4) representing the security response action.
    """
    action: int = Field(
        default=2,
        ge=0,
        le=4,
        description=(
            "0=no_op (advance tick), "
            "1=fast_patch (resolve fast, degrades proof), "
            "2=verified_patch (resolve with audit trail, costs 3 ticks), "
            "3=isolate_system (freeze SLA countdown, costs 15 SLA credits), "
            "4=escalate_human (flag for human review, adds ledger entry)"
        ),
    )


class SirenObservation(Observation):
    """Observation returned from the SIREN environment.

    Includes all state the agent needs to make a decision, plus
    rubric_reward for RL training signal and done_reason for diagnostics.
    """
    # Core incident state
    incidents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of active security incidents with severity, type, time_remaining",
    )
    sla_credits: float = Field(
        default=100.0,
        description="Remaining SLA credits [0, 100]. Episode fails if depleted.",
    )
    proof_score: float = Field(
        default=1.0,
        description="Audit proof quality [0, 1]. Degrades with fast_patch, improves with verified_patch.",
    )
    missing_logs: int = Field(
        default=0,
        description="Count of missing audit log entries. Increases with fast_patch.",
    )
    active_sessions: int = Field(
        default=0,
        description="Number of unresolved, non-isolated incidents draining SLA.",
    )

    # Ledger state
    ledger_entries: int = Field(default=0, description="Number of verified ledger entries.")
    ledger_hash: str = Field(default="", description="SHA-256 hash of the action ledger.")

    # Tick budget
    current_tick: int = Field(default=0, description="Current simulation tick.")
    max_ticks: int = Field(default=15, description="Maximum ticks before episode ends.")
    ticks_remaining: int = Field(
        default=15,
        description="Ticks remaining before budget exhausted (convenience field).",
    )

    # Audit state
    audit_triggered: bool = Field(
        default=False,
        description="Whether audit mode is active. fast_patch is penalized during audit.",
    )

    # Task context
    task_id: str = Field(default="easy", description="Task difficulty: easy, medium, or hard.")
    threat_intel: str = Field(
        default="",
        description="Optional threat intelligence context injected at reset time.",
    )

    # Scenario reasoning fields — the REAL PROBLEM for the LLM agent
    context: str = Field(
        default="",
        description="Natural language description of the security incident. The agent MUST read this.",
    )
    system_logs: List[str] = Field(
        default_factory=list,
        description="Simulated log lines showing what is happening on the affected systems.",
    )
    indicators: List[str] = Field(
        default_factory=list,
        description="Observable indicators of compromise the agent should detect.",
    )
    scenario_id: str = Field(
        default="",
        description="Scenario identifier for grading against correct_action_sequence.",
    )

    # System state layer (Phase 1) — always visible
    systems_status: Dict[str, Any] = Field(
        default_factory=dict,
        description="Real-time anomaly scores for each monitored system (db, web, auth).",
    )

    # Tool output (Phase 2) — result of last tool call
    tool_output: str = Field(
        default="",
        description="Output from the last tool action (query_logs, inspect_system, patch, etc.).",
    )

    # RL training signal (follows carla_env pattern)
    rubric_reward: Optional[float] = Field(
        default=None,
        description="Reward computed by SirenRubric for RL training. None on non-terminal steps.",
    )

    # Diagnostics
    done_reason: str = Field(
        default="",
        description="Why the episode ended: 'all_resolved', 'tick_budget', 'budget_exceeded', or ''.",
    )


class SirenState(State):
    """Server-side episode state for the SIREN environment.

    Tracks episode-level metrics for analytics and debugging.
    Mirrors the richness of carla_env/models.py:CarlaState.
    """
    task_id: str = Field(default="easy")
    seed: Optional[int] = Field(default=None)

    # Tick tracking
    current_tick: int = Field(default=0)
    max_ticks: int = Field(default=15)

    # Episode metrics
    total_reward: float = Field(default=0.0, description="Cumulative reward this episode.")
    incidents_resolved: int = Field(default=0, description="Total incidents resolved.")
    audit_triggered: bool = Field(default=False)
    sla_credits: float = Field(default=100.0)

    # Action tracking (mirrors carla_env:CarlaState.tool_call_counts)
    action_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each action type taken: no_op, fast_patch, verified_patch, isolate_system, escalate_human.",
    )

    # Threat context
    threat_intel: str = Field(default="")
