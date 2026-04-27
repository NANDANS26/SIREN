"""Reward computation for the SIREN RL environment."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.models import Incident


def compute_reward(
    action: int,
    resolved_incident,  # Incident | None — the incident resolved this step (None if no resolution)
    sla_delta: float,   # credits lost this step (old_sla - new_sla), always >= 0
    old_missing_logs: int,
    new_missing_logs: int,
    audit_triggered: bool,
    unresolved_incidents: list,  # list of Incident objects still unresolved after this step
    remaining_sla: float = 100.0,  # current SLA credits after action (for speed incentive)
) -> float:
    """Compute the scalar reward for a single environment step.

    Reward formula:
        reward = 0
        if incident resolved:
            reward += 0.5 + severity * 0.1          # base [0.5, 1.0]
            if action == 1 (fast_patch):
                reward -= 0.4
                if audit_triggered:
                    reward -= 0.6                   # total -1.0
            if action == 2 (verified_patch):
                reward += 0.2                       # reduced from 0.3
        if incident resolved (any action):
            reward += 0.1 * (remaining_sla / 100)  # speed incentive
        if action == 0 (no_op) and any(sev >= 4 for unresolved):
            reward -= 0.3
        reward -= sla_delta * 0.02
        reward -= (new_missing_logs - old_missing_logs) * 0.03   # delta only

    Requirements: 4.1–4.9
    """
    reward = 0.0

    # Incident resolution reward (Req 4.1)
    if resolved_incident is not None:
        reward += 0.5 + resolved_incident.severity * 0.1

        # fast_patch penalty (Req 4.2, 4.3)
        if action == 1:
            reward -= 0.4
            if audit_triggered:
                reward -= 0.6  # additional audit penalty, total -1.0

        # verified_patch bonus — reduced to 0.2 to prevent trivial dominance (Req 4.4)
        if action == 2:
            reward += 0.2

        # Speed incentive: reward resolving incidents while SLA is healthy
        reward += 0.1 * (remaining_sla / 100.0)

    # no_op idle penalty (Req 4.5)
    if action == 0 and any(inc.severity >= 4 for inc in unresolved_incidents):
        reward -= 0.3

    # SLA drain penalty (Req 4.6, 4.9)
    reward -= sla_delta * 0.02

    # Missing-log penalty — delta only, not cumulative (Req 4.7)
    missing_log_delta = new_missing_logs - old_missing_logs
    reward -= missing_log_delta * 0.03

    return reward
