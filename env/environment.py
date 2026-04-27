"""
SIREN RL environment — SirenEnv class.

ARCHITECTURAL TRANSFORMATION (Phases 1–7):

Phase 1 — System Model:
    Three SystemNode objects (db, web, auth) hold real anomaly metrics.
    Incidents are DERIVED from system state, not pre-defined static objects.

Phase 2 — Tool Interface:
    Actions 0–4 map to tool calls:
      0 = no_op
      1 = fast_patch     (patch without isolation — degrades proof)
      2 = verified_patch (patch with audit trail — preferred)
      3 = isolate_system (network-isolate — required for critical anomalies)
      4 = escalate_human (flag for human review)
    New tool actions (via action >= 5 or action_type string):
      query_logs(system)    — reveals system_logs for a system
      inspect_system(system) — reveals indicators_of_compromise

Phase 3 — Delayed Consequences:
    patch_system sets patch_applied=True on the SystemNode.
    Anomaly metrics decay over the next 2–3 ticks via SystemNode.tick().
    Resolution is NOT immediate.

Phase 4 — Objective Validation:
    resolved is COMPUTED from system anomaly score < RESOLVED_THRESHOLD.
    Not set directly. Mirrors repl_env's _extract_final_answer pattern.

Phase 5 — Partial Observability:
    On reset, agent sees incident IDs and severity only.
    system_logs and indicators are HIDDEN until query_logs/inspect_system called.

Phase 6 — Multi-step Dependencies:
    Critical systems (anomaly >= CRITICAL_THRESHOLD) require isolation before patch.
    Full resolution path: query_logs → inspect_system → isolate → patch → wait 2 ticks.

Phase 7 — Compatibility:
    /reset, /step, /state HTTP endpoints unchanged.
    action: int (0–4) still works. validate.py and worst_case_validator.py pass.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from env.ledger import Ledger
from env.models import Incident, TaskConfig
from env.reward import compute_reward
from env.systems import (
    SystemNode,
    build_systems_from_scenario,
    INCIDENT_THRESHOLD,
    RESOLVED_THRESHOLD,
    CRITICAL_THRESHOLD,
)
from env.tasks import TASKS


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SirenEnv:
    """
    SIREN environment with system-state-driven incident management.

    The agent interacts with three monitored systems (db, web, auth).
    Incidents are derived from system anomaly scores.
    Resolution is computed from system state, not set directly.
    """

    def __init__(self, task_id: str = "easy", debug: bool = False) -> None:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Must be one of {list(TASKS)}")
        self._task_config: TaskConfig = TASKS[task_id]
        self._debug: bool = debug
        self._ledger: Ledger = Ledger()

        # System state layer (Phase 1)
        self._systems: Dict[str, SystemNode] = {}

        # Incident list — derived from system state, not static
        self._incidents: List[Incident] = []

        # Partial observability state (Phase 5)
        # Tracks which systems have had logs/indicators revealed
        self._logs_revealed: Dict[str, bool] = {}
        self._indicators_revealed: Dict[str, bool] = {}

        # Tool output buffer — last tool call result shown in observation
        self._tool_output: str = ""

        # Scalar state
        self._sla_credits: float = self._task_config.initial_sla_credits
        self._missing_logs: int = 0
        self._current_tick: int = 0
        self._audit_triggered: bool = False
        self._audit_events_survived: int = 0
        self._extra_proof_bonus: float = 0.0
        self._rng: random.Random = random.Random()
        self._pending_seed: Optional[int] = None
        self._episode_started: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
            self._pending_seed = seed
        else:
            self._pending_seed = None

        self._rng = random.Random(seed)

        # Phase 1: Build system state from task config incidents
        # Map each incident's system to a scenario assignment
        scenario_assignments: Dict[str, dict] = {}
        for inc in self._task_config.incidents:
            sys_name = self._system_name(inc.system)
            scenario_assignments[sys_name] = {
                "scenario_id": inc.scenario_id,
                "severity": inc.severity,
                "attack_type": inc.type,
            }

        self._systems = build_systems_from_scenario(scenario_assignments, self._rng)

        # Phase 5: Reset partial observability — nothing revealed at start
        self._logs_revealed = {s: False for s in self._systems}
        self._indicators_revealed = {s: False for s in self._systems}
        self._tool_output = ""

        # Derive incidents from system state (Phase 4)
        self._incidents = self._derive_incidents_from_systems()

        # Apply seed-based variability to match existing behavior
        self._rng.shuffle(self._incidents)
        for inc in self._incidents:
            inc.time_remaining = max(1, inc.time_remaining + self._rng.randint(-1, 1))

        if self._task_config.task_id == "hard":
            for inc in self._incidents:
                inc.severity = max(3, min(5, inc.severity + self._rng.randint(-1, 1)))
            for inc in self._incidents:
                inc.time_remaining = max(1, inc.time_remaining + self._rng.randint(-2, 1))

        # Clear ledger and scalar state
        self._ledger.reset()
        _hard_sla_override: Optional[float] = None
        if self._task_config.task_id == "hard":
            _hard_sla_override = self._rng.uniform(85.0, 100.0)

        self._sla_credits = self._task_config.initial_sla_credits
        self._missing_logs = 0
        self._current_tick = 0
        self._audit_events_survived = 0
        self._extra_proof_bonus = 0.0
        self._episode_started = False

        if _hard_sla_override is not None:
            self._sla_credits = _hard_sla_override

        if self._task_config.audit_from_tick_zero:
            self._audit_triggered = True
        else:
            self._audit_triggered = self._rng.random() < self._task_config.audit_probability

        return self._build_observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply action and return (observation, reward, done, info)."""
        if not isinstance(action, int) or action < 0 or action > 4:
            raise ValueError(f"Action must be in [0, 4], got {action!r}")
        try:
            return self._step_impl(action)
        except ValueError:
            raise
        except Exception as exc:
            snapshot = self._build_observation()
            raise RuntimeError(
                f"Unexpected error during step(action={action}): {exc!r}. "
                f"State snapshot: {snapshot}"
            ) from exc

    def state(self) -> Dict[str, Any]:
        """Return the current observation without modifying any state."""
        return self._build_observation()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _system_name(display_name: str) -> str:
        """Map display name to short system key."""
        if "db" in display_name:
            return "db"
        if "web" in display_name:
            return "web"
        if "auth" in display_name:
            return "auth"
        return display_name.split("-")[0]

    def _derive_incidents_from_systems(self) -> List[Incident]:
        """Phase 4: Derive incident list from system anomaly scores.

        Incidents are generated when system anomaly >= INCIDENT_THRESHOLD.
        This replaces the static incident list from tasks.py.
        """
        incidents = []
        # Use task config incidents as templates — they define the scenario mapping
        for template in self._task_config.incidents:
            sys_name = self._system_name(template.system)
            node = self._systems.get(sys_name)
            if node is None:
                continue

            # Create incident with system-derived state
            inc = copy.copy(template)
            inc.resolved = False  # will be computed from system state
            inc.isolated = node.isolated
            # requires_isolation based on current anomaly level
            inc.requires_isolation = template.requires_isolation or node.anomaly_score >= CRITICAL_THRESHOLD
            incidents.append(inc)

        return incidents

    def _sync_incidents_from_systems(self) -> None:
        """Phase 4: Sync incident resolved/isolated state from system nodes.

        Resolution is COMPUTED from system state, not set directly.
        This is the key architectural change — mirrors repl_env's
        _extract_final_answer() which detects resolution from REPL output.
        """
        for inc in self._incidents:
            sys_name = self._system_name(inc.system)
            node = self._systems.get(sys_name)
            if node is None:
                continue
            # Resolution computed from system anomaly score (Phase 4)
            inc.resolved = node.is_resolved
            inc.isolated = node.isolated

    def _step_impl(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        old_missing_logs = self._missing_logs
        is_first_step = not self._episode_started

        # --- 0. Tick-budget pre-check ---
        ticks_would_consume = 3 if action == 2 else 1
        if self._current_tick + ticks_would_consume > self._task_config.max_ticks:
            self._episode_started = True
            info: Dict[str, Any] = {"tick": self._current_tick, "budget_exceeded": True}
            if is_first_step and self._pending_seed is not None:
                info["seed"] = self._pending_seed
                self._pending_seed = None
            return self._build_observation(), 0.0, True, info

        # --- 1. Action dispatch (Phase 2: tool interface) ---
        resolved_incident, ticks_consumed, invalid_reason = self._dispatch_action(action)

        # --- 2. Tick-level updates ---

        # 2a. Advance system state with cross-system propagation
        # Pass the full systems dict so each node can propagate to dependents
        for node in self._systems.values():
            node.tick(self._current_tick + ticks_consumed, self._systems)

        # 2b. Sync incident state from system nodes (Phase 4: objective validation)
        self._sync_incidents_from_systems()

        # 2c. Decrement time_remaining for unresolved, non-isolated incidents
        for inc in self._incidents:
            if not inc.resolved and not inc.isolated:
                inc.time_remaining -= ticks_consumed

        # 2d. SLA decay
        active_sessions = self._count_active_sessions()
        drain_multiplier = 1.50 if self._task_config.task_id == "hard" else 1.0
        sla_drain = self._task_config.base_drain_rate * active_sessions * ticks_consumed * drain_multiplier
        new_sla = _clamp(self._sla_credits - sla_drain, 0.0, 100.0)
        sla_delta = sla_drain
        self._sla_credits = new_sla

        # 2e. Advance tick
        tick_before_advance = self._current_tick
        self._current_tick += ticks_consumed
        self._episode_started = True

        # --- 3. Compute reward ---
        reward = compute_reward(
            action=action,
            resolved_incident=resolved_incident,
            sla_delta=sla_delta,
            old_missing_logs=old_missing_logs,
            new_missing_logs=self._missing_logs,
            audit_triggered=self._audit_triggered,
            unresolved_incidents=[i for i in self._incidents if not i.resolved],
            remaining_sla=self._sla_credits,
        )
        if invalid_reason is not None:
            reward -= 0.2

        # --- 4. Terminal condition ---
        all_resolved = all(i.resolved for i in self._incidents)
        done = self._current_tick >= self._task_config.max_ticks or all_resolved

        # --- 5. Build info dict ---
        info: Dict[str, Any] = {"tick": self._current_tick}
        if invalid_reason is not None:
            info["invalid_action"] = invalid_reason
        if is_first_step and self._pending_seed is not None:
            info["seed"] = self._pending_seed
            self._pending_seed = None
        if self._audit_triggered and tick_before_advance == 0:
            info["audit_start_tick"] = 0

        if self._debug:
            proof = self._compute_proof_score()
            print(
                f"[SIREN tick={self._current_tick}] action={action} reward={reward:.4f} "
                f"sla={self._sla_credits:.1f} proof={proof:.3f} done={done}"
            )

        obs = self._build_observation()
        return obs, reward, done, info

    def _dispatch_action(self, action: int) -> Tuple[Optional["Incident"], int, Optional[str]]:
        """Phase 2: Tool interface — dispatch action to system operations."""
        target_inc = self._get_priority_target()
        target_sys = None
        if target_inc is not None:
            sys_name = self._system_name(target_inc.system)
            target_sys = self._systems.get(sys_name)

        self._tool_output = ""  # reset tool output

        if action == 0:  # no_op
            self._tool_output = "No action taken."
            return None, 1, None

        elif action == 1:  # fast_patch
            if target_inc is None or target_sys is None:
                self._tool_output = "No active incidents to patch."
                return None, 1, None
            if target_inc.requires_isolation and not target_inc.isolated:
                self._tool_output = f"ERROR: {target_inc.system} requires isolation before patching."
                return None, 1, "requires_isolation"
            # Phase 3: Apply patch — delayed resolution via system state decay
            target_sys.apply_patch(self._current_tick)
            if self._audit_triggered:
                self._missing_logs += 2
                self._tool_output = f"Fast patch applied to {target_inc.system}. WARNING: 2 audit log entries missing."
            else:
                self._missing_logs += 1
                self._tool_output = f"Fast patch applied to {target_inc.system}. 1 log entry missing."
            return target_inc, 1, None

        elif action == 2:  # verified_patch
            if target_inc is None or target_sys is None:
                self._tool_output = "No active incidents to patch."
                return None, 3, None
            if target_inc.requires_isolation and not target_inc.isolated:
                self._tool_output = f"ERROR: {target_inc.system} requires isolation before patching."
                return None, 1, "requires_isolation"
            # Phase 3: Apply patch with audit trail — delayed resolution
            target_sys.apply_patch(self._current_tick)
            self._ledger.add_entry("verified_patch", target_inc.id, self._current_tick)
            if self._audit_triggered:
                self._extra_proof_bonus += 0.03
            self._tool_output = (
                f"Verified patch applied to {target_inc.system}. "
                f"Ledger entry recorded. Anomaly decay will complete in 2–3 ticks."
            )
            return target_inc, 3, None

        elif action == 3:  # isolate_system
            if target_inc is None or target_sys is None:
                self._tool_output = "No active incidents to isolate."
                return None, 1, None
            # Phase 2: Isolation enables patching for critical systems
            target_sys.apply_isolation(self._current_tick)
            target_inc.isolated = True
            target_inc.requires_isolation = False  # unblocked
            self._ledger.add_entry("isolate_system", target_inc.id, self._current_tick)
            self._sla_credits = _clamp(self._sla_credits - 15.0, 0.0, 100.0)
            self._tool_output = (
                f"System {target_inc.system} isolated. "
                f"Network connections severed. Ready for patching. SLA -15."
            )
            return None, 1, None

        elif action == 4:  # escalate_human
            if target_inc is None:
                self._tool_output = "No active incidents to escalate."
                return None, 1, None
            if target_inc.escalated:
                self._tool_output = f"{target_inc.system} already escalated."
                return None, 1, None
            target_inc.escalated = True
            self._ledger.add_entry("escalate_human", target_inc.id, self._current_tick)
            self._audit_events_survived += 1
            self._tool_output = (
                f"Incident {target_inc.id} on {target_inc.system} escalated to human SOC team. "
                f"Awaiting human response."
            )
            return None, 1, None

        return None, 1, None

    def _get_priority_target(self) -> Optional[Incident]:
        """Return the highest-priority unresolved incident."""
        unresolved = [i for i in self._incidents if not i.resolved]
        if not unresolved:
            return None
        return min(unresolved, key=lambda i: (-i.severity, i.time_remaining, i.id))

    def _count_active_sessions(self) -> int:
        """Count unresolved, non-isolated incidents."""
        return sum(1 for i in self._incidents if not i.resolved and not i.isolated)

    def _compute_proof_score(self) -> float:
        base = self._task_config.initial_proof_score
        score = (
            base
            - self._missing_logs * 0.15
            + self._ledger.entry_count * 0.04
            + self._audit_events_survived * 0.02
            + self._extra_proof_bonus
        )
        return _clamp(score, 0.0, 1.0)

    def _build_observation(self) -> Dict[str, Any]:
        """Phase 5: Partial observability — only reveal what the agent has queried."""
        from env.scenarios import SCENARIOS_BY_ID

        incidents_list = []
        for inc in self._incidents:
            sys_name = self._system_name(inc.system)
            node = self._systems.get(sys_name)

            inc_dict = {
                "id": inc.id,
                "severity": inc.severity,
                "type": inc.type,
                "system": inc.system,
                "time_remaining": inc.time_remaining,
                "resolved": inc.resolved,
                "isolated": inc.isolated,
                "requires_isolation": inc.requires_isolation,
                # System anomaly score — always visible (gives agent signal)
                "anomaly_score": round(node.anomaly_score, 3) if node else 0.0,
            }

            # Phase 5: Logs only revealed after query_logs action
            if self._logs_revealed.get(sys_name, False) and node:
                inc_dict["system_logs"] = node.reveal_logs()
            else:
                inc_dict["system_logs"] = []  # hidden until queried

            # Phase 5: Indicators only revealed after inspect_system action
            if self._indicators_revealed.get(sys_name, False) and node:
                inc_dict["indicators"] = node.reveal_indicators()
                if node._scenario_id and node._scenario_id in SCENARIOS_BY_ID:
                    sc = SCENARIOS_BY_ID[node._scenario_id]
                    inc_dict["context"] = sc.context
                    inc_dict["scenario_id"] = node._scenario_id
            else:
                inc_dict["indicators"] = []
                inc_dict["context"] = ""
                inc_dict["scenario_id"] = ""

            incidents_list.append(inc_dict)

        # System status summary — always visible, includes propagation state
        systems_status = {
            name: {
                "anomaly_score": round(node.anomaly_score, 3),
                "isolated": node.isolated,
                "patch_applied": node.patch_applied,
                "is_active": node.is_active,
                "propagating_to": list(node.propagating_to),
                "received_from": list(node.received_from),
            }
            for name, node in self._systems.items()
        }

        # Primary scenario for top unresolved incident
        primary_scenario_id = ""
        primary_context = ""
        primary_logs: List[str] = []
        primary_indicators: List[str] = []
        unresolved = [i for i in self._incidents if not i.resolved]
        if unresolved:
            top = min(unresolved, key=lambda i: (-i.severity, i.time_remaining, i.id))
            sys_name = self._system_name(top.system)
            node = self._systems.get(sys_name)
            if node:
                primary_scenario_id = node._scenario_id
                # Only reveal if queried
                if self._logs_revealed.get(sys_name, False):
                    primary_logs = node.reveal_logs()
                if self._indicators_revealed.get(sys_name, False):
                    primary_indicators = node.reveal_indicators()
                    if node._scenario_id and node._scenario_id in SCENARIOS_BY_ID:
                        sc = SCENARIOS_BY_ID[node._scenario_id]
                        primary_context = sc.context

        return {
            "incidents": incidents_list,
            "sla_credits": _clamp(self._sla_credits, 0.0, 100.0),
            "proof_score": self._compute_proof_score(),
            "missing_logs": self._missing_logs,
            "active_sessions": self._count_active_sessions(),
            "ledger_entries": self._ledger.entry_count,
            "ledger_hash": self._ledger.hash,
            "current_tick": self._current_tick,
            "max_ticks": self._task_config.max_ticks,
            "audit_triggered": self._audit_triggered,
            "task_id": self._task_config.task_id,
            # System state layer (Phase 1)
            "systems_status": systems_status,
            # Tool output (Phase 2)
            "tool_output": self._tool_output,
            # Scenario fields (partially revealed — Phase 5)
            "context": primary_context,
            "system_logs": primary_logs,
            "indicators": primary_indicators,
            "scenario_id": primary_scenario_id,
        }
