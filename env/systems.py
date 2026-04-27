"""
SIREN System Model — Phase 1 + Cross-System Propagation.

Introduces a real system state layer with dependency graph.
Incidents are DERIVED from system state, not pre-defined static objects.
Resolution is COMPUTED from system state, not set directly.

Three systems: db (database), web (web server), auth (auth service).
Dependency graph: db → auth → web
  - db compromise propagates to auth (credential leakage)
  - auth compromise propagates to web (session hijacking)

Propagation rules:
  - Only propagates when source is NOT isolated
  - Rate scales with source anomaly score
  - Isolation stops both local growth AND outbound propagation
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Anomaly thresholds
# ---------------------------------------------------------------------------
INCIDENT_THRESHOLD = 0.6   # anomaly score above this → incident active
RESOLVED_THRESHOLD = 0.2   # anomaly score below this → incident resolved
CRITICAL_THRESHOLD = 0.85  # anomaly score above this → requires isolation first

# ---------------------------------------------------------------------------
# Dependency graph: source → list of downstream systems it can infect
# db compromises auth (credential leakage), auth compromises web (session hijack)
# ---------------------------------------------------------------------------
DEPENDENCY_GRAPH: Dict[str, List[str]] = {
    "db":   ["auth"],
    "auth": ["web"],
    "web":  [],
}

# Propagation rate: fraction of source anomaly pushed to dependent per tick
PROPAGATION_RATE = 0.15   # unisolated high-anomaly source pushes 15% per tick
PROPAGATION_THRESHOLD = 0.5  # source must exceed this to propagate at all


@dataclass
class SystemNode:
    """A monitored system with real-time anomaly metrics and propagation tracking."""
    name: str
    display_name: str

    # Anomaly metrics
    error_rate: float = 0.0
    traffic_anomaly: float = 0.0
    process_anomaly: float = 0.0
    connection_anomaly: float = 0.0

    # Patch state
    patch_applied: bool = False
    patch_tick: int = -1
    patch_decay_rate: float = 0.4

    # Isolation state
    isolated: bool = False
    isolation_tick: int = -1

    # Escalation state
    escalated: bool = False

    # Propagation tracking — updated each tick for observation
    propagating_to: List[str] = field(default_factory=list)
    received_from: List[str] = field(default_factory=list)

    # Partial observability
    _revealed_logs: List[str] = field(default_factory=list)
    _revealed_indicators: List[str] = field(default_factory=list)
    _scenario_id: str = ""

    @property
    def anomaly_score(self) -> float:
        return (
            self.error_rate * 0.2
            + self.traffic_anomaly * 0.3
            + self.process_anomaly * 0.35
            + self.connection_anomaly * 0.15
        )

    @property
    def is_active(self) -> bool:
        return self.anomaly_score >= INCIDENT_THRESHOLD

    @property
    def is_resolved(self) -> bool:
        return self.anomaly_score < RESOLVED_THRESHOLD

    @property
    def requires_isolation(self) -> bool:
        return self.anomaly_score >= CRITICAL_THRESHOLD and not self.isolated

    def tick(self, current_tick: int, systems: "Dict[str, SystemNode]") -> None:
        """Advance system state by one tick, including cross-system propagation.

        Propagation rules:
          - Source must NOT be isolated to propagate
          - Source anomaly must exceed PROPAGATION_THRESHOLD
          - Propagation pushes connection_anomaly to downstream systems
          - Failure cascade: unisolated systems grow faster when neighbors are infected
        """
        # Reset propagation tracking for this tick
        self.propagating_to = []
        self.received_from = []

        if self.patch_applied:
            ticks_since_patch = current_tick - self.patch_tick
            if ticks_since_patch > 0:
                decay = self.patch_decay_rate * ticks_since_patch
                self.error_rate = max(0.0, self.error_rate - decay * 0.3)
                self.traffic_anomaly = max(0.0, self.traffic_anomaly - decay * 0.4)
                self.process_anomaly = max(0.0, self.process_anomaly - decay * 0.5)
                self.connection_anomaly = max(0.0, self.connection_anomaly - decay * 0.3)
        elif not self.isolated:
            # Unpatched, unisolated: local anomaly growth
            growth = 0.02
            self.error_rate = min(1.0, self.error_rate + growth * 0.5)
            self.process_anomaly = min(1.0, self.process_anomaly + growth * 0.3)

        # Cross-system propagation — only if not isolated and anomaly is high enough
        if not self.isolated and self.anomaly_score >= PROPAGATION_THRESHOLD:
            downstream = DEPENDENCY_GRAPH.get(self.name, [])
            for dep_name in downstream:
                dep = systems.get(dep_name)
                if dep is None or dep.isolated:
                    continue  # isolation stops propagation
                # Propagate connection anomaly — lateral movement / credential spread
                spread = PROPAGATION_RATE * self.anomaly_score
                dep.connection_anomaly = min(1.0, dep.connection_anomaly + spread)
                # Failure cascade: also raise error rate slightly
                dep.error_rate = min(1.0, dep.error_rate + spread * 0.3)
                self.propagating_to.append(dep_name)
                dep.received_from.append(self.name)

    def apply_patch(self, current_tick: int) -> None:
        self.patch_applied = True
        self.patch_tick = current_tick

    def apply_isolation(self, current_tick: int) -> None:
        """Isolate system — stops anomaly growth AND outbound propagation."""
        self.isolated = True
        self.isolation_tick = current_tick
        self.propagating_to = []  # immediately stops propagation

    def reveal_logs(self) -> List[str]:
        return list(self._revealed_logs)

    def reveal_indicators(self) -> List[str]:
        return list(self._revealed_indicators)


def build_systems_from_scenario(
    scenario_assignments: Dict[str, dict],
    rng: random.Random,
) -> Dict[str, SystemNode]:
    """Build system nodes with anomaly scores derived from scenario data.

    Args:
        scenario_assignments: Maps system name → {scenario_id, severity, attack_type}
        rng: Seeded RNG for reproducible anomaly initialization

    Returns:
        Dict of system_name → SystemNode
    """
    from env.scenarios import SCENARIOS_BY_ID

    system_display = {
        "db":   "db-prod-01",
        "web":  "web-server-02",
        "auth": "auth-service-03",
    }

    systems: Dict[str, SystemNode] = {}

    for sys_name, display in system_display.items():
        node = SystemNode(name=sys_name, display_name=display)

        if sys_name in scenario_assignments:
            assignment = scenario_assignments[sys_name]
            scenario_id = assignment.get("scenario_id", "")
            severity = assignment.get("severity", 3)
            attack_type = assignment.get("attack_type", "")

            # Set anomaly scores based on severity and attack type
            base = 0.5 + (severity - 3) * 0.12  # severity 3→0.5, 4→0.62, 5→0.74
            jitter = rng.uniform(-0.05, 0.05)

            if attack_type in ("ransomware", "privilege_escalation", "supply_chain", "lateral_movement"):
                # Process-heavy attacks
                node.process_anomaly = min(1.0, base + 0.15 + jitter)
                node.connection_anomaly = min(1.0, base * 0.8 + jitter)
                node.error_rate = min(1.0, base * 0.5 + jitter)
                node.traffic_anomaly = min(1.0, base * 0.4 + jitter)
            elif attack_type in ("ddos",):
                # Traffic-heavy attacks
                node.traffic_anomaly = min(1.0, base + 0.2 + jitter)
                node.connection_anomaly = min(1.0, base + 0.1 + jitter)
                node.error_rate = min(1.0, base * 0.7 + jitter)
                node.process_anomaly = min(1.0, base * 0.3 + jitter)
            elif attack_type in ("data_exfiltration", "sql_injection", "credential_stuffing"):
                # Connection/data attacks
                node.connection_anomaly = min(1.0, base + 0.15 + jitter)
                node.traffic_anomaly = min(1.0, base + 0.1 + jitter)
                node.error_rate = min(1.0, base * 0.6 + jitter)
                node.process_anomaly = min(1.0, base * 0.4 + jitter)
            else:
                # Default: balanced anomalies
                node.error_rate = min(1.0, base * 0.6 + jitter)
                node.traffic_anomaly = min(1.0, base * 0.5 + jitter)
                node.process_anomaly = min(1.0, base * 0.5 + jitter)
                node.connection_anomaly = min(1.0, base * 0.4 + jitter)

            # Attach scenario data for partial observability
            if scenario_id and scenario_id in SCENARIOS_BY_ID:
                sc = SCENARIOS_BY_ID[scenario_id]
                node._revealed_logs = sc.system_logs
                node._revealed_indicators = sc.indicators_of_compromise
                node._scenario_id = scenario_id

        systems[sys_name] = node

    return systems
