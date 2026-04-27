"""
SIREN Environment Client.

Follows the OpenEnv EnvClient pattern from reasoning_gym_env/client.py
and repl_env/client.py.

Maintains a persistent WebSocket connection to the environment server,
enabling efficient multi-step interactions with lower latency.
Each client instance has its own dedicated environment session on the server.

Example:
    >>> from client import SirenEnvClient
    >>> from models import SirenAction
    >>>
    >>> env = SirenEnvClient(base_url="http://localhost:7860")
    >>> result = env.reset(task_id="hard", seed=42)
    >>> print(result.observation.audit_triggered)   # True
    >>> print(result.observation.threat_intel)      # threat context
    >>>
    >>> result = env.step(SirenAction(action=3))    # isolate_system
    >>> print(result.observation.rubric_reward)     # RL training signal
    >>> print(result.observation.ticks_remaining)
    >>> env.close()

Example with Docker:
    >>> client = SirenEnvClient.from_docker_image("siren-env:latest")
    >>> try:
    ...     result = client.reset(task_id="hard", seed=42)
    ...     result = client.step(SirenAction(action=2))
    ... finally:
    ...     client.close()
"""
from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import SirenAction, SirenObservation, SirenState


class SirenEnvClient(EnvClient[SirenAction, SirenObservation, SirenState]):
    """
    WebSocket client for the SIREN environment.

    Connects to a running SIREN server and provides typed access to
    observations, rewards, and episode state.
    """

    def _step_payload(self, action: SirenAction) -> Dict[str, Any]:
        """Serialize SirenAction to JSON payload for the step message."""
        return {"action": action.action}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SirenObservation]:
        """Parse server response into typed StepResult[SirenObservation]."""
        obs_data = payload.get("observation", payload)

        observation = SirenObservation(
            incidents=obs_data.get("incidents", []),
            sla_credits=obs_data.get("sla_credits", 100.0),
            proof_score=obs_data.get("proof_score", 1.0),
            missing_logs=obs_data.get("missing_logs", 0),
            active_sessions=obs_data.get("active_sessions", 0),
            ledger_entries=obs_data.get("ledger_entries", 0),
            ledger_hash=obs_data.get("ledger_hash", ""),
            current_tick=obs_data.get("current_tick", 0),
            max_ticks=obs_data.get("max_ticks", 15),
            ticks_remaining=obs_data.get("ticks_remaining", 15),
            audit_triggered=obs_data.get("audit_triggered", False),
            task_id=obs_data.get("task_id", "easy"),
            threat_intel=obs_data.get("threat_intel", ""),
            rubric_reward=obs_data.get("rubric_reward"),
            done_reason=obs_data.get("done_reason", ""),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SirenState:
        """Parse server response into typed SirenState."""
        return SirenState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "easy"),
            seed=payload.get("seed"),
            current_tick=payload.get("current_tick", 0),
            max_ticks=payload.get("max_ticks", 15),
            total_reward=payload.get("total_reward", 0.0),
            incidents_resolved=payload.get("incidents_resolved", 0),
            audit_triggered=payload.get("audit_triggered", False),
            sla_credits=payload.get("sla_credits", 100.0),
            action_counts=payload.get("action_counts", {}),
            threat_intel=payload.get("threat_intel", ""),
        )
