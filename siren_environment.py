"""
SIREN Environment — OpenEnv-compatible implementation.

Extends openenv.core.env_server.interfaces.Environment.
Integrates SirenRubric for per-step RL training reward signal.
Follows the architectural pattern of reasoning_gym_environment.py and
carla_environment.py from the reference repos.
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from models import SirenAction, SirenObservation, SirenState
    from env.rubric import SirenRubric
except ImportError:
    from .models import SirenAction, SirenObservation, SirenState
    from .env.rubric import SirenRubric

from env.environment import SirenEnv

# Action name mapping for state tracking (mirrors carla_env tool_call_counts)
_ACTION_NAMES = {
    0: "no_op",
    1: "fast_patch",
    2: "verified_patch",
    3: "isolate_system",
    4: "escalate_human",
}

# Threat intelligence scenarios injected at reset (new capability)
_THREAT_INTEL = {
    "easy": "Low-severity phishing campaign detected. Standard response protocols apply.",
    "medium": "Active ransomware and data exfiltration campaign. Elevated threat posture.",
    "hard": "Critical multi-vector attack in progress. Audit mode active. All actions logged.",
}


class SirenEnvironment(Environment[SirenAction, SirenObservation, SirenState]):
    """
    SIREN Security Incident Response ENvironment.

    An RL training environment where an agent acts as an automated Security
    Operations Controller, resolving live security incidents while managing
    the Proof Gap tradeoff: fast actions sacrifice audit quality; thorough
    actions consume the tick budget.

    Three difficulty tiers:
        easy   — 1 incident, 15 ticks, no audit
        medium — 2 incidents, 8 ticks, 30% audit probability
        hard   — 3 incidents, 10 ticks, audit always active, 1.5× SLA drain

    Rubric:
        SirenRubric provides per-step RL training signal via obs.rubric_reward.
        Process reward: small per-step penalties for SLA drain, missing logs, idle.
        Outcome reward: grader score at episode termination.

    Threat Intelligence:
        Optional threat_intel context injected at reset time into observations,
        giving LLM agents domain-specific context for better decision-making.

    Example:
        >>> env = SirenEnvironment()
        >>> obs = env.reset(task_id="hard", seed=42)
        >>> print(obs.audit_triggered)   # True
        >>> print(obs.threat_intel)      # "Critical multi-vector attack..."
        >>> obs = env.step(SirenAction(action=3))  # isolate_system
        >>> print(obs.rubric_reward)     # per-step process reward
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._env: Optional[SirenEnv] = None
        self._rubric = SirenRubric()
        self._trajectory: list = []
        self._state = SirenState()
        self._threat_intel: str = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy",
        threat_intel: Optional[str] = None,
        **kwargs: Any,
    ) -> SirenObservation:
        """Reset the environment and return the initial observation.

        Args:
            seed: Random seed for reproducibility (None = random)
            episode_id: Optional episode identifier
            task_id: Difficulty — one of "easy", "medium", "hard"
            threat_intel: Optional threat context string. If None, uses
                          the default context for the task difficulty.

        Returns:
            Initial SirenObservation with threat_intel populated
        """
        self._env = SirenEnv(task_id=task_id)
        obs_dict = self._env.reset(seed=seed)
        self._trajectory = []

        # Reset rubric for new episode (follows repl_env pattern)
        self._rubric.reset()

        # Threat intelligence context (new capability — not in any reference repo)
        self._threat_intel = threat_intel if threat_intel is not None else _THREAT_INTEL.get(task_id, "")

        self._state = SirenState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            seed=seed,
            current_tick=obs_dict["current_tick"],
            max_ticks=obs_dict["max_ticks"],
            audit_triggered=obs_dict["audit_triggered"],
            sla_credits=obs_dict["sla_credits"],
            total_reward=0.0,
            incidents_resolved=0,
            action_counts={name: 0 for name in _ACTION_NAMES.values()},
            threat_intel=self._threat_intel,
        )

        return SirenObservation(
            **obs_dict,
            ticks_remaining=obs_dict["max_ticks"] - obs_dict["current_tick"],
            threat_intel=self._threat_intel,
            reward=0.0,
            done=False,
            done_reason="",
            rubric_reward=None,
        )

    def step(
        self,
        action: SirenAction,
        **kwargs: Any,
    ) -> SirenObservation:
        """Execute an action and return the resulting observation.

        Args:
            action: SirenAction with action integer (0–4)

        Returns:
            SirenObservation with reward, rubric_reward, and done_reason
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        obs_dict, reward, done, info = self._env.step(action.action)

        # Track action counts (mirrors carla_env:CarlaState.tool_call_counts)
        action_name = _ACTION_NAMES.get(action.action, "unknown")
        self._state.action_counts[action_name] = self._state.action_counts.get(action_name, 0) + 1
        self._state.step_count += 1
        self._state.current_tick = obs_dict["current_tick"]
        self._state.sla_credits = obs_dict["sla_credits"]
        self._state.audit_triggered = obs_dict["audit_triggered"]
        self._state.total_reward += reward

        # Determine done_reason for diagnostics
        done_reason = ""
        if done:
            if info.get("budget_exceeded"):
                done_reason = "budget_exceeded"
            elif all(inc.get("resolved", False) for inc in obs_dict["incidents"]):
                done_reason = "all_resolved"
                self._state.incidents_resolved = len(obs_dict["incidents"])
            else:
                done_reason = "tick_budget"

        obs = SirenObservation(
            **obs_dict,
            ticks_remaining=max(0, obs_dict["max_ticks"] - obs_dict["current_tick"]),
            threat_intel=self._threat_intel,
            reward=reward,
            done=done,
            done_reason=done_reason,
            rubric_reward=None,
        )
        step_record = {
            "action": action.action,
            "observation": obs_dict,
            "reward": reward,
            "done": done,
            "info": info,
        }
        self._trajectory.append(step_record)
        self._rubric.record_step(step_record)

        # Compute rubric reward (follows carla_env pattern: obs.rubric_reward)
        obs.rubric_reward = self._rubric(action.action, obs)

        # Attach grader score and info to metadata on terminal step
        obs.metadata = info
        if done:
            obs.metadata["grader_score"] = obs.rubric_reward
            obs.metadata["done_reason"] = done_reason
            obs.metadata["action_counts"] = dict(self._state.action_counts)

        return obs

    @property
    def state(self) -> SirenState:
        """Return current episode state with full metrics."""
        return self._state
