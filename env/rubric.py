"""
SIREN Rubric system — follows OpenEnv RFC 004.

Provides composable, per-step reward computation via openenv.core.rubrics.base.Rubric.
Mirrors the pattern in repl_env/rubrics.py and carla_env/server/rubrics.py.

Two components:
  SirenProcessRubric  — per-step signal (SLA drain, missing logs, idle penalty)
  SirenOutcomeRubric  — terminal-only signal (grader score on done=True)
  SirenRubric         — composite: process every step + outcome at terminal
"""
from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric

from env.grader import Grader


class SirenProcessRubric(Rubric):
    """Per-step reward signal based on action quality.

    Penalizes SLA drain, missing log creation, and idle actions.
    Returns 0.0 on terminal steps (outcome rubric handles those).
    """

    def forward(self, action: Any, observation: Any) -> float:
        if getattr(observation, "done", False):
            return 0.0

        reward = 0.0

        # Penalize SLA drain — read from observation
        sla = getattr(observation, "sla_credits", 100.0)
        if sla < 20.0:
            reward -= 0.05  # urgent SLA pressure signal

        # Penalize missing logs accumulation
        missing = getattr(observation, "missing_logs", 0)
        if missing > 0:
            reward -= missing * 0.01

        # Penalize idle when incidents are active
        active = getattr(observation, "active_sessions", 0)
        if action == 0 and active > 0:  # no_op with active incidents
            reward -= 0.05

        return reward

    def reset(self) -> None:
        pass


class SirenOutcomeRubric(Rubric):
    """Terminal-only reward: grader score on episode completion.

    Returns 0.0 on non-terminal steps.
    On done=True: returns grader score in (0, 1) range.
    """

    def __init__(self) -> None:
        super().__init__()
        self._grader = Grader()
        self._trajectory: list = []

    def record_step(self, step: dict) -> None:
        """Record a trajectory step for grading at episode end."""
        self._trajectory.append(step)

    def forward(self, action: Any, observation: Any) -> float:
        if not getattr(observation, "done", False):
            return 0.0

        task_id = getattr(observation, "task_id", "easy")
        scenario_id = getattr(observation, "scenario_id", "")
        if not self._trajectory:
            return 0.001

        return self._grader.grade(self._trajectory, task_id, scenario_id=scenario_id)

    def reset(self) -> None:
        self._trajectory = []


class SirenRubric(Rubric):
    """Composite rubric for the SIREN environment.

    Combines:
    - SirenProcessRubric: small per-step signals (SLA, logs, idle)
    - SirenOutcomeRubric: terminal grader score

    Usage:
        rubric = SirenRubric()
        rubric.reset()
        for action, obs in episode:
            reward = rubric(action, obs)
        # Terminal step reward = grader score
    """

    def __init__(self) -> None:
        super().__init__()
        self.process = SirenProcessRubric()
        self.outcome = SirenOutcomeRubric()

    def record_step(self, step: dict) -> None:
        """Forward trajectory recording to outcome rubric."""
        self.outcome.record_step(step)

    def forward(self, action: Any, observation: Any) -> float:
        done = getattr(observation, "done", False)
        if done:
            return self.outcome(action, observation)
        return self.process(action, observation)

    def reset(self) -> None:
        self.process.reset()
        self.outcome.reset()
