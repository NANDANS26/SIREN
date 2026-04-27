"""Behavior quality tests — validates the environment is non-trivial.

Ensures:
- Baseline policy does not trivially achieve perfect scores
- Mixed strategies produce different results than pure verified_patch
- Unresolved incident penalty is applied correctly
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import SirenEnv
from env.grader import Grader
from inference import baseline_policy


def _collect(env, action_fn, seed=42):
    obs = env.reset(seed=seed)
    trajectory = []
    done = False
    while not done:
        action = action_fn(obs)
        obs, reward, done, info = env.step(action)
        trajectory.append({"action": action, "observation": obs,
                            "reward": reward, "done": done, "info": info})
    return trajectory


# ---------------------------------------------------------------------------
# Test 1: baseline is NOT perfect on Hard task
# ---------------------------------------------------------------------------

def test_baseline_not_perfect_on_hard():
    """Baseline policy on Hard task should not achieve a perfect grader score.

    With max_ticks=8 and 3 incidents requiring 3 ticks each (9 total needed),
    the baseline runs over budget (tick 9 > max_ticks 8), so within_budget=False
    and the strict grader prevents a perfect score.
    """
    env = SirenEnv(task_id="hard")
    grader = Grader()

    trajectory = _collect(env, baseline_policy, seed=42)

    assert trajectory[-1]["done"] is True
    final_obs = trajectory[-1]["observation"]
    score = grader.grade(trajectory, "hard")

    # Hard task requires full resolution within budget for perfect score
    over_budget = final_obs["current_tick"] > final_obs["max_ticks"]
    unresolved = [i for i in final_obs["incidents"] if not i["resolved"]]
    assert over_budget or unresolved, (
        "Expected baseline to either run over budget or leave incidents unresolved"
    )
    assert score < 1.0, (
        f"Baseline on Hard task should not score 1.0 (environment is non-trivial), "
        f"got {score:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: mixed strategy produces a different score than pure verified_patch
# ---------------------------------------------------------------------------

def test_mixed_strategy_differs_from_pure_verified_patch():
    """isolate_system + verified_patch mix should produce a different score than
    always verified_patch on the Hard task (fixed seed=42).

    The strategic diversity bonus (+0.05) in _grade_hard() rewards use of
    isolate_system alongside verified_patch, making scores differ.
    """
    grader = Grader()

    # Strategy A: always verified_patch (no isolate_system used)
    env_a = SirenEnv(task_id="hard")
    traj_a = _collect(env_a, lambda obs: 2, seed=42)
    score_a = grader.grade(traj_a, "hard")

    # Strategy B: isolate first (tick 0), then verified_patch
    # Hard task has audit_from_tick_zero=True, so audit is always active.
    # isolate_system is still valid during audit (no penalty for it).
    def mixed_policy(obs):
        unresolved = [i for i in obs["incidents"] if not i["resolved"]]
        if not unresolved:
            return 0  # no_op
        # Use isolate on the very first step to reduce active sessions
        if obs["current_tick"] == 0 and obs["active_sessions"] >= 2:
            return 3  # isolate_system
        return 2  # verified_patch

    env_b = SirenEnv(task_id="hard")
    traj_b = _collect(env_b, mixed_policy, seed=42)
    score_b = grader.grade(traj_b, "hard")

    assert score_a != score_b, (
        f"Mixed strategy (isolate+vp, score={score_b:.4f}) and pure verified_patch "
        f"(score={score_a:.4f}) should produce different grader scores"
    )


# ---------------------------------------------------------------------------
# Test 3: unresolved penalty reduces score below 1.0
# ---------------------------------------------------------------------------

def test_unresolved_penalty_reduces_score():
    """When one incident remains unresolved, grader score must be < 1.0.

    Uses Easy task with no_op only so the incident stays unresolved,
    then verifies the unresolved penalty brings the score below 1.0.
    """
    grader = Grader()

    # Easy task (1 incident) — no_op until tick budget exhausted
    env = SirenEnv(task_id="easy")
    traj = _collect(env, lambda obs: 0, seed=42)

    assert traj[-1]["done"] is True
    final_obs = traj[-1]["observation"]

    unresolved = [i for i in final_obs["incidents"] if not i["resolved"]]
    assert len(unresolved) > 0, "Expected at least one unresolved incident after no_op episode"

    score = grader.grade(traj, "easy")
    assert score < 1.0, (
        f"Score must be < 1.0 when incidents remain unresolved, got {score:.4f}"
    )
