"""Grader tests for the SIREN RL environment.

Covers:
  - Property 12: Grader Output Range (Task 6.2)
  - Property 13: Hard Grader Trajectory Scan (Task 6.3)
  - Unit tests for Grader (Task 6.4)
  - Scenario-based reward signal tests (Task 6.5)
  - Chaos / adversarial robustness tests (Task 6.6)
"""
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from env.environment import SirenEnv
from env.grader import Grader


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def collect_trajectory(env, action_fn, seed=42):
    """Run an episode using action_fn(obs) -> action until done, return trajectory."""
    obs = env.reset(seed=seed)
    trajectory = []
    done = False
    while not done:
        action = action_fn(obs)
        obs, reward, done, info = env.step(action)
        trajectory.append({
            "action": action,
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        })
    return trajectory


# ---------------------------------------------------------------------------
# Hypothesis strategies for random trajectories
# ---------------------------------------------------------------------------

_REQUIRED_OBS_FIELDS = {
    "incidents": [],
    "sla_credits": 50.0,
    "proof_score": 0.5,
    "missing_logs": 0,
    "active_sessions": 0,
    "ledger_entries": 0,
    "ledger_hash": "a" * 64,
    "current_tick": 0,
    "max_ticks": 10,
    "audit_triggered": False,
    "task_id": "easy",
}

_obs_st = st.fixed_dictionaries({
    "incidents": st.lists(
        st.fixed_dictionaries({
            "id": st.text(min_size=1, max_size=10),
            "severity": st.integers(min_value=1, max_value=5),
            "type": st.text(min_size=1, max_size=20),
            "system": st.text(min_size=1, max_size=20),
            "time_remaining": st.integers(min_value=-10, max_value=20),
            "resolved": st.booleans(),
        }),
        max_size=5,
    ),
    "sla_credits": st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    "proof_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "missing_logs": st.integers(min_value=0, max_value=20),
    "active_sessions": st.integers(min_value=0, max_value=5),
    "ledger_entries": st.integers(min_value=0, max_value=20),
    "ledger_hash": st.just("a" * 64),
    "current_tick": st.integers(min_value=0, max_value=100),
    "max_ticks": st.integers(min_value=1, max_value=100),
    "audit_triggered": st.booleans(),
    "task_id": st.sampled_from(["easy", "medium", "hard"]),
})

_step_st = st.fixed_dictionaries({
    "action": st.integers(min_value=0, max_value=4),
    "observation": _obs_st,
    "reward": st.floats(allow_nan=False, allow_infinity=False),
    "done": st.booleans(),
    "info": st.fixed_dictionaries({}),
})


def _make_done_trajectory(steps):
    """Force the last step to have done=True so the grader doesn't short-circuit."""
    if not steps:
        return steps
    last = dict(steps[-1])
    last["done"] = True
    return steps[:-1] + [last]


# ===========================================================================
# Task 6.2 — Property 12: Grader Output Range
# ===========================================================================
# Feature: siren-env, Property 12: Grader Output Range

@given(
    steps=st.lists(_step_st, min_size=0, max_size=20),
    task_id=st.sampled_from(["easy", "medium", "hard"]),
)
@settings(max_examples=50)
def test_property_12_grader_output_range_random_trajectories(steps, task_id):
    """Property 12: Grader Output Range

    For any trajectory (including empty, incomplete, or adversarially constructed),
    the grader shall return a float in [0.0, 1.0].

    **Validates: Requirements 7.5, 7.6**
    """
    grader = Grader()
    score = grader.grade(steps, task_id)
    assert isinstance(score, float), f"Grader must return float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Grader score out of range: {score}"


@given(
    steps=st.lists(_step_st, min_size=1, max_size=20).map(_make_done_trajectory),
    task_id=st.sampled_from(["easy", "medium", "hard"]),
)
@settings(max_examples=50)
def test_property_12_grader_output_range_completed_trajectories(steps, task_id):
    """Property 12 (completed): Grader output in [0.0, 1.0] for completed trajectories.

    **Validates: Requirements 7.5, 7.6**
    """
    grader = Grader()
    score = grader.grade(steps, task_id)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Grader score out of range: {score}"


# ===========================================================================
# Task 6.3 — Property 13: Hard Grader Trajectory Scan
# ===========================================================================
# Feature: siren-env, Property 13: Hard Grader Trajectory Scan

def _make_hard_step(action, audit_triggered, resolved=False):
    """Build a minimal Hard-task step dict."""
    obs = {
        "incidents": [{"id": "INC-001", "severity": 5, "type": "ransomware",
                        "system": "db", "time_remaining": 0, "resolved": resolved}],
        "sla_credits": 50.0,
        "proof_score": 0.5,
        "missing_logs": 0,
        "active_sessions": 0 if resolved else 1,
        "ledger_entries": 0,
        "ledger_hash": "a" * 64,
        "current_tick": 1,
        "max_ticks": 6,
        "audit_triggered": audit_triggered,
        "task_id": "hard",
    }
    return {"action": action, "observation": obs, "reward": 0.0, "done": False, "info": {}}


@given(
    fast_patch_during_audit=st.integers(min_value=0, max_value=5),
    other_steps=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=50)
def test_property_13_hard_grader_trajectory_scan(fast_patch_during_audit, other_steps):
    """Property 13: Hard Grader Trajectory Scan

    The grader's count of fast_patch actions during audit shall equal the number
    of steps where action==1 AND observation["audit_triggered"]==True.

    When count > 0, the no_fast_patch_during_audit criterion is not met, so the
    grader must return < 1.0 (partial score, not full pass).

    **Validates: Requirements 7.7**
    """
    grader = Grader()

    # Build trajectory: fast_patch steps during audit + other non-fast_patch steps
    trajectory = []

    # Add fast_patch-during-audit steps
    for _ in range(fast_patch_during_audit):
        trajectory.append(_make_hard_step(action=1, audit_triggered=True))

    # Add other steps (verified_patch, not during audit or different action)
    for _ in range(other_steps):
        trajectory.append(_make_hard_step(action=2, audit_triggered=True))

    # Make last step done=True with severity-5 resolved so we can isolate the criterion
    if trajectory:
        last = dict(trajectory[-1])
        last_obs = dict(last["observation"])
        last_obs["resolved"] = True
        last_obs["incidents"] = [
            dict(inc, resolved=True) for inc in last_obs["incidents"]
        ]
        last_obs["proof_score"] = 0.9  # ensure proof_score criterion passes
        last["observation"] = last_obs
        last["done"] = True
        trajectory[-1] = last
    else:
        # Empty trajectory — grader returns 0.0
        score = grader.grade(trajectory, "hard")
        assert score == 0.0
        return

    score = grader.grade(trajectory, "hard")

    # Score must always be in [0.0, 1.0]
    assert 0.0 <= score <= 1.0

    # When fast_patch_during_audit > 0, the no_fast_patch_during_audit criterion fails
    # so the grader cannot return 1.0
    if fast_patch_during_audit > 0:
        assert score < 1.0, (
            f"Grader returned 1.0 despite {fast_patch_during_audit} fast_patch-during-audit steps"
        )


# ===========================================================================
# Task 6.4 — Unit tests for Grader
# ===========================================================================

def test_grader_easy_passing_trajectory():
    """Grader returns 1.0 for a passing Easy task trajectory (verified_patch only, seed=42)."""
    env = SirenEnv(task_id="easy")
    grader = Grader()

    trajectory = collect_trajectory(env, action_fn=lambda obs: 2, seed=42)

    assert trajectory[-1]["done"] is True
    score = grader.grade(trajectory, "easy")
    assert score == 1.0, f"Expected 1.0 for passing Easy trajectory, got {score}"


def test_grader_medium_passing_trajectory():
    """Grader returns 1.0 for a passing Medium task trajectory (verified_patch only, seed=42)."""
    # Medium task has 30% audit probability; seed=42 may or may not trigger audit.
    # verified_patch is always safe regardless of audit state.
    env = SirenEnv(task_id="medium")
    grader = Grader()

    trajectory = collect_trajectory(env, action_fn=lambda obs: 2, seed=42)

    assert trajectory[-1]["done"] is True
    score = grader.grade(trajectory, "medium")
    assert score == 1.0, f"Expected 1.0 for passing Medium trajectory, got {score}"


def test_grader_hard_passing_trajectory():
    """Hard task requires full resolution for perfect score.

    verified_patch-only on Hard task (max_ticks=8, 3 incidents × 3 ticks = 9 needed)
    runs over budget (tick 9 > max_ticks 8), so within_budget=False → score < 1.0.
    Any trajectory that leaves an incident unresolved OR runs over budget must score < 1.0.
    """
    env = SirenEnv(task_id="hard")
    grader = Grader()

    trajectory = collect_trajectory(env, action_fn=lambda obs: 2, seed=42)

    assert trajectory[-1]["done"] is True
    final_obs = trajectory[-1]["observation"]

    # The episode runs over budget (tick 9 > max_ticks 8), so strict grading applies
    over_budget = final_obs["current_tick"] > final_obs["max_ticks"]
    unresolved = [i for i in final_obs["incidents"] if not i["resolved"]]

    score = grader.grade(trajectory, "hard")

    # Hard task requires full resolution within budget for perfect score
    if over_budget or unresolved:
        assert score < 1.0, (
            f"Hard task must score < 1.0 when over budget or incidents unresolved "
            f"(over_budget={over_budget}, unresolved={len(unresolved)}), got {score}"
        )
    assert score >= 0.0, f"Score must be non-negative, got {score}"


def test_grader_empty_trajectory():
    """Grader returns 0.0 for an empty trajectory."""
    grader = Grader()
    score = grader.grade([], "easy")
    assert score == 0.0, f"Expected 0.0 for empty trajectory, got {score}"


def test_grader_incomplete_trajectory():
    """Grader returns 0.0 when last step has done=False."""
    env = SirenEnv(task_id="easy")
    grader = Grader()

    obs = env.reset(seed=42)
    # Take one step but don't finish the episode
    obs, reward, done, info = env.step(0)  # no_op — won't finish in one step
    trajectory = [{"action": 0, "observation": obs, "reward": reward, "done": False, "info": info}]

    score = grader.grade(trajectory, "easy")
    assert score == 0.0, f"Expected 0.0 for incomplete trajectory, got {score}"


# ===========================================================================
# Task 6.5 — Scenario-based reward signal tests
# ===========================================================================

def test_reward_verified_patch_beats_fast_patch():
    """verified_patch episode accumulates strictly higher total reward than fast_patch episode."""
    # Fast-patch episode
    env_fp = SirenEnv(task_id="easy")
    traj_fp = collect_trajectory(env_fp, action_fn=lambda obs: 1, seed=42)
    total_reward_fp = sum(step["reward"] for step in traj_fp)

    # Verified-patch episode
    env_vp = SirenEnv(task_id="easy")
    traj_vp = collect_trajectory(env_vp, action_fn=lambda obs: 2, seed=42)
    total_reward_vp = sum(step["reward"] for step in traj_vp)

    assert total_reward_vp > total_reward_fp, (
        f"verified_patch total reward ({total_reward_vp:.4f}) should be strictly higher "
        f"than fast_patch total reward ({total_reward_fp:.4f})"
    )


def test_reward_no_op_idle_penalty():
    """no_op episode has lower total reward than verified_patch episode on Medium task."""
    # Medium task has a severity-4 incident, so no_op incurs idle penalty each step.
    env_noop = SirenEnv(task_id="medium")
    traj_noop = collect_trajectory(env_noop, action_fn=lambda obs: 0, seed=42)
    total_reward_noop = sum(step["reward"] for step in traj_noop)

    env_vp = SirenEnv(task_id="medium")
    traj_vp = collect_trajectory(env_vp, action_fn=lambda obs: 2, seed=42)
    total_reward_vp = sum(step["reward"] for step in traj_vp)

    assert total_reward_noop < total_reward_vp, (
        f"no_op total reward ({total_reward_noop:.4f}) should be lower than "
        f"verified_patch total reward ({total_reward_vp:.4f})"
    )


# ===========================================================================
# Task 6.6 — Chaos / adversarial robustness tests
# ===========================================================================

_REQUIRED_OBS_KEYS = {
    "incidents", "sla_credits", "proof_score", "missing_logs",
    "active_sessions", "ledger_entries", "ledger_hash",
    "current_tick", "max_ticks", "audit_triggered", "task_id",
}


def _assert_valid_obs(obs):
    """Assert observation has all required fields and numeric bounds."""
    assert isinstance(obs, dict)
    for key in _REQUIRED_OBS_KEYS:
        assert key in obs, f"Missing observation field: {key}"
    assert 0.0 <= obs["sla_credits"] <= 100.0, f"sla_credits out of bounds: {obs['sla_credits']}"
    assert 0.0 <= obs["proof_score"] <= 1.0, f"proof_score out of bounds: {obs['proof_score']}"


def test_chaos_no_op_spam_all_tasks():
    """Spam no_op for the full tick budget on all tasks; no exception, valid final observation."""
    for task_id in ["easy", "medium", "hard"]:
        env = SirenEnv(task_id=task_id)
        obs = env.reset(seed=42)
        done = False
        final_obs = obs

        while not done:
            obs, reward, done, info = env.step(0)  # no_op
            final_obs = obs

        _assert_valid_obs(final_obs)
        assert done is True


def test_chaos_random_actions_bounds():
    """Random actions across multiple seeds keep sla_credits/proof_score in bounds and done eventually True."""
    rng = random.Random(42)

    for seed in [42, 123, 999, 7, 2024]:
        for task_id in ["easy", "medium", "hard"]:
            env = SirenEnv(task_id=task_id)
            env.reset(seed=seed)
            done = False
            steps = 0

            while not done and steps < 1000:
                action = rng.randint(0, 4)
                obs, reward, done, info = env.step(action)
                assert 0.0 <= obs["sla_credits"] <= 100.0, (
                    f"sla_credits out of bounds at step {steps}: {obs['sla_credits']}"
                )
                assert 0.0 <= obs["proof_score"] <= 1.0, (
                    f"proof_score out of bounds at step {steps}: {obs['proof_score']}"
                )
                steps += 1

            assert done is True, (
                f"Episode did not terminate within 1000 steps for task={task_id}, seed={seed}"
            )


def test_chaos_worst_case_hard_grader_zero():
    """Always fast_patch on Hard task → grader returns < 1.0 (no_fast_patch_during_audit criterion fails).

    The Hard task has audit active from tick 0, so every fast_patch violates the
    no_fast_patch_during_audit criterion. The grader cannot return 1.0 in this case.
    It may return a partial score for other criteria met (e.g. severity-5 resolved),
    but the no_fast_patch_during_audit criterion failure prevents a full pass.
    """
    env = SirenEnv(task_id="hard")
    grader = Grader()

    trajectory = collect_trajectory(env, action_fn=lambda obs: 1, seed=42)

    assert trajectory[-1]["done"] is True

    # Verify fast_patch-during-audit steps exist in the trajectory
    fast_patch_during_audit = sum(
        1 for step in trajectory
        if step["action"] == 1 and step["observation"]["audit_triggered"]
    )
    assert fast_patch_during_audit > 0, "Expected fast_patch steps during audit"

    score = grader.grade(trajectory, "hard")
    # The no_fast_patch_during_audit criterion is violated, so score must be < 1.0
    assert score < 1.0, (
        f"Expected grader score < 1.0 for always-fast_patch Hard trajectory, got {score}"
    )


def test_chaos_tick_budget_exhaustion():
    """no_op until done=True on Easy task; done is set and final observation is valid."""
    env = SirenEnv(task_id="easy")
    obs = env.reset(seed=42)
    done = False
    final_obs = obs

    while not done:
        obs, reward, done, info = env.step(0)  # no_op
        final_obs = obs

    assert done is True
    _assert_valid_obs(final_obs)
    # Tick budget should be exhausted (or all incidents resolved)
    assert (
        final_obs["current_tick"] >= final_obs["max_ticks"]
        or all(inc["resolved"] for inc in final_obs["incidents"])
    )
