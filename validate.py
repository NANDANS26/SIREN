#!/usr/bin/env python3
"""SIREN validation script — checks environment correctness."""
import sys
from env.environment import SirenEnv
from env.grader import Grader

CHECKS = []

REQUIRED_FIELDS = {
    "incidents": list,
    "sla_credits": float,
    "proof_score": float,
    "missing_logs": int,
    "active_sessions": int,
    "ledger_entries": int,
    "ledger_hash": str,
    "current_tick": int,
    "max_ticks": int,
    "audit_triggered": bool,
    "task_id": str,
}


def check(name):
    """Decorator to register a check function."""
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("reset_returns_valid_schema")
def check_reset_schema():
    env = SirenEnv(task_id="easy")
    obs = env.reset(seed=42)
    assert isinstance(obs, dict), "reset() must return a dict"
    for field, expected_type in REQUIRED_FIELDS.items():
        assert field in obs, f"Missing required field: '{field}'"
        assert isinstance(obs[field], expected_type), (
            f"Field '{field}' expected {expected_type.__name__}, "
            f"got {type(obs[field]).__name__}"
        )
    assert len(obs["ledger_hash"]) == 64, (
        f"ledger_hash must be 64 chars, got {len(obs['ledger_hash'])}"
    )


@check("step_returns_correct_types")
def check_step_types():
    for action in range(5):
        env = SirenEnv(task_id="easy")
        env.reset(seed=42)
        result = env.step(action)
        assert isinstance(result, tuple) and len(result) == 4, (
            f"step({action}) must return a 4-tuple, got {type(result)}"
        )
        obs, reward, done, info = result
        assert isinstance(obs, dict), (
            f"step({action}): obs must be dict, got {type(obs).__name__}"
        )
        assert isinstance(reward, float), (
            f"step({action}): reward must be float, got {type(reward).__name__}"
        )
        assert isinstance(done, bool), (
            f"step({action}): done must be bool, got {type(done).__name__}"
        )
        assert isinstance(info, dict), (
            f"step({action}): info must be dict, got {type(info).__name__}"
        )


@check("ledger_hash_changes_after_verified_patch")
def check_ledger_hash():
    # verified_patch (action 2) should change the ledger hash
    env = SirenEnv(task_id="easy")
    env.reset(seed=42)
    hash_before = env.state()["ledger_hash"]
    env.step(2)  # verified_patch
    hash_after = env.state()["ledger_hash"]
    assert hash_before != hash_after, (
        "ledger_hash must change after verified_patch action"
    )

    # fast_patch (action 1) should NOT change the ledger hash
    env2 = SirenEnv(task_id="easy")
    env2.reset(seed=42)
    hash_before2 = env2.state()["ledger_hash"]
    env2.step(1)  # fast_patch
    hash_after2 = env2.state()["ledger_hash"]
    assert hash_before2 == hash_after2, (
        "ledger_hash must NOT change after fast_patch action"
    )


@check("grader_returns_valid_range")
def check_grader_range():
    grader = Grader()
    for task_id in ("easy", "medium", "hard"):
        env = SirenEnv(task_id=task_id)
        env.reset(seed=42)
        trajectory = []
        done = False
        while not done:
            obs, reward, done, info = env.step(2)  # always verified_patch
            trajectory.append({
                "action": 2,
                "observation": obs,
                "reward": reward,
                "done": done,
                "info": info,
            })
        score = grader.grade(trajectory, task_id)
        assert isinstance(score, float), (
            f"Grader.grade() must return float for task '{task_id}', "
            f"got {type(score).__name__}"
        )
        assert 0.0 <= score <= 1.0, (
            f"Grader score out of range for task '{task_id}': {score}"
        )


@check("determinism_with_same_seed")
def check_determinism():
    def run_episode(task_id, seed, n_steps=5):
        env = SirenEnv(task_id=task_id)
        obs = env.reset(seed=seed)
        observations = [obs]
        for _ in range(n_steps - 1):
            obs, _, done, _ = env.step(2)
            observations.append(obs)
            if done:
                break
        return observations

    for task_id in ("easy",):
        obs_run1 = run_episode(task_id, seed=42, n_steps=5)
        obs_run2 = run_episode(task_id, seed=42, n_steps=5)
        assert len(obs_run1) == len(obs_run2), (
            f"Determinism check: runs produced different numbers of observations "
            f"({len(obs_run1)} vs {len(obs_run2)})"
        )
        for i, (o1, o2) in enumerate(zip(obs_run1, obs_run2)):
            assert o1 == o2, (
                f"Determinism check: observation at step {i} differs between runs "
                f"with seed=42.\nRun 1: {o1}\nRun 2: {o2}"
            )


def main():
    for name, fn in CHECKS:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL [{name}]: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"FAIL [{name}]: Unexpected error: {e}")
            sys.exit(1)
    print("All checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
