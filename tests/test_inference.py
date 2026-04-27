# Feature: siren-env, Property 17: Baseline Policy Never Uses fast_patch During Audit
"""Property-based tests for inference.py baseline policy."""
from hypothesis import given, settings
from hypothesis import strategies as st

from env.environment import SirenEnv
from inference import baseline_policy


@settings(max_examples=50)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    task_id=st.sampled_from(["easy", "medium", "hard"]),
)
def test_baseline_policy_never_fast_patch_during_audit(seed: int, task_id: str) -> None:
    """Property 17: Baseline Policy Never Uses fast_patch During Audit.

    For any episode run by the baseline policy, no step shall have
    action == 1 (fast_patch) when the observation's audit_triggered is True.

    Validates: Requirements 16.2
    """
    env = SirenEnv(task_id=task_id)
    obs = env.reset(seed=seed)
    done = False

    while not done:
        action = baseline_policy(obs)

        # The core property: fast_patch must never be chosen during an audit
        if obs["audit_triggered"]:
            assert action != 1, (
                f"baseline_policy chose fast_patch (action=1) while audit_triggered=True "
                f"on task={task_id}, seed={seed}, obs={obs}"
            )

        obs, _reward, done, _info = env.step(action)
