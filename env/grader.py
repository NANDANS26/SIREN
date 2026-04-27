"""Episode grader for the SIREN RL environment."""
from __future__ import annotations

from env.scenarios import SCENARIOS_BY_ID

# Action integer → name mapping for sequence comparison
_ACTION_NAMES = {
    0: "no_op",
    1: "fast_patch",
    2: "verified_patch",
    3: "isolate_system",
    4: "escalate_human",
}


class Grader:
    """Evaluates a completed episode trajectory and returns a score in (0, 1).

    Scoring has two components:
    1. Outcome score: resolution, proof, SLA health (existing logic)
    2. Sequence score: overlap between agent actions and correct_action_sequence
       from the scenario library — this is the REAL ground truth comparison,
       mirroring how reasoning_gym scores answers against correct_answer.
    """

    def grade(self, trajectory: list, task_id: str, scenario_id: str = "") -> float:
        """Grade a completed episode trajectory.

        Args:
            trajectory: List of step dicts with action, observation, reward, done, info.
            task_id: One of "easy", "medium", or "hard".
            scenario_id: Optional scenario ID for sequence-based grading.

        Returns:
            A float strictly in (0, 1).
        """
        if not trajectory:
            return 0.001
        if not trajectory[-1]["done"]:
            return 0.001

        final_obs = trajectory[-1]["observation"]

        if task_id == "easy":
            outcome = self._grade_easy(final_obs)
        elif task_id == "medium":
            outcome = self._grade_medium(final_obs)
        elif task_id == "hard":
            outcome = self._grade_hard(trajectory, final_obs)
        else:
            outcome = 0.0

        # Sequence score: compare agent actions against expert correct_action_sequence
        # This is the ground truth comparison — not circular reward
        sequence = self._grade_sequence(trajectory, scenario_id)

        # Blend: 70% outcome + 30% sequence alignment
        # When no scenario_id, sequence=0.5 (neutral) so outcome dominates
        score = outcome * 0.70 + sequence * 0.30

        return max(0.001, min(0.999, score))

    def _grade_sequence(self, trajectory: list, scenario_id: str) -> float:
        """Score agent action sequence against the expert correct_action_sequence.

        Computes the longest common subsequence overlap ratio.
        Returns 0.5 when no scenario_id is provided (neutral — doesn't penalize).
        """
        if not scenario_id or scenario_id not in SCENARIOS_BY_ID:
            return 0.5  # neutral when no ground truth available

        scenario = SCENARIOS_BY_ID[scenario_id]
        correct = scenario.correct_action_sequence  # e.g. ["isolate_system", "verified_patch"]

        # Extract agent action names (skip no_op for sequence comparison)
        agent_actions = [
            _ACTION_NAMES.get(step["action"], "unknown")
            for step in trajectory
            if step["action"] != 0  # exclude no_op
        ]

        if not correct:
            return 0.5
        if not agent_actions:
            return 0.001

        # Longest common subsequence length
        lcs = self._lcs_length(agent_actions, correct)
        # Score = LCS / max(len(correct), len(agent)) — penalizes both missing and extra actions
        score = lcs / max(len(correct), len(agent_actions))
        return score

    @staticmethod
    def _lcs_length(a: list, b: list) -> int:
        """Compute longest common subsequence length."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    # ------------------------------------------------------------------
    # Easy task
    # ------------------------------------------------------------------

    def _grade_easy(self, final_obs: dict) -> float:
        incidents = final_obs.get("incidents", [])
        incident_resolved = any(inc.get("resolved", False) for inc in incidents)
        missing_logs_zero = final_obs.get("missing_logs", 1) == 0
        min_episode_score = final_obs.get("proof_score", 0.0) >= 0.6
        criteria = [incident_resolved, missing_logs_zero, min_episode_score]
        criteria_met = sum(criteria)
        score = 1.0 if all(criteria) else criteria_met / 3
        unresolved_count = sum(1 for inc in incidents if not inc.get("resolved", False))
        score -= 0.3 * unresolved_count
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Medium task
    # ------------------------------------------------------------------

    def _grade_medium(self, final_obs: dict) -> float:
        incidents = final_obs.get("incidents", [])
        highest = max(incidents, key=lambda i: i.get("severity", 0)) if incidents else None
        high_severity_resolved = highest is not None and highest.get("resolved", False)
        min_proof_score = final_obs.get("proof_score", 0.0) >= 0.5
        min_sla_credits = final_obs.get("sla_credits", 0.0) >= 20.0
        criteria = [high_severity_resolved, min_proof_score, min_sla_credits]
        criteria_met = sum(criteria)
        score = 1.0 if all(criteria) else criteria_met / 3
        unresolved_count = sum(1 for inc in incidents if not inc.get("resolved", False))
        score -= 0.3 * unresolved_count
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Hard task
    # ------------------------------------------------------------------

    def _grade_hard(self, trajectory: list, final_obs: dict) -> float:
        incidents = final_obs.get("incidents", [])
        total = len(incidents) if incidents else 1
        resolved_count = sum(1 for inc in incidents if inc.get("resolved", False))
        unresolved_count = total - resolved_count

        resolution_score = (resolved_count / total) * 0.35
        severity5_resolved = any(
            inc.get("severity") == 5 and inc.get("resolved", False) for inc in incidents
        )
        severity5_score = 0.25 if severity5_resolved else 0.0
        proof = final_obs.get("proof_score", 0.0)
        proof_score = proof * 0.15
        fast_patch_count = sum(
            1 for step in trajectory
            if step["action"] == 1 and step["observation"].get("audit_triggered", False)
        )
        audit_score = 0.10 if fast_patch_count == 0 else max(0.0, 0.10 - fast_patch_count * 0.03)
        sla = final_obs.get("sla_credits", 0.0)
        sla_score = (sla / 100.0) * 0.15
        score = resolution_score + severity5_score + proof_score + audit_score + sla_score
        isolate_count = sum(1 for step in trajectory if step["action"] == 3)
        verified_count = sum(1 for step in trajectory if step["action"] == 2)
        if isolate_count > 0 and verified_count > 0 and total > 1:
            score += 0.05
        score -= (unresolved_count / total) * 0.15
        return max(0.0, min(1.0, score))
