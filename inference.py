#!/usr/bin/env python3
"""SIREN inference script — LLM-driven agent with deterministic fallback.

Uses an OpenAI-compatible client (OpenEnv submission format).
Environment variables (injected by the validator):
  API_BASE_URL  — base URL of the LiteLLM proxy endpoint
  API_KEY       — proxy API key (primary)
  MODEL_NAME    — model identifier to use for chat completions
  HF_TOKEN      — Hugging Face token (fallback key if API_KEY not set)

All LLM calls go through the provided API_BASE_URL proxy.
Falls back to the deterministic baseline_policy only on API errors.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict

from openai import OpenAI

from env.environment import SirenEnv
from siren_environment import SirenEnvironment
from models import SirenAction

# OpenEnv SDK client — now in standalone client.py (follows reference repo pattern)
try:
    from client import SirenEnvClient
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Environment variables — per submission spec:
#   API_BASE_URL and MODEL_NAME have placeholder defaults
#   API_KEY / HF_TOKEN have NO default (must be provided by validator)
#   LOCAL_IMAGE_NAME is optional (used when from_docker_image() is called)
# ---------------------------------------------------------------------------

API_BASE_URL     = os.getenv("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME       = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN         = os.getenv("HF_TOKEN")          # no default
API_KEY          = os.getenv("API_KEY") or HF_TOKEN  # validator injects API_KEY
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Internal aliases
_API_BASE_URL = API_BASE_URL
_MODEL_NAME   = MODEL_NAME
_HF_TOKEN     = API_KEY   # use whichever key was provided

_llm_client = None


def _get_client():
    """Return a cached OpenAI client configured via API_BASE_URL and API_KEY.

    Returns None only if both API_BASE_URL and API_KEY are genuinely absent,
    so the validator's injected credentials are always used when present.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    base_url = _API_BASE_URL
    api_key  = _HF_TOKEN

    # Only skip if no real base_url or key was injected
    if not api_key:
        return None
    if not base_url or base_url.startswith("<"):
        return None

    try:
        _llm_client = OpenAI(base_url=base_url, api_key=api_key)
        return _llm_client
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Deterministic fallback policy (kept intact per requirements)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Score-based baseline policy — dynamic, normalized, lookahead-aware
# ---------------------------------------------------------------------------

import math as _math


def _tanh(x: float) -> float:
    """Clamp a raw score to [-1, 1] via tanh for normalization."""
    return _math.tanh(x)


def _state_value(unresolved: list, sla: float, proof: float,
                 ticks_left: int, max_ticks: int, audit: bool) -> float:
    """Holistic value of a state — used for lookahead scoring.

    Higher = better state. All inputs are normalized internally.
    unresolved: list of incident dicts that are not yet resolved (may be isolated).
    """
    if max_ticks == 0:
        return 0.0
    # Fraction of incidents still unresolved (lower = better)
    unresolved_ratio = len(unresolved) / max(1, max_ticks)
    # Normalized SLA health [0,1]
    sla_health = sla / 100.0
    # Proof health [0,1]
    proof_health = proof
    # Remaining budget fraction [0,1]
    budget_health = ticks_left / max_ticks
    # Severity pressure: isolated incidents still need patching — count them too
    sev_pressure = sum(i["severity"] / 5.0 for i in unresolved)

    value = (
        proof_health * 1.5
        + sla_health * 0.8
        + budget_health * 0.5
        - sev_pressure * 1.2
        - unresolved_ratio * 2.0
    )
    # Audit amplifies proof importance
    if audit:
        value += proof_health * 0.5
    return value


def _score_action(action: int, obs: dict) -> float:
    """Score an action using dynamic thresholds derived entirely from obs.

    All intermediate values are normalized; final score is tanh-compressed
    to [-1, 1] before returning so no single factor dominates.
    """
    unresolved = [i for i in obs["incidents"] if not i["resolved"]]
    audit = obs["audit_triggered"]
    sla = obs["sla_credits"]
    proof = obs["proof_score"]
    current_tick = obs["current_tick"]
    max_ticks = obs["max_ticks"]
    ticks_left = max(0, max_ticks - current_tick)

    # ── Dynamic thresholds — all derived from obs, no magic numbers ──────────
    tick_urgency    = current_tick / max_ticks if max_ticks > 0 else 0.0
    budget_pressure = 1.0 - (ticks_left / max_ticks) if max_ticks > 0 else 1.0
    sla_pressure    = 1.0 - (sla / 100.0)
    proof_deficit   = 1.0 - proof
    # Audit influence: continuous multiplier [0, 1] — stronger when proof is degraded
    audit_weight    = (0.5 + 0.5 * proof_deficit) if audit else 0.0

    # ── No incidents left ─────────────────────────────────────────────────────
    if not unresolved:
        return 1.0 if action == 0 else -1.0

    top = min(unresolved, key=lambda i: (-i["severity"], i["time_remaining"], i["id"]))
    top_sev_norm    = top["severity"] / 5.0
    top_needs_iso   = top.get("requires_isolation", False) and not top.get("isolated", False)

    # ── Terminal mode: budget_pressure > 0.75 → resolve at all costs ─────────
    terminal = budget_pressure > 0.75

    # ── Simulate post-action state for lookahead ──────────────────────────────
    # We estimate the resulting state without calling env.step.

    def _lookahead(resolved_gain: int, proof_delta: float,
                   sla_delta: float, ticks_used: int) -> float:
        new_unresolved = unresolved[resolved_gain:]  # optimistic: top resolved first
        new_sla   = max(0.0, min(100.0, sla - sla_delta))
        new_proof = max(0.0, min(1.0, proof + proof_delta))
        new_ticks = max(0, ticks_left - ticks_used)
        return _state_value(new_unresolved, new_sla, new_proof,
                            new_ticks, max_ticks, audit)

    raw = 0.0

    # ── action 0: no_op ───────────────────────────────────────────────────────
    if action == 0:
        # Idle is only acceptable when no incidents remain
        idle_penalty = sum(i["severity"] / 5.0 for i in unresolved) * 1.5
        raw = _lookahead(0, 0.0, 0.0, 1) - idle_penalty
        if terminal:
            raw -= 2.0  # never idle in terminal mode

    # ── action 1: fast_patch ──────────────────────────────────────────────────
    elif action == 1:
        if audit:
            return -1.0  # hard block — never fast_patch during audit
        if top_needs_iso:
            return -1.0  # hard block — isolation dependency
        # proof cost: missing_logs +1 → proof drops by 0.15
        proof_delta = -0.15
        # SLA drain: 1 tick × active_sessions × base_rate (estimate 1.0)
        sla_drain = obs.get("active_sessions", 1) * 1.0
        resolution_bonus = top_sev_norm * 2.0
        raw = _lookahead(1, proof_delta, sla_drain, 1) + resolution_bonus
        # Proof penalty scaled by audit weight
        raw -= proof_deficit * audit_weight * 1.5

    # ── action 2: verified_patch ──────────────────────────────────────────────
    elif action == 2:
        if top_needs_iso:
            return -1.0  # hard block — isolation dependency
        # proof gain: ledger_entries +1 → +0.05; audit bonus +0.05 extra
        proof_delta = 0.10 if audit else 0.05
        sla_drain = obs.get("active_sessions", 1) * 3.0  # 3 ticks
        resolution_bonus = top_sev_norm * 2.0
        # Proof quality bonus — scales with audit weight
        proof_bonus = (0.5 + audit_weight) * proof_delta * 5.0
        raw = _lookahead(1, proof_delta, sla_drain, 3) + resolution_bonus + proof_bonus
        # Budget penalty: 3 ticks is expensive when budget is tight
        remaining_after = len(unresolved) - 1
        ticks_after = ticks_left - 3
        if remaining_after > 0 and ticks_after > 0:
            # Can we fit remaining incidents? Each needs at least 1 tick
            feasibility = ticks_after / (remaining_after * 1.0)
            raw -= _tanh(1.0 - feasibility) * 0.5
        elif remaining_after > 0 and ticks_after <= 0:
            raw -= 0.3  # budget exhausted after this action

    # ── action 3: isolate_system ──────────────────────────────────────────────
    elif action == 3:
        sla_cost = 15.0 + obs.get("active_sessions", 1) * 1.0  # isolation + 1 tick drain
        if top_needs_iso:
            # Unblocks the top incident — high value proportional to its severity
            unblock_bonus = top_sev_norm * 2.5
            # After isolation, top is patchable — estimate proof gain from future patch
            future_proof = 0.10 if audit else 0.05
            raw = _lookahead(0, future_proof * 0.5, sla_cost, 1) + unblock_bonus
        else:
            # Reduces active_sessions → slows SLA drain; only useful under SLA pressure
            drain_relief = sla_pressure * 0.8
            raw = _lookahead(0, 0.0, sla_cost, 1) + drain_relief
            # Penalise if SLA is healthy — wastes credits
            raw -= (1.0 - sla_pressure) * 0.5

    # ── action 4: escalate_human ──────────────────────────────────────────────
    elif action == 4:
        sla_drain = obs.get("active_sessions", 1) * 1.0
        # Adds approval ledger entry → proof +0.02
        proof_delta = 0.02
        proof_bonus = audit_weight * proof_deficit * 0.5
        raw = _lookahead(0, proof_delta, sla_drain, 1) + proof_bonus
        # Penalise: doesn't resolve anything, weak in terminal mode
        raw -= top_sev_norm * 0.8
        if terminal:
            raw -= 1.0

    # ── Normalize to [-1, 1] via tanh ─────────────────────────────────────────
    return _tanh(raw)


import copy as _copy
import random as _random


def _simulate_obs(action: int, obs: dict, noise_rng: "_random.Random | None" = None) -> dict:
    """Approximate next observation after *action* without calling env.step().

    noise_rng: if provided, injects ±10% uncertainty into SLA drain and
               time_remaining decay for uncertainty-aware planning.
    """
    unresolved = [i for i in obs["incidents"] if not i["resolved"]]
    top = (
        min(unresolved, key=lambda i: (-i["severity"], i["time_remaining"], i["id"]))
        if unresolved else None
    )

    incidents = _copy.deepcopy(obs["incidents"])
    sla   = obs["sla_credits"]
    proof = obs["proof_score"]
    tick  = obs["current_tick"]
    audit = obs["audit_triggered"]
    ticks_used = 1

    def _jitter(val: float) -> float:
        """Apply ±10% noise when rng is provided."""
        if noise_rng is None:
            return val
        return val * (1.0 + noise_rng.uniform(-0.10, 0.10))

    if action == 0:
        pass

    elif action == 1:  # fast_patch
        if top and not (top.get("requires_isolation") and not top.get("isolated")):
            for inc in incidents:
                if inc["id"] == top["id"]:
                    inc["resolved"] = True
                    break
            proof = max(0.0, proof - 0.15)

    elif action == 2:  # verified_patch
        if top and not (top.get("requires_isolation") and not top.get("isolated")):
            for inc in incidents:
                if inc["id"] == top["id"]:
                    inc["resolved"] = True
                    break
            proof = min(1.0, proof + (0.08 if audit else 0.04))
        ticks_used = 3

    elif action == 3:  # isolate_system
        if top:
            for inc in incidents:
                if inc["id"] == top["id"]:
                    inc["isolated"] = True
                    break
        sla = max(0.0, sla - 15.0)

    elif action == 4:  # escalate_human
        proof = min(1.0, proof + 0.02)

    # SLA drain with optional noise
    active = sum(1 for i in incidents if not i["resolved"] and not i.get("isolated"))
    sla = max(0.0, sla - _jitter(active * ticks_used))

    # time_remaining decay — no noise here, integer ticks must stay stable
    for inc in incidents:
        if not inc["resolved"] and not inc.get("isolated"):
            inc["time_remaining"] -= ticks_used

    return {
        "incidents":       incidents,
        "sla_credits":     max(0.0, min(100.0, sla)),
        "proof_score":     max(0.0, min(1.0, proof)),
        "missing_logs":    obs["missing_logs"],
        "active_sessions": active,
        "ledger_entries":  obs["ledger_entries"],
        "ledger_hash":     obs["ledger_hash"],
        "current_tick":    tick + ticks_used,
        "max_ticks":       obs["max_ticks"],
        "audit_triggered": audit,
        "task_id":         obs["task_id"],
    }


def _propagation_analysis(obs: dict) -> dict:
    """Analyse the current propagation state from systems_status.

    Returns a dict with:
      root_sources:    systems actively propagating to others (highest risk)
      downstream:      systems receiving anomaly from others
      chain_depth:     max depth of active propagation chain
      cascade_risk:    scalar [0,1] — how dangerous the current cascade is
    """
    status = obs.get("systems_status", {})
    if not status:
        return {"root_sources": [], "downstream": [], "chain_depth": 0, "cascade_risk": 0.0}

    root_sources = [
        name for name, s in status.items()
        if s.get("propagating_to") and not s.get("isolated")
    ]
    downstream = [
        name for name, s in status.items()
        if s.get("received_from")
    ]

    # Chain depth: how many hops are active (db→auth = 1, db→auth→web = 2)
    chain_depth = 0
    for src in root_sources:
        depth = 1
        visited = {src}
        frontier = list(status.get(src, {}).get("propagating_to", []))
        while frontier:
            nxt = []
            for dep in frontier:
                if dep not in visited and status.get(dep, {}).get("propagating_to"):
                    nxt.extend(status[dep]["propagating_to"])
                    depth += 1
                    visited.add(dep)
            frontier = nxt
        chain_depth = max(chain_depth, depth)

    # Cascade risk: weighted by anomaly scores of propagating systems
    cascade_risk = 0.0
    for name in root_sources:
        anomaly = status.get(name, {}).get("anomaly_score", 0.0)
        n_targets = len(status.get(name, {}).get("propagating_to", []))
        cascade_risk += anomaly * n_targets * 0.5
    cascade_risk = min(1.0, cascade_risk)

    return {
        "root_sources": root_sources,
        "downstream": downstream,
        "chain_depth": chain_depth,
        "cascade_risk": cascade_risk,
    }


def _incident_propagation_score(inc: dict, obs: dict, prop: dict) -> float:
    """Score a single incident with propagation awareness.

    Base = severity + urgency.
    Bonus if the incident's system is a root source (propagating to others).
    Bonus if the incident's system is downstream (receiving from others).
    Penalty if the incident is downstream AND its source is still active
    (deprioritize symptoms, prioritize root cause).
    """
    status = obs.get("systems_status", {})

    # Map incident system display name to short key
    sys_display = inc.get("system", "")
    if "db" in sys_display:
        sys_key = "db"
    elif "web" in sys_display:
        sys_key = "web"
    elif "auth" in sys_display:
        sys_key = "auth"
    else:
        sys_key = sys_display.split("-")[0]

    sev_w = inc["severity"] / 5.0
    tr = max(1, inc.get("time_remaining", 1))
    urgency_w = 1.0 / tr

    base = sev_w + urgency_w

    # Propagation bonus: root source systems get highest priority
    if sys_key in prop["root_sources"]:
        base += 0.8  # actively spreading — stop it first

    # Downstream bonus: receiving infection — needs attention but less than source
    if sys_key in prop["downstream"]:
        # Check if the source is still active — if so, deprioritize this symptom
        received_from = status.get(sys_key, {}).get("received_from", [])
        source_still_active = any(
            status.get(src, {}).get("anomaly_score", 0.0) > 0.5
            for src in received_from
        )
        if source_still_active:
            base -= 0.3  # deprioritize downstream symptom — fix root cause first
        else:
            base += 0.3  # source resolved, now fix downstream

    return base


def _global_priority_score(obs: dict) -> float:
    """Propagation-aware holistic urgency score across ALL unresolved incidents.

    Combines severity, urgency, isolation dependency, and propagation risk.
    Root-cause systems (propagating to others) score highest.
    Downstream symptoms score lower when their source is still active.
    """
    unresolved = [i for i in obs["incidents"] if not i["resolved"]]
    if not unresolved:
        return 0.0

    prop = _propagation_analysis(obs)
    total = 0.0
    for inc in unresolved:
        total += _incident_propagation_score(inc, obs, prop)
        # Isolation dependency penalty
        if inc.get("requires_isolation") and not inc.get("isolated"):
            total -= 0.2
    # Add cascade risk as a global multiplier
    total *= (1.0 + prop["cascade_risk"] * 0.5)
    return total


def _feasibility(obs: dict) -> float:
    """Estimate fraction of remaining incidents that can still be resolved."""
    unresolved = [i for i in obs["incidents"] if not i["resolved"]]
    if not unresolved:
        return 1.0
    ticks_left = max(0, obs["max_ticks"] - obs["current_tick"])
    min_ticks_needed = len(unresolved)
    if ticks_left == 0:
        return 0.0
    return min(1.0, ticks_left / max(1, min_ticks_needed))


def _plan_value(obs: dict, depth: int, gamma: float,
                noise_rng: "_random.Random | None" = None) -> float:
    """Recursively estimate the value of the best action sequence from obs."""
    if depth == 0:
        unresolved = [i for i in obs["incidents"] if not i["resolved"]]
        ticks_left = max(0, obs["max_ticks"] - obs["current_tick"])
        return _state_value(unresolved, obs["sla_credits"], obs["proof_score"],
                            ticks_left, obs["max_ticks"], obs["audit_triggered"])

    best = float("-inf")
    for action in range(5):
        if action == 1 and obs["audit_triggered"]:
            continue
        immediate = _score_action(action, obs)
        if immediate <= -1.0:
            continue
        next_obs  = _simulate_obs(action, obs, noise_rng)
        ticks_left_next = max(0, next_obs["max_ticks"] - next_obs["current_tick"])
        next_gamma = ticks_left_next / next_obs["max_ticks"] if next_obs["max_ticks"] > 0 else 0.0
        future = _plan_value(next_obs, depth - 1, next_gamma, noise_rng)
        total  = immediate + gamma * future
        if total > best:
            best = total
    return best if best > float("-inf") else 0.0


def _cascade_override(obs: dict, prop: dict) -> int | None:
    """Check if propagation state demands an immediate override action.

    If a root-source system is actively spreading AND not isolated AND
    its anomaly is critical, return isolate_system (3) immediately.
    This stops the cascade before it infects downstream systems.

    Returns action int if override applies, None otherwise.
    """
    if not prop["root_sources"]:
        return None

    status = obs.get("systems_status", {})
    unresolved = [i for i in obs["incidents"] if not i["resolved"]]

    for src_key in prop["root_sources"]:
        src_status = status.get(src_key, {})
        anomaly = src_status.get("anomaly_score", 0.0)
        isolated = src_status.get("isolated", False)

        if isolated or anomaly < 0.7:
            continue

        # Find the incident for this source system
        src_inc = next(
            (i for i in unresolved if (
                ("db" in i["system"] and src_key == "db") or
                ("web" in i["system"] and src_key == "web") or
                ("auth" in i["system"] and src_key == "auth")
            )),
            None,
        )
        if src_inc is None:
            continue

        # If source requires isolation and isn't isolated → isolate immediately
        if src_inc.get("requires_isolation") and not src_inc.get("isolated"):
            return 3  # isolate_system

        # If source is propagating and anomaly is very high → isolate to stop cascade
        if anomaly >= 0.75 and len(src_status.get("propagating_to", [])) > 0:
            return 3  # isolate_system

    return None


def baseline_policy(obs: dict, _action_history: "list | None" = None) -> int:
    """Propagation-aware depth-3 planning agent.

    Extends the previous policy with:
    1. Cascade override: if a root-source system is actively spreading at
       high anomaly, immediately isolate it before running the planner.
    2. Propagation-aware priority: _global_priority_score now weights
       root-source incidents higher and deprioritizes downstream symptoms
       when their source is still active.
    3. Propagation bonus in action scoring: actions that target root-source
       systems get an additional score boost proportional to cascade_risk.

    All randomness is seeded from observation state — deterministic per seed.
    """
    PLAN_DEPTH = 3
    N_SAMPLES  = 3

    try:
        prop = _propagation_analysis(obs)

        # --- Cascade override: stop active propagation first ---
        override = _cascade_override(obs, prop)
        if override is not None:
            return override

        ticks_left  = max(0, obs["max_ticks"] - obs["current_tick"])
        gamma       = ticks_left / obs["max_ticks"] if obs["max_ticks"] > 0 else 0.0
        feasible    = _feasibility(obs)
        global_prio = _global_priority_score(obs)

        _noise_seed = int(obs["current_tick"] * 1000 + obs["sla_credits"] * 7)
        noise_rng   = _random.Random(_noise_seed)

        best_action = 2
        best_score  = float("-inf")

        for action in range(5):
            # Hard constraint: never fast_patch during audit
            if action == 1 and obs["audit_triggered"]:
                continue

            # Hard constraint: never isolate when top incident doesn't need it
            # (unless cascade override already fired above)
            if action == 3:
                unresolved = [i for i in obs["incidents"] if not i["resolved"]]
                top = min(unresolved, key=lambda i: (-i["severity"], i["time_remaining"], i["id"])) if unresolved else None
                if top and not (top.get("requires_isolation") and not top.get("isolated")):
                    if obs["sla_credits"] > 20.0 and not prop["root_sources"]:
                        continue  # skip isolation unless cascade is active

            # Uncertainty-aware evaluation
            sample_scores = []
            for s in range(N_SAMPLES):
                sample_rng = _random.Random(_noise_seed + s * 31337)
                immediate  = _score_action(action, obs)
                next_obs   = _simulate_obs(action, obs, sample_rng)
                future_val = _plan_value(next_obs, PLAN_DEPTH - 1, gamma * 0.9, sample_rng)
                sample_scores.append(immediate + gamma * future_val)

            score = sum(sample_scores) / len(sample_scores)

            # Global priority bonus: reward actions that reduce system-wide risk
            unresolved_after = [i for i in _simulate_obs(action, obs)["incidents"]
                                 if not i["resolved"]]
            prio_after = sum(
                _incident_propagation_score(i, obs, prop) for i in unresolved_after
            )
            score += (global_prio - prio_after) * 0.15

            # Cascade risk bonus: extra reward for actions targeting root sources
            if prop["cascade_risk"] > 0.2:
                # Isolating or patching a root-source system is worth more
                target_inc = next(
                    (i for i in obs["incidents"]
                     if not i["resolved"] and (
                         ("db" in i["system"] and "db" in prop["root_sources"]) or
                         ("auth" in i["system"] and "auth" in prop["root_sources"]) or
                         ("web" in i["system"] and "web" in prop["root_sources"])
                     )),
                    None,
                )
                if target_inc and action in (2, 3):
                    score += prop["cascade_risk"] * 0.4

            # Feasibility adjustment
            if feasible < 0.5 and action in (0, 4):
                score -= (1.0 - feasible) * 0.5

            # Action diversity penalty
            if _action_history and len(_action_history) >= 2:
                if _action_history[-1] == action and _action_history[-2] == action:
                    score -= 0.3

            if score > best_score:
                best_score  = score
                best_action = action

        return best_action

    except Exception:
        return 2  # safe fallback

    except Exception:
        return 2  # safe fallback: verified_patch


# ---------------------------------------------------------------------------
# LLM-based action selection
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an automated Security Operations Controller (SOC) responding to live security incidents.
You will receive a situation report including:
- A natural language description of what is happening (context)
- Actual system log lines showing the attack in progress (system_logs)
- Observable indicators of compromise (indicators)
- Current incident status and resource levels

Your job is to choose the best response action based on the SITUATION, not just the numbers.

Actions:
  0 = no_op          (advance tick, no effect — use only if no action needed)
  1 = fast_patch     (resolve incident fast — NEVER use during audit, degrades proof)
  2 = verified_patch (resolve with full audit trail — preferred during audit, costs 3 ticks)
  3 = isolate_system (network-isolate the system — required before patching ransomware/rootkits)
  4 = escalate_human (flag for human review — use for insider threats, policy decisions)

Critical rules:
- NEVER use action 1 (fast_patch) when audit_triggered is true
- ALWAYS use action 3 (isolate_system) before patching if requires_isolation is true
- Read the context and system_logs to understand WHAT is happening before deciding
- Respond with ONLY a single integer (0, 1, 2, 3, or 4). No explanation.
"""


def llm_policy(obs: dict) -> int:
    """Ask the LLM to pick an action. Returns None on any failure."""
    client = _get_client()
    if client is None:
        return None

    unresolved = [i for i in obs["incidents"] if not i["resolved"]]

    # Build situation report — the LLM reasons about the SCENARIO, not just numbers
    situation = {
        "situation": obs.get("context", ""),
        "system_logs": obs.get("system_logs", []),
        "indicators_of_compromise": obs.get("indicators", []),
        "tool_output": obs.get("tool_output", ""),
        "systems_status": obs.get("systems_status", {}),
        "current_tick": obs["current_tick"],
        "ticks_remaining": obs.get("ticks_remaining", obs["max_ticks"] - obs["current_tick"]),
        "audit_active": obs["audit_triggered"],
        "sla_credits_remaining": round(obs["sla_credits"], 1),
        "proof_score": round(obs["proof_score"], 3),
        "missing_logs": obs["missing_logs"],
        "unresolved_incidents": [
            {
                "id": i["id"],
                "severity": i["severity"],
                "type": i["type"],
                "system": i["system"],
                "time_remaining": i["time_remaining"],
                "requires_isolation": i.get("requires_isolation", False),
                "isolated": i.get("isolated", False),
                "anomaly_score": i.get("anomaly_score", 0.0),
            }
            for i in unresolved
        ],
    }
    user_msg = f"Situation Report:\n{json.dumps(situation, indent=2)}\n\nAction:"

    try:
        response = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        action = int(text)
        if 0 <= action <= 4:
            return action
    except Exception:
        pass

    return None  # signal fallback


def select_action(obs: dict, action_history: "list | None" = None) -> int:
    """Select action via LLM, falling back to baseline_policy on any failure."""
    action = llm_policy(obs)
    if action is None:
        action = baseline_policy(obs, _action_history=action_history)
    return action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

MAX_STEPS = 19  # hard cap < 20 as required


def run_episode(task_id: str, seed: int = 42) -> tuple:
    """Run a complete episode using SirenEnvironment for typed observations.

    Uses SirenEnvironment (not raw SirenEnv) so rubric_reward, threat_intel,
    ticks_remaining, and done_reason are available on every observation.
    """
    env = SirenEnvironment()

    obs = env.reset(task_id=task_id, seed=seed)
    # Convert typed observation to dict for policy functions
    obs_dict = {
        "incidents": obs.incidents,
        "sla_credits": obs.sla_credits,
        "proof_score": obs.proof_score,
        "missing_logs": obs.missing_logs,
        "active_sessions": obs.active_sessions,
        "ledger_entries": obs.ledger_entries,
        "ledger_hash": obs.ledger_hash,
        "current_tick": obs.current_tick,
        "max_ticks": obs.max_ticks,
        "audit_triggered": obs.audit_triggered,
        "task_id": obs.task_id,
        "systems_status": obs.systems_status,
        "tool_output": obs.tool_output,
        "context": obs.context,
        "system_logs": obs.system_logs,
        "indicators": obs.indicators,
        "ticks_remaining": obs.ticks_remaining,
    }

    trajectory = []
    action_history = []
    total_reward = 0.0
    cumulative = 0.0
    done = False

    using_llm = _get_client() is not None
    policy_label = f"LLM ({_MODEL_NAME})" if using_llm else "baseline (fallback)"

    print(f"[START] task={task_id} seed={seed} policy={policy_label}", flush=True)
    if obs.threat_intel:
        print(f"[INFO] threat_intel={obs.threat_intel}", flush=True)

    step = 0
    while not done and step < MAX_STEPS:
        action = select_action(obs_dict, action_history=action_history)
        typed_obs = env.step(SirenAction(action=action))
        done = typed_obs.done
        reward = typed_obs.reward
        action_history.append(action)
        total_reward += reward
        cumulative += reward

        # Convert back to dict for policy functions
        obs_dict = {
            "incidents": typed_obs.incidents,
            "sla_credits": typed_obs.sla_credits,
            "proof_score": typed_obs.proof_score,
            "missing_logs": typed_obs.missing_logs,
            "active_sessions": typed_obs.active_sessions,
            "ledger_entries": typed_obs.ledger_entries,
            "ledger_hash": typed_obs.ledger_hash,
            "current_tick": typed_obs.current_tick,
            "max_ticks": typed_obs.max_ticks,
            "audit_triggered": typed_obs.audit_triggered,
            "task_id": typed_obs.task_id,
            "systems_status": typed_obs.systems_status,
            "tool_output": typed_obs.tool_output,
            "context": typed_obs.context,
            "system_logs": typed_obs.system_logs,
            "indicators": typed_obs.indicators,
            "ticks_remaining": typed_obs.ticks_remaining,
        }

        trajectory.append({
            "action": action,
            "observation": obs_dict,
            "reward": reward,
            "done": done,
            "info": typed_obs.metadata or {},
        })

        rubric_str = f" rubric={typed_obs.rubric_reward:.4f}" if typed_obs.rubric_reward is not None else ""
        print(
            f"[STEP] step={step} action={action} reward={reward:.4f}"
            f" cumulative_reward={cumulative:.4f} done={done}{rubric_str}",
            flush=True,
        )
        step += 1

    grader_score = typed_obs.rubric_reward if (done and typed_obs.rubric_reward is not None) else 0.001
    done_reason = typed_obs.done_reason if done else ""

    print(
        f"[END] task={task_id} score={grader_score:.4f}"
        f" steps={step} total_reward={total_reward:.4f}"
        + (f" reason={done_reason}" if done_reason else ""),
        flush=True,
    )

    return trajectory, total_reward, grader_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    task_ids = ["easy", "medium", "hard"]
    results = {}

    start_time = time.time()

    try:
        for task_id in task_ids:
            _trajectory, total_reward, grader_score = run_episode(task_id, seed=seed)
            results[task_id] = {
                "total_reward": total_reward,
                "grader_score": grader_score,
            }
    except Exception as exc:
        print(f"Error during episode execution: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start_time

    print("\n=== Summary ===", flush=True)
    print(f"{'task_id':<10} | {'total_reward':>12} | {'grader_score':>12}", flush=True)
    print(f"{'-'*10}-|-{'-'*14}-|-{'-'*13}", flush=True)
    for task_id in task_ids:
        r = results[task_id]
        print(f"{task_id:<10} | {r['total_reward']:>12.4f} | {r['grader_score']:>12.4f}", flush=True)
    print(f"\nElapsed time: {elapsed:.2f}s", flush=True)

    if elapsed >= 1200:
        print(
            "Error: all three episodes exceeded the 1200-second time limit.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Unhandled exception: {exc}", file=sys.stderr)
        sys.exit(1)
