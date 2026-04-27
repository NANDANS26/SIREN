"""
FastAPI application for the SIREN Environment.

Exposes the SirenEnvironment over HTTP endpoints compatible with the
OpenEnv validator, plus a WebSocket endpoint for SDK-based clients.

Endpoints:
    GET  /health    — liveness probe
    POST /reset     — start or restart an episode
    POST /step      — submit action (0–4)
    GET  /state     — read current observation without advancing state
    GET  /schema    — action/observation JSON schemas
    GET  /metadata  — environment metadata
    WS   /ws        — WebSocket endpoint for persistent sessions (OpenEnv SDK)

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request

from env.environment import SirenEnv
from server.schemas import (
    ObservationSchema,
    ResetRequest,
    StepRequest,
    StepResponse,
)

app = FastAPI(
    title="OpenEnv Environment HTTP API",
    description="SIREN Security Incident Response ENvironment",
    version="1.0.0",
)

# Global environment instance — stateful for HTTP endpoints
_env: SirenEnv | None = None
_env_initialized: bool = False


# ---------------------------------------------------------------------------
# HTTP endpoints (stateful — shared instance)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/reset", response_model=ObservationSchema)
def reset(body: ResetRequest = ResetRequest()) -> ObservationSchema:
    global _env, _env_initialized
    _env = SirenEnv(task_id=body.task_id)
    obs = _env.reset(seed=body.seed)
    _env_initialized = True
    return ObservationSchema(**obs)


@app.post("/step", response_model=StepResponse)
def step(body: StepRequest) -> StepResponse:
    global _env, _env_initialized
    if not _env_initialized or _env is None:
        raise HTTPException(
            status_code=400,
            detail="Episode not initialized. Call POST /reset first.",
        )
    try:
        obs, reward, done, info = _env.step(body.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StepResponse(
        observation=ObservationSchema(**obs),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=ObservationSchema)
def state() -> ObservationSchema:
    global _env, _env_initialized
    if not _env_initialized or _env is None:
        raise HTTPException(
            status_code=400,
            detail="Episode not initialized. Call POST /reset first.",
        )
    obs = _env.state()
    return ObservationSchema(**obs)


@app.get("/schema")
def schema() -> dict:
    """Return action, observation, and state JSON schemas (OpenEnv SDK compatibility)."""
    return {
        "action": {
            "type": "object",
            "title": "SirenAction",
            "description": "Action for the SIREN environment.",
            "properties": {
                "action": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                    "default": 2,
                    "description": (
                        "0=no_op, 1=fast_patch, 2=verified_patch, "
                        "3=isolate_system, 4=escalate_human"
                    ),
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Additional metadata for the action",
                },
            },
            "additionalProperties": False,
        },
        "observation": {
            "type": "object",
            "title": "SirenObservation",
            "description": "Observation from the SIREN environment.",
            "properties": {
                "incidents": {"type": "array"},
                "sla_credits": {"type": "number"},
                "proof_score": {"type": "number"},
                "missing_logs": {"type": "integer"},
                "active_sessions": {"type": "integer"},
                "ledger_entries": {"type": "integer"},
                "ledger_hash": {"type": "string"},
                "current_tick": {"type": "integer"},
                "max_ticks": {"type": "integer"},
                "audit_triggered": {"type": "boolean"},
                "task_id": {"type": "string"},
                "reward": {"type": "number"},
                "done": {"type": "boolean"},
            },
        },
        "state": {
            "type": "object",
            "title": "SirenState",
            "description": "Server-side state for a SIREN episode.",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "task_id": {"type": "string"},
                "seed": {"type": "integer"},
                "current_tick": {"type": "integer"},
                "max_ticks": {"type": "integer"},
                "audit_triggered": {"type": "boolean"},
                "sla_credits": {"type": "number"},
                "total_reward": {"type": "number"},
                "incidents_resolved": {"type": "integer"},
                "action_counts": {"type": "object"},
                "threat_intel": {"type": "string"},
            },
        },
    }


@app.get("/metadata")
def metadata() -> dict:
    """Return environment metadata (OpenEnv SDK compatibility)."""
    return {
        "name": "siren-env",
        "description": (
            "SIREN Security Incident Response ENvironment — "
            "RL training environment for security operations"
        ),
        "version": "1.0.0",
        "action_space": {"type": "discrete", "n": 5},
        "tasks": ["easy", "medium", "hard"],
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> dict:
    """MCP JSON-RPC endpoint (OpenEnv SDK compatibility)."""
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "name": "siren-env",
            "description": "SIREN Security Incident Response ENvironment",
            "tools": [],
        },
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint — per-session stateful environment (OpenEnv SDK)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for persistent session-based interactions.

    Supports the OpenEnv EnvClient SDK protocol:
      - {"type": "reset", ...kwargs} → observation
      - {"type": "step", "action": {...}} → {observation, reward, done}
      - {"type": "state"} → observation
    """
    await websocket.accept()
    ws_env: SirenEnv | None = None
    ws_initialized = False

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_id = msg.get("task_id", "easy")
                seed = msg.get("seed")
                ws_env = SirenEnv(task_id=task_id)
                obs = ws_env.reset(seed=seed)
                ws_initialized = True
                await websocket.send_json({
                    "observation": obs,
                    "reward": 0.0,
                    "done": False,
                })

            elif msg_type == "step":
                if not ws_initialized or ws_env is None:
                    await websocket.send_json({"error": "Call reset first"})
                    continue
                action_data = msg.get("action", {})
                action_int = (
                    action_data.get("action", 2)
                    if isinstance(action_data, dict)
                    else int(action_data)
                )
                try:
                    obs, reward, done, info = ws_env.step(action_int)
                    await websocket.send_json({
                        "observation": obs,
                        "reward": reward,
                        "done": done,
                        "info": info,
                    })
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})

            elif msg_type == "state":
                if not ws_initialized or ws_env is None:
                    await websocket.send_json({"error": "Call reset first"})
                    continue
                obs = ws_env.state()
                await websocket.send_json({"observation": obs})

            else:
                await websocket.send_json({"error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        pass


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main()  # entry point — port configurable via --port arg above