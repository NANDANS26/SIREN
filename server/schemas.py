"""Pydantic v2 request/response schemas for the SIREN FastAPI server."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IncidentSchema(BaseModel):
    id: str
    severity: int
    type: str
    system: str
    time_remaining: int
    resolved: bool
    isolated: bool = False
    requires_isolation: bool = False
    # Phase 1: system anomaly score — always visible
    anomaly_score: float = 0.0
    # Phase 5: partial observability — only populated after tool calls
    system_logs: List[str] = Field(default_factory=list)
    indicators: List[str] = Field(default_factory=list)
    context: str = ""
    scenario_id: str = ""


class ObservationSchema(BaseModel):
    incidents: List[IncidentSchema]
    sla_credits: float
    proof_score: float
    missing_logs: int
    active_sessions: int
    ledger_entries: int
    ledger_hash: str
    current_tick: int
    max_ticks: int
    audit_triggered: bool
    task_id: str
    # Phase 1: system state layer
    systems_status: Dict[str, Any] = Field(default_factory=dict)
    # Phase 2: tool output
    tool_output: str = ""
    # Scenario fields (partially revealed)
    context: str = ""
    system_logs: List[str] = Field(default_factory=list)
    indicators: List[str] = Field(default_factory=list)
    scenario_id: str = ""


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=4)


class StepResponse(BaseModel):
    observation: ObservationSchema
    reward: float
    done: bool
    info: dict
