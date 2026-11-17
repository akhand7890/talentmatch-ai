"""
Pydantic schemas for Agent interaction, tool execution logs, and structured outputs.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    output: Any
    duration_ms: float = 0.0


class JobMatchRecommendation(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    remote_type: str
    fit_score: float = Field(..., ge=0, le=100, description="Match score from 0 to 100")
    why_recommended: str = Field(..., description="Personalized rationale tailored to candidate's background")
    matching_skills: List[str]
    missing_skills: List[str]
    salary_fit: str = Field(..., description="Salary comparison with candidate's target")
    culture_or_remote_fit: str


class StructuredMatchReport(BaseModel):
    candidate_id: str
    candidate_name: str
    executive_summary: str
    top_matches: List[JobMatchRecommendation]
    identified_gaps: List[str]
    recommended_upskilling: List[str]
    memory_updates_stored: List[str] = Field(default_factory=list)


class AgentChatRequest(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "Can you find me senior AI engineer jobs that allow remote work and pay at least $160k?"})
    candidate_id: Optional[str] = Field(None, description="Candidate ID to load persistent context", json_schema_extra={"example": "cand_01_alex"})
    session_id: Optional[str] = Field("default_session", description="Session ID for multi-turn short-term memory")


class AgentChatResponse(BaseModel):
    reply: str
    session_id: str
    candidate_id: Optional[str] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    structured_report: Optional[StructuredMatchReport] = None
    active_memories: List[str] = Field(default_factory=list, description="Memories retrieved and utilized by the agent")
