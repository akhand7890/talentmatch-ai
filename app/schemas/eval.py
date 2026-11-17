"""
Pydantic schemas for LLM Evaluation and Benchmark Tracking.
Enables quantitative validation of retrieval, ranking, tool calling, and LLM output quality.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    description: str
    query: str
    candidate_profile: Dict[str, Any]
    candidate_memories: List[str] = Field(default_factory=list)
    golden_job_ids: List[str] = Field(..., description="Ground-truth relevant job IDs")
    negative_job_ids: List[str] = Field(default_factory=list, description="Jobs that must be excluded (e.g. filtered by memory or constraint)")
    expected_tools: List[str] = Field(default_factory=list, description="Tools the agent is expected to invoke")


class EvalMetricResult(BaseModel):
    name: str
    score: float
    threshold: float
    passed: bool
    description: str


class EvalCaseResult(BaseModel):
    case_id: str
    case_description: str
    retrieved_job_ids: List[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    tool_accuracy: float
    faithfulness_score: float
    passed: bool
    notes: List[str] = Field(default_factory=list)


class EvalRunSummary(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_cases: int
    passed_cases: int
    pass_rate_percent: float
    avg_precision_at_k: float
    avg_recall_at_k: float
    mean_reciprocal_rank: float
    avg_tool_accuracy: float
    avg_faithfulness: float
    avg_latency_ms: float
    metrics: List[EvalMetricResult]
    case_results: List[EvalCaseResult]
    quality_verdict: str  # e.g., "PASSED: Production Ready" or "FAILED: Needs Prompt Tuning"
