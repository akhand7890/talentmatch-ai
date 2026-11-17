"""
Pydantic schemas for Job data models, search requests, and index updates.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class JobBase(BaseModel):
    title: str = Field(..., json_schema_extra={"example": "Senior AI Backend Engineer"})
    company: str = Field(..., json_schema_extra={"example": "Anthropic"})
    location: str = Field(default="Remote", json_schema_extra={"example": "San Francisco, CA"})
    remote_type: str = Field(default="remote", json_schema_extra={"example": "remote"})  # remote, hybrid, on-site
    experience_level: str = Field(default="senior", json_schema_extra={"example": "senior"})  # junior, mid, senior, lead
    min_salary: Optional[int] = Field(default=None, json_schema_extra={"example": 160000})
    max_salary: Optional[int] = Field(default=None, json_schema_extra={"example": 220000})
    currency: str = Field(default="USD", json_schema_extra={"example": "USD"})
    required_skills: List[str] = Field(default_factory=list, json_schema_extra={"example": ["Python", "FastAPI", "LLMs", "Vector Databases"]})
    nice_to_have_skills: List[str] = Field(default_factory=list, json_schema_extra={"example": ["Docker", "Kubernetes", "PyTorch"]})
    description: str = Field(..., json_schema_extra={"example": "We are looking for an AI Backend Engineer to build tool-calling agents..."})
    benefits: List[str] = Field(default_factory=list, json_schema_extra={"example": ["Health insurance", "Remote stipend", "Equity"]})
    is_active: bool = Field(default=True)


class JobCreate(JobBase):
    pass


class JobUpdate(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    remote_type: Optional[str] = None
    experience_level: Optional[str] = None
    min_salary: Optional[int] = None
    max_salary: Optional[int] = None
    currency: Optional[str] = None
    required_skills: Optional[List[str]] = None
    nice_to_have_skills: Optional[List[str]] = None
    description: Optional[str] = None
    benefits: Optional[List[str]] = None
    is_active: Optional[bool] = None


class JobResponse(JobBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class JobSearchFilter(BaseModel):
    query: Optional[str] = Field(None, description="Natural language search query")
    remote_type: Optional[str] = Field(None, description="remote, hybrid, or on-site")
    experience_level: Optional[str] = Field(None, description="junior, mid, senior, lead")
    min_salary: Optional[int] = Field(None, description="Minimum acceptable salary")
    skills: Optional[List[str]] = Field(None, description="Filtering skills")
    top_k: int = Field(default=10, ge=1, le=50)


class JobSearchResult(BaseModel):
    job: JobResponse
    score: float = Field(..., description="Combined retrieval or re-rank score")
    semantic_score: float = Field(..., description="Cosine similarity score from vector index")
    skill_match_percentage: float = Field(..., description="Overlap percentage with candidate/query skills")
    matching_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
