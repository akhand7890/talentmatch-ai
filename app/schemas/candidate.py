"""
Pydantic schemas for Candidate profiles and Agent Memories.
"""
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, ConfigDict


class CandidateMemoryBase(BaseModel):
    category: str = Field(..., description="preference, constraint, rejection, goal, feedback", json_schema_extra={"example": "preference"})
    content: str = Field(..., description="Details of constraint/preference", json_schema_extra={"example": "Candidate prefers fully remote roles and will not relocate."})
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CandidateMemoryCreate(CandidateMemoryBase):
    pass


class CandidateMemoryResponse(CandidateMemoryBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    candidate_id: str
    created_at: Optional[datetime] = None


class CandidateBase(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Alex Rivera"})
    email: EmailStr = Field(..., json_schema_extra={"example": "alex.rivera@example.com"})
    headline: Optional[str] = Field(None, json_schema_extra={"example": "Senior Python & LLM Engineer"})
    years_of_experience: float = Field(default=0.0, json_schema_extra={"example": 5.5})
    skills: List[str] = Field(default_factory=list, json_schema_extra={"example": ["Python", "FastAPI", "Docker", "PostgreSQL", "LangChain"]})
    desired_roles: List[str] = Field(default_factory=list, json_schema_extra={"example": ["AI Backend Engineer", "LLM Systems Engineer"]})
    preferred_locations: List[str] = Field(default_factory=list, json_schema_extra={"example": ["Remote", "San Francisco"]})
    remote_preference: str = Field(default="remote", json_schema_extra={"example": "remote"})
    min_desired_salary: Optional[int] = Field(default=None, json_schema_extra={"example": 150000})
    summary: Optional[str] = Field(
        None,
        json_schema_extra={"example": "Built RAG systems and microservices in production. Experienced with embeddings and vector databases."}
    )


class CandidateCreate(CandidateBase):
    pass


class CandidateUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    headline: Optional[str] = None
    years_of_experience: Optional[float] = None
    skills: Optional[List[str]] = None
    desired_roles: Optional[List[str]] = None
    preferred_locations: Optional[List[str]] = None
    remote_preference: Optional[str] = None
    min_desired_salary: Optional[int] = None
    summary: Optional[str] = None


class CandidateResponse(CandidateBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    memories: List[CandidateMemoryResponse] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
