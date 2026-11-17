"""
Database models for TalentMatch AI Agent.
Demonstrates relational SQL modeling, JSON attributes for flexible skill matching,
and persistent memory storage for the LLM agent.
"""
import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Job(Base):
    """
    Represents an employment opportunity in the jobs index.
    Supports dynamic changes (is_active, updated_at, upserting).
    """
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    location = Column(String(255), nullable=False, default="Remote")
    remote_type = Column(String(50), nullable=False, default="remote")  # remote, hybrid, on-site
    experience_level = Column(String(50), nullable=False, default="mid")  # junior, mid, senior, lead
    min_salary = Column(Integer, nullable=True)
    max_salary = Column(Integer, nullable=True)
    currency = Column(String(10), default="USD")
    required_skills = Column(JSON, default=list)  # list of strings
    nice_to_have_skills = Column(JSON, default=list)  # list of strings
    description = Column(Text, nullable=False)
    benefits = Column(JSON, default=list)
    is_active = Column(Boolean, default=True, index=True)  # for dynamic invalidation
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    matches = relationship("MatchRecord", back_populates="job", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "remote_type": self.remote_type,
            "experience_level": self.experience_level,
            "min_salary": self.min_salary,
            "max_salary": self.max_salary,
            "currency": self.currency,
            "required_skills": self.required_skills or [],
            "nice_to_have_skills": self.nice_to_have_skills or [],
            "description": self.description,
            "benefits": self.benefits or [],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_search_text(self) -> str:
        """Constructs an information-dense text representation for vector embedding."""
        req_skills = ", ".join(self.required_skills or [])
        nice_skills = ", ".join(self.nice_to_have_skills or [])
        salary_str = f"${self.min_salary or 0} - ${self.max_salary or 0} {self.currency}"
        return (
            f"Job Title: {self.title}\n"
            f"Company: {self.company}\n"
            f"Level: {self.experience_level} | Work Mode: {self.remote_type} | Location: {self.location}\n"
            f"Salary Range: {salary_str}\n"
            f"Required Skills: {req_skills}\n"
            f"Nice-to-Have Skills: {nice_skills}\n"
            f"Description: {self.description}"
        )


class Candidate(Base):
    """
    Represents a candidate profile with skills, experience, and baseline career preferences.
    """
    __tablename__ = "candidates"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    headline = Column(String(255), nullable=True)
    years_of_experience = Column(Float, default=0.0)
    skills = Column(JSON, default=list)  # ["Python", "FastAPI", "Docker", "PyTorch"]
    desired_roles = Column(JSON, default=list)  # ["AI Engineer", "Backend Developer"]
    preferred_locations = Column(JSON, default=list)  # ["Remote", "New York"]
    remote_preference = Column(String(50), default="remote")
    min_desired_salary = Column(Integer, nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Agent memory and matching associations
    memories = relationship("CandidateMemory", back_populates="candidate", cascade="all, delete-orphan")
    matches = relationship("MatchRecord", back_populates="candidate", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "headline": self.headline,
            "years_of_experience": self.years_of_experience,
            "skills": self.skills or [],
            "desired_roles": self.desired_roles or [],
            "preferred_locations": self.preferred_locations or [],
            "remote_preference": self.remote_preference,
            "min_desired_salary": self.min_desired_salary,
            "summary": self.summary,
            "memories_count": len(self.memories) if self.memories else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_profile_summary(self) -> str:
        """Synthesizes profile into an agent prompt context string."""
        skills_str = ", ".join(self.skills or [])
        roles_str = ", ".join(self.desired_roles or [])
        return (
            f"Candidate: {self.name} ({self.headline or 'Candidate'})\n"
            f"Experience: {self.years_of_experience} years\n"
            f"Skills: {skills_str}\n"
            f"Desired Roles: {roles_str}\n"
            f"Remote Pref: {self.remote_preference} | Min Salary: ${self.min_desired_salary or 'Open'}\n"
            f"Summary: {self.summary or 'None provided'}"
        )


class CandidateMemory(Base):
    """
    Persistent memory store for the AI Agent.
    Remembers context across sessions: user preferences, company rejections,
    salary requirements changes, interview notes, and domain constraints.
    """
    __tablename__ = "candidate_memories"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False, index=True)
    category = Column(String(50), nullable=False)  # "preference", "constraint", "rejection", "goal", "feedback"
    content = Column(Text, nullable=False)  # e.g., "Refuses to work in crypto or gambling companies"
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="memories")

    def to_dict(self):
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "category": self.category,
            "content": self.content,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MatchRecord(Base):
    """
    Stores historical matching results, fit scores, and LLM generated rationales.
    Allows measuring performance, ranking calibration, and user feedback.
    """
    __tablename__ = "match_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False, index=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False, index=True)
    fit_score = Column(Float, nullable=False)  # 0.0 to 100.0
    retrieval_rank = Column(Integer, nullable=True)
    semantic_similarity = Column(Float, nullable=True)
    matching_skills = Column(JSON, default=list)
    missing_skills = Column(JSON, default=list)
    match_rationale = Column(Text, nullable=True)
    feedback_status = Column(String(50), default="suggested")  # "suggested", "shortlisted", "applied", "dismissed"
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="matches")
    job = relationship("Job", back_populates="matches")


class EvaluationRecord(Base):
    """
    Stores LLM output quality evaluations, benchmark metrics, and judge results.
    """
    __tablename__ = "evaluation_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    benchmark_name = Column(String(255), nullable=False)
    precision_at_k = Column(Float, nullable=False)
    recall_at_k = Column(Float, nullable=False)
    mrr = Column(Float, nullable=False)
    tool_selection_accuracy = Column(Float, nullable=False)
    faithfulness_score = Column(Float, nullable=False)
    total_eval_cases = Column(Integer, nullable=False)
    details = Column(JSON, default=dict)
    run_at = Column(DateTime, default=datetime.utcnow)
