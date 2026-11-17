"""
Candidate profile and agent memory management endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Candidate, CandidateMemory
from app.schemas.candidate import (
    CandidateCreate,
    CandidateUpdate,
    CandidateResponse,
    CandidateMemoryCreate,
    CandidateMemoryResponse,
)
from app.schemas.agent import StructuredMatchReport
from app.recommender.matcher import get_recommender
from app.core.logger import logger

router = APIRouter(prefix="/candidates", tags=["Candidates & Memories"])


@router.post("/", response_model=CandidateResponse, status_code=status.HTTP_201_CREATED)
def create_candidate(candidate_in: CandidateCreate, db: Session = Depends(get_db)):
    existing = db.query(Candidate).filter(Candidate.email == candidate_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Candidate with this email already exists")

    candidate = Candidate(**candidate_in.model_dump())
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    logger.info(f"Registered candidate: {candidate.name} (ID: {candidate.id})")
    return candidate


@router.get("/", response_model=List[CandidateResponse])
def list_candidates(db: Session = Depends(get_db)):
    return db.query(Candidate).all()


@router.get("/{candidate_id}", response_model=CandidateResponse)
def get_candidate(candidate_id: str, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@router.post("/{candidate_id}/memories", response_model=CandidateMemoryResponse, status_code=status.HTTP_201_CREATED)
def add_candidate_memory(candidate_id: str, memory_in: CandidateMemoryCreate, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    mem = CandidateMemory(
        candidate_id=candidate_id,
        category=memory_in.category,
        content=memory_in.content,
        confidence=memory_in.confidence
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)
    logger.info(f"Added memory for candidate {candidate_id}: [{mem.category}] {mem.content}")
    return mem


@router.get("/{candidate_id}/matches", response_model=StructuredMatchReport)
def get_candidate_matches(
    candidate_id: str,
    custom_query: Optional[str] = None,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """
    Directly runs two-stage retrieval and personalized re-ranking for a candidate.
    Incorporates all persistent memories and constraints from database.
    """
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    memories = [m.content for m in candidate.memories]
    recommender = get_recommender()
    report = recommender.match_candidate(
        candidate_data=candidate.to_dict(),
        memories=memories,
        custom_query=custom_query,
        top_k=limit
    )
    return report
