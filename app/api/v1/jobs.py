"""
Job management and semantic search endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Job
from app.schemas.job import JobCreate, JobUpdate, JobResponse, JobSearchFilter, JobSearchResult
from app.search.indexer import get_jobs_index
from app.search.hybrid_search import get_hybrid_search_engine
from app.core.logger import logger

router = APIRouter(prefix="/jobs", tags=["Jobs & Dynamic Search Index"])


@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job(job_in: JobCreate, db: Session = Depends(get_db)):
    """
    Ingests a new job posting.
    Immediately indexes it into the live vector search index.
    """
    job = Job(**job_in.model_dump())
    db.add(job)
    db.commit()
    db.refresh(job)

    # Incrementally update live vector index
    index = get_jobs_index()
    index.upsert_job(job_id=job.id, text=job.to_search_text(), metadata=job.to_dict())
    logger.info(f"Ingested and indexed new job: {job.title} (ID: {job.id})")
    return job


@router.get("/", response_model=List[JobResponse])
def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    query = db.query(Job)
    if active_only:
        query = query.filter(Job.is_active == True)
    return query.offset(skip).limit(limit).all()


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.put("/{job_id}", response_model=JobResponse)
def update_job(job_id: str, job_update: JobUpdate, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    update_data = job_update.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        setattr(job, field, val)

    db.commit()
    db.refresh(job)

    # Sync to vector index
    index = get_jobs_index()
    if job.is_active:
        index.upsert_job(job_id=job.id, text=job.to_search_text(), metadata=job.to_dict())
    else:
        index.remove_job(job.id)

    return job


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str, db: Session = Depends(get_db)):
    """
    Deactivates and removes job from the active search index.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.is_active = False
    db.commit()

    index = get_jobs_index()
    index.remove_job(job_id)
    return None


@router.post("/search", response_model=List[JobSearchResult])
def search_jobs(filters: JobSearchFilter):
    """
    Runs semantic + lexical hybrid search over the live jobs index.
    """
    engine = get_hybrid_search_engine()
    results = engine.search(
        query=filters.query or "software engineer",
        filters=filters,
        candidate_skills=filters.skills,
        top_k=filters.top_k
    )
    return results
