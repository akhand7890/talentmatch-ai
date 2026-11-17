"""
Main FastAPI Application Entrypoint for TalentMatch AI.
Coordinates lifespan startup, database seeding, vector indexing, CORS, and routing.
"""
import os
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logger import logger
from app.db.session import init_db, SessionLocal
from app.db.models import Job, Candidate, CandidateMemory
from app.search.indexer import get_jobs_index
from app.api.v1.jobs import router as jobs_router
from app.api.v1.candidates import router as candidates_router
from app.api.v1.agent import router as agent_router
from app.api.v1.evals import router as evals_router

BASE_DIR = Path(__file__).resolve().parent.parent


def seed_data_if_empty():
    """Seeds sample jobs and candidates if database is fresh."""
    db = SessionLocal()
    try:
        # 1. Seed Jobs
        job_count = db.query(Job).count()
        jobs_file = BASE_DIR / "data" / "sample_jobs.json"

        if job_count == 0 and jobs_file.exists():
            logger.info("Database jobs table empty. Seeding sample jobs corpus...")
            with open(jobs_file, "r", encoding="utf-8") as f:
                jobs_data = json.load(f)

            for j in jobs_data:
                db_job = Job(
                    id=j["id"],
                    title=j["title"],
                    company=j["company"],
                    location=j["location"],
                    remote_type=j["remote_type"],
                    experience_level=j["experience_level"],
                    min_salary=j.get("min_salary"),
                    max_salary=j.get("max_salary"),
                    currency=j.get("currency", "USD"),
                    required_skills=j.get("required_skills", []),
                    nice_to_have_skills=j.get("nice_to_have_skills", []),
                    description=j["description"],
                    benefits=j.get("benefits", []),
                    is_active=j.get("is_active", True)
                )
                db.add(db_job)
            db.commit()
            logger.info(f"Successfully seeded {len(jobs_data)} jobs into database.")

        # 2. Synchronize all active jobs into the live vector index
        all_active_jobs = db.query(Job).filter(Job.is_active == True).all()
        index = get_jobs_index()
        batch = [(j.id, j.to_search_text(), j.to_dict()) for j in all_active_jobs]
        index.upsert_batch(batch)
        logger.info(f"Vector Index populated with {index.count()} active jobs.")

        # 3. Seed Candidates
        cand_count = db.query(Candidate).count()
        candidates_file = BASE_DIR / "data" / "sample_candidates.json"

        if cand_count == 0 and candidates_file.exists():
            logger.info("Database candidates table empty. Seeding candidates...")
            with open(candidates_file, "r", encoding="utf-8") as f:
                cands_data = json.load(f)

            for c in cands_data:
                db_cand = Candidate(
                    id=c["id"],
                    name=c["name"],
                    email=c["email"],
                    headline=c.get("headline"),
                    years_of_experience=c.get("years_of_experience", 0.0),
                    skills=c.get("skills", []),
                    desired_roles=c.get("desired_roles", []),
                    preferred_locations=c.get("preferred_locations", []),
                    remote_preference=c.get("remote_preference", "remote"),
                    min_desired_salary=c.get("min_desired_salary"),
                    summary=c.get("summary")
                )
                db.add(db_cand)
                db.flush()

                # Add memories
                for m in c.get("memories", []):
                    mem = CandidateMemory(
                        candidate_id=db_cand.id,
                        category=m["category"],
                        content=m["content"],
                        confidence=1.0
                    )
                    db.add(mem)

            db.commit()
            logger.info(f"Successfully seeded {len(cands_data)} candidates with persistent memories.")

    except Exception as e:
        logger.error(f"Error seeding data: {e}", exc_info=True)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up TalentMatch AI Backend...")
    init_db()
    seed_data_if_empty()
    yield
    logger.info("Shutting down TalentMatch AI Backend...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade AI Agent for Semantic Job Matching, Retrieval & Personalization.",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(agent_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(candidates_router, prefix="/api/v1")
app.include_router(evals_router, prefix="/api/v1")


@app.get("/health", tags=["System Health"])
def health_check():
    idx = get_jobs_index()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "indexed_jobs": idx.count(),
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER
    }


# Static UI mount
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def read_root():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Welcome to TalentMatch AI. Access /docs for interactive Swagger UI."}
