"""
Tool registry and execution engine for TalentMatch AI Agent.
Includes OpenAI-compatible JSON tool schemas and concrete execution handlers.
"""
import time
from typing import Dict, Any, List, Optional
from app.search.hybrid_search import get_hybrid_search_engine
from app.schemas.job import JobSearchFilter, JobCreate
from app.search.indexer import get_jobs_index
from app.recommender.matcher import get_recommender
from app.db.session import SessionLocal
from app.db.models import Job, Candidate, CandidateMemory
from app.core.logger import logger

# OpenAI-compatible function/tool definitions
AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Performs semantic search with optional filters over the constantly updated jobs index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query describing role, technology, or domain (e.g. 'Senior Python Backend LLM Engineer')"
                    },
                    "remote_type": {
                        "type": "string",
                        "enum": ["remote", "hybrid", "on-site"],
                        "description": "Workplace mode filter"
                    },
                    "min_salary": {
                        "type": "integer",
                        "description": "Minimum compensation in USD (e.g. 150000)"
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of required or preferred skills to filter/match"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of jobs to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_details",
            "description": "Fetches complete details, description, salary, and requirements for a specific job ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The unique ID of the job"
                    }
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "match_candidate_profile",
            "description": "Executes the 2-stage retrieval and personalized re-ranking engine for a candidate profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "description": "Unique identifier of the candidate profile"
                    },
                    "custom_query": {
                        "type": "string",
                        "description": "Optional custom query or user prompt refining the match"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of top personalized recommendations (default: 5)"
                    }
                },
                "required": ["candidate_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_candidate_fact",
            "description": "Saves an extracted candidate preference, constraint, rejection, or career goal to persistent memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "description": "Candidate ID"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["preference", "constraint", "rejection", "goal", "feedback"],
                        "description": "Category of the memory fact"
                    },
                    "content": {
                        "type": "string",
                        "description": "The exact fact or preference to remember (e.g. 'Must have remote work; refuses to relocate')"
                    }
                },
                "required": ["candidate_id", "category", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_or_update_job",
            "description": "Dynamically ingests or updates a job posting into the live index and SQL database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "location": {"type": "string"},
                    "remote_type": {"type": "string", "enum": ["remote", "hybrid", "on-site"]},
                    "min_salary": {"type": "integer"},
                    "max_salary": {"type": "integer"},
                    "required_skills": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"}
                },
                "required": ["title", "company", "description"]
            }
        }
    }
]


class ToolExecutor:
    """Executes agent tool calls against backend engines and databases."""

    @staticmethod
    def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        logger.info(f"Executing tool '{name}' with args: {arguments}")

        try:
            if name == "search_jobs":
                res = ToolExecutor._search_jobs(**arguments)
            elif name == "get_job_details":
                res = ToolExecutor._get_job_details(**arguments)
            elif name == "match_candidate_profile":
                res = ToolExecutor._match_candidate(**arguments)
            elif name == "remember_candidate_fact":
                res = ToolExecutor._remember_fact(**arguments)
            elif name == "add_or_update_job":
                res = ToolExecutor._add_or_update_job(**arguments)
            else:
                res = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}", exc_info=True)
            res = {"error": str(e)}

        duration_ms = round((time.time() - start) * 1000, 2)
        return {
            "tool_name": name,
            "arguments": arguments,
            "result": res,
            "duration_ms": duration_ms
        }

    @staticmethod
    def _search_jobs(
        query: str,
        remote_type: Optional[str] = None,
        min_salary: Optional[int] = None,
        skills: Optional[List[str]] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        search_engine = get_hybrid_search_engine()
        filters = JobSearchFilter(
            query=query,
            remote_type=remote_type,
            min_salary=min_salary,
            skills=skills,
            top_k=limit
        )
        results = search_engine.search(
            query=query,
            filters=filters,
            candidate_skills=skills,
            top_k=limit
        )

        formatted = []
        for r in results:
            formatted.append({
                "id": r.job.id,
                "title": r.job.title,
                "company": r.job.company,
                "location": r.job.location,
                "remote_type": r.job.remote_type,
                "experience_level": r.job.experience_level,
                "salary_range": f"${r.job.min_salary or 0:,} - ${r.job.max_salary or 0:,} {r.job.currency}",
                "required_skills": r.job.required_skills,
                "match_score": r.score,
                "matching_skills": r.matching_skills,
                "missing_skills": r.missing_skills
            })

        return {
            "total_found": len(formatted),
            "jobs": formatted
        }

    @staticmethod
    def _get_job_details(job_id: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                return {"error": f"Job with id '{job_id}' not found."}
            return job.to_dict()
        finally:
            db.close()

    @staticmethod
    def _match_candidate(
        candidate_id: str,
        custom_query: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
            if not candidate:
                return {"error": f"Candidate with id '{candidate_id}' not found."}

            candidate_dict = candidate.to_dict()
            memories = [m.content for m in candidate.memories]

            recommender = get_recommender()
            report = recommender.match_candidate(
                candidate_data=candidate_dict,
                memories=memories,
                custom_query=custom_query,
                top_k=limit
            )
            return report.model_dump()
        finally:
            db.close()

    @staticmethod
    def _remember_fact(candidate_id: str, category: str, content: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
            if not candidate:
                return {"error": f"Candidate with id '{candidate_id}' not found."}

            # Check if duplicate already exists
            existing = db.query(CandidateMemory).filter(
                CandidateMemory.candidate_id == candidate_id,
                CandidateMemory.content == content
            ).first()

            if existing:
                return {"status": "already_exists", "memory_id": existing.id}

            memory = CandidateMemory(
                candidate_id=candidate_id,
                category=category,
                content=content,
                confidence=1.0
            )
            db.add(memory)
            db.commit()
            db.refresh(memory)
            logger.info(f"Saved memory for candidate {candidate_id}: [{category}] {content}")
            return {"status": "saved", "memory_id": memory.id, "content": content}
        finally:
            db.close()

    @staticmethod
    def _add_or_update_job(
        title: str,
        company: str,
        description: str,
        location: str = "Remote",
        remote_type: str = "remote",
        min_salary: Optional[int] = None,
        max_salary: Optional[int] = None,
        required_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            job = Job(
                title=title,
                company=company,
                location=location,
                remote_type=remote_type,
                min_salary=min_salary,
                max_salary=max_salary,
                required_skills=required_skills or [],
                description=description,
                is_active=True
            )
            db.add(job)
            db.commit()
            db.refresh(job)

            # Synchronize to Vector Index immediately
            idx = get_jobs_index()
            idx.upsert_job(job_id=job.id, text=job.to_search_text(), metadata=job.to_dict())

            return {"status": "success", "job_id": job.id, "title": job.title}
        finally:
            db.close()
