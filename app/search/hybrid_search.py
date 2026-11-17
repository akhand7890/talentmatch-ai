"""
Hybrid Retrieval Engine combining:
1. Dense semantic vector similarity (embeddings)
2. Lexical keyword / skill matching
3. Structured metadata SQL filtering (salary, remote, seniority)
"""
from typing import List, Optional, Dict, Any
from app.search.indexer import get_jobs_index, JobsVectorIndex
from app.schemas.job import JobSearchFilter, JobSearchResult, JobResponse
from app.core.logger import logger


class HybridSearchEngine:
    def __init__(self, index: Optional[JobsVectorIndex] = None):
        self.index = index or get_jobs_index()

    def search(
        self,
        query: str,
        filters: Optional[JobSearchFilter] = None,
        candidate_skills: Optional[List[str]] = None,
        alpha: float = 0.65,  # Weight for vector similarity
        beta: float = 0.35,   # Weight for skill/lexical overlap
        top_k: int = 10,
    ) -> List[JobSearchResult]:
        """
        Executes hybrid retrieval:
        1. Retrieves candidate jobs via vector cosine similarity.
        2. Filters out jobs not meeting hard constraints (remote_type, min_salary, etc.).
        3. Computes lexical / skill overlap percentage.
        4. Produces blended score and returns ranked results.
        """
        # Retrieve more candidates from vector search to allow filtering headroom
        fetch_k = max(top_k * 3, 25)
        raw_results = self.index.search_vector(query_text=query, top_k=fetch_k)

        if not raw_results:
            return []

        search_skills = set(s.lower() for s in (candidate_skills or []))
        if filters and filters.skills:
            search_skills.update(s.lower() for s in filters.skills)

        filtered_results: List[JobSearchResult] = []

        for job_id, semantic_score, meta in raw_results:
            # 1. Hard Filter Checks
            if filters:
                if filters.remote_type:
                    target_mode = filters.remote_type.lower()
                    job_mode = meta.get("remote_type", "").lower()
                    if target_mode == "remote" and job_mode == "on-site":
                        continue
                    elif target_mode == "on-site" and job_mode != "on-site":
                        continue

                if filters.experience_level:
                    target_lvl = filters.experience_level.lower()
                    job_lvl = meta.get("experience_level", "").lower()
                    # Senior/Lead exclusion for junior
                    if target_lvl == "junior" and job_lvl in ["lead", "principal", "staff"]:
                        continue
                    if target_lvl in ["lead", "principal"] and job_lvl == "junior":
                        continue

                if filters.min_salary:
                    job_max_salary = meta.get("max_salary") or meta.get("min_salary") or 0
                    if job_max_salary < filters.min_salary:
                        continue

            # 2. Skill Overlap & Gap Analysis
            job_req_skills = [s.lower() for s in (meta.get("required_skills") or [])]
            job_nice_skills = [s.lower() for s in (meta.get("nice_to_have_skills") or [])]
            all_job_skills = set(job_req_skills + job_nice_skills)

            matching_skills = []
            missing_skills = []

            if search_skills:
                matching_skills = [s for s in job_req_skills if s in search_skills]
                missing_skills = [s for s in job_req_skills if s not in search_skills]
                skill_match_ratio = len(matching_skills) / max(len(job_req_skills), 1)
            else:
                skill_match_ratio = 0.5  # Neutral if no skills specified

            # 3. Score Fusion: Normalize semantic score from [-1, 1] to [0, 1]
            norm_semantic = max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))
            blended_score = (alpha * norm_semantic) + (beta * skill_match_ratio)

            # Map to JobSearchResult
            job_resp = JobResponse(
                id=job_id,
                title=meta.get("title", ""),
                company=meta.get("company", ""),
                location=meta.get("location", "Remote"),
                remote_type=meta.get("remote_type", "remote"),
                experience_level=meta.get("experience_level", "mid"),
                min_salary=meta.get("min_salary"),
                max_salary=meta.get("max_salary"),
                currency=meta.get("currency", "USD"),
                required_skills=meta.get("required_skills", []),
                nice_to_have_skills=meta.get("nice_to_have_skills", []),
                description=meta.get("description", ""),
                benefits=meta.get("benefits", []),
                is_active=meta.get("is_active", True)
            )

            filtered_results.append(
                JobSearchResult(
                    job=job_resp,
                    score=round(blended_score, 4),
                    semantic_score=round(float(semantic_score), 4),
                    skill_match_percentage=round(skill_match_ratio * 100, 1),
                    matching_skills=matching_skills,
                    missing_skills=missing_skills
                )
            )

        # Sort descending by blended score
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        return filtered_results[:top_k]


_hybrid_search_instance: Optional[HybridSearchEngine] = None


def get_hybrid_search_engine() -> HybridSearchEngine:
    global _hybrid_search_instance
    if _hybrid_search_instance is None:
        _hybrid_search_instance = HybridSearchEngine()
    return _hybrid_search_instance
