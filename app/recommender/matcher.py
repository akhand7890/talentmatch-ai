"""
Two-Stage Recommendation & Personalization Engine:
Stage 1: Candidate Profile / Query Retrieval (Vector + Keyword)
Stage 2: Multi-Factor Re-Ranking & Personalization with Candidate Memory Integration
"""
from typing import List, Dict, Any, Optional
from app.search.hybrid_search import get_hybrid_search_engine, HybridSearchEngine
from app.schemas.job import JobSearchFilter, JobResponse
from app.schemas.candidate import CandidateResponse
from app.schemas.agent import JobMatchRecommendation, StructuredMatchReport
from app.core.logger import logger


class TwoStageRecommender:
    def __init__(self, search_engine: Optional[HybridSearchEngine] = None):
        self.search_engine = search_engine or get_hybrid_search_engine()

    def match_candidate(
        self,
        candidate_data: Dict[str, Any],
        memories: Optional[List[str]] = None,
        custom_query: Optional[str] = None,
        top_k: int = 5
    ) -> StructuredMatchReport:
        """
        Executes end-to-end personalized matching:
        Stage 1: Retrieval via dense vector + keyword filters
        Stage 2: Re-ranking with weighted personalization, memory constraints, and gap analysis
        """
        candidate_name = candidate_data.get("name", "Candidate")
        candidate_id = candidate_data.get("id", "anonymous")
        skills = candidate_data.get("skills", [])
        experience_years = float(candidate_data.get("years_of_experience", 0.0))
        desired_roles = candidate_data.get("desired_roles", [])
        min_salary = candidate_data.get("min_desired_salary")
        remote_pref = candidate_data.get("remote_preference")
        memories = memories or []

        # Construct Stage 1 search query
        roles_text = " ".join(desired_roles) if desired_roles else "Software Engineer"
        skills_text = " ".join(skills)
        search_query = custom_query or f"{roles_text} {skills_text} {candidate_data.get('headline', '')}"

        # Setup filters based on candidate preferences
        exp_level = "junior" if experience_years <= 2 else None
        filters = JobSearchFilter(
            remote_type=remote_pref if remote_pref and remote_pref.lower() not in ["any", "hybrid", "flexible"] else None,
            experience_level=exp_level,
            min_salary=int(min_salary * 0.85) if min_salary else None,  # slightly relaxed for retrieval stage
            skills=skills,
            top_k=top_k * 3
        )

        # Stage 1: Retrieval
        retrieved_results = self.search_engine.search(
            query=search_query,
            filters=filters,
            candidate_skills=skills,
            top_k=top_k * 3
        )

        if not retrieved_results:
            return StructuredMatchReport(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                executive_summary=f"No matching active jobs found matching preferences for {candidate_name}.",
                top_matches=[],
                identified_gaps=["No matching roles in current jobs index."],
                recommended_upskilling=["Expand search criteria or update skill profile."]
            )

        # Stage 2: Re-ranking & Personalization Scoring
        scored_recommendations: List[JobMatchRecommendation] = []
        all_identified_gaps = set()
        cand_skills_lower = {s.lower() for s in skills}

        # Check negative memories / constraints (e.g., "no crypto", "no defense")
        disliked_keywords = []
        for mem in memories:
            m_low = mem.lower()
            if "reject" in m_low or "no " in m_low or "dislike" in m_low or "avoid" in m_low or "refuse" in m_low:
                for word in ["crypto", "web3", "gambling", "defense", "consulting", "onsite", "on-site"]:
                    if word in m_low:
                        disliked_keywords.append(word)

        for item in retrieved_results:
            job = item.job
            job_text_lower = (job.title + " " + job.description + " " + job.company).lower()

            # Memory constraint filtering / hard penalty
            violates_memory = False
            for bad_kw in disliked_keywords:
                if bad_kw in job_text_lower:
                    violates_memory = True
                    break
            if violates_memory:
                continue  # Exclude job based on candidate's remembered constraints

            # Factor A: Semantic & Skill fit (0 - 50 points)
            req_skills = [s.lower() for s in (job.required_skills or [])]
            matched_skills = [s for s in job.required_skills if s.lower() in cand_skills_lower]
            missing_skills = [s for s in job.required_skills if s.lower() not in cand_skills_lower]
            all_identified_gaps.update(missing_skills)

            skill_ratio = len(matched_skills) / max(len(req_skills), 1)
            semantic_norm = max(0.0, min(1.0, (item.semantic_score + 1.0) / 2.0))
            tech_fit_score = (0.5 * skill_ratio + 0.5 * semantic_norm) * 50.0

            # Factor B: Experience Level Alignment (0 - 20 points)
            exp_score = 15.0
            job_level = (job.experience_level or "mid").lower()
            if job_level == "junior":
                exp_score = 20.0 if experience_years <= 3 else 12.0
            elif job_level == "mid":
                exp_score = 20.0 if 2 <= experience_years <= 7 else 15.0
            elif job_level == "senior":
                exp_score = 20.0 if experience_years >= 5 else (10.0 if experience_years >= 3 else 5.0)
            elif job_level == "lead":
                exp_score = 20.0 if experience_years >= 8 else (8.0 if experience_years >= 5 else 3.0)

            # Factor C: Compensation Fit (0 - 20 points)
            comp_score = 15.0
            salary_fit_desc = "Competitive"
            if min_salary and job.max_salary:
                if job.max_salary >= min_salary:
                    comp_score = 20.0
                    salary_fit_desc = f"Exceeds/Meets target (${min_salary:,} desired vs up to ${job.max_salary:,} offered)"
                elif job.max_salary >= min_salary * 0.9:
                    comp_score = 12.0
                    salary_fit_desc = f"Slightly below target (${min_salary:,} vs max ${job.max_salary:,})"
                else:
                    comp_score = 5.0
                    salary_fit_desc = f"Below target compensation range (${job.min_salary or 0:,} - ${job.max_salary:,})"
            elif job.min_salary:
                salary_fit_desc = f"Salary offered: ${job.min_salary:,} - ${job.max_salary or 'Open':,}"

            # Factor D: Work Mode / Remote Fit (0 - 10 points)
            mode_score = 10.0
            culture_fit_desc = f"Matches work mode: {job.remote_type}"
            if remote_pref:
                if remote_pref.lower() == job.remote_type.lower():
                    mode_score = 10.0
                elif remote_pref.lower() == "remote" and job.remote_type.lower() != "remote":
                    mode_score = 2.0
                    culture_fit_desc = f"Job is {job.remote_type}, but candidate prefers remote"

            # Final Personalized Fit Score (0 - 100)
            final_fit = round(tech_fit_score + exp_score + comp_score + mode_score, 1)
            final_fit = min(100.0, max(0.0, final_fit))

            # Personalized Rationale
            match_reasons = []
            if matched_skills:
                match_reasons.append(f"Strong overlap in core skills: {', '.join(matched_skills[:4])}")
            if final_fit >= 80:
                match_reasons.append(f"High seniority and role profile alignment for {job.title}")
            if job.remote_type == "remote":
                match_reasons.append("Fully remote flexibility")
            why_text = ". ".join(match_reasons) + "."

            rec = JobMatchRecommendation(
                job_id=job.id,
                title=job.title,
                company=job.company,
                location=job.location,
                remote_type=job.remote_type,
                fit_score=final_fit,
                why_recommended=why_text,
                matching_skills=matched_skills,
                missing_skills=missing_skills,
                salary_fit=salary_fit_desc,
                culture_or_remote_fit=culture_fit_desc
            )
            scored_recommendations.append(rec)

        # Stage 2 Final Ranking
        scored_recommendations.sort(key=lambda x: x.fit_score, reverse=True)
        top_matches = scored_recommendations[:top_k]

        # Executive summary
        summary = (
            f"Analyzed {len(retrieved_results)} candidate positions. Selected {len(top_matches)} top-tier matches "
            f"for {candidate_name} tailored to {experience_years}y experience and target roles."
        )

        return StructuredMatchReport(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            executive_summary=summary,
            top_matches=top_matches,
            identified_gaps=sorted(list(all_identified_gaps))[:6],
            recommended_upskilling=[f"Gain proficiency in {skill}" for skill in sorted(list(all_identified_gaps))[:4]]
        )


_recommender_instance: Optional[TwoStageRecommender] = None


def get_recommender() -> TwoStageRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = TwoStageRecommender()
    return _recommender_instance
