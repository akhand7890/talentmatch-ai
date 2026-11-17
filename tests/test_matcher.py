"""
Unit tests for 2-Stage Recommender and constraint personalization.
"""
import pytest
from app.search.indexer import JobsVectorIndex
from app.search.embeddings import FastLocalEmbeddingProvider
from app.search.hybrid_search import HybridSearchEngine
from app.recommender.matcher import TwoStageRecommender


@pytest.fixture
def test_recommender():
    provider = FastLocalEmbeddingProvider(dimension=128)
    index = JobsVectorIndex(embedding_provider=provider)

    # Add mock jobs
    index.upsert_job(
        "job_ai",
        "Senior AI Engineer Python LLMs FastAPI Remote 180k",
        {
            "id": "job_ai",
            "title": "Senior AI Engineer",
            "company": "AI Labs",
            "location": "Remote",
            "remote_type": "remote",
            "experience_level": "senior",
            "min_salary": 160000,
            "max_salary": 200000,
            "required_skills": ["Python", "LLMs", "FastAPI"],
            "is_active": True
        }
    )
    index.upsert_job(
        "job_crypto",
        "Lead Web3 Solidity Developer Crypto Smart Contracts Remote 190k",
        {
            "id": "job_crypto",
            "title": "Lead Web3 Developer",
            "company": "Crypto DeFi",
            "location": "Remote",
            "remote_type": "remote",
            "experience_level": "senior",
            "min_salary": 170000,
            "max_salary": 210000,
            "required_skills": ["Solidity", "Rust", "Crypto"],
            "is_active": True
        }
    )
    search_engine = HybridSearchEngine(index=index)
    return TwoStageRecommender(search_engine=search_engine)


def test_recommender_scoring_and_memory_constraint(test_recommender):
    candidate = {
        "id": "cand_1",
        "name": "Jordan",
        "years_of_experience": 6,
        "skills": ["Python", "FastAPI", "LLMs"],
        "desired_roles": ["AI Engineer"],
        "remote_preference": "remote",
        "min_desired_salary": 150000
    }

    # Test without constraint: should match AI job at the top
    report = test_recommender.match_candidate(candidate, memories=[], top_k=5)
    assert len(report.top_matches) > 0
    assert report.top_matches[0].job_id == "job_ai"
    assert report.top_matches[0].fit_score >= 80

    # Test WITH negative memory constraint: "strictly refuses crypto"
    report_constrained = test_recommender.match_candidate(
        candidate,
        memories=["strictly refuses crypto and web3 roles"],
        top_k=5
    )
    matched_job_ids = [m.job_id for m in report_constrained.top_matches]
    assert "job_ai" in matched_job_ids
    assert "job_crypto" not in matched_job_ids  # Negative constraint must filter out crypto!
