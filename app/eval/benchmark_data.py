"""
Golden Evaluation Benchmark Dataset for TalentMatch AI.
Used for automated LLM output quality, RAG retrieval recall, ranking MRR,
and constraint adherence evaluations.
"""
from typing import List
from app.schemas.eval import EvalCase

BENCHMARK_CASES: List[EvalCase] = [
    EvalCase(
        id="case_01_senior_python_ai",
        description="Senior AI backend engineer seeking high-comp remote role",
        query="Looking for senior Python and LLM backend jobs that are remote and pay over $150k",
        candidate_profile={
            "name": "Jordan Lee",
            "headline": "Senior AI Backend Engineer",
            "years_of_experience": 6.0,
            "skills": ["Python", "FastAPI", "LLMs", "Vector Databases", "Docker", "PyTorch"],
            "desired_roles": ["AI Engineer", "Backend Developer"],
            "remote_preference": "remote",
            "min_desired_salary": 160000
        },
        candidate_memories=["Prefers AI research or agentic startups"],
        golden_job_ids=["job_01_ai_eng", "job_03_rag_backend", "job_07_agent_dev"],
        negative_job_ids=["job_05_crypto_solidity", "job_06_onsite_junior"],
        expected_tools=["search_jobs", "match_candidate_profile"]
    ),
    EvalCase(
        id="case_02_memory_constraint_adherence",
        description="Candidate with hard negative constraint against crypto/blockchain companies",
        query="Find me backend engineer roles, but remember my strict refusal for crypto",
        candidate_profile={
            "name": "Samira Khan",
            "headline": "Backend Systems Engineer",
            "years_of_experience": 4.5,
            "skills": ["Python", "Go", "PostgreSQL", "Docker", "Kubernetes", "Redis"],
            "desired_roles": ["Backend Systems Engineer"],
            "remote_preference": "remote",
            "min_desired_salary": 140000
        },
        candidate_memories=["Candidate strictly refuses crypto and blockchain roles"],
        golden_job_ids=["job_04_cloud_backend", "job_08_distributed_systems"],
        negative_job_ids=["job_05_crypto_solidity"],
        expected_tools=["remember_candidate_fact", "search_jobs"]
    ),
    EvalCase(
        id="case_03_junior_entry_level",
        description="Junior developer should not be matched with staff/lead roles",
        query="Entry level junior developer positions in Python or Frontend",
        candidate_profile={
            "name": "Leo Martin",
            "headline": "Junior Software Engineer",
            "years_of_experience": 1.0,
            "skills": ["Python", "JavaScript", "HTML/CSS", "Git", "SQL"],
            "desired_roles": ["Junior Software Engineer", "Junior Frontend Developer"],
            "remote_preference": "hybrid",
            "min_desired_salary": 75000
        },
        candidate_memories=[],
        golden_job_ids=["job_06_onsite_junior", "job_09_junior_frontend"],
        negative_job_ids=["job_01_ai_eng", "job_08_distributed_systems"],
        expected_tools=["search_jobs"]
    ),
    EvalCase(
        id="case_04_data_ml_ops",
        description="MLOps engineer focusing on Kubernetes, PyTorch, and pipeline orchestration",
        query="MLOps and platform engineering jobs with Kubernetes and model serving",
        candidate_profile={
            "name": "Chen Wei",
            "headline": "MLOps & Infrastructure Engineer",
            "years_of_experience": 5.0,
            "skills": ["Python", "Kubernetes", "PyTorch", "Docker", "MLflow", "Triton"],
            "desired_roles": ["MLOps Engineer", "AI Infrastructure Engineer"],
            "remote_preference": "remote",
            "min_desired_salary": 165000
        },
        candidate_memories=[],
        golden_job_ids=["job_02_mlops_platform", "job_07_agent_dev"],
        negative_job_ids=["job_09_junior_frontend"],
        expected_tools=["search_jobs"]
    )
]
