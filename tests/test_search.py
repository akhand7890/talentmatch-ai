"""
Unit tests for embeddings and dynamic vector indexing.
"""
import pytest
from app.search.embeddings import FastLocalEmbeddingProvider
from app.search.indexer import JobsVectorIndex
from app.search.hybrid_search import HybridSearchEngine
from app.schemas.job import JobSearchFilter


def test_fast_local_embedding():
    provider = FastLocalEmbeddingProvider(dimension=128)
    vec1 = provider.embed_text("Senior Python and AI Engineer")
    vec2 = provider.embed_text("Senior Python and Machine Learning Developer")
    vec3 = provider.embed_text("Certified Public Accountant Tax Auditor")

    assert len(vec1) == 128
    assert len(vec2) == 128

    import numpy as np
    sim_similar = float(np.dot(vec1, vec2))
    sim_dissimilar = float(np.dot(vec1, vec3))

    # Semantic similarity between AI jobs should be noticeably higher than accounting
    assert sim_similar > sim_dissimilar


def test_dynamic_vector_indexing():
    provider = FastLocalEmbeddingProvider(dimension=128)
    index = JobsVectorIndex(embedding_provider=provider)

    # Ingest 2 jobs
    index.upsert_job(
        job_id="test_01",
        text="Senior Python Backend Developer with FastAPI and Postgres",
        metadata={"id": "test_01", "title": "Senior Python Dev", "is_active": True}
    )
    index.upsert_job(
        job_id="test_02",
        text="Lead React and Frontend Engineer with TypeScript",
        metadata={"id": "test_02", "title": "Lead React Dev", "is_active": True}
    )

    assert index.count() == 2

    # Query for Python
    results = index.search_vector("Python FastAPI developer", top_k=2)
    assert len(results) == 2
    assert results[0][0] == "test_01"  # First match should be python

    # Dynamic removal
    index.remove_job("test_01")
    assert index.count() == 1
    remaining = index.search_vector("Python", top_k=2)
    assert remaining[0][0] == "test_02"
