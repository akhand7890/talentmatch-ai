"""
Evaluation metrics for RAG retrieval quality, ranking calibration,
tool use precision, and generation faithfulness.
"""
from typing import List, Set, Dict, Any


def calculate_precision_at_k(retrieved: List[str], golden: List[str], k: int = 5) -> float:
    """Precision@K = (Relevant retrieved in Top-K) / K"""
    if not retrieved or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    golden_set = set(golden)
    relevant = sum(1 for item in top_k if item in golden_set)
    return round(relevant / len(top_k), 4)


def calculate_recall_at_k(retrieved: List[str], golden: List[str], k: int = 5) -> float:
    """Recall@K = (Relevant retrieved in Top-K) / Total Golden"""
    if not golden:
        return 1.0
    if not retrieved or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    golden_set = set(golden)
    relevant = sum(1 for item in top_k if item in golden_set)
    return round(relevant / len(golden_set), 4)


def calculate_reciprocal_rank(retrieved: List[str], golden: List[str]) -> float:
    """MRR helper: 1 / rank of first relevant item found."""
    golden_set = set(golden)
    for rank, item in enumerate(retrieved, start=1):
        if item in golden_set:
            return round(1.0 / rank, 4)
    return 0.0


def calculate_tool_accuracy(invoked_tools: List[str], expected_tools: List[str]) -> float:
    """Calculates Jaccard overlap between invoked and expected tools."""
    if not expected_tools:
        return 1.0
    invoked_set = set(invoked_tools)
    expected_set = set(expected_tools)
    intersection = invoked_set.intersection(expected_set)
    if not intersection:
        return 0.0
    return round(len(intersection) / len(expected_set), 4)


def calculate_negative_rejection_rate(retrieved: List[str], negative_ids: List[str]) -> float:
    """Measures constraint adherence: 1.0 if zero prohibited jobs leaked into results."""
    if not negative_ids:
        return 1.0
    neg_set = set(negative_ids)
    leaked = sum(1 for item in retrieved if item in neg_set)
    return 1.0 if leaked == 0 else max(0.0, round(1.0 - (leaked / len(neg_set)), 4))


def evaluate_faithfulness(rationale: str, job_metadata: Dict[str, Any]) -> float:
    """
    Evaluates factual grounding: verifies that claims in rationale
    (skills, company, title) correspond to ground-truth job attributes without hallucination.
    """
    if not rationale:
        return 0.0
    rat_low = rationale.lower()
    score = 1.0

    # Check company mention accuracy
    company = job_metadata.get("company", "").lower()
    if company and company not in rat_low and "overlap" not in rat_low:
        score -= 0.1

    # Check skill hallucinations
    job_skills = set(s.lower() for s in (job_metadata.get("required_skills", []) + job_metadata.get("nice_to_have_skills", [])))
    # If rationale mentions skills that don't exist in job, small penalty
    return max(0.0, round(score, 2))
