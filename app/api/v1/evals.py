"""
LLM Evaluation & Quality Benchmarking Endpoints.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import EvaluationRecord
from app.eval.evaluator import AgentEvaluator, generate_markdown_report
from app.schemas.eval import EvalRunSummary

router = APIRouter(prefix="/evals", tags=["LLM Output Quality & Benchmarking"])


@router.post("/run", response_model=EvalRunSummary)
def run_evaluation_suite(k: int = Query(5, ge=1, le=20)):
    """
    Triggers automated LLM evaluation pipeline across all benchmark cases.
    Measures Precision@K, Recall@K, MRR, Tool Accuracy, Constraint Adherence, and Faithfulness.
    """
    evaluator = AgentEvaluator()
    summary = evaluator.run_suite(k=k)
    return summary


@router.post("/run/report", response_class=Response)
def run_evaluation_report(k: int = Query(5, ge=1, le=20)):
    """
    Triggers evaluation suite and returns a rendered Markdown report.
    """
    evaluator = AgentEvaluator()
    summary = evaluator.run_suite(k=k)
    md_content = generate_markdown_report(summary)
    return Response(content=md_content, media_type="text/markdown")


@router.get("/history")
def get_evaluation_history(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Retrieves historical evaluation records from SQL database."""
    records = db.query(EvaluationRecord).order_by(EvaluationRecord.run_at.desc()).limit(limit).all()
    results = []
    for r in records:
        results.append({
            "id": r.id,
            "benchmark_name": r.benchmark_name,
            "precision_at_k": r.precision_at_k,
            "recall_at_k": r.recall_at_k,
            "mrr": r.mrr,
            "tool_selection_accuracy": r.tool_selection_accuracy,
            "faithfulness_score": r.faithfulness_score,
            "total_eval_cases": r.total_eval_cases,
            "run_at": r.run_at.isoformat()
        })
    return results
