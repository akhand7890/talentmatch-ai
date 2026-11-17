"""
CLI Evaluation Runner for TalentMatch AI.
Usage:
    python evaluate.py
    python evaluate.py --report
"""
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.eval.evaluator import AgentEvaluator, generate_markdown_report
from app.db.session import init_db
from app.main import seed_data_if_empty


def main():
    parser = argparse.ArgumentParser(description="Run TalentMatch AI LLM Evaluation Suite")
    parser.add_argument("--report", action="store_true", help="Print full markdown report")
    parser.add_argument("--k", type=int, default=5, help="Top-K for retrieval evaluation")
    args = parser.parse_args()

    print("\n🚀 Initializing database & index for evaluation run...")
    init_db()
    seed_data_if_empty()

    print(f"📊 Running LLM Quality Evaluation Suite (Top-K = {args.k})...\n")
    evaluator = AgentEvaluator()
    summary = evaluator.run_suite(k=args.k)

    if args.report:
        report_md = generate_markdown_report(summary)
        print(report_md)
        # Also save to disk
        out_file = Path("data") / f"{summary.run_id}_report.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"\n📄 Saved evaluation report to: {out_file.resolve()}")
    else:
        print(f"Verdict: {summary.quality_verdict}")
        print(f"Pass Rate: {summary.pass_rate_percent}% ({summary.passed_cases}/{summary.total_cases} cases passed)")
        print(f"Mean Reciprocal Rank (MRR): {summary.mean_reciprocal_rank}")
        print(f"Avg Precision@K: {summary.avg_precision_at_k}")
        print(f"Avg Recall@K: {summary.avg_recall_at_k}")
        print(f"Tool Selection Accuracy: {summary.avg_tool_accuracy}")
        print(f"Faithfulness Score: {summary.avg_faithfulness}")
        print(f"Average Latency: {summary.avg_latency_ms} ms")


if __name__ == "__main__":
    main()
