from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("JOB_CHROMA_PERSIST_DIR", "/tmp/resumetooffer_eval_chroma")

from app.agent_orchestrator import (  # noqa: E402
    canonicalize_goal,
    fallback_goal,
    force_partial_rerun_for_feedback,
    parse_agent_goal,
)
from app.input_router import fallback_route, route_user_input  # noqa: E402
from app.job_matching import (  # noqa: E402
    bm25_retrieve,
    chroma_retrieve,
    merge_candidates,
    metadata_filter_jobs,
    match_jobs,
    user_document,
)
from app.model_logging import get_model_metrics, reset_model_metrics  # noqa: E402
from app.schemas import Job, JobMatchRequest, Profile, Project  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / "reports"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_tools(plan_steps: list[Any]) -> list[str]:
    return [
        step.step_id
        for step in plan_steps
        if step.step_id != "goal_understanding"
    ]


def plan_for_case(case: dict[str, Any]) -> Any:
    draft = canonicalize_goal(fallback_goal(case["message"]), case["message"])
    if case.get("feedback_rerun"):
        force_partial_rerun_for_feedback(draft)
    return draft


def live_plan_for_case(case: dict[str, Any]) -> Any:
    draft = parse_agent_goal(case["message"])
    if case.get("feedback_rerun"):
        force_partial_rerun_for_feedback(draft)
    return draft


def require_live_environment() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Live eval requires OPENAI_API_KEY. Use --mode offline for local deterministic eval.")


def elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def evaluate_intent(mode: str) -> dict[str, Any]:
    cases = load_json(EVAL_DIR / "intent_routing_cases.json")
    rows = []
    correct = 0
    latencies = []
    for case in cases:
        started_at = time.perf_counter()
        actual = (
            route_user_input(case["message"], Profile(), [])
            if mode == "live"
            else fallback_route(case["message"])
        )
        latencies.append(elapsed_ms(started_at))
        passed = (
            actual.intent == case["expected_intent"]
            and actual.route == case["expected_route"]
        )
        correct += int(passed)
        rows.append(
            {
                "id": case["id"],
                "passed": passed,
                "expected_intent": case["expected_intent"],
                "actual_intent": actual.intent,
                "expected_route": case["expected_route"],
                "actual_route": actual.route,
                "latency_ms": latencies[-1],
            }
        )
    return {
        "metric": "Intent Accuracy",
        "mode": mode,
        "score": correct / len(cases) if cases else 0.0,
        "passed": correct,
        "total": len(cases),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "cases": rows,
    }


def evaluate_planner(mode: str) -> dict[str, Any]:
    cases = load_json(EVAL_DIR / "planner_cases.json")
    rows = []
    correct = 0
    latencies = []
    for case in cases:
        started_at = time.perf_counter()
        draft = live_plan_for_case(case) if mode == "live" else plan_for_case(case)
        latencies.append(elapsed_ms(started_at))
        actual_tools = selected_tools(draft.task_plan.steps)
        expected_tools = case["expected_tools"]
        target_ok = True
        if "expected_target_direction" in case:
            target_ok = draft.target_direction == case["expected_target_direction"]
        tools_ok = set(actual_tools) == set(expected_tools)
        passed = target_ok and tools_ok
        correct += int(passed)
        rows.append(
            {
                "id": case["id"],
                "passed": passed,
                "expected_target_direction": case.get("expected_target_direction", ""),
                "actual_target_direction": draft.target_direction,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "latency_ms": latencies[-1],
            }
        )
    return {
        "metric": "Tool Accuracy",
        "mode": mode,
        "score": correct / len(cases) if cases else 0.0,
        "passed": correct,
        "total": len(cases),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "cases": rows,
    }


def evaluate_tool_traces(mode: str) -> dict[str, Any]:
    cases = load_json(EVAL_DIR / "tool_trace_cases.json")
    rows = []
    correct = 0
    latencies = []
    for case in cases:
        started_at = time.perf_counter()
        draft = live_plan_for_case(case) if mode == "live" else plan_for_case(case)
        latencies.append(elapsed_ms(started_at))
        actual_trace = [
            {
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "depends_on": step.depends_on,
            }
            for step in draft.task_plan.steps
        ]
        passed = actual_trace == case["expected_trace"]
        correct += int(passed)
        rows.append(
            {
                "id": case["id"],
                "passed": passed,
                "expected_trace": case["expected_trace"],
                "actual_trace": actual_trace,
                "latency_ms": latencies[-1],
            }
        )
    return {
        "metric": "Trace Accuracy",
        "mode": mode,
        "score": correct / len(cases) if cases else 0.0,
        "passed": correct,
        "total": len(cases),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "cases": rows,
    }


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = ranked_ids[:k]
    return len([job_id for job_id in retrieved if job_id in relevant_ids]) / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def dcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    score = 0.0
    for index, job_id in enumerate(ranked_ids[:k], start=1):
        relevance = 1.0 if job_id in relevant_ids else 0.0
        if relevance:
            score += relevance / math.log2(index + 1)
    return score


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    ideal_count = min(len(relevant_ids), k)
    ideal_ids = list(relevant_ids)[:ideal_count]
    ideal = dcg_at_k(ideal_ids, relevant_ids, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant_ids, k) / ideal


def load_seed_jobs() -> list[Job]:
    return [
        Job.model_validate({**item, "status": item.get("status", "active")})
        for item in load_json(BACKEND_DIR / "data" / "jobs_seed.json")
    ]


def retrieval_source_comparison(
    profile: Profile,
    projects: list[Project],
    jobs: list[Job],
    request: JobMatchRequest,
    relevant_ids: set[str],
) -> dict[str, Any]:
    filtered_jobs, _ = metadata_filter_jobs(jobs, request, profile)
    jobs_by_id = {job.id: job for job in filtered_jobs}
    query = user_document(profile, projects, None, request.target_direction)
    bm25_hits = bm25_retrieve(query, filtered_jobs, 50)
    vector_hits = chroma_retrieve(query, filtered_jobs, 50)
    hybrid_candidates = merge_candidates(jobs_by_id, bm25_hits, vector_hits)
    bm25_ids = [hit.job_id for hit in bm25_hits]
    vector_ids = [hit.job_id for hit in vector_hits]
    hybrid_ids = [
        job_id
        for job_id, _scores in sorted(
            hybrid_candidates.items(),
            key=lambda item: item[1].get("fusion", 0.0),
            reverse=True,
        )
    ]
    return {
        "metadata_candidate_count": len(filtered_jobs),
        "bm25_candidate_count": len(bm25_ids),
        "vector_candidate_count": len(vector_ids),
        "hybrid_candidate_count": len(hybrid_ids),
        "hybrid_overlap_count": len(set(bm25_ids) & set(vector_ids)),
        "bm25_recall_at_50": recall_at_k(bm25_ids, relevant_ids, 50),
        "vector_recall_at_50": recall_at_k(vector_ids, relevant_ids, 50),
        "hybrid_recall_at_50": recall_at_k(hybrid_ids, relevant_ids, 50),
    }


def evaluate_ranking(mode: str) -> dict[str, Any]:
    cases = load_json(EVAL_DIR / "ranking_labeled_dataset.json")
    jobs = load_seed_jobs()
    rows = []
    precision_scores = []
    ndcg_scores = []
    recall_10_scores = []
    bm25_recall_50_scores = []
    vector_recall_50_scores = []
    hybrid_recall_50_scores = []
    latencies = []
    for case in cases:
        profile = Profile.model_validate(case["profile"])
        projects = [Project.model_validate(project) for project in case.get("projects", [])]
        request = JobMatchRequest.model_validate(
            {
                **case["request"],
                "top_k": 10,
                "llm_candidate_count": case["request"].get("llm_candidate_count", 0 if mode == "offline" else 5),
            }
        )
        started_at = time.perf_counter()
        response = match_jobs(profile, projects, None, jobs, request)
        latencies.append(elapsed_ms(started_at))
        ranked_ids = [match.job.id for match in response.matches]
        relevant_ids = set(case["relevant_job_ids"])
        precision = precision_at_k(ranked_ids, relevant_ids, 10)
        recall_10 = recall_at_k(ranked_ids, relevant_ids, 10)
        ndcg = ndcg_at_k(ranked_ids, relevant_ids, 10)
        source_comparison = retrieval_source_comparison(profile, projects, jobs, request, relevant_ids)
        precision_scores.append(precision)
        recall_10_scores.append(recall_10)
        ndcg_scores.append(ndcg)
        bm25_recall_50_scores.append(source_comparison["bm25_recall_at_50"])
        vector_recall_50_scores.append(source_comparison["vector_recall_at_50"])
        hybrid_recall_50_scores.append(source_comparison["hybrid_recall_at_50"])
        rows.append(
            {
                "id": case["id"],
                "precision_at_10": precision,
                "recall_at_10": recall_10,
                "ndcg_at_10": ndcg,
                "retrieval_source_comparison": source_comparison,
                "ranked_ids": ranked_ids,
                "relevant_job_ids": sorted(relevant_ids),
                "latency_ms": latencies[-1],
            }
        )
    return {
        "metric": "Ranking Evaluation",
        "mode": mode,
        "precision_at_10": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        "recall_at_10": sum(recall_10_scores) / len(recall_10_scores) if recall_10_scores else 0.0,
        "ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "source_comparison": {
            "bm25_recall_at_50": sum(bm25_recall_50_scores) / len(bm25_recall_50_scores)
            if bm25_recall_50_scores
            else 0.0,
            "vector_recall_at_50": sum(vector_recall_50_scores) / len(vector_recall_50_scores)
            if vector_recall_50_scores
            else 0.0,
            "hybrid_recall_at_50": sum(hybrid_recall_50_scores) / len(hybrid_recall_50_scores)
            if hybrid_recall_50_scores
            else 0.0,
        },
        "total": len(cases),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "cases": rows,
    }


def evaluate_partial_rerun_reuse(mode: str) -> dict[str, Any]:
    cases = [case for case in load_json(EVAL_DIR / "planner_cases.json") if case.get("feedback_rerun")]
    rows = []
    reuse_rates = []
    expected_rerun_tools = {"job_ranking", "skill_gap_analysis", "next_step_plan"}
    reusable_steps = {"project_profiling", "career_direction", "resume_generation"}
    for case in cases:
        started_at = time.perf_counter()
        draft = live_plan_for_case(case) if mode == "live" else plan_for_case(case)
        latency_ms = elapsed_ms(started_at)
        actual_tools = set(selected_tools(draft.task_plan.steps))
        rerun_tools = actual_tools & expected_rerun_tools
        protected_reused_tools = reusable_steps - actual_tools
        denominator = len(rerun_tools) + len(protected_reused_tools)
        reuse_rate = len(protected_reused_tools) / denominator if denominator else 0.0
        reuse_rates.append(reuse_rate)
        rows.append(
            {
                "id": case["id"],
                "actual_tools": sorted(actual_tools),
                "rerun_tools": sorted(rerun_tools),
                "reused_tools": sorted(protected_reused_tools),
                "reuse_rate": reuse_rate,
                "latency_ms": latency_ms,
            }
        )
    return {
        "metric": "Partial Rerun Reuse Rate",
        "mode": mode,
        "reuse_rate": sum(reuse_rates) / len(reuse_rates) if reuse_rates else 0.0,
        "total": len(cases),
        "cases": rows,
    }


def print_summary(results: dict[str, Any], include_cases: bool) -> None:
    print(f"Agent Evaluation Harness ({results['metadata']['mode']})")
    print("========================")
    intent = results["intent"]
    planner = results["planner"]
    trace = results["tool_traces"]
    ranking = results["ranking"]
    print(f"Intent Accuracy: {intent['score']:.3f} ({intent['passed']}/{intent['total']})")
    print(f"Tool Accuracy: {planner['score']:.3f} ({planner['passed']}/{planner['total']})")
    print(f"Trace Accuracy: {trace['score']:.3f} ({trace['passed']}/{trace['total']})")
    print(f"Precision@10: {ranking['precision_at_10']:.3f}")
    print(f"Recall@10: {ranking['recall_at_10']:.3f}")
    print(f"NDCG@10: {ranking['ndcg_at_10']:.3f}")
    source = ranking["source_comparison"]
    print(
        "Recall@50 BM25/Vector/Hybrid: "
        f"{source['bm25_recall_at_50']:.3f}/"
        f"{source['vector_recall_at_50']:.3f}/"
        f"{source['hybrid_recall_at_50']:.3f}"
    )
    model_metrics = results["model_metrics"]
    print(f"Fallback Rate: {model_metrics['fallback_rate']:.3f} ({model_metrics['fallbacks']}/{model_metrics['starts']})")
    partial = results["partial_rerun"]
    print(f"Partial Rerun Reuse Rate: {partial['reuse_rate']:.3f} ({partial['total']} cases)")
    if results.get("output_path"):
        print(f"Report: {results['output_path']}")
    if include_cases:
        print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline agent and ranking evaluations.")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline", help="offline uses deterministic fallback paths; live calls model-backed router/planner paths.")
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    parser.add_argument("--cases", action="store_true", help="Print full per-case details after summary.")
    parser.add_argument("--output", help="Write full JSON report to this path.")
    args = parser.parse_args()

    if args.mode == "live":
        require_live_environment()

    reset_model_metrics()
    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "openai_api_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
            "job_chroma_persist_dir": os.getenv("JOB_CHROMA_PERSIST_DIR", ""),
        },
        "intent": evaluate_intent(args.mode),
        "planner": evaluate_planner(args.mode),
        "tool_traces": evaluate_tool_traces(args.mode),
        "ranking": evaluate_ranking(args.mode),
        "partial_rerun": evaluate_partial_rerun_reuse(args.mode),
    }
    results["model_metrics"] = get_model_metrics()
    timestamp = results["metadata"]["timestamp"].replace(":", "").replace("+", "Z")
    output_path = Path(args.output) if args.output else REPORT_DIR / f"eval_{args.mode}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["output_path"] = str(output_path)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_summary(results, include_cases=args.cases)


if __name__ == "__main__":
    main()
