from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
)
from app.input_router import fallback_route  # noqa: E402
from app.job_matching import match_jobs  # noqa: E402
from app.schemas import Job, JobMatchRequest, Profile, Project  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent


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


def evaluate_intent() -> dict[str, Any]:
    cases = load_json(EVAL_DIR / "intent_routing_cases.json")
    rows = []
    correct = 0
    for case in cases:
        actual = fallback_route(case["message"])
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
            }
        )
    return {
        "metric": "Intent Accuracy",
        "score": correct / len(cases) if cases else 0.0,
        "passed": correct,
        "total": len(cases),
        "cases": rows,
    }


def evaluate_planner() -> dict[str, Any]:
    cases = load_json(EVAL_DIR / "planner_cases.json")
    rows = []
    correct = 0
    for case in cases:
        draft = plan_for_case(case)
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
            }
        )
    return {
        "metric": "Tool Accuracy",
        "score": correct / len(cases) if cases else 0.0,
        "passed": correct,
        "total": len(cases),
        "cases": rows,
    }


def evaluate_tool_traces() -> dict[str, Any]:
    cases = load_json(EVAL_DIR / "tool_trace_cases.json")
    rows = []
    correct = 0
    for case in cases:
        draft = plan_for_case(case)
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
            }
        )
    return {
        "metric": "Trace Accuracy",
        "score": correct / len(cases) if cases else 0.0,
        "passed": correct,
        "total": len(cases),
        "cases": rows,
    }


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = ranked_ids[:k]
    return len([job_id for job_id in retrieved if job_id in relevant_ids]) / k


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


def evaluate_ranking() -> dict[str, Any]:
    cases = load_json(EVAL_DIR / "ranking_labeled_dataset.json")
    jobs = load_seed_jobs()
    rows = []
    precision_scores = []
    ndcg_scores = []
    for case in cases:
        profile = Profile.model_validate(case["profile"])
        projects = [Project.model_validate(project) for project in case.get("projects", [])]
        request = JobMatchRequest.model_validate(
            {
                **case["request"],
                "top_k": 10,
                "llm_candidate_count": 0,
            }
        )
        response = match_jobs(profile, projects, None, jobs, request)
        ranked_ids = [match.job.id for match in response.matches]
        relevant_ids = set(case["relevant_job_ids"])
        precision = precision_at_k(ranked_ids, relevant_ids, 10)
        ndcg = ndcg_at_k(ranked_ids, relevant_ids, 10)
        precision_scores.append(precision)
        ndcg_scores.append(ndcg)
        rows.append(
            {
                "id": case["id"],
                "precision_at_10": precision,
                "ndcg_at_10": ndcg,
                "ranked_ids": ranked_ids,
                "relevant_job_ids": sorted(relevant_ids),
            }
        )
    return {
        "metric": "Ranking Evaluation",
        "precision_at_10": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        "ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "total": len(cases),
        "cases": rows,
    }


def print_summary(results: dict[str, Any], include_cases: bool) -> None:
    print("Agent Evaluation Harness")
    print("========================")
    intent = results["intent"]
    planner = results["planner"]
    trace = results["tool_traces"]
    ranking = results["ranking"]
    print(f"Intent Accuracy: {intent['score']:.3f} ({intent['passed']}/{intent['total']})")
    print(f"Tool Accuracy: {planner['score']:.3f} ({planner['passed']}/{planner['total']})")
    print(f"Trace Accuracy: {trace['score']:.3f} ({trace['passed']}/{trace['total']})")
    print(f"Precision@10: {ranking['precision_at_10']:.3f}")
    print(f"NDCG@10: {ranking['ndcg_at_10']:.3f}")
    if include_cases:
        print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline agent and ranking evaluations.")
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    parser.add_argument("--cases", action="store_true", help="Print full per-case details after summary.")
    args = parser.parse_args()

    results = {
        "intent": evaluate_intent(),
        "planner": evaluate_planner(),
        "tool_traces": evaluate_tool_traces(),
        "ranking": evaluate_ranking(),
    }
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_summary(results, include_cases=args.cases)


if __name__ == "__main__":
    main()
