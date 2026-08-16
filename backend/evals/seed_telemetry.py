from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("JOB_CHROMA_PERSIST_DIR", "/tmp/resumetooffer_eval_chroma")
os.environ["TELEMETRY_DB_ENABLED"] = "1"

from app.agent.planner import (  # noqa: E402
    canonicalize_goal,
    fallback_goal,
    force_partial_rerun_for_feedback,
    parse_agent_goal,
)
from app.database import create_tables  # noqa: E402
from app.job_matching import match_jobs  # noqa: E402
from app.schemas import Job, JobMatchRequest, Profile, Project  # noqa: E402
from app.telemetry import log_event, telemetry_context  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
REUSABLE_PARTIAL_STEPS = {"project_profiling", "career_direction", "resume_generation"}
RERUN_PARTIAL_STEPS = {"job_ranking", "skill_gap_analysis", "next_step_plan"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_tools(plan_steps: list[Any]) -> list[str]:
    return [
        step.step_id
        for step in plan_steps
        if step.step_id != "goal_understanding"
    ]


def plan_for_case(case: dict[str, Any], mode: str) -> Any:
    draft = (
        parse_agent_goal(case["message"])
        if mode == "live"
        else canonicalize_goal(fallback_goal(case["message"]), case["message"])
    )
    if case.get("feedback_rerun"):
        force_partial_rerun_for_feedback(draft)
    return draft


def load_seed_jobs() -> list[Job]:
    return [
        Job.model_validate({**item, "status": item.get("status", "active")})
        for item in load_json(BACKEND_DIR / "data" / "jobs_seed.json")
    ]


def replay_planner_cases(mode: str, repeat: int) -> int:
    cases = load_json(EVAL_DIR / "planner_cases.json")
    event_count = 0
    for iteration in range(repeat):
        for case in cases:
            agent_run_id = str(uuid4())
            with telemetry_context(agent_run_id):
                draft = plan_for_case(case, mode)
                tools = selected_tools(draft.task_plan.steps)
                is_feedback = bool(case.get("feedback_rerun"))
                reused_steps = sorted(REUSABLE_PARTIAL_STEPS - set(tools)) if is_feedback else []
                rerun_steps = sorted(RERUN_PARTIAL_STEPS & set(tools)) if is_feedback else []
                log_event(
                    "agent_run_summary",
                    agent_run_id=agent_run_id,
                    seed_case_id=case["id"],
                    seed_iteration=iteration + 1,
                    profile_id=1,
                    target_direction=draft.target_direction,
                    selected_tools=tools,
                    step_count=len(draft.task_plan.steps),
                    reused_step_count=len(reused_steps),
                    rerun_step_count=len(rerun_steps),
                    is_feedback_rerun=is_feedback,
                    latest_resume_id="seed-resume" if "resume_generation" not in tools else "",
                    latest_job_match_count=10 if "job_ranking" in tools else 0,
                    gap_severity="seed",
                )
                event_count += 1

                for step in draft.task_plan.steps:
                    log_event(
                        "task_step_trace",
                        agent_run_id=agent_run_id,
                        seed_case_id=case["id"],
                        seed_iteration=iteration + 1,
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        depends_on=step.depends_on,
                        status="completed",
                        rerun_policy=step.rerun_policy,
                        input_refs=step.input_refs,
                        output_refs=step.output_refs,
                        is_reused=step.step_id in reused_steps,
                        is_rerun=step.step_id in rerun_steps,
                    )
                    event_count += 1

                if is_feedback:
                    denominator = len(reused_steps) + len(rerun_steps)
                    log_event(
                        "partial_rerun_event",
                        agent_run_id=agent_run_id,
                        seed_case_id=case["id"],
                        seed_iteration=iteration + 1,
                        feedback_text=case["message"],
                        changed_preferences={
                            "target_direction": draft.target_direction,
                            "locations": draft.locations,
                            "levels": draft.levels,
                            "role_families": draft.role_families,
                        },
                        reused_steps=reused_steps,
                        rerun_steps=rerun_steps,
                        saved_step_count=len(reused_steps),
                        reuse_rate=len(reused_steps) / denominator if denominator else 0.0,
                    )
                    event_count += 1
    return event_count


def replay_ranking_cases(mode: str, repeat: int) -> int:
    cases = load_json(EVAL_DIR / "ranking_labeled_dataset.json")
    jobs = load_seed_jobs()
    event_count = 0
    for iteration in range(repeat):
        for case in cases:
            profile = Profile.model_validate(case["profile"])
            projects = [Project.model_validate(project) for project in case.get("projects", [])]
            request = JobMatchRequest.model_validate(
                {
                    **case["request"],
                    "top_k": 10,
                    "llm_candidate_count": case["request"].get(
                        "llm_candidate_count",
                        0 if mode == "offline" else 5,
                    ),
                }
            )
            agent_run_id = str(uuid4())
            with telemetry_context(agent_run_id):
                log_event(
                    "agent_run_summary",
                    agent_run_id=agent_run_id,
                    seed_case_id=case["id"],
                    seed_iteration=iteration + 1,
                    profile_id=1,
                    target_direction=request.target_direction,
                    selected_tools=["job_ranking"],
                    step_count=1,
                    reused_step_count=0,
                    rerun_step_count=0,
                    is_feedback_rerun=False,
                    latest_resume_id="",
                    latest_job_match_count=0,
                    gap_severity="seed",
                )
                match_jobs(profile, projects, None, jobs, request)
                event_count += 2
    return event_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay eval cases into telemetry database tables.")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "live" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Live telemetry seeding requires OPENAI_API_KEY.")

    create_tables()
    repeat = max(1, args.repeat)
    planner_events = replay_planner_cases(args.mode, repeat)
    ranking_events = replay_ranking_cases(args.mode, repeat)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "repeat": repeat,
                "planner_related_events": planner_events,
                "ranking_related_events": ranking_events,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
