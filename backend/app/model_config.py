import os
import warnings


warnings.filterwarnings(
    "ignore",
    message="Streaming with Pydantic response_format not yet supported.",
)

PROJECT_PROFILING_MODEL = os.getenv("PROJECT_PROFILING_MODEL", "gpt-5-nano")
CAREER_DIRECTION_MODEL = os.getenv("CAREER_DIRECTION_MODEL", "gpt-5-nano")
RESUME_GENERATION_MODEL = os.getenv("RESUME_GENERATION_MODEL", "gpt-5-mini")
JOB_MATCHING_MODEL = os.getenv("JOB_MATCHING_MODEL", "gpt-5-nano")
NEXT_STEP_PLAN_MODEL = os.getenv(
    "NEXT_STEP_PLAN_MODEL",
    os.getenv("LEARNING_PLAN_MODEL", "gpt-5-nano"),
)
AGENT_ORCHESTRATOR_MODEL = os.getenv("AGENT_ORCHESTRATOR_MODEL", "gpt-5-nano")
PROFILE_PROJECT_EDITING_MODEL = os.getenv("PROFILE_PROJECT_EDITING_MODEL", "gpt-5-nano")
INPUT_ROUTER_MODEL = os.getenv("INPUT_ROUTER_MODEL", "gpt-5-nano")


def configured_models() -> dict[str, str]:
    return {
        "project_profiling": PROJECT_PROFILING_MODEL,
        "career_direction_reason": CAREER_DIRECTION_MODEL,
        "resume_generation": RESUME_GENERATION_MODEL,
        "job_matching": JOB_MATCHING_MODEL,
        "next_step_plan": NEXT_STEP_PLAN_MODEL,
        "agent_orchestrator": AGENT_ORCHESTRATOR_MODEL,
        "profile_project_editing": PROFILE_PROJECT_EDITING_MODEL,
        "input_router": INPUT_ROUTER_MODEL,
    }
