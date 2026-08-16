from app.agent.planner import (
    AgentPlanDraft,
    canonicalize_goal,
    fallback_goal,
    force_partial_rerun_for_feedback,
    message_updates_preferences,
)
from app.agent.scheduler import reused_steps_for_partial_rerun
from app.schemas import AgentState, GeneratedResume, UserPreference


def step_ids(draft: AgentPlanDraft) -> list[str]:
    return [step.step_id for step in draft.task_plan.steps]


def test_planner_fallback_resume_only_uses_resume_generation() -> None:
    draft = canonicalize_goal(fallback_goal("帮我生成一版后端开发简历"), "帮我生成一版后端开发简历")

    assert draft.target_direction == "Backend Developer"
    assert step_ids(draft) == ["goal_understanding", "resume_generation"]


def test_planner_fallback_jobs_gap_plan_selects_downstream_tools() -> None:
    message = "我想投北京初级后端岗位，推荐岗位、分析差距并给我学习计划"
    draft = canonicalize_goal(fallback_goal(message), message)

    assert draft.target_direction == "Backend Developer"
    assert step_ids(draft) == [
        "goal_understanding",
        "job_ranking",
        "skill_gap_analysis",
        "next_step_plan",
    ]


def test_partial_rerun_for_feedback_keeps_only_downstream_steps() -> None:
    previous = AgentState(
        last_agent_run_id="run-1",
        preference=UserPreference(target_direction="Frontend Developer"),
    )
    assert message_updates_preferences("这些岗位太偏前端，我想做后端", previous)

    draft = canonicalize_goal(fallback_goal("这些岗位太偏前端，我想做后端"), "这些岗位太偏前端，我想做后端")
    force_partial_rerun_for_feedback(draft)

    assert draft.target_direction == "Backend Developer"
    assert step_ids(draft) == [
        "goal_understanding",
        "job_ranking",
        "skill_gap_analysis",
        "next_step_plan",
    ]
    assert not draft.run_resume_generation


def test_reused_steps_for_partial_rerun_reports_resume_reuse() -> None:
    previous = AgentState(
        last_target_direction="Backend Developer",
        latest_resume_id="resume-1",
    )
    resume = GeneratedResume(id="resume-1", target_direction="Backend Developer")

    reused = reused_steps_for_partial_rerun(previous, resume)

    assert [step.step_id for step in reused] == [
        "project_profiling",
        "career_direction",
        "resume_generation",
    ]
    assert reused[-1].status == "reused"
    assert "resume-1" in reused[-1].reason
