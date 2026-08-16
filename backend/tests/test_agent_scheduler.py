from app.agent.planner import canonicalize_goal, fallback_goal
from app.agent.scheduler import (
    force_partial_rerun_for_feedback,
    message_updates_preferences,
    ordered_steps,
    partial_rerun_context,
    rerun_steps_from_task_plan,
    reused_steps_for_partial_rerun,
)
from app.schemas import AgentState, GeneratedResume, TaskPlan, TaskPlanStep, UserPreference


def step_ids(steps: list[TaskPlanStep]) -> list[str]:
    return [step.step_id for step in steps]


def test_ordered_steps_respects_dependencies_not_list_order() -> None:
    task_plan = TaskPlan(
        steps=[
            TaskPlanStep(step_id="next_step_plan", tool_name="next_step_plan_tool", depends_on=["skill_gap_analysis"]),
            TaskPlanStep(step_id="goal_understanding", tool_name="agent_goal_parser"),
            TaskPlanStep(step_id="job_ranking", tool_name="job_matching_tool", depends_on=["goal_understanding"]),
            TaskPlanStep(step_id="skill_gap_analysis", tool_name="skill_gap_analysis_tool", depends_on=["job_ranking"]),
        ]
    )

    assert step_ids(ordered_steps(task_plan)) == [
        "goal_understanding",
        "job_ranking",
        "skill_gap_analysis",
        "next_step_plan",
    ]


def test_feedback_rerun_forces_downstream_task_plan() -> None:
    previous = AgentState(
        last_agent_run_id="run-1",
        preference=UserPreference(target_direction="Frontend Developer"),
    )
    assert message_updates_preferences("这些岗位太偏前端，我想做后端", previous)

    draft = canonicalize_goal(fallback_goal("这些岗位太偏前端，我想做后端"), "这些岗位太偏前端，我想做后端")
    force_partial_rerun_for_feedback(draft)

    assert not draft.run_project_profiling
    assert not draft.run_career_direction
    assert not draft.run_resume_generation
    assert draft.run_job_ranking
    assert draft.run_skill_gap_analysis
    assert draft.run_next_step_plan
    assert step_ids(draft.task_plan.steps) == [
        "goal_understanding",
        "job_ranking",
        "skill_gap_analysis",
        "next_step_plan",
    ]
    assert "feedback_memory.last_feedback" in draft.task_plan.steps[1].input_refs


def test_partial_rerun_context_reuses_upstream_and_reruns_downstream() -> None:
    previous = AgentState(
        last_target_direction="Backend Developer",
        latest_resume_id="resume-1",
    )
    resume = GeneratedResume(id="resume-1", target_direction="Backend Developer")
    draft = canonicalize_goal(
        fallback_goal("我想投北京初级后端岗位，推荐岗位、分析差距并给我学习计划"),
        "我想投北京初级后端岗位，推荐岗位、分析差距并给我学习计划",
    )
    force_partial_rerun_for_feedback(draft)

    reused_steps, rerun_steps = partial_rerun_context(
        draft.task_plan,
        previous,
        resume,
        is_feedback_rerun=True,
    )

    assert [step.step_id for step in reused_steps] == [
        "project_profiling",
        "career_direction",
        "resume_generation",
    ]
    assert step_ids(rerun_steps) == [
        "job_ranking",
        "skill_gap_analysis",
        "next_step_plan",
    ]


def test_partial_rerun_context_is_empty_without_feedback() -> None:
    previous = AgentState(last_target_direction="Backend Developer")
    task_plan = TaskPlan(steps=[TaskPlanStep(step_id="goal_understanding", tool_name="agent_goal_parser")])

    assert partial_rerun_context(task_plan, previous, None, is_feedback_rerun=False) == ([], [])


def test_reused_steps_keep_resume_ref_when_resume_body_is_missing() -> None:
    previous = AgentState(
        last_target_direction="Backend Developer",
        latest_resume_id="resume-1",
    )

    reused = reused_steps_for_partial_rerun(previous, None)

    assert [step.step_id for step in reused] == [
        "project_profiling",
        "career_direction",
        "resume_generation",
    ]
    assert reused[-1].output_refs == ["agent_state.latest_resume_id"]
    assert "resume-1" in reused[-1].reason


def test_rerun_steps_follow_dependency_order() -> None:
    task_plan = TaskPlan(
        steps=[
            TaskPlanStep(step_id="next_step_plan", tool_name="next_step_plan_tool", depends_on=["skill_gap_analysis"]),
            TaskPlanStep(step_id="skill_gap_analysis", tool_name="skill_gap_analysis_tool", depends_on=["job_ranking"]),
            TaskPlanStep(step_id="job_ranking", tool_name="job_matching_tool", depends_on=["goal_understanding"]),
            TaskPlanStep(step_id="goal_understanding", tool_name="agent_goal_parser"),
        ]
    )

    assert step_ids(rerun_steps_from_task_plan(task_plan)) == [
        "job_ranking",
        "skill_gap_analysis",
        "next_step_plan",
    ]
