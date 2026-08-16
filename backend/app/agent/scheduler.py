import re
from typing import Any

from app.schemas import AgentState, GeneratedResume, TaskPlan, TaskPlanStep, WorkflowStepReuse


PARTIAL_RERUN_STEP_IDS = {"job_ranking", "skill_gap_analysis", "next_step_plan"}


def ordered_steps(task_plan: TaskPlan) -> list[TaskPlanStep]:
    steps_by_id = {step.step_id: step for step in task_plan.steps}
    ordered: list[TaskPlanStep] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(step: TaskPlanStep) -> None:
        if step.step_id in visited:
            return
        if step.step_id in visiting:
            raise ValueError(f"TaskPlan contains a dependency cycle at {step.step_id}")
        visiting.add(step.step_id)
        for dependency_id in step.depends_on:
            dependency = steps_by_id.get(dependency_id)
            if dependency is not None:
                visit(dependency)
        visiting.remove(step.step_id)
        visited.add(step.step_id)
        ordered.append(step)

    for step in task_plan.steps:
        visit(step)
    return ordered


def update_task_status(task_plan: TaskPlan, step_id: str, status: str) -> None:
    for step in task_plan.steps:
        if step.step_id == step_id:
            step.status = status
            return


def message_updates_preferences(message: str, previous_state: AgentState) -> bool:
    if not previous_state.last_agent_run_id:
        return False
    return bool(
        re.search(
            r"这些岗位|岗位.*太偏|太偏|不想|不要|换成|改成|更想|更偏|我想做|偏前端|偏后端|preference|prefer",
            message,
            flags=re.IGNORECASE,
        )
    )


def force_partial_rerun_for_feedback(draft: Any) -> None:
    draft.run_project_profiling = False
    draft.run_career_direction = False
    draft.run_resume_generation = False
    draft.run_job_ranking = True
    draft.run_skill_gap_analysis = True
    draft.run_next_step_plan = True

    from app.agent.planner import refresh_task_plan

    refresh_task_plan(draft, feedback_rerun=True)


def rerun_steps_from_task_plan(task_plan: TaskPlan) -> list[TaskPlanStep]:
    return [
        step
        for step in ordered_steps(task_plan)
        if step.step_id in PARTIAL_RERUN_STEP_IDS
    ]


def reused_steps_for_partial_rerun(
    previous_state: AgentState,
    reusable_resume: GeneratedResume | None,
) -> list[WorkflowStepReuse]:
    reused = [
        WorkflowStepReuse(
            step_id="project_profiling",
            tool_name="project_profiling_tool",
            source_refs=["profile", "projects"],
            output_refs=["projects"],
            reason="用户反馈只调整岗位偏好，未提供新的项目经历，因此保留已有 Profile/Projects。",
        )
    ]
    if previous_state.last_target_direction:
        reused.append(
            WorkflowStepReuse(
                step_id="career_direction",
                tool_name="career_direction_tool",
                source_refs=["agent_state.last_target_direction", "agent_state.preference"],
                output_refs=["goal.target_direction", "goal.role_families"],
                reason="本次反馈直接给出目标偏好，跳过职业方向重新推荐。",
            )
        )
    if reusable_resume:
        reused.append(
            WorkflowStepReuse(
                step_id="resume_generation",
                tool_name="resume_generation_tool",
                source_refs=["agent_state.latest_resume_id"],
                output_refs=["generated_resume"],
                reason=f"复用最近简历版本 {reusable_resume.id}，不重新生成简历。",
            )
        )
    elif previous_state.latest_resume_id:
        reused.append(
            WorkflowStepReuse(
                step_id="resume_generation",
                tool_name="resume_generation_tool",
                source_refs=["agent_state.latest_resume_id"],
                output_refs=["agent_state.latest_resume_id"],
                reason=f"保留最近简历版本 {previous_state.latest_resume_id}，但本次未加载到对应简历内容。",
            )
        )
    return reused


def partial_rerun_context(
    task_plan: TaskPlan,
    previous_state: AgentState,
    reusable_resume: GeneratedResume | None,
    is_feedback_rerun: bool,
) -> tuple[list[WorkflowStepReuse], list[TaskPlanStep]]:
    if not is_feedback_rerun:
        return [], []
    return (
        reused_steps_for_partial_rerun(previous_state, reusable_resume),
        rerun_steps_from_task_plan(task_plan),
    )
