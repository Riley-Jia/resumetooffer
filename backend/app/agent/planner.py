import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.model_config import AGENT_ORCHESTRATOR_MODEL
from app.model_logging import invoke_model_with_logging, log_model_fallback
from app.schemas import AgentState, TaskPlan, TaskPlanStep


class AgentPlanDraft(BaseModel):
    target_direction: str = ""
    locations: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    role_families: list[str] = Field(default_factory=list)
    timeline_weeks: int | None = None
    project_notes: str = ""
    execution_plan: list[str] = Field(default_factory=list)
    run_project_profiling: bool = False
    run_career_direction: bool = False
    run_resume_generation: bool = False
    run_job_ranking: bool = False
    run_skill_gap_analysis: bool = False
    run_next_step_plan: bool = False
    task_plan: TaskPlan = Field(default_factory=TaskPlan)


AGENT_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Career Agent Orchestrator for a resume-to-offer product.
Parse the user's natural-language request into an execution goal.

Return only structured data. Do not invent profile facts, projects, education,
companies, or skills. project_notes should only contain actual project
experience details if the user provided them, not the user's target command.

Use these canonical role families when possible:
Backend, AI Application, Data Analyst, Full Stack, Frontend, Product Manager,
Graduate Software Engineer, Software Engineer.

Execution plan should usually include:
Project Profiling, Career Direction, Resume Generation, Job Ranking,
Skill Gap Analysis, Next Step Plan.

task_plan may be empty. The backend will convert selected run_* flags into a
validated TaskPlan with stable step ids, tool names, dependencies, rerun policy,
input refs, and output refs.

Before selecting tools, infer the minimum tool set needed for the user's request.
Set each run_* boolean independently:
- run_project_profiling: only if the user provides new project/work experience details
  that should be extracted into a project preview.
- run_career_direction: if the user asks for career direction recommendations,
  or if a downstream requested tool needs a target direction and the user did not provide one.
- run_resume_generation: if the user asks to generate, rewrite, update, or tailor a resume.
- run_job_ranking: if the user asks for job recommendations, job ranking, applications,
  or matching roles.
- run_skill_gap_analysis: if the user asks for gaps, missing skills, fit analysis,
  or comparison against jobs.
- run_next_step_plan: if the user asks for a learning plan, next steps, preparation plan,
  or if skill gap analysis is requested.

If the user only asks to generate a resume for a specified direction, run only
Resume Generation and any cheap prerequisite parsing. Do not run Job Ranking,
Skill Gap Analysis, or Next Step Plan.
""",
        ),
        ("human", "{message}"),
    ]
)


def message_updates_preferences(message: str, previous_state: AgentState) -> bool:
    from app.agent.scheduler import message_updates_preferences as scheduler_message_updates_preferences

    return scheduler_message_updates_preferences(message, previous_state)


def force_partial_rerun_for_feedback(draft: AgentPlanDraft) -> None:
    from app.agent.scheduler import force_partial_rerun_for_feedback as scheduler_force_partial_rerun_for_feedback

    scheduler_force_partial_rerun_for_feedback(draft)


def fallback_goal(message: str) -> AgentPlanDraft:
    lowered = message.lower()
    locations: list[str] = []
    for city in ["Sydney", "Remote", "北京", "上海", "深圳"]:
        if city.lower() in lowered or city in message:
            locations.append(city)

    levels: list[str] = []
    if "intern" in lowered or "实习" in message:
        levels.append("Intern")
    if "graduate" in lowered or "校招" in message or "应届" in message:
        levels.append("Graduate")
    if "junior" in lowered or "初级" in message:
        levels.append("Junior")

    role_families: list[str] = []
    target_direction = ""
    role_checks = [
        ("backend", "Backend", "Backend Developer"),
        ("后端", "Backend", "Backend Developer"),
        ("ai application", "AI Application", "AI Application Developer"),
        ("llm", "AI Application", "AI Application Developer"),
        ("data analyst", "Data Analyst", "Data Analyst"),
        ("数据分析", "Data Analyst", "Data Analyst"),
        ("full stack", "Full Stack", "Full Stack Developer"),
        ("frontend", "Frontend", "Frontend Developer"),
        ("前端", "Frontend", "Frontend Developer"),
        ("product", "Product Manager", "Product Manager"),
        ("产品", "Product Manager", "Product Manager"),
    ]
    for keyword, family, direction in role_checks:
        if keyword in lowered or keyword in message:
            if family not in role_families:
                role_families.append(family)
            target_direction = target_direction or direction

    timeline_weeks = None
    weeks_match = re.search(r"(\d+)\s*(?:周|week|weeks)", lowered)
    if weeks_match:
        timeline_weeks = int(weeks_match.group(1))

    tool_plan = infer_tool_plan_from_message(message, bool(target_direction))
    return AgentPlanDraft(
        target_direction=target_direction,
        locations=locations,
        levels=levels,
        role_families=role_families,
        timeline_weeks=timeline_weeks,
        execution_plan=execution_plan_from_tools(tool_plan),
        **tool_plan,
    )


def default_execution_plan() -> list[str]:
    return [
        "理解用户目标",
        "Project Profiling: 如果用户提供了新的项目经历，抽取结构化预览",
        "Career Direction: 根据 Profile 和 Projects 生成/覆盖职业方向",
        "Resume Generation: 选择相关项目并生成新版简历",
        "Job Ranking: 根据目标城市、职级、方向推荐岗位",
        "Skill Gap Analysis: 分析 Top3 岗位缺口",
        "Next Step Plan: 根据 gap 输出学习、提升或投递准备计划",
    ]


def infer_tool_plan_from_message(message: str, has_target_direction: bool = False) -> dict[str, bool]:
    lowered = message.lower()
    wants_resume = bool(re.search(r"生成.*简历|简历|resume|cv", lowered))
    wants_jobs = bool(re.search(r"推荐岗位|岗位推荐|岗位|职位|投递|申请|job|jobs|ranking|match", lowered))
    wants_gap = bool(re.search(r"差距|gap|缺口|missing|fit|匹配度", lowered))
    wants_plan = bool(re.search(r"学习计划|下一步|准备计划|计划|next step|roadmap", lowered))
    wants_direction = bool(re.search(r"职业方向|方向推荐|适合.*方向|career direction", lowered))
    has_project_notes = bool(
        re.search(r"项目|系统|平台|工具|应用|网站|Agent|实习|工作经历|技术栈|负责|实现|开发", message, flags=re.IGNORECASE)
        and not wants_resume
        and not wants_jobs
        and not wants_gap
        and not wants_plan
    )

    if not any([wants_resume, wants_jobs, wants_gap, wants_plan, wants_direction, has_project_notes]):
        wants_direction = True
        wants_resume = True
        wants_jobs = True
        wants_gap = True
        wants_plan = True

    return {
        "run_project_profiling": has_project_notes,
        "run_career_direction": wants_direction or ((wants_resume or wants_jobs or wants_gap) and not has_target_direction),
        "run_resume_generation": wants_resume,
        "run_job_ranking": wants_jobs or wants_gap,
        "run_skill_gap_analysis": wants_gap,
        "run_next_step_plan": wants_plan or wants_gap,
    }


def execution_plan_from_tools(tool_plan: dict[str, bool]) -> list[str]:
    plan = ["理解用户目标"]
    if tool_plan.get("run_project_profiling"):
        plan.append("Project Profiling: 抽取用户提供的新项目/工作经历预览")
    if tool_plan.get("run_career_direction"):
        plan.append("Career Direction: 生成或更新职业方向评分")
    if tool_plan.get("run_resume_generation"):
        plan.append("Resume Generation: 生成目标方向简历")
    if tool_plan.get("run_job_ranking"):
        plan.append("Job Ranking: 召回并排序岗位")
    if tool_plan.get("run_skill_gap_analysis"):
        plan.append("Skill Gap Analysis: 分析岗位技能差距")
    if tool_plan.get("run_next_step_plan"):
        plan.append("Next Step Plan: 生成下一步计划")
    return plan


def task_plan_from_tools(
    tool_plan: dict[str, bool],
    feedback_rerun: bool = False,
) -> TaskPlan:
    steps = [
        TaskPlanStep(
            step_id="goal_understanding",
            tool_name="agent_goal_parser",
            status="planned",
            rerun_policy="always",
            input_refs=["request.message", "agent_state.preference"],
            output_refs=["goal", "tool_flags"],
        )
    ]
    previous_step = "goal_understanding"

    if tool_plan.get("run_project_profiling"):
        steps.append(
            TaskPlanStep(
                step_id="project_profiling",
                tool_name="project_profiling_tool",
                depends_on=[previous_step],
                rerun_policy="on_new_project_notes",
                input_refs=["goal.project_notes"],
                output_refs=["project_profile_preview"],
            )
        )
        previous_step = "project_profiling"

    if tool_plan.get("run_career_direction"):
        steps.append(
            TaskPlanStep(
                step_id="career_direction",
                tool_name="career_direction_tool",
                depends_on=["goal_understanding"],
                rerun_policy="on_profile_or_project_change",
                input_refs=["profile", "projects"],
                output_refs=["career_directions"],
            )
        )
        previous_step = "career_direction"

    if tool_plan.get("run_resume_generation"):
        depends_on = ["career_direction"] if tool_plan.get("run_career_direction") else ["goal_understanding"]
        steps.append(
            TaskPlanStep(
                step_id="resume_generation",
                tool_name="resume_generation_tool",
                depends_on=depends_on,
                rerun_policy="on_target_direction_or_profile_change",
                input_refs=["profile", "projects", "goal.target_direction"],
                output_refs=["generated_resume", "agent_state.latest_resume_id"],
            )
        )
        previous_step = "resume_generation"

    if tool_plan.get("run_job_ranking"):
        depends_on = ["resume_generation"] if tool_plan.get("run_resume_generation") else ["goal_understanding"]
        input_refs = [
            "profile",
            "projects",
            "goal.target_direction",
            "goal.locations",
            "goal.levels",
            "goal.role_families",
            "agent_state.latest_resume_id",
        ]
        if feedback_rerun:
            input_refs.append("feedback_memory.last_feedback")
        steps.append(
            TaskPlanStep(
                step_id="job_ranking",
                tool_name="job_matching_tool",
                depends_on=depends_on,
                rerun_policy="on_preference_or_resume_change",
                input_refs=input_refs,
                output_refs=["job_matches", "agent_state.latest_job_match_ids"],
            )
        )
        previous_step = "job_ranking"

    if tool_plan.get("run_skill_gap_analysis"):
        depends_on = ["job_ranking"] if tool_plan.get("run_job_ranking") else [previous_step]
        steps.append(
            TaskPlanStep(
                step_id="skill_gap_analysis",
                tool_name="skill_gap_analysis_tool",
                depends_on=depends_on,
                rerun_policy="on_job_matches_or_profile_skills_change",
                input_refs=["profile.skills", "projects.technologies", "job_matches.top3"],
                output_refs=["skill_gap", "agent_state.latest_gap_result"],
            )
        )
        previous_step = "skill_gap_analysis"

    if tool_plan.get("run_next_step_plan"):
        depends_on = ["skill_gap_analysis"] if tool_plan.get("run_skill_gap_analysis") else [previous_step]
        steps.append(
            TaskPlanStep(
                step_id="next_step_plan",
                tool_name="next_step_plan_tool",
                depends_on=depends_on,
                rerun_policy="on_gap_result_change",
                input_refs=["skill_gap", "goal.timeline_weeks", "goal.target_direction"],
                output_refs=["skill_gap.next_step_plan"],
            )
        )

    return TaskPlan(steps=steps)


def tool_plan_from_draft(draft: AgentPlanDraft) -> dict[str, bool]:
    return {
        "run_project_profiling": draft.run_project_profiling,
        "run_career_direction": draft.run_career_direction,
        "run_resume_generation": draft.run_resume_generation,
        "run_job_ranking": draft.run_job_ranking,
        "run_skill_gap_analysis": draft.run_skill_gap_analysis,
        "run_next_step_plan": draft.run_next_step_plan,
    }


def refresh_task_plan(draft: AgentPlanDraft, feedback_rerun: bool = False) -> None:
    tool_plan = tool_plan_from_draft(draft)
    draft.execution_plan = execution_plan_from_tools(tool_plan)
    draft.task_plan = task_plan_from_tools(tool_plan, feedback_rerun)


def update_task_status(task_plan: TaskPlan, step_id: str, status: str) -> None:
    from app.agent.scheduler import update_task_status as scheduler_update_task_status

    scheduler_update_task_status(task_plan, step_id, status)


def rerun_steps_from_task_plan(task_plan: TaskPlan) -> list[TaskPlanStep]:
    from app.agent.scheduler import rerun_steps_from_task_plan as scheduler_rerun_steps_from_task_plan

    return scheduler_rerun_steps_from_task_plan(task_plan)


def normalize_tool_plan(draft: AgentPlanDraft, message: str = "") -> AgentPlanDraft:
    inferred = infer_tool_plan_from_message(message, bool(draft.target_direction))
    message_inferred = None
    if not any(
        [
            draft.run_project_profiling,
            draft.run_career_direction,
            draft.run_resume_generation,
            draft.run_job_ranking,
            draft.run_skill_gap_analysis,
            draft.run_next_step_plan,
        ]
    ):
        message_inferred = inferred

    if message_inferred:
        for field, value in message_inferred.items():
            setattr(draft, field, value)

    if (draft.run_resume_generation or draft.run_job_ranking or draft.run_skill_gap_analysis) and not draft.target_direction:
        draft.run_career_direction = True
    if draft.run_skill_gap_analysis:
        draft.run_job_ranking = True
    if draft.run_next_step_plan and draft.run_skill_gap_analysis:
        draft.run_job_ranking = True

    refresh_task_plan(draft)
    return draft


def canonicalize_goal(draft: AgentPlanDraft, message: str = "") -> AgentPlanDraft:
    combined_text = " ".join(
        [
            draft.target_direction,
            " ".join(draft.role_families),
        ]
    ).lower()
    direction_map = [
        ("backend", "Backend", "Backend Developer"),
        ("后端", "Backend", "Backend Developer"),
        ("ai application", "AI Application", "AI Application Developer"),
        ("llm", "AI Application", "AI Application Developer"),
        ("data analyst", "Data Analyst", "Data Analyst"),
        ("数据分析", "Data Analyst", "Data Analyst"),
        ("full stack", "Full Stack", "Full Stack Developer"),
        ("frontend", "Frontend", "Frontend Developer"),
        ("前端", "Frontend", "Frontend Developer"),
        ("product", "Product Manager", "Product Manager"),
        ("产品", "Product Manager", "Product Manager"),
    ]
    canonical_direction = draft.target_direction
    canonical_families = list(draft.role_families)
    for keyword, family, direction in direction_map:
        if keyword in combined_text or keyword in draft.target_direction:
            canonical_direction = direction
            if family not in canonical_families:
                canonical_families.append(family)
            break

    level_aliases = {
        "intern": ["Intern", "实习"],
        "graduate": ["Graduate", "校招", "应届"],
        "junior": ["Junior", "初级", "应届", "1年以内"],
        "实习": ["Intern", "实习"],
        "校招": ["Graduate", "校招", "应届"],
        "应届": ["Graduate", "校招", "应届"],
        "初级": ["Junior", "初级", "应届", "1年以内"],
    }
    canonical_levels: list[str] = []
    for level in draft.levels:
        aliases = level_aliases.get(level.lower(), level_aliases.get(level, [level]))
        for alias in aliases:
            if alias not in canonical_levels:
                canonical_levels.append(alias)

    draft.target_direction = canonical_direction
    draft.role_families = canonical_families
    draft.levels = canonical_levels
    return normalize_tool_plan(draft, message)


def parse_agent_goal(message: str) -> AgentPlanDraft:
    try:
        draft = invoke_model_with_logging(
            "agent_orchestrator",
            AGENT_ORCHESTRATOR_MODEL,
            lambda: (
                AGENT_PLAN_PROMPT
                | ChatOpenAI(model=AGENT_ORCHESTRATOR_MODEL).with_structured_output(
                    AgentPlanDraft,
                    method="json_schema",
                )
            ).invoke({"message": message}),
        )
        if not draft.execution_plan:
            draft.execution_plan = default_execution_plan()
        return canonicalize_goal(draft, message)
    except Exception:
        log_model_fallback("agent_orchestrator", AGENT_ORCHESTRATOR_MODEL, "regex_goal_parser")
        return canonicalize_goal(fallback_goal(message), message)
