from uuid import uuid4

from sqlalchemy.orm import Session

from app.agent.planner import (
    AGENT_PLAN_PROMPT,
    AgentPlanDraft,
    canonicalize_goal,
    default_execution_plan,
    execution_plan_from_tools,
    fallback_goal,
    infer_tool_plan_from_message,
    normalize_tool_plan,
    parse_agent_goal,
    refresh_task_plan,
    task_plan_from_tools,
    tool_plan_from_draft,
)
from app.agent.scheduler import (
    force_partial_rerun_for_feedback,
    message_updates_preferences,
    partial_rerun_context,
    reused_steps_for_partial_rerun,
    rerun_steps_from_task_plan,
    update_task_status,
)
from app.career_direction import recommend_career_directions
from app.job_matching import match_jobs
from app.models import (
    AgentRunModel,
    AgentStateModel,
    CareerDirectionsModel,
    GeneratedResumeModel,
    JobModel,
    ProfileModel,
    ProjectModel,
)
from app.profiling import profile_project_text
from app.resume_generation import generate_resume_content
from app.schemas import (
    AgentExecutionStep,
    AgentState,
    CareerAgentGoal,
    CareerAgentRequest,
    CareerAgentResponse,
    CareerDirectionRecommendation,
    CareerDirectionsSnapshot,
    FeedbackEntry,
    FeedbackMemory,
    GeneratedResume,
    Job,
    JobMatchRequest,
    JobMatchResponse,
    Profile,
    Project,
    ProjectInput,
    ResumeProjectSection,
    SkillGapAnalysisRequest,
    SkillGapAnalysisResponse,
    UserPreference,
)
from app.skill_gap import run_skill_gap_analysis
from app.telemetry import log_event


DEFAULT_PROFILE_ID = 1


def profile_from_model(profile: ProfileModel | None) -> Profile:
    if profile is None:
        return Profile()
    return Profile(
        name=profile.name,
        headline=profile.headline,
        email=profile.email,
        phone=profile.phone,
        wechat=profile.wechat,
        location=profile.location,
        summary=profile.summary,
        skills=profile.skills or [],
        education=profile.education or [],
        experience=profile.experience or [],
    )


def project_from_model(project: ProjectModel) -> Project:
    return Project(
        id=project.id,
        category=project.category or "project",
        title=project.title,
        role=project.role,
        start_date=project.start_date,
        end_date=project.end_date,
        description=project.description,
        technologies=project.technologies or [],
        highlights=project.highlights or [],
    )


def job_from_model(job: JobModel) -> Job:
    return Job(
        id=job.id,
        title=job.title,
        company=job.company,
        location=job.location,
        level=job.level,
        role_family=job.role_family,
        status=job.status,
        required_skills=job.required_skills or [],
        nice_to_have_skills=job.nice_to_have_skills or [],
        description=job.description,
    )


def resume_from_model(resume: GeneratedResumeModel) -> GeneratedResume:
    return GeneratedResume(
        id=resume.id,
        target_direction=resume.target_direction,
        created_at=resume.created_at.isoformat() if resume.created_at else "",
        introduction=resume.introduction,
        skills=resume.skills or [],
        projects=[
            ResumeProjectSection.model_validate(project)
            for project in resume.projects or []
        ],
        selected_project_ids=resume.selected_project_ids or [],
    )


def state_from_model(state_model: AgentStateModel | None) -> AgentState:
    if state_model is None:
        return AgentState(profile_id=DEFAULT_PROFILE_ID)

    return AgentState(
        profile_id=state_model.profile_id,
        preference=UserPreference.model_validate(state_model.preferences or {}),
        latest_resume_id=state_model.latest_resume_id or "",
        latest_job_match_ids=state_model.latest_job_match_ids or [],
        latest_gap_result=SkillGapAnalysisResponse.model_validate(state_model.latest_gap_result or {}),
        feedback_memory=FeedbackMemory.model_validate(state_model.feedback_memory or {}),
        last_agent_run_id=state_model.last_agent_run_id or "",
        last_target_direction=state_model.last_target_direction or "",
        project_count=state_model.project_count or 0,
        updated_at=state_model.updated_at.isoformat() if state_model.updated_at else "",
    )


def get_agent_state(db: Session) -> AgentState:
    return state_from_model(db.get(AgentStateModel, DEFAULT_PROFILE_ID))


def preference_from_goal(goal: CareerAgentGoal) -> UserPreference:
    return UserPreference(
        target_direction=goal.target_direction,
        locations=goal.locations,
        levels=goal.levels,
        role_families=goal.role_families,
    )


def merge_goal_with_state(draft: AgentPlanDraft, previous_state: AgentState) -> None:
    preference = previous_state.preference
    if not draft.target_direction:
        draft.target_direction = preference.target_direction or previous_state.last_target_direction
    if not draft.locations:
        draft.locations = list(preference.locations)
    if not draft.levels:
        draft.levels = list(preference.levels)
    if not draft.role_families:
        draft.role_families = list(preference.role_families)


def load_resume_for_rerun(
    db: Session,
    previous_state: AgentState,
    target_direction: str,
) -> GeneratedResume | None:
    resume: GeneratedResumeModel | None = None
    if previous_state.latest_resume_id:
        resume = db.get(GeneratedResumeModel, previous_state.latest_resume_id)

    if resume is None:
        resume_query = db.query(GeneratedResumeModel).filter(
            GeneratedResumeModel.profile_id == DEFAULT_PROFILE_ID
        )
        if target_direction:
            resume_query = resume_query.filter(
                GeneratedResumeModel.target_direction == target_direction
            )
        resume = resume_query.order_by(GeneratedResumeModel.created_at.desc()).first()

    if resume is None or resume.profile_id != DEFAULT_PROFILE_ID:
        return None
    return resume_from_model(resume)


def build_feedback_entry(
    message: str,
    preference: UserPreference,
    rerun_tools: list[str],
) -> FeedbackEntry:
    return FeedbackEntry(
        message=message,
        extracted_preferences=preference,
        rerun_tools=rerun_tools,
    )


def save_agent_state(
    db: Session,
    state: AgentState,
) -> AgentState:
    state_model = db.get(AgentStateModel, state.profile_id)
    if state_model is None:
        state_model = AgentStateModel(profile_id=state.profile_id)
        db.add(state_model)

    state_model.preferences = state.preference.model_dump(mode="json")
    state_model.latest_resume_id = state.latest_resume_id
    state_model.latest_job_match_ids = state.latest_job_match_ids
    state_model.latest_gap_result = state.latest_gap_result.model_dump(mode="json")
    state_model.feedback_memory = state.feedback_memory.model_dump(mode="json")
    state_model.last_agent_run_id = state.last_agent_run_id
    state_model.last_target_direction = state.last_target_direction
    state_model.project_count = state.project_count
    db.flush()
    db.refresh(state_model)
    return state_from_model(state_model)


def save_career_directions(
    db: Session,
    recommendations: list[CareerDirectionRecommendation],
) -> CareerDirectionsSnapshot:
    career_directions = db.get(CareerDirectionsModel, DEFAULT_PROFILE_ID)
    if career_directions is None:
        career_directions = CareerDirectionsModel(profile_id=DEFAULT_PROFILE_ID)
        db.add(career_directions)

    career_directions.recommendations = [
        recommendation.model_dump() for recommendation in recommendations
    ]
    db.commit()
    db.refresh(career_directions)
    return CareerDirectionsSnapshot(
        recommendations=recommendations,
        updated_at=(
            career_directions.updated_at.isoformat()
            if career_directions.updated_at
            else ""
        ),
    )


def save_generated_resume(db: Session, generated_resume: GeneratedResume) -> GeneratedResume:
    resume_model = GeneratedResumeModel(
        id=str(uuid4()),
        profile_id=DEFAULT_PROFILE_ID,
        target_direction=generated_resume.target_direction,
        introduction=generated_resume.introduction,
        skills=generated_resume.skills,
        projects=[project.model_dump() for project in generated_resume.projects],
        selected_project_ids=generated_resume.selected_project_ids,
    )
    db.add(resume_model)
    db.commit()
    db.refresh(resume_model)
    return resume_from_model(resume_model)


def run_career_agent(
    request: CareerAgentRequest,
    db: Session,
    agent_run_id: str | None = None,
) -> CareerAgentResponse:
    agent_run_id = agent_run_id or str(uuid4())
    steps: list[AgentExecutionStep] = []
    previous_state = get_agent_state(db)
    draft = parse_agent_goal(request.message)
    is_feedback_rerun = message_updates_preferences(request.message, previous_state)
    merge_goal_with_state(draft, previous_state)
    draft = canonicalize_goal(draft, request.message)
    if is_feedback_rerun:
        force_partial_rerun_for_feedback(draft)
    task_plan = draft.task_plan
    update_task_status(task_plan, "goal_understanding", "completed")

    goal = CareerAgentGoal(
        target_direction=draft.target_direction,
        locations=draft.locations,
        levels=draft.levels,
        role_families=draft.role_families,
        timeline_weeks=draft.timeline_weeks,
        project_notes=draft.project_notes,
    )
    steps.append(
        AgentExecutionStep(
            name="理解目标",
            status="completed",
            detail=(
                "已读取旧 Agent State，并根据用户反馈调整偏好，只重跑岗位排序及下游步骤。"
                if is_feedback_rerun
                else "已解析用户目标和执行计划。"
            ),
        )
    )

    profile = profile_from_model(db.get(ProfileModel, DEFAULT_PROFILE_ID))
    projects = [
        project_from_model(project)
        for project in db.query(ProjectModel)
        .filter(ProjectModel.profile_id == DEFAULT_PROFILE_ID)
        .order_by(ProjectModel.title)
        .all()
    ]

    project_preview: ProjectInput | None = None
    if draft.run_project_profiling and goal.project_notes.strip():
        project_preview = profile_project_text(goal.project_notes)
        update_task_status(task_plan, "project_profiling", "preview")
        steps.append(
            AgentExecutionStep(
                name="Project Profiling",
                status="preview",
                detail="已从用户输入中抽取项目预览，等待用户确认后再保存到项目表。",
            )
        )
    else:
        update_task_status(task_plan, "project_profiling", "skipped")
        steps.append(
            AgentExecutionStep(
                name="Project Profiling",
                status="skipped",
                detail=(
                    "本次输入没有包含新的项目经历描述，沿用数据库中已有项目。"
                    if draft.run_project_profiling
                    else "本次目标不需要 Project Profiling，已跳过。"
                ),
            )
        )

    recommendations: list[CareerDirectionRecommendation] = []
    career_snapshot = CareerDirectionsSnapshot()
    if draft.run_career_direction:
        recommendations = recommend_career_directions(profile, projects)
        career_snapshot = save_career_directions(db, recommendations)
        update_task_status(task_plan, "career_direction", "completed")
        steps.append(
            AgentExecutionStep(
                name="Career Direction",
                status="completed",
                detail=f"已生成并覆盖保存 {len(recommendations)} 个职业方向评分。",
            )
        )
    else:
        update_task_status(task_plan, "career_direction", "skipped")
        steps.append(
            AgentExecutionStep(
                name="Career Direction",
                status="skipped",
                detail="本次目标不需要职业方向推荐，已跳过。",
            )
        )

    target_direction = goal.target_direction
    if not target_direction and recommendations:
        target_direction = recommendations[0].direction
        goal.target_direction = target_direction

    generated_resume: GeneratedResume | None = None
    if draft.run_resume_generation and target_direction:
        generated_resume = save_generated_resume(
            db,
            generate_resume_content(profile, projects, target_direction),
        )
        update_task_status(task_plan, "resume_generation", "completed")
        steps.append(
            AgentExecutionStep(
                name="Resume Generation",
                status="completed",
                detail=f"已生成并保存新版 {target_direction} 简历。",
            )
        )
    else:
        update_task_status(task_plan, "resume_generation", "skipped")
        steps.append(
            AgentExecutionStep(
                name="Resume Generation",
                status="skipped",
                detail=(
                    "没有可用目标方向，未生成简历。"
                    if draft.run_resume_generation
                    else "本次目标不需要生成简历，已跳过。"
                ),
            )
        )

    reusable_resume = generated_resume
    if reusable_resume is None and draft.run_job_ranking:
        reusable_resume = load_resume_for_rerun(db, previous_state, target_direction)
        if is_feedback_rerun and reusable_resume:
            steps.append(
                AgentExecutionStep(
                    name="State Reuse",
                    status="completed",
                    detail=f"已复用上一版简历 {reusable_resume.id} 进行局部重跑。",
                )
            )

    job_matches = JobMatchResponse()
    if draft.run_job_ranking:
        jobs = [
            job_from_model(job)
            for job in db.query(JobModel)
            .filter(JobModel.status == "active")
            .order_by(JobModel.role_family, JobModel.location, JobModel.title)
            .all()
        ]
        resume_data = reusable_resume.model_dump() if reusable_resume else None
        job_request = JobMatchRequest(
            target_direction=target_direction,
            locations=goal.locations,
            levels=goal.levels,
            role_families=goal.role_families,
            status="active",
            top_k=10,
            llm_candidate_count=20,
        )
        job_matches = match_jobs(profile, projects, resume_data, jobs, job_request)
        requested_locations = {location.lower() for location in goal.locations}
        returned_locations = {match.job.location.lower() for match in job_matches.matches}
        location_detail = ""
        if requested_locations and job_matches.matches and not (requested_locations & returned_locations):
            location_detail = " 未找到目标城市岗位，已按职级和方向回退到其它城市岗位。"
        update_task_status(task_plan, "job_ranking", "completed")
        steps.append(
            AgentExecutionStep(
                name="Job Ranking",
                status="completed",
                detail=f"已完成岗位召回和排序，返回 Top{len(job_matches.matches)}。{location_detail}",
            )
        )
    else:
        update_task_status(task_plan, "job_ranking", "skipped")
        steps.append(
            AgentExecutionStep(
                name="Job Ranking",
                status="skipped",
                detail="本次目标不需要岗位召回和排序，已跳过。",
            )
        )

    skill_gap = SkillGapAnalysisResponse()
    if draft.run_skill_gap_analysis:
        skill_gap = run_skill_gap_analysis(
            SkillGapAnalysisRequest(
                jobs=[match.job for match in job_matches.matches[:3]],
                top_matches=job_matches.matches[:3],
                user_skills=profile.skills,
                target_direction=target_direction,
            ),
            profile,
            projects,
        )
        update_task_status(task_plan, "skill_gap_analysis", "completed")
        steps.append(
            AgentExecutionStep(
                name="Skill Gap Analysis",
                status="completed",
                detail=f"Gap severity: {skill_gap.gap_severity}。",
            )
        )
    else:
        update_task_status(task_plan, "skill_gap_analysis", "skipped")
        steps.append(
            AgentExecutionStep(
                name="Skill Gap Analysis",
                status="skipped",
                detail="本次目标不需要技能差距分析，已跳过。",
            )
        )

    if draft.run_next_step_plan:
        if not draft.run_skill_gap_analysis:
            skill_gap = run_skill_gap_analysis(
                SkillGapAnalysisRequest(
                    jobs=[match.job for match in job_matches.matches[:3]],
                    top_matches=job_matches.matches[:3],
                    user_skills=profile.skills,
                    target_direction=target_direction,
                ),
                profile,
                projects,
            )
        update_task_status(task_plan, "next_step_plan", "completed")
        steps.append(
            AgentExecutionStep(
                name="Next Step Plan",
                status="completed",
                detail=f"已生成 {len(skill_gap.next_step_plan)} 周下一步计划。",
            )
        )
    else:
        update_task_status(task_plan, "next_step_plan", "skipped")
        steps.append(
            AgentExecutionStep(
                name="Next Step Plan",
                status="skipped",
                detail="本次目标不需要下一步计划，已跳过。",
            )
        )

    rerun_tools = [
        name
        for name, enabled in [
            ("job_ranking", draft.run_job_ranking),
            ("skill_gap_analysis", draft.run_skill_gap_analysis),
            ("next_step_plan", draft.run_next_step_plan),
        ]
        if enabled
    ]
    preference = preference_from_goal(goal)
    if not preference.target_direction:
        preference.target_direction = previous_state.preference.target_direction
    if not preference.locations:
        preference.locations = list(previous_state.preference.locations)
    if not preference.levels:
        preference.levels = list(previous_state.preference.levels)
    if not preference.role_families:
        preference.role_families = list(previous_state.preference.role_families)

    feedback_entries = list(previous_state.feedback_memory.entries)
    last_feedback = previous_state.feedback_memory.last_feedback
    if is_feedback_rerun:
        feedback_entries.append(build_feedback_entry(request.message, preference, rerun_tools))
        feedback_entries = feedback_entries[-20:]
        last_feedback = request.message

    latest_resume_id = (
        generated_resume.id
        if generated_resume
        else reusable_resume.id
        if reusable_resume
        else previous_state.latest_resume_id
    )
    latest_job_match_ids = (
        [match.job.id for match in job_matches.matches]
        if draft.run_job_ranking
        else previous_state.latest_job_match_ids
    )
    latest_gap_result = skill_gap if draft.run_skill_gap_analysis or draft.run_next_step_plan else previous_state.latest_gap_result
    reused_steps, rerun_steps = partial_rerun_context(
        task_plan,
        previous_state,
        reusable_resume,
        is_feedback_rerun,
    )
    summary = {
        "profile_id": DEFAULT_PROFILE_ID,
        "project_count": len(projects),
        "resume_id": latest_resume_id or None,
        "target_direction": target_direction,
        "job_match_count": len(job_matches.matches),
        "gap_severity": latest_gap_result.gap_severity,
        "timeline_weeks": goal.timeline_weeks,
        "is_feedback_rerun": is_feedback_rerun,
        "run_project_profiling": draft.run_project_profiling,
        "run_career_direction": draft.run_career_direction,
        "run_resume_generation": draft.run_resume_generation,
        "run_job_ranking": draft.run_job_ranking,
        "run_skill_gap_analysis": draft.run_skill_gap_analysis,
        "run_next_step_plan": draft.run_next_step_plan,
        "reused_step_count": len(reused_steps),
        "rerun_step_count": len(rerun_steps),
        "agent_run_id": agent_run_id,
    }
    current_state = save_agent_state(
        db,
        AgentState(
            profile_id=DEFAULT_PROFILE_ID,
            preference=preference,
            latest_resume_id=latest_resume_id,
            latest_job_match_ids=latest_job_match_ids,
            latest_gap_result=latest_gap_result,
            feedback_memory=FeedbackMemory(
                entries=feedback_entries,
                last_feedback=last_feedback,
            ),
            last_agent_run_id=agent_run_id,
            last_target_direction=target_direction,
            project_count=len(projects),
            summary=summary,
        ),
    )
    current_state.summary = summary

    response = CareerAgentResponse(
        user_message=request.message,
        goal=goal,
        execution_plan=draft.execution_plan or default_execution_plan(),
        task_plan=task_plan,
        reused_steps=reused_steps,
        rerun_steps=rerun_steps,
        steps=steps,
        project_profile_preview=project_preview,
        career_directions=career_snapshot,
        generated_resume=generated_resume,
        job_matches=job_matches,
        skill_gap=skill_gap,
        state=current_state,
    )
    selected_tools = [
        step.step_id
        for step in task_plan.steps
        if step.step_id != "goal_understanding" and step.status == "completed"
    ]
    log_event(
        "agent_run_summary",
        agent_run_id=agent_run_id,
        profile_id=DEFAULT_PROFILE_ID,
        target_direction=target_direction,
        selected_tools=selected_tools,
        step_count=len(task_plan.steps),
        reused_step_count=len(reused_steps),
        rerun_step_count=len(rerun_steps),
        is_feedback_rerun=is_feedback_rerun,
        latest_resume_id=latest_resume_id,
        latest_job_match_count=len(latest_job_match_ids),
        gap_severity=latest_gap_result.gap_severity,
    )
    for step in task_plan.steps:
        log_event(
            "task_step_trace",
            agent_run_id=agent_run_id,
            step_id=step.step_id,
            tool_name=step.tool_name,
            depends_on=step.depends_on,
            status=step.status,
            rerun_policy=step.rerun_policy,
            input_refs=step.input_refs,
            output_refs=step.output_refs,
            is_reused=any(reused.step_id == step.step_id for reused in reused_steps),
            is_rerun=any(rerun.step_id == step.step_id for rerun in rerun_steps),
        )
    if is_feedback_rerun:
        log_event(
            "partial_rerun_event",
            agent_run_id=agent_run_id,
            feedback_text=request.message,
            changed_preferences=preference.model_dump(mode="json"),
            reused_steps=[step.step_id for step in reused_steps],
            rerun_steps=[step.step_id for step in rerun_steps],
            saved_step_count=len(reused_steps),
            reuse_rate=len(reused_steps) / (len(reused_steps) + len(rerun_steps))
            if reused_steps or rerun_steps
            else 0.0,
        )
    db.add(
        AgentRunModel(
            id=agent_run_id,
            profile_id=DEFAULT_PROFILE_ID,
            user_message=request.message,
            goal=goal.model_dump(mode="json"),
            steps=[step.model_dump(mode="json") for step in steps],
            result=response.model_dump(mode="json"),
        )
    )
    db.commit()
    return response
