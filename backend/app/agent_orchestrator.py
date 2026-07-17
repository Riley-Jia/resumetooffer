import re
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.career_direction import recommend_career_directions
from app.job_matching import match_jobs
from app.model_config import AGENT_ORCHESTRATOR_MODEL
from app.model_logging import invoke_model_with_logging, log_model_fallback
from app.models import (
    AgentRunModel,
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
    CareerAgentGoal,
    CareerAgentRequest,
    CareerAgentResponse,
    CareerDirectionRecommendation,
    CareerDirectionsSnapshot,
    GeneratedResume,
    Job,
    JobMatchRequest,
    JobMatchResponse,
    Profile,
    Project,
    ProjectInput,
    ResumeProjectSection,
    SkillGapAnalysisRequest,
)
from app.skill_gap import run_skill_gap_analysis


DEFAULT_PROFILE_ID = 1


class AgentPlanDraft(BaseModel):
    target_direction: str = ""
    locations: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    role_families: list[str] = Field(default_factory=list)
    timeline_weeks: int | None = None
    project_notes: str = ""
    execution_plan: list[str] = Field(default_factory=list)


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
""",
        ),
        ("human", "{message}"),
    ]
)


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

    return AgentPlanDraft(
        target_direction=target_direction,
        locations=locations,
        levels=levels,
        role_families=role_families,
        timeline_weeks=timeline_weeks,
        execution_plan=default_execution_plan(),
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


def canonicalize_goal(draft: AgentPlanDraft) -> AgentPlanDraft:
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
    return draft


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
        return canonicalize_goal(draft)
    except Exception:
        log_model_fallback("agent_orchestrator", AGENT_ORCHESTRATOR_MODEL, "regex_goal_parser")
        return canonicalize_goal(fallback_goal(message))


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
) -> CareerAgentResponse:
    steps: list[AgentExecutionStep] = []
    draft = parse_agent_goal(request.message)
    goal = CareerAgentGoal(
        target_direction=draft.target_direction,
        locations=draft.locations,
        levels=draft.levels,
        role_families=draft.role_families,
        timeline_weeks=draft.timeline_weeks,
        project_notes=draft.project_notes,
    )
    steps.append(AgentExecutionStep(name="理解目标", status="completed", detail="已解析用户目标和执行计划。"))

    profile = profile_from_model(db.get(ProfileModel, DEFAULT_PROFILE_ID))
    projects = [
        project_from_model(project)
        for project in db.query(ProjectModel)
        .filter(ProjectModel.profile_id == DEFAULT_PROFILE_ID)
        .order_by(ProjectModel.title)
        .all()
    ]

    project_preview: ProjectInput | None = None
    if goal.project_notes.strip():
        project_preview = profile_project_text(goal.project_notes)
        steps.append(
            AgentExecutionStep(
                name="Project Profiling",
                status="preview",
                detail="已从用户输入中抽取项目预览，等待用户确认后再保存到项目表。",
            )
        )
    else:
        steps.append(
            AgentExecutionStep(
                name="Project Profiling",
                status="skipped",
                detail="本次输入没有包含新的项目经历描述，沿用数据库中已有项目。",
            )
        )

    recommendations = recommend_career_directions(profile, projects)
    career_snapshot = save_career_directions(db, recommendations)
    steps.append(
        AgentExecutionStep(
            name="Career Direction",
            status="completed",
            detail=f"已生成并覆盖保存 {len(recommendations)} 个职业方向评分。",
        )
    )

    target_direction = goal.target_direction
    if not target_direction and recommendations:
        target_direction = recommendations[0].direction
        goal.target_direction = target_direction

    generated_resume: GeneratedResume | None = None
    if target_direction:
        generated_resume = save_generated_resume(
            db,
            generate_resume_content(profile, projects, target_direction),
        )
        steps.append(
            AgentExecutionStep(
                name="Resume Generation",
                status="completed",
                detail=f"已生成并保存新版 {target_direction} 简历。",
            )
        )
    else:
        steps.append(
            AgentExecutionStep(
                name="Resume Generation",
                status="skipped",
                detail="没有可用目标方向，未生成简历。",
            )
        )

    jobs = [
        job_from_model(job)
        for job in db.query(JobModel)
        .filter(JobModel.status == "active")
        .order_by(JobModel.role_family, JobModel.location, JobModel.title)
        .all()
    ]
    resume_data = generated_resume.model_dump() if generated_resume else None
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
    steps.append(
        AgentExecutionStep(
            name="Job Ranking",
            status="completed",
            detail=f"已完成岗位召回和排序，返回 Top{len(job_matches.matches)}。{location_detail}",
        )
    )

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
    steps.append(
        AgentExecutionStep(
            name="Skill Gap Analysis",
            status="completed",
            detail=f"Gap severity: {skill_gap.gap_severity}。",
        )
    )
    steps.append(
        AgentExecutionStep(
            name="Next Step Plan",
            status="completed",
            detail=f"已生成 {len(skill_gap.next_step_plan)} 周下一步计划。",
        )
    )

    response = CareerAgentResponse(
        user_message=request.message,
        goal=goal,
        execution_plan=draft.execution_plan or default_execution_plan(),
        steps=steps,
        project_profile_preview=project_preview,
        career_directions=career_snapshot,
        generated_resume=generated_resume,
        job_matches=job_matches,
        skill_gap=skill_gap,
        state={
            "profile_id": DEFAULT_PROFILE_ID,
            "project_count": len(projects),
            "resume_id": generated_resume.id if generated_resume else None,
            "target_direction": target_direction,
            "job_match_count": len(job_matches.matches),
            "gap_severity": skill_gap.gap_severity,
            "timeline_weeks": goal.timeline_weeks,
        },
    )
    agent_run_id = str(uuid4())
    response.state["agent_run_id"] = agent_run_id
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
