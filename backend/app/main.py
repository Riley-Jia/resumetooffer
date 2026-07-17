import logging
import os
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.agent_orchestrator import run_career_agent
from app.career_direction import CAREER_DIRECTION_TOOL
from app.database import SessionLocal, create_tables, get_db
from app.input_router import route_user_input
from app.job_matching import match_jobs
from app.job_seed import seed_jobs
from app.model_config import configured_models
from app.models import (
    CareerDirectionsModel,
    GeneratedResumeModel,
    JobModel,
    ProfileModel,
    ProjectModel,
)
from app.profile_project_editing import build_changes_for_patch, build_edit_preview, clean_patch
from app.profiling import PROJECT_PROFILING_TOOL
from app.resume_generation import RESUME_GENERATION_TOOL
from app.schemas import (
    CareerDirectionRecommendation,
    CareerDirectionsSnapshot,
    CareerAgentRequest,
    CareerAgentResponse,
    ProfileProjectEditApplyRequest,
    ProfileProjectEditApplyResponse,
    ProfileProjectEditPreview,
    ProfileProjectEditRequest,
    GeneratedResume,
    InputRouterRequest,
    InputRouterResponse,
    Job,
    JobMatchRequest,
    JobMatchResponse,
    Profile,
    ProfileProjects,
    Project,
    ProjectInput,
    ProjectProfilingRequest,
    ResumeGenerationRequest,
    ResumeProjectSection,
    SkillGapAnalysisRequest,
    SkillGapAnalysisResponse,
)
from app.skill_gap import run_skill_gap_analysis


app = FastAPI(title="Resume to Offer API")
DEFAULT_PROFILE_ID = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    create_tables()
    with SessionLocal() as db:
        seed_jobs(db)


@app.get("/debug/config")
def debug_config() -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    return {
        "models": configured_models(),
        "openai_api_key_loaded": bool(api_key),
        "openai_api_key_prefix": api_key[:8] if api_key else "",
        "database_url_prefix": os.getenv("DATABASE_URL", "")[:48],
    }


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


def upsert_profile(db: Session, profile: Profile) -> ProfileModel:
    profile_model = db.get(ProfileModel, DEFAULT_PROFILE_ID)

    if profile_model is None:
        profile_model = ProfileModel(id=DEFAULT_PROFILE_ID)
        db.add(profile_model)

    profile_model.name = profile.name
    profile_model.headline = profile.headline
    profile_model.email = profile.email
    profile_model.phone = profile.phone
    profile_model.wechat = profile.wechat
    profile_model.location = profile.location
    profile_model.summary = profile.summary
    profile_model.skills = profile.skills
    profile_model.education = profile.education
    profile_model.experience = profile.experience

    db.commit()
    db.refresh(profile_model)
    return profile_model


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


def career_directions_from_model(
    career_directions: CareerDirectionsModel | None,
) -> CareerDirectionsSnapshot:
    if career_directions is None:
        return CareerDirectionsSnapshot()

    return CareerDirectionsSnapshot(
        recommendations=[
            CareerDirectionRecommendation.model_validate(item)
            for item in career_directions.recommendations or []
        ],
        updated_at=(
            career_directions.updated_at.isoformat()
            if career_directions.updated_at
            else ""
        ),
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-status")
def model_status() -> dict[str, Any]:
    return {
        "openai_api_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "models": configured_models(),
    }


@app.post("/agent/run")
def run_agent(
    request: CareerAgentRequest,
    db: Session = Depends(get_db),
) -> CareerAgentResponse:
    return run_career_agent(request, db)


@app.post("/input/route")
def route_input(
    request: InputRouterRequest,
    db: Session = Depends(get_db),
) -> InputRouterResponse:
    profile_projects = get_profile_projects(db)
    return route_user_input(
        request.message,
        profile_projects.profile,
        profile_projects.projects,
    )


@app.get("/profile-projects")
def get_profile_projects(db: Session = Depends(get_db)) -> ProfileProjects:
    profile = db.get(ProfileModel, DEFAULT_PROFILE_ID)
    projects = (
        db.query(ProjectModel)
        .filter(ProjectModel.profile_id == DEFAULT_PROFILE_ID)
        .order_by(ProjectModel.title)
        .all()
    )

    return ProfileProjects(
        profile=profile_from_model(profile),
        projects=[project_from_model(project) for project in projects],
    )


@app.post("/profile-projects/edit/preview")
def preview_profile_project_edit(
    request: ProfileProjectEditRequest,
    db: Session = Depends(get_db),
) -> ProfileProjectEditPreview:
    profile_projects = get_profile_projects(db)
    return build_edit_preview(
        request.message,
        profile_projects.profile,
        profile_projects.projects,
        request.router_intent,
        request.router_content_type,
        request.normalized_instruction,
    )


@app.post("/profile-projects/edit/apply")
def apply_profile_project_edit(
    request: ProfileProjectEditApplyRequest,
    db: Session = Depends(get_db),
) -> ProfileProjectEditApplyResponse:
    profile_projects = get_profile_projects(db)
    warnings: list[str] = []
    patch = clean_patch(request.patch, profile_projects.projects, warnings)
    applied_changes = build_changes_for_patch(
        profile_projects.profile,
        profile_projects.projects,
        patch,
    )

    profile_model = db.get(ProfileModel, DEFAULT_PROFILE_ID)
    if profile_model is None:
        profile_model = ProfileModel(id=DEFAULT_PROFILE_ID)
        db.add(profile_model)
        db.flush()

    for field, value in patch.profile.set_fields.items():
        setattr(profile_model, field, value)

    profile_model.skills = apply_list_edit(
        profile_model.skills or [],
        patch.profile.skills_add,
        patch.profile.skills_remove,
    )
    profile_model.education = apply_list_edit(
        profile_model.education or [],
        patch.profile.education_add,
        patch.profile.education_remove,
    )
    for project_patch in patch.projects:
        if project_patch.action == "create" and project_patch.create_project:
            project_model = ProjectModel(
                id=str(uuid4()),
                profile_id=DEFAULT_PROFILE_ID,
                **project_patch.create_project.model_dump(),
            )
            db.add(project_model)
            continue

        project_model = db.get(ProjectModel, project_patch.project_id)
        if project_model is None or project_model.profile_id != DEFAULT_PROFILE_ID:
            warnings.append(f"应用时没有找到项目：{project_patch.match_title or project_patch.project_id}")
            continue

        for field, value in project_patch.set_fields.items():
            setattr(project_model, field, value)
        project_model.technologies = apply_list_edit(
            project_model.technologies or [],
            project_patch.technologies_add,
            project_patch.technologies_remove,
        )
        project_model.highlights = apply_list_edit(
            project_model.highlights or [],
            project_patch.highlights_add,
            project_patch.highlights_remove,
        )

    db.commit()
    updated = get_profile_projects(db)
    return ProfileProjectEditApplyResponse(
        profile=updated.profile,
        projects=updated.projects,
        applied_changes=applied_changes,
        warnings=warnings,
    )


def apply_list_edit(current: list[str], additions: list[str], removals: list[str]) -> list[str]:
    removed = {item.strip().lower() for item in removals}
    updated = [item for item in current if item.strip().lower() not in removed]
    existing = {item.strip().lower() for item in updated}
    for item in additions:
        key = item.strip().lower()
        if item.strip() and key not in existing:
            updated.append(item.strip())
            existing.add(key)
    return updated


@app.get("/jobs")
def list_jobs(db: Session = Depends(get_db)) -> list[Job]:
    jobs = (
        db.query(JobModel)
        .order_by(JobModel.role_family, JobModel.location, JobModel.title)
        .all()
    )
    return [job_from_model(job) for job in jobs]


@app.post("/job-matches/generate")
def generate_job_matches(
    request: JobMatchRequest,
    db: Session = Depends(get_db),
) -> JobMatchResponse:
    profile_projects = get_profile_projects(db)
    jobs = (
        db.query(JobModel)
        .filter(JobModel.status == request.status)
        .order_by(JobModel.role_family, JobModel.location, JobModel.title)
        .all()
    )
    resume_query = db.query(GeneratedResumeModel).filter(
        GeneratedResumeModel.profile_id == DEFAULT_PROFILE_ID
    )
    if request.target_direction:
        resume_query = resume_query.filter(
            GeneratedResumeModel.target_direction == request.target_direction
        )
    latest_resume = resume_query.order_by(GeneratedResumeModel.created_at.desc()).first()
    resume_data = resume_from_model(latest_resume).model_dump() if latest_resume else None

    return match_jobs(
        profile_projects.profile,
        profile_projects.projects,
        resume_data,
        [job_from_model(job) for job in jobs],
        request,
    )


@app.post("/skill-gap/analyze")
def analyze_skill_gap(
    request: SkillGapAnalysisRequest,
    db: Session = Depends(get_db),
) -> SkillGapAnalysisResponse:
    profile_projects = get_profile_projects(db)
    return run_skill_gap_analysis(
        request,
        profile_projects.profile,
        profile_projects.projects,
    )


@app.get("/profile")
def get_profile(db: Session = Depends(get_db)) -> Profile:
    return profile_from_model(db.get(ProfileModel, DEFAULT_PROFILE_ID))


@app.post("/profile")
def create_profile(profile: Profile, db: Session = Depends(get_db)) -> Profile:
    return profile_from_model(upsert_profile(db, profile))


@app.put("/profile")
def update_profile(profile: Profile, db: Session = Depends(get_db)) -> Profile:
    return profile_from_model(upsert_profile(db, profile))


@app.get("/projects")
def list_projects(db: Session = Depends(get_db)) -> list[Project]:
    projects = (
        db.query(ProjectModel)
        .filter(ProjectModel.profile_id == DEFAULT_PROFILE_ID)
        .order_by(ProjectModel.title)
        .all()
    )
    return [project_from_model(project) for project in projects]


@app.get("/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)) -> Project:
    project = db.get(ProjectModel, project_id)
    if project is None or project.profile_id != DEFAULT_PROFILE_ID:
        raise HTTPException(status_code=404, detail="Project not found")

    return project_from_model(project)


@app.post("/projects")
def create_project(project: ProjectInput, db: Session = Depends(get_db)) -> Project:
    profile = db.get(ProfileModel, DEFAULT_PROFILE_ID)
    if profile is None:
        profile = ProfileModel(id=DEFAULT_PROFILE_ID)
        db.add(profile)
        db.flush()

    project_model = ProjectModel(
        id=str(uuid4()),
        profile_id=DEFAULT_PROFILE_ID,
        **project.model_dump(),
    )
    db.add(project_model)
    db.commit()
    db.refresh(project_model)
    return project_from_model(project_model)


@app.put("/projects/{project_id}")
def update_project(
    project_id: str,
    project: ProjectInput,
    db: Session = Depends(get_db),
) -> Project:
    project_model = db.get(ProjectModel, project_id)
    if project_model is None or project_model.profile_id != DEFAULT_PROFILE_ID:
        raise HTTPException(status_code=404, detail="Project not found")

    project_model.title = project.title
    project_model.category = project.category
    project_model.role = project.role
    project_model.start_date = project.start_date
    project_model.end_date = project.end_date
    project_model.description = project.description
    project_model.technologies = project.technologies
    project_model.highlights = project.highlights

    db.commit()
    db.refresh(project_model)
    return project_from_model(project_model)


@app.post("/projects/profile/preview")
def preview_project_profile(
    request: ProjectProfilingRequest,
) -> ProjectInput:
    result = PROJECT_PROFILING_TOOL.invoke({"text": request.text})
    return ProjectInput.model_validate(result)


@app.get("/career-directions")
def get_career_directions(
    db: Session = Depends(get_db),
) -> CareerDirectionsSnapshot:
    return career_directions_from_model(db.get(CareerDirectionsModel, DEFAULT_PROFILE_ID))


@app.post("/career-directions/generate")
def generate_career_directions(
    db: Session = Depends(get_db),
) -> CareerDirectionsSnapshot:
    profile_projects = get_profile_projects(db)
    result = CAREER_DIRECTION_TOOL.invoke(
        {
            "profile": profile_projects.profile.model_dump(),
            "projects": [
                project.model_dump() for project in profile_projects.projects
            ],
        }
    )
    recommendations = [
        CareerDirectionRecommendation.model_validate(item)
        for item in result
    ]
    career_directions = db.get(CareerDirectionsModel, DEFAULT_PROFILE_ID)
    if career_directions is None:
        career_directions = CareerDirectionsModel(profile_id=DEFAULT_PROFILE_ID)
        db.add(career_directions)

    career_directions.recommendations = [
        recommendation.model_dump() for recommendation in recommendations
    ]
    db.commit()
    db.refresh(career_directions)
    return career_directions_from_model(career_directions)


@app.post("/resumes/generate")
def generate_resume(
    request: ResumeGenerationRequest,
    db: Session = Depends(get_db),
) -> GeneratedResume:
    profile_projects = get_profile_projects(db)
    result = RESUME_GENERATION_TOOL.invoke(
        {
            "profile": profile_projects.profile.model_dump(),
            "projects": [
                project.model_dump() for project in profile_projects.projects
            ],
            "target_direction": request.target_direction,
        }
    )
    generated_resume = GeneratedResume.model_validate(result)

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


@app.get("/resumes")
def list_generated_resumes(db: Session = Depends(get_db)) -> list[GeneratedResume]:
    resumes = (
        db.query(GeneratedResumeModel)
        .filter(GeneratedResumeModel.profile_id == DEFAULT_PROFILE_ID)
        .order_by(GeneratedResumeModel.created_at.desc())
        .all()
    )
    return [resume_from_model(resume) for resume in resumes]


@app.get("/resumes/{resume_id}")
def get_generated_resume(
    resume_id: str,
    db: Session = Depends(get_db),
) -> GeneratedResume:
    resume = db.get(GeneratedResumeModel, resume_id)
    if resume is None or resume.profile_id != DEFAULT_PROFILE_ID:
        raise HTTPException(status_code=404, detail="Resume not found")

    return resume_from_model(resume)
