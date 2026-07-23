from typing import Any

from pydantic import BaseModel, Field


class Profile(BaseModel):
    name: str = ""
    headline: str = ""
    email: str = ""
    phone: str = ""
    wechat: str = ""
    location: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)


class ProjectInput(BaseModel):
    category: str = "project"
    title: str = ""
    role: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)


class Project(ProjectInput):
    id: str


class ProfileProjects(BaseModel):
    profile: Profile = Field(default_factory=Profile)
    projects: list[Project] = Field(default_factory=list)


class ProjectProfilingRequest(BaseModel):
    text: str


class CareerDirectionRecommendation(BaseModel):
    direction: str
    match_score: int
    reason: str
    related_projects: list[str] = Field(default_factory=list)


class CareerDirectionsSnapshot(BaseModel):
    recommendations: list[CareerDirectionRecommendation] = Field(default_factory=list)
    updated_at: str = ""


class ResumeGenerationRequest(BaseModel):
    target_direction: str


class ResumeProjectSection(BaseModel):
    title: str = ""
    role: str = ""
    period: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)


class GeneratedResume(BaseModel):
    id: str
    target_direction: str
    created_at: str = ""
    introduction: str = ""
    skills: list[str] = Field(default_factory=list)
    projects: list[ResumeProjectSection] = Field(default_factory=list)
    selected_project_ids: list[str] = Field(default_factory=list)


class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    level: str
    role_family: str
    status: str = "active"
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    description: str = ""


class JobMatchRequest(BaseModel):
    target_direction: str = ""
    locations: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    role_families: list[str] = Field(default_factory=list)
    status: str = "active"
    top_k: int = 10
    llm_candidate_count: int = 20


class JobMatchResult(BaseModel):
    job: Job
    final_score: float
    rule_score: float
    llm_score: float
    retrieval_fusion_score: float = 0.0
    skill_coverage: float
    location_score: float
    level_score: float
    role_family_score: float
    match_reason: str
    missing_skills: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    retrieval_sources: list[str] = Field(default_factory=list)
    retrieval_source_scores: dict[str, float] = Field(default_factory=dict)


class JobMatchResponse(BaseModel):
    matches: list[JobMatchResult] = Field(default_factory=list)
    metadata_filter: dict[str, list[str] | str] = Field(default_factory=dict)
    candidate_counts: dict[str, int] = Field(default_factory=dict)


class SkillGapAnalysisRequest(BaseModel):
    jobs: list[Job] = Field(default_factory=list)
    top_matches: list[JobMatchResult] = Field(default_factory=list)
    user_skills: list[str] = Field(default_factory=list)
    target_direction: str = ""


class JobSkillGap(BaseModel):
    job_id: str
    title: str
    company: str
    missing_skills: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    evaluation_description: str = ""


class NextStepPlanWeek(BaseModel):
    week: int
    focus: str
    plan_type: str = "learning"
    goals: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    deliverable: str = ""


LearningPlanWeek = NextStepPlanWeek


class SkillGapAnalysisResponse(BaseModel):
    has_gap: bool = False
    gap_severity: str = "none"
    gap_summary: str = ""
    common_missing_skills: list[str] = Field(default_factory=list)
    priority_skills: list[str] = Field(default_factory=list)
    per_job_gaps: list[JobSkillGap] = Field(default_factory=list)
    next_step_plan: list[NextStepPlanWeek] = Field(default_factory=list)
    learning_plan: list[LearningPlanWeek] = Field(default_factory=list)


class UserPreference(BaseModel):
    target_direction: str = ""
    locations: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    role_families: list[str] = Field(default_factory=list)


class FeedbackEntry(BaseModel):
    message: str
    feedback_type: str = "preference_update"
    extracted_preferences: UserPreference = Field(default_factory=UserPreference)
    rerun_tools: list[str] = Field(default_factory=list)


class FeedbackMemory(BaseModel):
    entries: list[FeedbackEntry] = Field(default_factory=list)
    last_feedback: str = ""


class AgentState(BaseModel):
    profile_id: int = 1
    preference: UserPreference = Field(default_factory=UserPreference)
    latest_resume_id: str = ""
    latest_job_match_ids: list[str] = Field(default_factory=list)
    latest_gap_result: SkillGapAnalysisResponse = Field(default_factory=SkillGapAnalysisResponse)
    feedback_memory: FeedbackMemory = Field(default_factory=FeedbackMemory)
    last_agent_run_id: str = ""
    last_target_direction: str = ""
    project_count: int = 0
    updated_at: str = ""
    summary: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class CareerAgentRequest(BaseModel):
    message: str


class InputRouterRequest(BaseModel):
    message: str


class InputRouterResponse(BaseModel):
    intent: str = "unknown"
    content_type: str = "unknown"
    route: str = "need_confirmation"
    confidence: float = 0.0
    reason: str = ""
    normalized_instruction: str = ""
    follow_up_route: str = ""
    follow_up_instruction: str = ""


class ProfileProjectEditRequest(BaseModel):
    message: str
    router_intent: str = ""
    router_content_type: str = ""
    normalized_instruction: str = ""


class ProfileEditPatch(BaseModel):
    set_fields: dict[str, str] = Field(default_factory=dict)
    skills_add: list[str] = Field(default_factory=list)
    skills_remove: list[str] = Field(default_factory=list)
    education_add: list[str] = Field(default_factory=list)
    education_remove: list[str] = Field(default_factory=list)
    experience_add: list[str] = Field(default_factory=list)
    experience_remove: list[str] = Field(default_factory=list)


class ProjectEditPatch(BaseModel):
    action: str = "update"
    project_id: str = ""
    match_title: str = ""
    set_fields: dict[str, str] = Field(default_factory=dict)
    technologies_add: list[str] = Field(default_factory=list)
    technologies_remove: list[str] = Field(default_factory=list)
    highlights_add: list[str] = Field(default_factory=list)
    highlights_remove: list[str] = Field(default_factory=list)
    create_project: ProjectInput | None = None


class ProfileProjectEditPatch(BaseModel):
    profile: ProfileEditPatch = Field(default_factory=ProfileEditPatch)
    projects: list[ProjectEditPatch] = Field(default_factory=list)


class EditChangePreview(BaseModel):
    target: str
    action: str
    before: Any = ""
    after: Any = ""


class FollowUpQuestion(BaseModel):
    field: str
    question: str
    priority: str = "medium"
    scope: str = "profile"


class InformationCompletenessResult(BaseModel):
    score: int = 0
    status: str = "incomplete"
    can_continue: bool = True
    missing_required: list[str] = Field(default_factory=list)
    missing_recommended: list[str] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
    follow_up_questions: list[FollowUpQuestion] = Field(default_factory=list)


class ProfileProjectEditPreview(BaseModel):
    message: str
    patch: ProfileProjectEditPatch = Field(default_factory=ProfileProjectEditPatch)
    changes: list[EditChangePreview] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    completeness: InformationCompletenessResult = Field(default_factory=InformationCompletenessResult)
    debug: dict[str, Any] = Field(default_factory=dict)
    has_changes: bool = False


class ProfileProjectEditApplyRequest(BaseModel):
    patch: ProfileProjectEditPatch


class ProfileProjectEditApplyResponse(BaseModel):
    profile: Profile = Field(default_factory=Profile)
    projects: list[Project] = Field(default_factory=list)
    applied_changes: list[EditChangePreview] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CareerAgentGoal(BaseModel):
    target_direction: str = ""
    locations: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    role_families: list[str] = Field(default_factory=list)
    timeline_weeks: int | None = None
    project_notes: str = ""


class AgentExecutionStep(BaseModel):
    name: str
    status: str
    detail: str = ""


class TaskPlanStep(BaseModel):
    step_id: str
    tool_name: str
    depends_on: list[str] = Field(default_factory=list)
    status: str = "planned"
    rerun_policy: str = "always"
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    steps: list[TaskPlanStep] = Field(default_factory=list)


class WorkflowStepReuse(BaseModel):
    step_id: str
    tool_name: str
    status: str = "reused"
    source_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    reason: str = ""


class CareerAgentResponse(BaseModel):
    user_message: str
    goal: CareerAgentGoal
    execution_plan: list[str] = Field(default_factory=list)
    task_plan: TaskPlan = Field(default_factory=TaskPlan)
    reused_steps: list[WorkflowStepReuse] = Field(default_factory=list)
    rerun_steps: list[TaskPlanStep] = Field(default_factory=list)
    steps: list[AgentExecutionStep] = Field(default_factory=list)
    project_profile_preview: ProjectInput | None = None
    career_directions: CareerDirectionsSnapshot = Field(default_factory=CareerDirectionsSnapshot)
    generated_resume: GeneratedResume | None = None
    job_matches: JobMatchResponse = Field(default_factory=JobMatchResponse)
    skill_gap: SkillGapAnalysisResponse = Field(default_factory=SkillGapAnalysisResponse)
    state: AgentState = Field(default_factory=AgentState)
