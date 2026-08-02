from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ProfileModel(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    name: Mapped[str] = mapped_column(String(255), default="")
    headline: Mapped[str] = mapped_column(String(255), default="")
    email: Mapped[str] = mapped_column(String(255), default="")
    phone: Mapped[str] = mapped_column(String(80), default="")
    wechat: Mapped[str] = mapped_column(String(120), default="")
    location: Mapped[str] = mapped_column(String(255), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    skills: Mapped[list[str]] = mapped_column(JSONB, default=list)
    education: Mapped[list[str]] = mapped_column(JSONB, default=list)
    experience: Mapped[list[str]] = mapped_column(JSONB, default=list)
    projects: Mapped[list["ProjectModel"]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
    )


class ProjectModel(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
        default=1,
    )
    category: Mapped[str] = mapped_column(String(40), default="project")
    title: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(255), default="")
    start_date: Mapped[str] = mapped_column(String(40), default="")
    end_date: Mapped[str] = mapped_column(String(40), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    technologies: Mapped[list[str]] = mapped_column(JSONB, default=list)
    highlights: Mapped[list[str]] = mapped_column(JSONB, default=list)
    profile: Mapped[ProfileModel] = relationship(back_populates="projects")


class GeneratedResumeModel(Base):
    __tablename__ = "generated_resumes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
        default=1,
    )
    target_direction: Mapped[str] = mapped_column(String(160))
    introduction: Mapped[str] = mapped_column(Text, default="")
    skills: Mapped[list[str]] = mapped_column(JSONB, default=list)
    projects: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    selected_project_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)


class CareerDirectionsModel(Base):
    __tablename__ = "career_directions"

    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"),
        primary_key=True,
        default=1,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    recommendations: Mapped[list[dict]] = mapped_column(JSONB, default=list)


class JobModel(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    company: Mapped[str] = mapped_column(String(255))
    location: Mapped[str] = mapped_column(String(120))
    level: Mapped[str] = mapped_column(String(120))
    role_family: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(40), default="active", index=True)
    required_skills: Mapped[list[str]] = mapped_column(JSONB, default=list)
    nice_to_have_skills: Mapped[list[str]] = mapped_column(JSONB, default=list)
    description: Mapped[str] = mapped_column(Text, default="")


class AgentRunModel(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
        default=1,
    )
    user_message: Mapped[str] = mapped_column(Text, default="")
    goal: Mapped[dict] = mapped_column(JSONB, default=dict)
    steps: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)


class AgentStateModel(Base):
    __tablename__ = "agent_states"

    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"),
        primary_key=True,
        default=1,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    preferences: Mapped[dict] = mapped_column(JSONB, default=dict)
    latest_resume_id: Mapped[str] = mapped_column(String(36), default="")
    latest_job_match_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    latest_gap_result: Mapped[dict] = mapped_column(JSONB, default=dict)
    feedback_memory: Mapped[dict] = mapped_column(JSONB, default=dict)
    last_agent_run_id: Mapped[str] = mapped_column(String(36), default="")
    last_target_direction: Mapped[str] = mapped_column(String(160), default="")
    project_count: Mapped[int] = mapped_column(Integer, default=0)


class TelemetryEventModel(Base):
    __tablename__ = "telemetry_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    agent_run_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)


class AgentRunEventModel(Base):
    __tablename__ = "agent_run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    agent_run_id: Mapped[str] = mapped_column(String(36), index=True)
    profile_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    target_direction: Mapped[str] = mapped_column(String(160), default="")
    selected_tools: Mapped[list[str]] = mapped_column(JSONB, default=list)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    reused_step_count: Mapped[int] = mapped_column(Integer, default=0)
    rerun_step_count: Mapped[int] = mapped_column(Integer, default=0)
    is_feedback_rerun: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    latest_resume_id: Mapped[str] = mapped_column(String(36), default="")
    latest_job_match_count: Mapped[int] = mapped_column(Integer, default=0)
    gap_severity: Mapped[str] = mapped_column(String(32), default="")


class AgentStepEventModel(Base):
    __tablename__ = "agent_step_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    agent_run_id: Mapped[str] = mapped_column(String(36), index=True)
    step_id: Mapped[str] = mapped_column(String(80), index=True)
    tool_name: Mapped[str] = mapped_column(String(128), default="")
    depends_on: Mapped[list[str]] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(32), default="", index=True)
    rerun_policy: Mapped[str] = mapped_column(String(128), default="")
    input_refs: Mapped[list[str]] = mapped_column(JSONB, default=list)
    output_refs: Mapped[list[str]] = mapped_column(JSONB, default=list)
    is_reused: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_rerun: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class PartialRerunEventModel(Base):
    __tablename__ = "partial_rerun_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    agent_run_id: Mapped[str] = mapped_column(String(36), index=True)
    feedback_text: Mapped[str] = mapped_column(Text, default="")
    changed_preferences: Mapped[dict] = mapped_column(JSONB, default=dict)
    reused_steps: Mapped[list[str]] = mapped_column(JSONB, default=list)
    rerun_steps: Mapped[list[str]] = mapped_column(JSONB, default=list)
    saved_step_count: Mapped[int] = mapped_column(Integer, default=0)
    reuse_rate: Mapped[float] = mapped_column(Numeric(6, 4), default=0)


class RetrievalRankingEventModel(Base):
    __tablename__ = "retrieval_ranking_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    agent_run_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    target_direction: Mapped[str] = mapped_column(String(160), default="", index=True)
    top_k: Mapped[int] = mapped_column(Integer, default=10)
    metadata_candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    bm25_candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    vector_candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    merged_candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    hybrid_overlap_count: Mapped[int] = mapped_column(Integer, default=0)
    llm_rerank_candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    top_matches: Mapped[list[dict]] = mapped_column(JSONB, default=list)


class ModelCallEventModel(Base):
    __tablename__ = "model_call_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    agent_run_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    task: Mapped[str] = mapped_column(String(128), index=True)
    model: Mapped[str] = mapped_column(String(128), default="", index=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fallback: Mapped[str] = mapped_column(String(128), default="")
    error_type: Mapped[str] = mapped_column(String(128), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(Numeric(12, 6), nullable=True)
