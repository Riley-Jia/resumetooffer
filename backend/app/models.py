from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
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
