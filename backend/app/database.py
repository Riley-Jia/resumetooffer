import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://resume_user:resume_password@localhost:5432/resumetooffer",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS wechat VARCHAR(120) DEFAULT ''"))
        connection.execute(text("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS education JSONB DEFAULT '[]'::jsonb"))
        connection.execute(text("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS experience JSONB DEFAULT '[]'::jsonb"))
        connection.execute(
            text(
                "INSERT INTO profiles "
                "(id, name, headline, email, phone, wechat, location, summary, "
                "skills, education, experience) "
                "VALUES (1, '', '', '', '', '', '', '', '[]'::jsonb, "
                "'[]'::jsonb, '[]'::jsonb) "
                "ON CONFLICT (id) DO NOTHING"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE projects "
                "ADD COLUMN IF NOT EXISTS profile_id INTEGER "
                "REFERENCES profiles(id) ON DELETE CASCADE"
            )
        )
        connection.execute(text("UPDATE projects SET profile_id = 1 WHERE profile_id IS NULL"))
        connection.execute(text("ALTER TABLE projects ALTER COLUMN profile_id SET NOT NULL"))
        connection.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS category VARCHAR(40) DEFAULT 'project'"))
        connection.execute(text("UPDATE projects SET category = 'project' WHERE category IS NULL OR category = ''"))
        connection.execute(text("ALTER TABLE projects DROP COLUMN IF EXISTS skills"))
        connection.execute(text("ALTER TABLE projects DROP COLUMN IF EXISTS role_type"))
        connection.execute(text("ALTER TABLE projects DROP COLUMN IF EXISTS strengths"))
        connection.execute(text("ALTER TABLE projects DROP COLUMN IF EXISTS suitable_directions"))
        connection.execute(
            text(
                "ALTER TABLE generated_resumes "
                "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS career_directions ("
                "profile_id INTEGER PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE, "
                "updated_at TIMESTAMPTZ DEFAULT now(), "
                "recommendations JSONB DEFAULT '[]'::jsonb"
                ")"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS agent_runs ("
                "id VARCHAR(36) PRIMARY KEY, "
                "created_at TIMESTAMPTZ DEFAULT now(), "
                "profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE, "
                "user_message TEXT DEFAULT '', "
                "goal JSONB DEFAULT '{}'::jsonb, "
                "steps JSONB DEFAULT '[]'::jsonb, "
                "result JSONB DEFAULT '{}'::jsonb"
                ")"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS agent_states ("
                "profile_id INTEGER PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE, "
                "updated_at TIMESTAMPTZ DEFAULT now(), "
                "preferences JSONB DEFAULT '{}'::jsonb, "
                "latest_resume_id VARCHAR(36) DEFAULT '', "
                "latest_job_match_ids JSONB DEFAULT '[]'::jsonb, "
                "latest_gap_result JSONB DEFAULT '{}'::jsonb, "
                "feedback_memory JSONB DEFAULT '{}'::jsonb, "
                "last_agent_run_id VARCHAR(36) DEFAULT '', "
                "last_target_direction VARCHAR(160) DEFAULT '', "
                "project_count INTEGER DEFAULT 0"
                ")"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS telemetry_events ("
                "id SERIAL PRIMARY KEY, "
                "created_at TIMESTAMPTZ DEFAULT now(), "
                "event_type VARCHAR(80) NOT NULL, "
                "agent_run_id VARCHAR(36) DEFAULT '', "
                "payload JSONB DEFAULT '{}'::jsonb"
                ")"
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_telemetry_events_created_at ON telemetry_events(created_at)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_telemetry_events_event_type ON telemetry_events(event_type)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_telemetry_events_agent_run_id ON telemetry_events(agent_run_id)"))
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS agent_run_events ("
                "id SERIAL PRIMARY KEY, "
                "created_at TIMESTAMPTZ DEFAULT now(), "
                "agent_run_id VARCHAR(36) NOT NULL, "
                "profile_id INTEGER DEFAULT 1, "
                "target_direction VARCHAR(160) DEFAULT '', "
                "selected_tools JSONB DEFAULT '[]'::jsonb, "
                "step_count INTEGER DEFAULT 0, "
                "reused_step_count INTEGER DEFAULT 0, "
                "rerun_step_count INTEGER DEFAULT 0, "
                "is_feedback_rerun BOOLEAN DEFAULT false, "
                "latest_resume_id VARCHAR(36) DEFAULT '', "
                "latest_job_match_count INTEGER DEFAULT 0, "
                "gap_severity VARCHAR(32) DEFAULT ''"
                ")"
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_run_events_agent_run_id ON agent_run_events(agent_run_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_run_events_created_at ON agent_run_events(created_at)"))
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS agent_step_events ("
                "id SERIAL PRIMARY KEY, "
                "created_at TIMESTAMPTZ DEFAULT now(), "
                "agent_run_id VARCHAR(36) NOT NULL, "
                "step_id VARCHAR(80) NOT NULL, "
                "tool_name VARCHAR(128) DEFAULT '', "
                "depends_on JSONB DEFAULT '[]'::jsonb, "
                "status VARCHAR(32) DEFAULT '', "
                "rerun_policy VARCHAR(128) DEFAULT '', "
                "input_refs JSONB DEFAULT '[]'::jsonb, "
                "output_refs JSONB DEFAULT '[]'::jsonb, "
                "is_reused BOOLEAN DEFAULT false, "
                "is_rerun BOOLEAN DEFAULT false"
                ")"
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_step_events_agent_run_id ON agent_step_events(agent_run_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_step_events_step_id ON agent_step_events(step_id)"))
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS partial_rerun_events ("
                "id SERIAL PRIMARY KEY, "
                "created_at TIMESTAMPTZ DEFAULT now(), "
                "agent_run_id VARCHAR(36) NOT NULL, "
                "feedback_text TEXT DEFAULT '', "
                "changed_preferences JSONB DEFAULT '{}'::jsonb, "
                "reused_steps JSONB DEFAULT '[]'::jsonb, "
                "rerun_steps JSONB DEFAULT '[]'::jsonb, "
                "saved_step_count INTEGER DEFAULT 0, "
                "reuse_rate NUMERIC(6, 4) DEFAULT 0"
                ")"
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_partial_rerun_events_agent_run_id ON partial_rerun_events(agent_run_id)"))
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS retrieval_ranking_events ("
                "id SERIAL PRIMARY KEY, "
                "created_at TIMESTAMPTZ DEFAULT now(), "
                "agent_run_id VARCHAR(36) DEFAULT '', "
                "target_direction VARCHAR(160) DEFAULT '', "
                "top_k INTEGER DEFAULT 10, "
                "metadata_candidate_count INTEGER DEFAULT 0, "
                "bm25_candidate_count INTEGER DEFAULT 0, "
                "vector_candidate_count INTEGER DEFAULT 0, "
                "merged_candidate_count INTEGER DEFAULT 0, "
                "hybrid_overlap_count INTEGER DEFAULT 0, "
                "llm_rerank_candidate_count INTEGER DEFAULT 0, "
                "top_matches JSONB DEFAULT '[]'::jsonb"
                ")"
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_retrieval_ranking_events_agent_run_id ON retrieval_ranking_events(agent_run_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_retrieval_ranking_events_target_direction ON retrieval_ranking_events(target_direction)"))
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS model_call_events ("
                "id SERIAL PRIMARY KEY, "
                "created_at TIMESTAMPTZ DEFAULT now(), "
                "agent_run_id VARCHAR(36) DEFAULT '', "
                "task VARCHAR(128) NOT NULL, "
                "model VARCHAR(128) DEFAULT '', "
                "event_type VARCHAR(40) NOT NULL, "
                "success BOOLEAN, "
                "elapsed_ms INTEGER, "
                "fallback VARCHAR(128) DEFAULT '', "
                "error_type VARCHAR(128) DEFAULT '', "
                "error_message TEXT DEFAULT '', "
                "prompt_tokens INTEGER, "
                "completion_tokens INTEGER, "
                "estimated_cost NUMERIC(12, 6)"
                ")"
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_model_call_events_agent_run_id ON model_call_events(agent_run_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_model_call_events_task ON model_call_events(task)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_model_call_events_event_type ON model_call_events(event_type)"))
        connection.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS status VARCHAR(40) DEFAULT 'active'"))
        connection.execute(text("UPDATE jobs SET status = 'active' WHERE status IS NULL OR status = ''"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
