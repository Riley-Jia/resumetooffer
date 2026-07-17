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
        connection.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS status VARCHAR(40) DEFAULT 'active'"))
        connection.execute(text("UPDATE jobs SET status = 'active' WHERE status IS NULL OR status = ''"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
