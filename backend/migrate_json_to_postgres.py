import json
from pathlib import Path

from app.database import SessionLocal, create_tables
from app.main import DEFAULT_PROFILE_ID, upsert_profile
from app.models import ProjectModel
from app.schemas import ProfileProjects


DATA_FILE = Path(__file__).resolve().parent / "data" / "profile_projects.json"


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"JSON data file not found: {DATA_FILE}")

    with DATA_FILE.open("r", encoding="utf-8") as file:
        data = ProfileProjects.model_validate(json.load(file))

    create_tables()

    with SessionLocal() as db:
        upsert_profile(db, data.profile)

        for project in data.projects:
            project_model = db.get(ProjectModel, project.id)
            if project_model is None:
                project_model = ProjectModel(id=project.id, profile_id=DEFAULT_PROFILE_ID)
                db.add(project_model)

            project_model.profile_id = DEFAULT_PROFILE_ID
            project_model.title = project.title
            project_model.role = project.role
            project_model.start_date = project.start_date
            project_model.end_date = project.end_date
            project_model.description = project.description
            project_model.technologies = project.technologies
            project_model.highlights = project.highlights

        db.commit()

    print("Migrated profile and projects JSON data to PostgreSQL.")


if __name__ == "__main__":
    main()
