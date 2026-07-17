import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import JobModel


SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "jobs_seed.json"


def seed_jobs(db: Session) -> None:
    seed_jobs_data = json.loads(SEED_PATH.read_text(encoding="utf-8"))

    for item in seed_jobs_data:
        job = db.get(JobModel, item["id"])
        if job is None:
            job = JobModel(id=item["id"])
            db.add(job)

        job.title = item["title"]
        job.company = item["company"]
        job.location = item["location"]
        job.level = item["level"]
        job.role_family = item["role_family"]
        job.status = item.get("status", "active")
        job.required_skills = item["required_skills"]
        job.nice_to_have_skills = item["nice_to_have_skills"]
        job.description = item["description"]

    db.commit()
