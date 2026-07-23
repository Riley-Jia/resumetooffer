from app.job_matching import (
    bm25_retrieve,
    local_vector_retrieve,
    merge_candidates,
    rule_rerank,
)
from app.schemas import Job, JobMatchRequest, Profile


def sample_jobs() -> list[Job]:
    return [
        Job(
            id="backend",
            title="Backend Developer",
            company="A",
            location="Sydney",
            level="Junior",
            role_family="Backend",
            required_skills=["Python", "FastAPI", "SQL"],
            nice_to_have_skills=["PostgreSQL"],
            description="Build REST APIs with Python and FastAPI.",
        ),
        Job(
            id="frontend",
            title="Frontend Developer",
            company="B",
            location="Sydney",
            level="Junior",
            role_family="Frontend",
            required_skills=["React", "TypeScript"],
            description="Build React user interfaces.",
        ),
    ]


def test_bm25_vector_merge_keeps_source_scores() -> None:
    jobs = sample_jobs()
    jobs_by_id = {job.id: job for job in jobs}

    bm25_hits = bm25_retrieve("Python FastAPI backend API", jobs, top_n=2)
    vector_hits = local_vector_retrieve("Python FastAPI backend API", jobs, top_n=2)
    candidates = merge_candidates(jobs_by_id, bm25_hits, vector_hits)

    assert "backend" in candidates
    assert candidates["backend"]["fusion"] > 0
    assert "bm25" in candidates["backend"]
    assert "local_vector" in candidates["backend"]


def test_job_ranking_rule_score_rewards_skill_location_level_and_role_match() -> None:
    jobs = sample_jobs()
    jobs_by_id = {job.id: job for job in jobs}
    candidates = {
        "backend": {"bm25": 1.0, "local_vector": 1.0, "fusion": 1.0},
        "frontend": {"bm25": 0.2, "local_vector": 0.2, "fusion": 0.2},
    }
    request = JobMatchRequest(
        target_direction="Backend Developer",
        locations=["Sydney"],
        levels=["Junior"],
        role_families=["Backend"],
    )
    profile = Profile(location="Sydney", skills=["Python", "FastAPI", "SQL"])

    ranked = rule_rerank(candidates, jobs_by_id, request, profile, [], None)

    assert ranked[0].job.id == "backend"
    assert ranked[0].rule_score == 100.0
    assert ranked[0].retrieval_fusion_score == 100.0
    assert ranked[0].retrieval_source_scores["fusion"] == 100.0
    assert ranked[1].rule_score < ranked[0].rule_score
