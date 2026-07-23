from pydantic import ValidationError
import pytest

from app.schemas import (
    AgentState,
    Job,
    JobMatchResult,
    SkillGapAnalysisResponse,
    TaskPlan,
    TaskPlanStep,
    UserPreference,
)


def test_task_plan_schema_defaults_and_refs() -> None:
    plan = TaskPlan(
        steps=[
            TaskPlanStep(
                step_id="job_ranking",
                tool_name="job_matching_tool",
                depends_on=["goal_understanding"],
                input_refs=["profile", "projects"],
                output_refs=["job_matches"],
            )
        ]
    )

    assert plan.steps[0].status == "planned"
    assert plan.steps[0].rerun_policy == "always"
    assert plan.steps[0].depends_on == ["goal_understanding"]


def test_agent_state_schema_holds_preference_and_gap_result() -> None:
    state = AgentState(
        preference=UserPreference(
            target_direction="Backend Developer",
            locations=["Sydney"],
            levels=["Junior"],
        ),
        latest_job_match_ids=["job-1"],
        latest_gap_result=SkillGapAnalysisResponse(gap_severity="mild"),
    )

    assert state.preference.target_direction == "Backend Developer"
    assert state.latest_job_match_ids == ["job-1"]
    assert state.latest_gap_result.gap_severity == "mild"


def test_job_match_result_schema_requires_nested_job() -> None:
    with pytest.raises(ValidationError):
        JobMatchResult.model_validate(
            {
                "final_score": 1,
                "rule_score": 1,
                "llm_score": 1,
                "skill_coverage": 1,
                "location_score": 1,
                "level_score": 1,
                "role_family_score": 1,
                "match_reason": "missing job",
            }
        )

    result = JobMatchResult(
        job=Job(
            id="job-1",
            title="Backend Developer",
            company="A",
            location="Sydney",
            level="Junior",
            role_family="Backend",
        ),
        final_score=100,
        rule_score=100,
        llm_score=100,
        skill_coverage=1,
        location_score=20,
        level_score=15,
        role_family_score=15,
        match_reason="matched",
    )

    assert result.retrieval_fusion_score == 0.0
    assert result.retrieval_source_scores == {}
