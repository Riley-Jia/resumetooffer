from app.input_router import fallback_route


def test_router_fallback_routes_resume_command_to_agent() -> None:
    result = fallback_route("帮我生成一版后端开发简历，并分析差距")

    assert result.intent == "run_career_agent"
    assert result.route == "career_agent_run"
    assert result.confidence >= 0.8


def test_router_fallback_routes_work_experience_to_project_edit() -> None:
    result = fallback_route("我在云帆科技有限公司实习，负责 FastAPI 后端接口开发")

    assert result.intent == "edit_project"
    assert result.route == "profile_project_edit_preview"
    assert result.content_type == "work_experience_project"


def test_router_fallback_routes_job_posting_to_confirmation() -> None:
    result = fallback_route("JD：岗位职责包括开发 REST API，任职要求 Python 和 SQL")

    assert result.intent == "analyze_job_posting"
    assert result.route == "need_confirmation"
    assert result.content_type == "job_posting"
