import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.model_config import INPUT_ROUTER_MODEL
from app.model_logging import invoke_model_with_logging, log_model_fallback
from app.schemas import InputRouterResponse, Profile, Project


ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Input Router Tool for a resume-to-offer product.
Classify what the user entered and what they want the system to do.

Return:
- intent: edit_profile, edit_project, create_project_preview, run_career_agent,
  analyze_job_posting, need_confirmation, unknown
- content_type: profile_education, profile_experience, profile_contact,
  profile_skills, profile_summary, project_experience, project_update,
  work_experience_project, job_search_goal, job_posting, command, mixed, unknown
- route: profile_project_edit_preview, career_agent_run,
  project_profile_preview, job_posting_analysis, need_confirmation
- confidence: 0 to 1
- reason: concise explanation
- normalized_instruction: rewrite the instruction for the selected tool

Routing rules:
- Personal education, contact info, skills, location, or summary should route
  to profile_project_edit_preview as Profile edits.
- User's own company work/internship experience should route to
  profile_project_edit_preview as a Project edit/create with category work_experience,
  not profile.experience.
- Existing project updates or new project facts should route to
  profile_project_edit_preview unless the user only asks for extraction preview.
- A command like "generate resume, recommend jobs, analyze gaps" should route to
  career_agent_run.
- A pasted job description should not be added to Profile. Route it to
  job_posting_analysis or need_confirmation if that tool is not available.
- If mixed with profile/project edits and career-agent commands, route first to
  profile_project_edit_preview, and set follow_up_route to career_agent_run.
  If unclear, use need_confirmation with confidence below 0.7.
""",
        ),
        (
            "human",
            """
User input:
{message}

Current profile:
{profile}

Known project titles:
{project_titles}
""",
        ),
    ]
)


def fallback_route(message: str) -> InputRouterResponse:
    normalized = message.lower()
    job_posting_terms = ["招聘", "职位描述", "岗位职责", "任职要求", "jd", "hiring", "requirements"]
    career_command_terms = ["生成简历", "简历", "推荐岗位", "岗位推荐", "推荐", "岗位", "分析差距", "学习计划", "投", "申请", "run agent"]
    profile_terms = [
        "教育经历",
        "学历",
        "毕业",
        "就读",
        "大学",
        "本科",
        "硕士",
        "博士",
        "工作经历",
        "实习",
        "联系方式",
        "电话",
        "邮箱",
        "微信",
        "技能",
        "location",
        "profile",
    ]
    project_update_terms = ["项目", "系统", "平台", "工具", "应用", "网站", "Agent", "技术栈", "亮点", "负责", "实现", "开发", "我在"]

    if any(term in normalized for term in job_posting_terms):
        return InputRouterResponse(
            intent="analyze_job_posting",
            content_type="job_posting",
            route="need_confirmation",
            confidence=0.72,
            reason="输入看起来像岗位 JD，不应该直接写入 Profile。",
            normalized_instruction=message,
        )

    has_career_command = any(term in normalized for term in career_command_terms)
    has_work_experience = bool(
        re.search(
            r"实习|工作经历|工作经验|我在.+(?:公司|有限公司|科技|Tech|Ltd).*(?:工作|实习|担任|做过)",
            message,
            flags=re.IGNORECASE,
        )
    )

    if any(term in normalized for term in profile_terms):
        content_type = "work_experience_project" if has_work_experience else "profile_education"
        if re.search(r"技能|skill", normalized):
            content_type = "profile_skills"
        response = InputRouterResponse(
            intent="edit_project" if has_work_experience else "edit_profile",
            content_type=content_type,
            route="profile_project_edit_preview",
            confidence=0.82,
            reason=(
                "输入描述的是用户自己的工作/实习经历，应作为 work_experience 类型项目先生成编辑预览。"
                if has_work_experience
                else "输入描述的是用户自己的 Profile 信息，应先生成编辑预览。"
            ),
            normalized_instruction=message,
        )
        if has_career_command:
            response.content_type = "mixed"
            response.follow_up_route = "career_agent_run"
            response.follow_up_instruction = message
        return response

    if any(term in normalized for term in project_update_terms) and not any(
        term in normalized for term in career_command_terms
    ):
        return InputRouterResponse(
            intent="edit_project",
            content_type="project_update",
            route="profile_project_edit_preview",
            confidence=0.75,
            reason="输入看起来是在描述或修改项目经历。",
            normalized_instruction=message,
        )

    if any(term in normalized for term in career_command_terms):
        return InputRouterResponse(
            intent="run_career_agent",
            content_type="command",
            route="career_agent_run",
            confidence=0.84,
            reason="输入是求职流程命令，应运行 Career Agent。",
            normalized_instruction=message,
        )

    return InputRouterResponse(
        intent="unknown",
        content_type="unknown",
        route="need_confirmation",
        confidence=0.4,
        reason="无法稳定判断用户想编辑资料还是运行求职流程。",
        normalized_instruction=message,
    )


def route_user_input(
    message: str,
    profile: Profile,
    projects: list[Project],
) -> InputRouterResponse:
    def correct_route(response: InputRouterResponse) -> InputRouterResponse:
        has_career_command = bool(re.search(r"生成.*简历|简历|推荐.*岗位|岗位推荐|岗位|分析差距|学习计划|投|申请", message))
        has_profile_or_project_info = bool(
            re.search(r"教育经历|学历|毕业|就读|大学|本科|硕士|博士|实习|工作经历|技能|技术栈|location|项目|系统|平台|工具|应用|网站|Agent", message)
        )
        if has_career_command and has_profile_or_project_info:
            response.route = "profile_project_edit_preview"
            response.content_type = "mixed"
            response.follow_up_route = "career_agent_run"
            response.follow_up_instruction = message
        elif response.follow_up_route:
            response.follow_up_route = ""
            response.follow_up_instruction = ""
        if re.search(
            r"实习|工作经历|工作经验|我在.+(?:公司|有限公司|科技|Tech|Ltd).*(?:工作|实习|担任|做过)",
            message,
            flags=re.IGNORECASE,
        ):
            if response.route == "profile_project_edit_preview":
                response.intent = "edit_project"
                response.content_type = "mixed" if has_career_command else "work_experience_project"
                response.reason = "输入描述的是用户自己的工作/实习经历，应作为 work_experience 类型项目先生成编辑预览。"
                if has_career_command:
                    response.follow_up_route = "career_agent_run"
                    response.follow_up_instruction = message
        return response

    try:
        routed = invoke_model_with_logging(
            "input_router",
            INPUT_ROUTER_MODEL,
            lambda: (
                ROUTE_PROMPT
                | ChatOpenAI(model=INPUT_ROUTER_MODEL).with_structured_output(
                    InputRouterResponse,
                    method="json_schema",
                )
            ).invoke(
                {
                    "message": message,
                    "profile": profile.model_dump(mode="json"),
                    "project_titles": ", ".join(project.title for project in projects),
                }
            ),
        )
        if routed.confidence >= 0.55 and routed.route:
            return correct_route(routed)
        fallback = fallback_route(message)
        if fallback.confidence > routed.confidence:
            return correct_route(fallback)
        return correct_route(routed)
    except Exception:
        log_model_fallback("input_router", INPUT_ROUTER_MODEL, "rule_router")
        return correct_route(fallback_route(message))
