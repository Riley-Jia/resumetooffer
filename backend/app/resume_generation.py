from typing import Any, ClassVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.career_direction import DIRECTION_RULES, matched_keywords, project_search_text
from app.language import detect_output_language, language_instruction
from app.model_config import RESUME_GENERATION_MODEL
from app.model_logging import invoke_model_with_logging, log_model_fallback
from app.schemas import GeneratedResume, Profile, Project, ResumeProjectSection


class ResumeGenerationToolInput(BaseModel):
    profile: Profile
    projects: list[Project] = Field(default_factory=list)
    target_direction: str


class ResumeDraft(BaseModel):
    introduction: str = ""
    skills: list[str] = Field(default_factory=list)
    projects: list[ResumeProjectSection] = Field(default_factory=list)


RESUME_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是中文简历生成助手。你必须严格基于用户提供的 Profile 和项目经历改写，
不能编造学校、公司、项目、时间、技术栈、指标、职责或成果。
输出语言必须和用户输入/Profile/项目经历的主要语言一致。{language_instruction}

输出中文简历文本内容的数据结构：
- introduction: 2-3 句中文个人介绍，面向目标方向
- skills: 技能列表，只能来自 Profile skills 与项目 technologies/description/highlights
- projects: 2-3 个项目，每个项目包含 title、role、period、description、technologies、details

写法要求：
- 项目 description 用“项目简介：...”风格
- details 写成职责/项目细节要点，优先使用用户原始 highlights
- 每条 details 先描述做了什么功能，再补充约 30 字说明实现方式、处理逻辑或对简历表达的价值
- 每条 details 只能扩写已有项目事实，不能补充用户没有提供的新技术、新指标或新成果
- 如果没有证据，留空或弱化表达，不要补事实
- 不输出 Markdown
""",
        ),
        (
            "human",
            """
目标方向：{target_direction}

Profile:
姓名：{name}
标题：{headline}
简介：{summary}
技能：{profile_skills}
教育背景：{education}

已按规则选中的项目：
{projects}
""",
        ),
    ]
)


def direction_keywords(target_direction: str) -> tuple[str, ...]:
    normalized_target = target_direction.lower()
    for rule in DIRECTION_RULES:
        if rule.direction.lower() == normalized_target:
            return rule.keywords

    for rule in DIRECTION_RULES:
        if normalized_target in rule.direction.lower() or rule.direction.lower() in normalized_target:
            return rule.keywords

    return tuple()


def content_score(project: Project) -> int:
    return (
        len(project.description)
        + sum(len(item) for item in project.highlights)
        + len(project.technologies) * 20
    )


def select_resume_projects(
    projects: list[Project],
    target_direction: str,
) -> list[Project]:
    keywords = direction_keywords(target_direction)
    scored_projects: list[tuple[int, Project]] = []

    for project in projects:
        matches = matched_keywords(project_search_text(project), keywords)
        if matches:
            scored_projects.append((len(set(matches)) * 100 + content_score(project), project))

    selected = [project for _, project in sorted(scored_projects, reverse=True, key=lambda item: item[0])]

    if len(selected) < 2:
        selected_ids = {project.id for project in selected}
        fallback_projects = sorted(
            [project for project in projects if project.id not in selected_ids],
            key=content_score,
            reverse=True,
        )
        selected.extend(fallback_projects[: 2 - len(selected)])

    return selected[:3]


def format_projects_for_prompt(projects: list[Project]) -> str:
    blocks: list[str] = []
    for project in projects:
        period = " - ".join(
            item for item in [project.start_date, project.end_date] if item
        )
        blocks.append(
            "\n".join(
                [
                    f"项目ID：{project.id}",
                    f"分类：{project.category}",
                    f"标题：{project.title}",
                    f"角色：{project.role}",
                    f"时间：{period}",
                    f"描述：{project.description}",
                    f"技术栈：{', '.join(project.technologies)}",
                    "要点：",
                    *[f"- {highlight}" for highlight in project.highlights],
                ]
            )
        )

    return "\n\n".join(blocks)


def fallback_resume(
    profile: Profile,
    selected_projects: list[Project],
    target_direction: str,
    output_language: str,
) -> ResumeDraft:
    project_sections = [
        ResumeProjectSection(
            title=project.title,
            role=project.role,
            period=" - ".join(
                item for item in [project.start_date, project.end_date] if item
            ),
            description=project.description,
            technologies=project.technologies,
            details=project.highlights,
        )
        for project in selected_projects
    ]

    return ResumeDraft(
        introduction=profile.summary
        or (
            f"目标方向为{target_direction}，具备相关项目实践经验。"
            if output_language == "Chinese"
            else f"Targeting {target_direction}, with relevant project experience."
        ),
        skills=profile.skills,
        projects=project_sections,
    )


def generate_resume_content(
    profile: Profile,
    projects: list[Project],
    target_direction: str,
) -> GeneratedResume:
    selected_projects = select_resume_projects(projects, target_direction)
    output_language = detect_output_language(
        profile.name,
        profile.headline,
        profile.summary,
        profile.skills,
        profile.education,
        [project.model_dump() for project in selected_projects],
    )

    try:
        model = RESUME_GENERATION_MODEL
        chain = RESUME_PROMPT | ChatOpenAI(model=model).with_structured_output(
            ResumeDraft,
            method="json_schema",
        )
        draft = invoke_model_with_logging(
            "resume_generation",
            model,
            lambda: chain.invoke(
                {
                    "target_direction": target_direction,
                    "name": profile.name,
                    "headline": profile.headline,
                    "summary": profile.summary,
                    "profile_skills": ", ".join(profile.skills),
                    "education": "\n".join(profile.education),
                    "projects": format_projects_for_prompt(selected_projects),
                    "language_instruction": language_instruction(output_language),
                }
            ),
        )
    except Exception:
        log_model_fallback("resume_generation", RESUME_GENERATION_MODEL, "template_resume")
        draft = fallback_resume(profile, selected_projects, target_direction, output_language)

    return GeneratedResume(
        id="",
        target_direction=target_direction,
        created_at="",
        introduction=draft.introduction,
        skills=draft.skills,
        projects=draft.projects,
        selected_project_ids=[project.id for project in selected_projects],
    )


class ResumeGenerationTool(BaseTool):
    name: str = "resume_generation_tool"
    description: str = (
        "Select the most relevant 2-3 projects for a target career direction and "
        "generate Chinese resume text. Scores and project selection are grounded "
        "in user-provided profile and project data; the model rewrites but must "
        "not invent experiences."
    )
    args_schema: ClassVar[type[BaseModel]] = ResumeGenerationToolInput

    def _run(
        self,
        profile: Profile | dict[str, Any],
        projects: list[Project | dict[str, Any]],
        target_direction: str,
    ) -> dict[str, Any]:
        if isinstance(profile, dict):
            profile = Profile.model_validate(profile)

        parsed_projects = [
            project if isinstance(project, Project) else Project.model_validate(project)
            for project in projects
        ]
        return generate_resume_content(
            profile,
            parsed_projects,
            target_direction,
        ).model_dump()


RESUME_GENERATION_TOOL = ResumeGenerationTool()
