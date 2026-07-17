from dataclasses import dataclass
from typing import Any, ClassVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.language import detect_output_language, language_instruction
from app.model_config import CAREER_DIRECTION_MODEL
from app.model_logging import invoke_model_with_logging, log_model_fallback
from app.schemas import CareerDirectionRecommendation, Profile, Project


@dataclass(frozen=True)
class DirectionRule:
    direction: str
    keywords: tuple[str, ...]


class CareerDirectionToolInput(BaseModel):
    profile: Profile
    projects: list[Project] = Field(default_factory=list)


class ReasonOutput(BaseModel):
    reason: str


DIRECTION_RULES = [
    DirectionRule(
        direction="Backend Developer",
        keywords=(
            "fastapi",
            "sql",
            "postgresql",
            "mysql",
            "rest api",
            "api",
            "backend",
            "database",
            "sqlalchemy",
            "docker",
            "python",
        ),
    ),
    DirectionRule(
        direction="AI Application Developer",
        keywords=(
            "langchain",
            "llm",
            "gpt",
            "openai",
            "chromadb",
            "chroma",
            "rag",
            "vector",
            "embedding",
            "prompt",
            "ai",
        ),
    ),
    DirectionRule(
        direction="Data Analyst",
        keywords=(
            "python",
            "pandas",
            "tableau",
            "sql",
            "excel",
            "power bi",
            "data analysis",
            "analytics",
            "visualization",
            "dashboard",
        ),
    ),
    DirectionRule(
        direction="Full Stack Developer",
        keywords=(
            "react",
            "typescript",
            "javascript",
            "fastapi",
            "api",
            "frontend",
            "backend",
            "postgresql",
            "database",
            "full stack",
        ),
    ),
    DirectionRule(
        direction="Frontend Developer",
        keywords=(
            "react",
            "typescript",
            "javascript",
            "html",
            "css",
            "vite",
            "frontend",
            "ui",
            "ux",
        ),
    ),
    DirectionRule(
        direction="Product Manager",
        keywords=(
            "product",
            "roadmap",
            "user research",
            "requirements",
            "stakeholder",
            "metrics",
            "strategy",
            "prioritization",
            "launch",
            "market",
        ),
    ),
    DirectionRule(
        direction="Technical Product Manager",
        keywords=(
            "api",
            "technical",
            "requirements",
            "stakeholder",
            "metrics",
            "roadmap",
            "data",
            "system design",
            "integration",
        ),
    ),
]

REASON_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You write concise, evidence-based career direction reasons.
The score and direction are already determined by deterministic rules.
Do not change the score. Do not invent skills, companies, metrics, or project names.
Explain the fit in one sentence using the matched skills and related projects.
Match the user's language. {language_instruction}
""",
        ),
        (
            "human",
            """
Direction: {direction}
Score: {score}
Profile skills: {profile_skills}
Matched skills: {matched_skills}
Related projects: {related_projects}
""",
        ),
    ]
)


def normalize(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ")


def project_search_text(project: Project) -> str:
    parts = [
        project.category,
        project.title,
        project.role,
        project.description,
        " ".join(project.technologies),
        " ".join(project.highlights),
    ]
    return normalize(" ".join(parts))


def matched_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def score_direction(
    rule: DirectionRule,
    profile: Profile,
    projects: list[Project],
) -> tuple[int, list[str], list[str]]:
    related_projects: list[str] = []
    matched_skills: set[str] = set()
    project_points: list[int] = []

    for project in projects:
        matches = matched_keywords(project_search_text(project), rule.keywords)
        if not matches:
            continue

        related_projects.append(project.title or "Untitled project")
        matched_skills.update(matches)
        project_points.append(min(22, 8 + len(set(matches)) * 4))

    profile_text = normalize(" ".join(profile.skills + [profile.headline, profile.summary]))
    profile_matches = matched_keywords(profile_text, rule.keywords)
    matched_skills.update(profile_matches)

    score = sum(sorted(project_points, reverse=True)[:4])
    score += len(related_projects) * 5
    score += min(12, len(set(profile_matches)) * 3)

    if len(related_projects) == 1:
        score = min(score, 55)

    return min(100, score), sorted(matched_skills), related_projects


def fallback_reason(
    direction: str,
    score: int,
    matched_skills: list[str],
    related_projects: list[str],
    output_language: str,
) -> str:
    if not related_projects:
        if output_language == "Chinese":
            return "暂时没有匹配到足够强的项目证据。"
        return "No strong project evidence matched this direction yet."

    skills = ", ".join(matched_skills[:5]) or "relevant project signals"
    projects = ", ".join(related_projects[:2])
    if output_language == "Chinese":
        return f"{direction} 得分 {score}，因为 {projects} 体现了 {skills} 等相关能力。"

    return (
        f"{direction} scores {score} because {projects} show evidence of "
        f"{skills}."
    )


def build_reason(
    direction: str,
    score: int,
    profile: Profile,
    matched_skills: list[str],
    related_projects: list[str],
) -> str:
    output_language = detect_output_language(
        profile.name,
        profile.headline,
        profile.summary,
        profile.skills,
        related_projects,
    )
    if score == 0 or not related_projects:
        return fallback_reason(direction, score, matched_skills, related_projects, output_language)

    try:
        model = CAREER_DIRECTION_MODEL
        chain = REASON_PROMPT | ChatOpenAI(model=model).with_structured_output(
            ReasonOutput,
            method="json_schema",
        )
        result = invoke_model_with_logging(
            "career_direction_reason",
            model,
            lambda: chain.invoke(
                {
                    "direction": direction,
                    "score": score,
                    "profile_skills": ", ".join(profile.skills),
                    "matched_skills": ", ".join(matched_skills),
                    "related_projects": ", ".join(related_projects[:3]),
                    "language_instruction": language_instruction(output_language),
                }
            ),
        )
        return result.reason
    except Exception:
        log_model_fallback("career_direction_reason", CAREER_DIRECTION_MODEL, "rule_reason")
        return fallback_reason(direction, score, matched_skills, related_projects, output_language)


def recommend_career_directions(
    profile: Profile,
    projects: list[Project],
) -> list[CareerDirectionRecommendation]:
    recommendations: list[CareerDirectionRecommendation] = []

    for rule in DIRECTION_RULES:
        score, matched_skills, related_projects = score_direction(rule, profile, projects)
        recommendations.append(
            CareerDirectionRecommendation(
                direction=rule.direction,
                match_score=score,
                reason=build_reason(
                    rule.direction,
                    score,
                    profile,
                    matched_skills,
                    related_projects,
                ),
                related_projects=related_projects[:3],
            )
        )

    return sorted(recommendations, key=lambda item: item.match_score, reverse=True)


class CareerDirectionTool(BaseTool):
    name: str = "career_direction_tool"
    description: str = (
        "Recommend career directions from a user's profile and projects. "
        "Deterministic rules calculate match_score from project skills and "
        "related project count; gpt-5-nano only writes the evidence-based reason."
    )
    args_schema: ClassVar[type[BaseModel]] = CareerDirectionToolInput

    def _run(
        self,
        profile: Profile | dict[str, Any],
        projects: list[Project | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(profile, dict):
            profile = Profile.model_validate(profile)

        parsed_projects = [
            project if isinstance(project, Project) else Project.model_validate(project)
            for project in projects
        ]

        return [
            recommendation.model_dump()
            for recommendation in recommend_career_directions(profile, parsed_projects)
        ]


CAREER_DIRECTION_TOOL = CareerDirectionTool()
