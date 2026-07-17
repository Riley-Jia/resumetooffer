from typing import Any, ClassVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.language import detect_output_language, language_instruction
from app.model_config import PROJECT_PROFILING_MODEL
from app.model_logging import invoke_model_with_logging
from app.schemas import ProjectInput


PROFILE_PROJECT_SYSTEM_PROMPT = """
You are a project intake assistant for a resume-to-offer product.
Extract a structured project entry from the user's natural-language project experience.

Return:
- category: "work_experience" if it is company internship/work experience, otherwise "project"
- title: concise project name
- role: user's role in the project
- start_date: start date if provided, otherwise empty string
- end_date: end date if provided, otherwise empty string
- description: resume-ready summary of what the project did and the user's contribution
- technologies: concrete tools, frameworks, languages, platforms, and methods mentioned
- highlights: achievement bullets, outcomes, metrics, or notable responsibilities

Use only evidence from the user's text. Do not invent metrics, dates, tools, or employers.
If a field is not present, return an empty string or empty list.
Keep the language concise and suitable for a resume project section.
Match the user's language. {language_instruction}
"""


def profile_project_text(text: str) -> ProjectInput:
    model = PROJECT_PROFILING_MODEL
    output_language = detect_output_language(text)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROFILE_PROJECT_SYSTEM_PROMPT),
            (
                "human",
                """
User project experience:
{text}
""",
            ),
        ]
    )
    llm = ChatOpenAI(model=model)
    chain = prompt | llm.with_structured_output(ProjectInput, method="json_schema")

    return invoke_model_with_logging(
        "project_profiling",
        model,
        lambda: chain.invoke(
            {
                "text": text,
                "language_instruction": language_instruction(output_language),
            }
        ),
    )


class ProjectProfilingToolInput(BaseModel):
    text: str


class ProjectProfilingTool(BaseTool):
    name: str = "project_profiling_tool"
    description: str = (
        "Extract the standard project fields from a user's natural-language "
        "project experience: title, role, start_date, end_date, description, "
        "technologies, and highlights. Return a preview for user confirmation "
        "before saving it to the projects table."
    )
    args_schema: ClassVar[type[BaseModel]] = ProjectProfilingToolInput

    def _run(self, text: str, **_: Any) -> dict[str, Any]:
        return profile_project_text(text).model_dump()


PROJECT_PROFILING_TOOL = ProjectProfilingTool()
