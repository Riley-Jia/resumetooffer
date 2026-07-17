import re
from difflib import SequenceMatcher

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.profiling import profile_project_text
from app.model_config import PROFILE_PROJECT_EDITING_MODEL
from app.model_logging import invoke_model_with_logging, log_model_fallback
from app.information_completeness import check_information_completeness
from app.schemas import (
    EditChangePreview,
    Profile,
    ProfileEditPatch,
    ProfileProjectEditPatch,
    ProfileProjectEditPreview,
    Project,
    ProjectInput,
    ProjectEditPatch,
)


PROFILE_SCALAR_FIELDS = {"name", "headline", "email", "phone", "wechat", "location", "summary"}
PROJECT_SCALAR_FIELDS = {"category", "title", "role", "start_date", "end_date", "description"}
PROJECT_ROUTE_CONTENT_TYPES = {
    "project_experience",
    "project_update",
    "work_experience_project",
    "mixed",
}


class EditDraft(BaseModel):
    patch: ProfileProjectEditPatch = Field(default_factory=ProfileProjectEditPatch)
    warnings: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    school: str = ""
    degree: str = ""
    start_date: str = ""
    end_date: str = ""
    raw: str = ""

    def item(self) -> str:
        return build_education_item(self.school, self.degree, self.start_date, self.end_date)


class EditOperation(BaseModel):
    action: str = "infer"
    explicit: bool = False


EDIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Profile & Project Editing Tool.
Convert the user's instruction into a conservative edit patch.

Rules:
- Do not apply changes. Only return a patch for preview.
- Only include edits explicitly requested by the user.
- Do not invent profile facts, project facts, dates, metrics, companies, schools, or skills.
- For profile scalar fields, use set_fields with keys limited to:
  name, headline, email, phone, wechat, location, summary.
- For profile list fields, use skills_add/remove and education_add/remove.
- User company work/internship experience must be represented as a Project with
  category="work_experience", not as profile.experience.
- For projects, update existing projects by project_id when a matching title is clear.
- If the user describes a brand-new project, use action=create and create_project.
- For project set_fields, use keys limited to:
  category, title, role, start_date, end_date, description.
- For project list fields, use technologies_add/remove and highlights_add/remove.
- If the requested edit is ambiguous, leave it out and add a warning.
""",
        ),
        (
            "human",
            """
User instruction:
{message}

Current profile:
{profile}

Current projects:
{projects}
""",
        ),
    ]
)


def format_projects(projects: list[Project]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"id: {project.id}",
                f"title: {project.title}",
                f"role: {project.role}",
                f"period: {project.start_date} - {project.end_date}",
                f"description: {project.description}",
                f"technologies: {', '.join(project.technologies)}",
                f"highlights: {'; '.join(project.highlights)}",
            ]
        )
        for project in projects
    )


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))


def split_items(value: str) -> list[str]:
    return dedupe(re.split(r"[,，、和与/]+", value))


def detect_edit_operation(message: str) -> EditOperation:
    if re.search(r"删除|移除|去掉|不要|remove|delete", message, flags=re.IGNORECASE):
        return EditOperation(action="remove", explicit=True)
    if re.search(r"替换为|覆盖为|全部改成|更新为|设置为|设为|replace|set to", message, flags=re.IGNORECASE):
        return EditOperation(action="replace", explicit=True)
    if re.search(r"修改|改成|改为|更新|更正|纠正|修正|其实|应该是|不是|换成|改一下|change", message, flags=re.IGNORECASE):
        return EditOperation(action="update", explicit=True)
    if re.search(r"新增|添加|加上|追加|另一个|另一段|还有|另外|也在|add", message, flags=re.IGNORECASE):
        return EditOperation(action="add", explicit=True)
    return EditOperation(action="infer", explicit=False)


def normalize_month(value: str) -> str:
    match = re.search(r"(\d{4})\s*(?:年|[./-])\s*(\d{1,2})", value)
    if not match:
        return value.strip()
    return f"{match.group(1)}.{int(match.group(2)):02d}"


def find_date_range(message: str) -> tuple[str, str] | None:
    date_pattern = r"\d{4}\s*(?:年|[./-])\s*\d{1,2}\s*(?:月)?"
    match = re.search(
        rf"({date_pattern})\s*(?:到|至|-|—|~)\s*({date_pattern})",
        message,
    )
    if not match:
        return None
    return normalize_month(match.group(1)), normalize_month(match.group(2))


def normalize_year_or_month(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        return value
    return normalize_month(value)


def normalize_school_name(value: str) -> str:
    school = value.strip(" ，,。.;；")
    school = re.sub(
        r"^(?:新增|添加|加上|追加|修改|更新|更正|纠正|修正|把|将|我的|学历|教育经历|我是|本人在|本人|我在|在|于)+\s*",
        "",
        school,
    )
    return school.strip(" ，,。.;；")


def build_education_item(school: str, degree: str, start: str, end: str) -> str:
    return " | ".join(
        item
        for item in [
            normalize_school_name(school),
            degree.strip(),
            f"{normalize_year_or_month(start)} - {normalize_year_or_month(end)}",
        ]
        if item
    )


def normalize_degree(value: str) -> str:
    return value.strip().lower()


def infer_degree_from_message(message: str) -> str:
    match = re.search(
        r"本科|硕士|博士|研究生|学士|master|bachelor|phd",
        message,
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else ""


def infer_education_entry_from_message(message: str) -> EducationEntry | None:
    degree_pattern = r"(本科|硕士|博士|研究生|学士|master|bachelor|phd)"
    school_pattern = r"((?:我是|本人在|本人|我在|在|于)?\s*[\u4e00-\u9fffA-Za-z\s]+大学)"
    date_pattern = r"(\d{4}(?:\s*(?:年|[./-])\s*\d{1,2})?)"

    school_first = re.search(
        rf"{school_pattern}[^0-9。；;\n]*?{degree_pattern}?[^0-9。；;\n]*(?:毕业|在读|就读|完成)?[^0-9]*{date_pattern}\s*(?:到|至|-|—|~)\s*{date_pattern}",
        message,
        flags=re.IGNORECASE,
    )
    if school_first:
        return EducationEntry(
            school=normalize_school_name(school_first.group(1)),
            degree=school_first.group(2) or infer_degree_from_message(message),
            start_date=normalize_year_or_month(school_first.group(3)),
            end_date=normalize_year_or_month(school_first.group(4)),
            raw=message,
        )

    date_first = re.search(
        rf"{date_pattern}\s*(?:到|至|-|—|~)\s*{date_pattern}[^。；;\n]*?(?:在|于)?\s*{school_pattern}[^。；;\n]*?{degree_pattern}?",
        message,
        flags=re.IGNORECASE,
    )
    if date_first:
        return EducationEntry(
            school=normalize_school_name(date_first.group(3)),
            degree=date_first.group(4) or infer_degree_from_message(message),
            start_date=normalize_year_or_month(date_first.group(1)),
            end_date=normalize_year_or_month(date_first.group(2)),
            raw=message,
        )

    return None


def infer_education_from_message(message: str) -> str:
    entry = infer_education_entry_from_message(message)
    return entry.item() if entry else ""


def parse_education_item(value: str) -> EducationEntry:
    parts = [part.strip() for part in value.split("|")]
    entry = EducationEntry(raw=value)
    for part in parts:
        if "大学" in part and not entry.school:
            entry.school = normalize_school_name(part)
        if re.fullmatch(r"本科|硕士|博士|研究生|学士|master|bachelor|phd", part, flags=re.IGNORECASE):
            entry.degree = part
        period_match = re.search(
            r"(\d{4}(?:\.\d{1,2})?)\s*(?:-|到|至|—|~)\s*(\d{4}(?:\.\d{1,2})?)",
            part,
        )
        if period_match:
            entry.start_date = normalize_year_or_month(period_match.group(1))
            entry.end_date = normalize_year_or_month(period_match.group(2))
    return entry


def compact_key(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())


def education_period_key(entry: EducationEntry) -> tuple[str, str]:
    return entry.start_date, entry.end_date


def find_education_update_target(
    new_entry: EducationEntry,
    current_items: list[str],
    operation: EditOperation,
) -> str:
    candidates: list[tuple[int, str]] = []
    new_school = compact_key(new_entry.school)
    new_degree = normalize_degree(new_entry.degree)
    new_period = education_period_key(new_entry)

    for item in current_items:
        current = parse_education_item(item)
        if compact_key(current.raw) == compact_key(new_entry.item()):
            continue

        score = 0
        same_school = new_school and compact_key(current.school) == new_school
        same_degree = new_degree and normalize_degree(current.degree) == new_degree
        same_period = all(new_period) and education_period_key(current) == new_period

        if same_school:
            score += 3
        if same_degree:
            score += 2
        if same_period:
            score += 4
        if operation.action in {"update", "replace"} and same_school:
            score += 2

        if same_school and (same_period or same_degree or operation.action in {"update", "replace"}):
            candidates.append((score, item))

    if not candidates:
        return ""
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if candidates[0][0] >= 5:
        return candidates[0][1]
    return ""


def resolve_education_edit_intent(
    message: str,
    profile: Profile,
    patch: ProfileProjectEditPatch,
) -> None:
    operation = detect_edit_operation(message)
    if operation.action == "add":
        return

    removals = {compact_key(item) for item in patch.profile.education_remove}
    for new_item in patch.profile.education_add:
        new_entry = parse_education_item(new_item)
        if not new_entry.school:
            continue
        update_target = find_education_update_target(new_entry, profile.education, operation)
        if update_target and compact_key(update_target) not in removals:
            removals.add(compact_key(update_target))
            patch.profile.education_remove.append(update_target)


def resolve_profile_list_edit_intents(
    message: str,
    profile: Profile,
    patch: ProfileProjectEditPatch,
) -> None:
    operation = detect_edit_operation(message)
    resolve_education_edit_intent(message, profile, patch)

    if operation.action == "replace" and patch.profile.skills_add and not patch.profile.skills_remove:
        patch.profile.skills_remove = list(profile.skills)


def infer_technologies(message: str) -> list[str]:
    known = [
        "Python",
        "FastAPI",
        "SQL",
        "PostgreSQL",
        "MySQL",
        "REST API",
        "API",
        "LangChain",
        "LLM",
        "ChromaDB",
        "向量检索",
        "AI 搜索",
        "AI",
    ]
    lowered = message.lower()
    technologies: list[str] = []
    for skill in known:
        if skill.lower() in lowered:
            technologies.append(skill)
    return dedupe(technologies)


def build_work_experience_project(message: str) -> ProjectInput | None:
    if not re.search(r"实习|工作经历|我在.+(公司|实习|负责)", message):
        return None

    date_range = find_date_range(message)
    company_match = re.search(
        r"我(?:之前|曾经|曾)?在\s*(.+?)\s*(?:做过|担任|进行了|有过|实习|工作|负责)",
        message,
    )
    if not company_match:
        return None

    company = company_match.group(1).strip(" ，,。.;；")
    if "的 " in company:
        company = company.rsplit("的 ", 1)[1].strip()
    if not company:
        return None

    role_match = re.search(r"((?:软件开发|后端开发|前端开发|全栈开发|数据分析|产品|AI应用|算法)[^，。,.;；\n]{0,12}?实习)", message)
    if not role_match:
        role_match = re.search(
            r"((?:软件开发|后端开发|前端开发|全栈开发|数据分析|产品|AI应用|算法)[^，。,.;；\n]{0,12}?(?:工程师|助理|经理|岗位))",
            message,
        )
    role = role_match.group(1).strip(" ，,。.;；") if role_match else "实习"

    description = message.strip()
    highlights = dedupe(
        item.strip(" ，,。.;；")
        for item in re.split(r"[。；;\n]", message)
        if re.search(r"负责|参与|开发|设计|实现|接入|让用户|熟悉", item)
    )

    start_date = date_range[0] if date_range else ""
    end_date = date_range[1] if date_range else ""
    return ProjectInput(
        category="work_experience",
        title=company,
        role=role,
        start_date=start_date,
        end_date=end_date,
        description=description,
        technologies=infer_technologies(message),
        highlights=highlights or [description],
    )


def build_generic_project(message: str) -> ProjectInput:
    title = ""
    title_patterns = [
        r"我(?:做了|开发了|实现了|参与了|设计了)\s*(?:一个|一款|一套)?\s*([^。；;\n，,]{2,40}?)(?:项目|系统|平台|工具|应用|网站|Agent)",
        r"(?:项目|系统|平台|工具|应用|网站|Agent)[叫名为是：:\s]+([^。；;\n，,]{2,40})",
        r"([^。；;\n，,]{2,40}?)(?:项目|系统|平台|工具|应用|网站|Agent)",
    ]
    for pattern in title_patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            title = match.group(1).strip(" ，,。.;；")
            break

    role = ""
    role_match = re.search(
        r"(?:我的角色|我担任|作为|role)[是为：:\s]*([^。；;\n，,]{2,24})",
        message,
        flags=re.IGNORECASE,
    )
    if role_match:
        role = role_match.group(1).strip(" ，,。.;；")
    elif re.search(r"负责|开发|实现|设计", message):
        role = "开发者"

    date_range = find_date_range(message)
    highlights = dedupe(
        item.strip(" ，,。.;；")
        for item in re.split(r"[。；;\n]", message)
        if re.search(r"负责|参与|开发|设计|实现|接入|优化|上线|提升|支持", item)
    )
    return ProjectInput(
        category="project",
        title=title,
        role=role,
        start_date=date_range[0] if date_range else "",
        end_date=date_range[1] if date_range else "",
        description=message.strip(),
        technologies=infer_technologies(message),
        highlights=highlights or [message.strip()],
    )


def merge_project_input(base: ProjectInput, fallback: ProjectInput) -> ProjectInput:
    data = base.model_dump()
    fallback_data = fallback.model_dump()
    for field in ["category", "title", "role", "start_date", "end_date", "description"]:
        if not str(data.get(field, "")).strip() and str(fallback_data.get(field, "")).strip():
            data[field] = fallback_data[field]
    for field in ["technologies", "highlights"]:
        if not data.get(field) and fallback_data.get(field):
            data[field] = fallback_data[field]
    return ProjectInput(**data)


def complete_project_input_from_message(
    project_input: ProjectInput,
    message: str,
    router_content_type: str,
) -> ProjectInput:
    fallbacks: list[ProjectInput] = []
    work_project = build_work_experience_project(message)
    generic_project = build_generic_project(message)
    if router_content_type == "work_experience_project" and work_project:
        fallbacks.append(work_project)
    fallbacks.append(generic_project)
    if work_project and work_project not in fallbacks:
        fallbacks.append(work_project)

    completed = project_input
    for fallback in fallbacks:
        completed = merge_project_input(completed, fallback)

    if router_content_type == "work_experience_project" or (
        not router_content_type and work_project is not None
    ):
        completed.category = "work_experience"
    elif router_content_type in {"project_experience", "project_update"} and completed.category != "work_experience":
        completed.category = "project"

    if not completed.description.strip():
        completed.description = message.strip()
    if not completed.technologies:
        completed.technologies = infer_technologies(message)
    if not completed.highlights:
        completed.highlights = generic_project.highlights
    return completed


def is_project_route(router_intent: str = "", router_content_type: str = "") -> bool:
    return router_intent in {"edit_project", "create_project_preview"} or router_content_type in PROJECT_ROUTE_CONTENT_TYPES


def message_looks_like_project(message: str) -> bool:
    return bool(
        re.search(
            r"项目|系统|平台|工具|应用|网站|Agent|实习|工作经历|技术栈|负责|实现|开发",
            message,
            flags=re.IGNORECASE,
        )
    )


def title_similarity(left: str, right: str) -> float:
    left_key = re.sub(r"\s+", "", left.strip().lower())
    right_key = re.sub(r"\s+", "", right.strip().lower())
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    if left_key in right_key or right_key in left_key:
        return 0.86
    return SequenceMatcher(None, left_key, right_key).ratio()


def find_matching_project(project_input: ProjectInput, projects: list[Project], message: str) -> Project | None:
    candidates: list[tuple[float, Project]] = []
    for project in projects:
        score = title_similarity(project_input.title, project.title)
        if project.title and project.title in message:
            score = max(score, 0.92)
        if project.role and project.role in message:
            score = max(score, 0.72)
        candidates.append((score, project))

    if not candidates:
        return None
    score, project = max(candidates, key=lambda item: item[0])
    if score >= 0.78:
        return project
    return None


def project_scalar_field_requested(message: str, field: str) -> bool:
    patterns = {
        "title": r"标题|名称|名字|项目名|叫",
        "role": r"角色|担任|职位|岗位|role",
        "start_date": r"开始|起始|时间|日期|周期|从",
        "end_date": r"结束|截止|时间|日期|周期|到|至",
        "description": r"描述|介绍|背景|概述|summary|description",
        "category": r"类型|类别|分类|category",
    }
    return bool(re.search(patterns.get(field, ""), message, flags=re.IGNORECASE))


def project_list_field_requested(message: str, field: str) -> bool:
    patterns = {
        "technologies": r"技术栈|技术|工具|框架|technologies|technology|stack",
        "highlights": r"亮点|要点|成果|产出|负责|实现|优化|上线|提升|highlights|bullets",
    }
    return bool(re.search(patterns.get(field, ""), message, flags=re.IGNORECASE))


def build_update_patch_from_project(
    project_input: ProjectInput,
    matched: Project,
    message: str = "",
) -> ProjectEditPatch:
    patch = ProjectEditPatch(
        action="update",
        project_id=matched.id,
        match_title=matched.title,
    )
    for field in PROJECT_SCALAR_FIELDS:
        value = getattr(project_input, field)
        if value and value != getattr(matched, field) and project_scalar_field_requested(message, field):
            patch.set_fields[field] = value

    existing_tech = {item.strip().lower() for item in matched.technologies}
    if project_list_field_requested(message, "technologies"):
        patch.technologies_add = [
            item for item in project_input.technologies if item.strip().lower() not in existing_tech
        ]
    existing_highlights = {item.strip().lower() for item in matched.highlights}
    if project_list_field_requested(message, "highlights"):
        patch.highlights_add = [
            item for item in project_input.highlights if item.strip().lower() not in existing_highlights
        ]
    return patch


def build_project_patch_from_router(
    message: str,
    projects: list[Project],
    router_content_type: str,
) -> EditDraft:
    warnings: list[str] = []
    operation = detect_edit_operation(message)
    try:
        project_input = profile_project_text(message)
    except Exception:
        log_model_fallback("project_profiling", "project_profiling", "regex_project_parser")
        project_input = build_work_experience_project(message) or build_generic_project(message)

    project_input = complete_project_input_from_message(project_input, message, router_content_type)

    if not project_input.title.strip():
        fallback_title = re.search(r"(?:项目|系统|平台|工具)[：: ]?([^。；;\n，,]{2,30})", message)
        if fallback_title:
            project_input.title = fallback_title.group(1).strip(" ，,。.;；")
    if not project_input.title.strip():
        generic_project = build_generic_project(message)
        project_input.title = generic_project.title
        project_input.role = project_input.role or generic_project.role
        project_input.technologies = project_input.technologies or generic_project.technologies
        project_input.highlights = project_input.highlights or generic_project.highlights
    if not project_input.description.strip():
        project_input.description = message.strip()

    matched = None if operation.action == "add" else find_matching_project(project_input, projects, message)
    if matched:
        project_patch = build_update_patch_from_project(project_input, matched, message)
    elif operation.action in {"update", "replace", "remove"} and operation.explicit:
        warnings.append("识别到用户想修改已有项目，但没有匹配到明确的项目，请补充项目名称。")
        project_patch = None
    else:
        project_patch = ProjectEditPatch(action="create", create_project=project_input)

    patch = ProfileProjectEditPatch(projects=[project_patch] if project_patch else [])
    if not patch_has_changes(patch):
        warnings.append("项目输入已识别，但没有形成可应用的新增或修改。")
    return EditDraft(patch=patch, warnings=warnings)


def build_project_recovery_patch(
    message: str,
    projects: list[Project],
    router_content_type: str,
) -> ProfileProjectEditPatch:
    operation = detect_edit_operation(message)
    project_input = complete_project_input_from_message(
        ProjectInput(),
        message,
        router_content_type,
    )
    matched = None if operation.action == "add" else find_matching_project(project_input, projects, message)
    if matched:
        return ProfileProjectEditPatch(
            projects=[build_update_patch_from_project(project_input, matched, message)]
        )
    if operation.action in {"update", "replace", "remove"} and operation.explicit:
        return ProfileProjectEditPatch()
    return ProfileProjectEditPatch(
        projects=[
            ProjectEditPatch(
                action="create",
                create_project=project_input,
            )
        ]
    )


def fallback_edit_patch(message: str, projects: list[Project]) -> EditDraft:
    profile_patch = ProfileEditPatch()
    project_patches: list[ProjectEditPatch] = []
    warnings: list[str] = []

    work_experience_project = build_work_experience_project(message)
    if work_experience_project:
        project_patches.append(
            ProjectEditPatch(
                action="create",
                create_project=work_experience_project,
            )
        )

    location_match = re.search(
        r"(?:location|地点|城市|所在地)[^。；;\n]*(?:改成|修改为|设置为|设为|change to|set to)\s*([A-Za-z\u4e00-\u9fff\s,-]+)",
        message,
        flags=re.IGNORECASE,
    )
    if location_match:
        profile_patch.set_fields["location"] = location_match.group(1).strip(" ，,。.;；")

    skill_replace_match = re.search(
        r"(?:技能|skills)[^。；;\n]*(?:替换为|覆盖为|全部改成|更新为|改成|设置为|设为|replace|set to)\s*([A-Za-z0-9+#.\u4e00-\u9fff\s,，、/]+)",
        message,
        flags=re.IGNORECASE,
    )
    if skill_replace_match:
        profile_patch.skills_add = split_items(skill_replace_match.group(1).strip(" 。.;；"))

    skill_add_match = re.search(
        r"(?:技能|skills)[^。；;\n]*(?:加上|添加|新增|add)\s*([A-Za-z0-9+#.\u4e00-\u9fff\s,，、/]+)",
        message,
        flags=re.IGNORECASE,
    )
    if skill_add_match and not profile_patch.skills_add:
        profile_patch.skills_add = split_items(skill_add_match.group(1).strip(" 。.;；"))

    skill_remove_match = re.search(
        r"(?:技能|skills)[^。；;\n]*(?:删除|移除|去掉|remove)\s*([A-Za-z0-9+#.\u4e00-\u9fff\s,，、/]+)",
        message,
        flags=re.IGNORECASE,
    )
    if skill_remove_match:
        profile_patch.skills_remove = split_items(skill_remove_match.group(1).strip(" 。.;；"))

    education_entry = infer_education_entry_from_message(message)
    if education_entry:
        profile_patch.education_add = [education_entry.item()]

    education_add_match = re.search(
        r"(?:教育经历|education)[^。；;\n]*(?:加上|添加|新增|更新为|改成|设置为|add|set to)\s*([^。；;\n]+)",
        message,
        flags=re.IGNORECASE,
    )
    if education_add_match and not profile_patch.education_add:
        profile_patch.education_add = [education_add_match.group(1).strip()]

    experience_match = re.search(
        r"我在\s*([^，。,.;；\s]+)\s*(?:做过|担任|进行了|有过)\s*([^，。,.;；]*?(?:实习|工作|开发|工程师|分析师|经理))[^0-9]*(\d{4}[./-]\d{1,2})\s*(?:到|至|-|—|~)\s*(\d{4}[./-]\d{1,2})[，,]?\s*(?:负责|主要负责)?\s*([^。；;\n]*)",
        message,
        flags=re.IGNORECASE,
    )
    if not experience_match:
        experience_match = re.search(
            r"我在\s*([^，。,.;；\s]+)\s*([^，。,.;；]*?(?:实习|工作|开发|工程师|分析师|经理))[^0-9]*(\d{4}[./-]\d{1,2})\s*(?:到|至|-|—|~)\s*(\d{4}[./-]\d{1,2})[，,]?\s*(?:负责|主要负责)?\s*([^。；;\n]*)",
            message,
            flags=re.IGNORECASE,
        )
    if experience_match and not work_experience_project:
        company = experience_match.group(1).strip()
        role = experience_match.group(2).strip()
        start = normalize_month(experience_match.group(3))
        end = normalize_month(experience_match.group(4))
        responsibility = experience_match.group(5).strip(" ，,。.;；")
        project_patches.append(
            ProjectEditPatch(
                action="create",
                create_project=ProjectInput(
                    category="work_experience",
                    title=company,
                    role=role,
                    start_date=start,
                    end_date=end,
                    description=responsibility or f"{company} {role}",
                    technologies=[],
                    highlights=[responsibility] if responsibility else [],
                ),
            )
        )

    experience_add_match = re.search(
        r"(?:工作经历|实习经历|experience)[^。；;\n]*(?:加上|添加|新增|更新为|改成|设置为|add|set to)\s*([^。；;\n]+)",
        message,
        flags=re.IGNORECASE,
    )
    if experience_add_match and not work_experience_project:
        raw_experience = experience_add_match.group(1).strip()
        project_patches.append(
            ProjectEditPatch(
                action="create",
                create_project=ProjectInput(
                    category="work_experience",
                    title=raw_experience,
                    description=raw_experience,
                ),
            )
        )

    for project in projects:
        if project.title and project.title in message:
            tech_add_match = re.search(
                rf"{re.escape(project.title)}[^。；;\n]*(?:技术栈|technologies|technology)[^。；;\n]*(?:加上|添加|新增|add)\s*([A-Za-z0-9+#.\u4e00-\u9fff\s,，、/]+)",
                message,
                flags=re.IGNORECASE,
            )
            tech_remove_match = re.search(
                rf"{re.escape(project.title)}[^。；;\n]*(?:技术栈|technologies|technology)[^。；;\n]*(?:删除|移除|去掉|remove)\s*([A-Za-z0-9+#.\u4e00-\u9fff\s,，、/]+)",
                message,
                flags=re.IGNORECASE,
            )
            highlight_add_match = re.search(
                rf"{re.escape(project.title)}[^。；;\n]*(?:亮点|highlights|要点)[^。；;\n]*(?:加上|添加|新增|add)\s*([^。；;\n]+)",
                message,
                flags=re.IGNORECASE,
            )

            project_patch = ProjectEditPatch(
                action="update",
                project_id=project.id,
                match_title=project.title,
            )
            if tech_add_match:
                project_patch.technologies_add = split_items(tech_add_match.group(1).strip(" 。.;；"))
            if tech_remove_match:
                project_patch.technologies_remove = split_items(tech_remove_match.group(1).strip(" 。.;；"))
            if highlight_add_match:
                project_patch.highlights_add = [highlight_add_match.group(1).strip()]
            if (
                project_patch.set_fields
                or project_patch.technologies_add
                or project_patch.technologies_remove
                or project_patch.highlights_add
                or project_patch.highlights_remove
            ):
                project_patches.append(project_patch)

    patch = ProfileProjectEditPatch(profile=profile_patch, projects=project_patches)
    if not patch_has_changes(patch):
        warnings.append("没有从自然语言中识别到明确可应用的 Profile / Project 修改。")
    return EditDraft(patch=patch, warnings=warnings)


def parse_edit_patch(
    message: str,
    profile: Profile,
    projects: list[Project],
    router_intent: str = "",
    router_content_type: str = "",
    normalized_instruction: str = "",
) -> EditDraft:
    effective_message = normalized_instruction.strip() or message
    if is_project_route(router_intent, router_content_type):
        return build_project_patch_from_router(effective_message, projects, router_content_type)

    if re.search(r"实习|工作经历|我在.+(公司|实习|负责)", message):
        fallback = fallback_edit_patch(message, projects)
        if patch_has_changes(fallback.patch):
            return fallback

    try:
        draft = invoke_model_with_logging(
            "profile_project_editing",
            PROFILE_PROJECT_EDITING_MODEL,
            lambda: (
                EDIT_PROMPT
                | ChatOpenAI(model=PROFILE_PROJECT_EDITING_MODEL).with_structured_output(
                    EditDraft,
                    method="json_schema",
                )
            ).invoke(
                {
                    "message": effective_message,
                    "profile": profile.model_dump(),
                    "projects": format_projects(projects),
                }
            ),
        )
        if patch_has_changes(draft.patch):
            return draft
        fallback = fallback_edit_patch(message, projects)
        if patch_has_changes(fallback.patch):
            fallback.warnings.extend(draft.warnings)
            return fallback
        return draft
    except Exception:
        log_model_fallback(
            "profile_project_editing",
            PROFILE_PROJECT_EDITING_MODEL,
            "regex_edit_parser",
        )
        return fallback_edit_patch(message, projects)


def clean_profile_patch(patch: ProfileEditPatch, warnings: list[str]) -> ProfileEditPatch:
    set_fields: dict[str, str] = {}
    for field, value in patch.set_fields.items():
        if field in PROFILE_SCALAR_FIELDS:
            set_fields[field] = value
        else:
            warnings.append(f"忽略不支持的 Profile 字段：{field}")

    patch.set_fields = set_fields
    patch.skills_add = dedupe(patch.skills_add)
    patch.skills_remove = dedupe(patch.skills_remove)
    patch.education_add = dedupe(patch.education_add)
    patch.education_remove = dedupe(patch.education_remove)
    if patch.experience_add or patch.experience_remove:
        warnings.append("工作/实习经历不再写入 Profile experience，请作为 work_experience 类型项目保存。")
    patch.experience_add = []
    patch.experience_remove = []
    return patch


def clean_project_patch(
    patch: ProjectEditPatch,
    projects_by_id: dict[str, Project],
    projects_by_title: dict[str, Project],
    warnings: list[str],
) -> ProjectEditPatch | None:
    if patch.action not in {"update", "create"}:
        warnings.append(f"忽略不支持的项目操作：{patch.action}")
        return None

    if patch.action == "create":
        if patch.create_project is None or not patch.create_project.title.strip():
            warnings.append("新建项目缺少标题，已跳过。")
            return None
        return patch

    if patch.project_id not in projects_by_id and patch.match_title:
        matched = projects_by_title.get(patch.match_title.strip().lower())
        if matched:
            patch.project_id = matched.id

    if patch.project_id not in projects_by_id:
        warnings.append(f"没有找到要修改的项目：{patch.match_title or patch.project_id}")
        return None

    set_fields: dict[str, str] = {}
    for field, value in patch.set_fields.items():
        if field in PROJECT_SCALAR_FIELDS:
            set_fields[field] = value
        else:
            warnings.append(f"忽略不支持的 Project 字段：{field}")

    patch.set_fields = set_fields
    patch.technologies_add = dedupe(patch.technologies_add)
    patch.technologies_remove = dedupe(patch.technologies_remove)
    patch.highlights_add = dedupe(patch.highlights_add)
    patch.highlights_remove = dedupe(patch.highlights_remove)
    return patch


def clean_patch(
    patch: ProfileProjectEditPatch,
    projects: list[Project],
    warnings: list[str],
) -> ProfileProjectEditPatch:
    projects_by_id = {project.id: project for project in projects}
    projects_by_title = {project.title.strip().lower(): project for project in projects}
    cleaned_projects: list[ProjectEditPatch] = []

    for project_patch in patch.projects:
        cleaned = clean_project_patch(
            project_patch,
            projects_by_id,
            projects_by_title,
            warnings,
        )
        if cleaned:
            cleaned_projects.append(cleaned)

    return ProfileProjectEditPatch(
        profile=clean_profile_patch(patch.profile, warnings),
        projects=cleaned_projects,
    )


def apply_list_changes(current: list[str], additions: list[str], removals: list[str]) -> list[str]:
    removed = {item.strip().lower() for item in removals}
    updated = [item for item in current if item.strip().lower() not in removed]
    existing = {item.strip().lower() for item in updated}
    for item in additions:
        if item.strip().lower() not in existing:
            updated.append(item)
            existing.add(item.strip().lower())
    return updated


def list_change_preview(
    target: str,
    current: list[str],
    additions: list[str],
    removals: list[str],
) -> EditChangePreview | None:
    updated = apply_list_changes(current, additions, removals)
    if updated == current:
        return None
    return EditChangePreview(target=target, action="update_list", before=current, after=updated)


def build_edit_preview(
    message: str,
    profile: Profile,
    projects: list[Project],
    router_intent: str = "",
    router_content_type: str = "",
    normalized_instruction: str = "",
) -> ProfileProjectEditPreview:
    draft = parse_edit_patch(
        message,
        profile,
        projects,
        router_intent,
        router_content_type,
        normalized_instruction,
    )
    warnings = list(draft.warnings)
    patch = clean_patch(draft.patch, projects, warnings)
    resolve_profile_list_edit_intents(message, profile, patch)
    project_route = is_project_route(router_intent, router_content_type) or message_looks_like_project(message)
    if project_route and not patch.projects:
        recovery_warnings: list[str] = []
        recovery_patch = clean_patch(
            build_project_recovery_patch(
                normalized_instruction.strip() or message,
                projects,
                router_content_type,
            ),
            projects,
            recovery_warnings,
        )
        if recovery_patch.projects:
            patch = recovery_patch
            warnings = [
                warning
                for warning in warnings
                if warning != "新建项目缺少标题，已跳过。"
            ]
            warnings.extend(recovery_warnings)
    changes = build_changes_for_patch(profile, projects, patch)
    completeness = check_information_completeness(
        profile,
        projects,
        patch,
        include_profile=not project_route,
    )
    debug = {
        "router_intent": router_intent,
        "router_content_type": router_content_type,
        "project_route": project_route,
        "patch_project_count": len(patch.projects),
        "patch_actions": [project_patch.action for project_patch in patch.projects],
        "create_titles": [
            project_patch.create_project.title
            for project_patch in patch.projects
            if project_patch.create_project
        ],
        "change_count": len(changes),
    }
    return ProfileProjectEditPreview(
        message=message,
        patch=patch,
        changes=changes,
        warnings=warnings,
        completeness=completeness,
        debug=debug,
        has_changes=bool(changes),
    )


def build_changes_for_patch(
    profile: Profile,
    projects: list[Project],
    patch: ProfileProjectEditPatch,
) -> list[EditChangePreview]:
    changes: list[EditChangePreview] = []
    projects_by_id = {project.id: project for project in projects}

    for field, value in patch.profile.set_fields.items():
        before = getattr(profile, field)
        if before != value:
            changes.append(EditChangePreview(target=f"profile.{field}", action="set", before=before, after=value))

    profile_lists = [
        ("profile.skills", profile.skills, patch.profile.skills_add, patch.profile.skills_remove),
        ("profile.education", profile.education, patch.profile.education_add, patch.profile.education_remove),
    ]
    for target, current, additions, removals in profile_lists:
        preview = list_change_preview(target, current, additions, removals)
        if preview:
            changes.append(preview)

    for project_patch in patch.projects:
        if project_patch.action == "create" and project_patch.create_project:
            changes.append(
                EditChangePreview(
                    target="projects",
                    action="create",
                    before="",
                    after=project_patch.create_project.model_dump(mode="json"),
                )
            )
            continue

        project = projects_by_id.get(project_patch.project_id)
        if not project:
            continue

        for field, value in project_patch.set_fields.items():
            before = getattr(project, field)
            if before != value:
                changes.append(
                    EditChangePreview(
                        target=f"project[{project.title}].{field}",
                        action="set",
                        before=before,
                        after=value,
                    )
                )

        project_lists = [
            (
                f"project[{project.title}].technologies",
                project.technologies,
                project_patch.technologies_add,
                project_patch.technologies_remove,
            ),
            (
                f"project[{project.title}].highlights",
                project.highlights,
                project_patch.highlights_add,
                project_patch.highlights_remove,
            ),
        ]
        for target, current, additions, removals in project_lists:
            preview = list_change_preview(target, current, additions, removals)
            if preview:
                changes.append(preview)

    return changes


def patch_has_changes(patch: ProfileProjectEditPatch) -> bool:
    profile_patch = patch.profile
    if (
        profile_patch.set_fields
        or profile_patch.skills_add
        or profile_patch.skills_remove
        or profile_patch.education_add
        or profile_patch.education_remove
    ):
        return True

    return bool(patch.projects)
