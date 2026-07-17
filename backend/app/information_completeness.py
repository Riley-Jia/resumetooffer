from app.schemas import (
    FollowUpQuestion,
    InformationCompletenessResult,
    Profile,
    ProfileProjectEditPatch,
    Project,
    ProjectInput,
)


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def list_after_changes(current: list[str], additions: list[str], removals: list[str]) -> list[str]:
    removed = {item.strip().lower() for item in removals}
    updated = [item for item in current if item.strip().lower() not in removed]
    existing = {item.strip().lower() for item in updated}
    for item in additions:
        key = item.strip().lower()
        if key and key not in existing:
            updated.append(item)
            existing.add(key)
    return updated


def profile_after_patch(profile: Profile, patch: ProfileProjectEditPatch) -> Profile:
    data = profile.model_dump()
    for field, value in patch.profile.set_fields.items():
        if field in data:
            data[field] = value
    data["skills"] = list_after_changes(
        profile.skills,
        patch.profile.skills_add,
        patch.profile.skills_remove,
    )
    data["education"] = list_after_changes(
        profile.education,
        patch.profile.education_add,
        patch.profile.education_remove,
    )
    return Profile(**data)


def project_after_update(project: Project, patch: ProfileProjectEditPatch) -> Project:
    data = project.model_dump()
    for project_patch in patch.projects:
        if project_patch.action != "update" or project_patch.project_id != project.id:
            continue
        for field, value in project_patch.set_fields.items():
            if field in data:
                data[field] = value
        data["technologies"] = list_after_changes(
            data["technologies"],
            project_patch.technologies_add,
            project_patch.technologies_remove,
        )
        data["highlights"] = list_after_changes(
            data["highlights"],
            project_patch.highlights_add,
            project_patch.highlights_remove,
        )
    return Project(**data)


def projects_after_patch(projects: list[Project], patch: ProfileProjectEditPatch) -> list[Project | ProjectInput]:
    updated: list[Project | ProjectInput] = [project_after_update(project, patch) for project in projects]
    for project_patch in patch.projects:
        if project_patch.action == "create" and project_patch.create_project:
            updated.append(project_patch.create_project)
    return updated


def touched_projects_after_patch(projects: list[Project], patch: ProfileProjectEditPatch) -> list[Project | ProjectInput]:
    projects_by_id = {project.id: project for project in projects}
    touched: list[Project | ProjectInput] = []
    for project_patch in patch.projects:
        if project_patch.action == "create" and project_patch.create_project:
            touched.append(project_patch.create_project)
        elif project_patch.action == "update":
            project = projects_by_id.get(project_patch.project_id)
            if project:
                touched.append(project_after_update(project, patch))
    return touched


def profile_patch_has_changes(patch: ProfileProjectEditPatch) -> bool:
    profile_patch = patch.profile
    return bool(
        profile_patch.set_fields
        or profile_patch.skills_add
        or profile_patch.skills_remove
        or profile_patch.education_add
        or profile_patch.education_remove
        or profile_patch.experience_add
        or profile_patch.experience_remove
    )


def add_question(
    questions: list[FollowUpQuestion],
    field: str,
    question: str,
    priority: str,
    scope: str,
) -> None:
    if any(existing.field == field and existing.scope == scope for existing in questions):
        return
    questions.append(
        FollowUpQuestion(
            field=field,
            question=question,
            priority=priority,
            scope=scope,
        )
    )


def evaluate_profile(profile: Profile, result: InformationCompletenessResult) -> None:
    if not profile.name.strip():
        result.missing_required.append("profile.name")
        add_question(result.follow_up_questions, "name", "你的姓名应该写成什么？", "high", "profile")

    if not profile.email.strip() and not profile.phone.strip() and not profile.wechat.strip():
        result.missing_required.append("profile.contact")
        add_question(
            result.follow_up_questions,
            "contact",
            "请补充至少一种联系方式，例如邮箱、手机号或微信。",
            "high",
            "profile",
        )

    if not profile.education:
        result.missing_required.append("profile.education")
        add_question(
            result.follow_up_questions,
            "education",
            "请补充学校、专业、学历和起止时间，例如：悉尼大学，计算机科学硕士，2025.07 - 2026.09。",
            "high",
            "profile",
        )

    if not profile.location.strip():
        result.missing_recommended.append("profile.location")
        add_question(result.follow_up_questions, "location", "你现在主要投递哪个城市或地区？", "medium", "profile")

    if len(profile.skills) < 5:
        result.missing_recommended.append("profile.skills")
        add_question(
            result.follow_up_questions,
            "skills",
            "请补充 5 个左右主要技能或技术栈，例如 Python、FastAPI、SQL、React、LangChain。",
            "medium",
            "profile",
        )

    if not profile.summary.strip() or len(profile.summary.strip()) < 30:
        result.missing_recommended.append("profile.summary")
        result.quality_notes.append("个人简介偏短，后续生成简历介绍时可用信息较少。")


def evaluate_project(project: Project | ProjectInput, index: int, result: InformationCompletenessResult) -> None:
    label = project.title.strip() or f"项目 {index + 1}"
    scope = f"project:{label}"

    if not project.title.strip():
        result.missing_required.append(f"{scope}.title")
        add_question(result.follow_up_questions, "title", "这段项目或经历的名称是什么？", "high", scope)

    if not project.role.strip():
        result.missing_required.append(f"{scope}.role")
        add_question(result.follow_up_questions, "role", f"{label} 中你的角色是什么？", "high", scope)

    if not project.description.strip() or len(project.description.strip()) < 30:
        result.missing_required.append(f"{scope}.description")
        add_question(
            result.follow_up_questions,
            "description",
            f"请补充 {label} 的背景、目标和你参与的核心工作。",
            "high",
            scope,
        )

    if not project.start_date.strip() or not project.end_date.strip():
        result.missing_recommended.append(f"{scope}.period")
        add_question(
            result.follow_up_questions,
            "period",
            f"{label} 的起止时间是什么？例如 2025.06 - 2025.08。",
            "medium",
            scope,
        )

    if len(project.technologies) < 2:
        result.missing_recommended.append(f"{scope}.technologies")
        add_question(
            result.follow_up_questions,
            "technologies",
            f"{label} 主要用了哪些技术栈、工具或平台？",
            "medium",
            scope,
        )

    if len(project.highlights) < 2:
        result.missing_recommended.append(f"{scope}.highlights")
        add_question(
            result.follow_up_questions,
            "highlights",
            f"{label} 里你具体负责了哪些功能？有没有上线、效率提升、用户使用或其他结果？",
            "medium",
            scope,
        )

    if project.highlights and all(len(item.strip()) < 18 for item in project.highlights):
        result.quality_notes.append(f"{label} 的项目要点偏短，生成简历 bullet 时可能缺少细节。")


def calculate_score(result: InformationCompletenessResult) -> int:
    score = 100
    score -= len(result.missing_required) * 12
    score -= len(result.missing_recommended) * 6
    score -= len(result.quality_notes) * 3
    return max(0, min(100, score))


def check_information_completeness(
    profile: Profile,
    projects: list[Project],
    patch: ProfileProjectEditPatch,
    include_profile: bool = True,
) -> InformationCompletenessResult:
    result = InformationCompletenessResult()
    next_profile = profile_after_patch(profile, patch)
    next_projects = touched_projects_after_patch(projects, patch)

    if include_profile and (profile_patch_has_changes(patch) or not patch.projects):
        evaluate_profile(next_profile, result)
    for index, project in enumerate(next_projects):
        evaluate_project(project, index, result)

    result.missing_required = dedupe(result.missing_required)
    result.missing_recommended = dedupe(result.missing_recommended)
    result.quality_notes = dedupe(result.quality_notes)
    result.follow_up_questions = result.follow_up_questions[:3]
    result.score = calculate_score(result)
    result.can_continue = len(result.missing_required) == 0
    if result.missing_required:
        result.status = "needs_required_info"
    elif result.missing_recommended or result.quality_notes:
        result.status = "needs_optional_info"
    else:
        result.status = "complete"
    return result
