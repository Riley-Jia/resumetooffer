from collections import Counter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.job_matching import SKILL_ALIASES, normalize
from app.language import detect_output_language, language_instruction
from app.model_config import NEXT_STEP_PLAN_MODEL
from app.model_logging import invoke_model_with_logging, log_model_fallback
from app.schemas import (
    Job,
    JobMatchResult,
    JobSkillGap,
    NextStepPlanWeek,
    Profile,
    Project,
    SkillGapAnalysisRequest,
    SkillGapAnalysisResponse,
)


class NextStepPlanDraft(BaseModel):
    next_step_plan: list[NextStepPlanWeek] = Field(default_factory=list)


NEXT_STEP_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Next Step Plan Tool for junior IT job matching.
Use only the provided missing skills and user context. Do not invent work
experience, certificates, school names, companies, or completed projects.

Create exactly 3 weeks of next steps based on the gap severity:
- none: application preparation plan, not a learning plan
- mild: resume/project polishing and targeted improvement plan
- moderate or severe: focused learning + project evidence plan

Each week must include plan_type, focus, goals, tasks, and deliverable.
Allowed plan_type values: learning, improvement, application_prep.

Match the user's language. {language_instruction}
""",
        ),
        (
            "human",
            """
Target direction: {target_direction}
User skills: {user_skills}
Common missing skills: {common_missing_skills}
Priority skills: {priority_skills}
Gap severity: {gap_severity}
Gap summary: {gap_summary}
Top jobs:
{jobs_text}
""",
        ),
    ]
)


def expand_skills(skills: list[str]) -> set[str]:
    normalized = {normalize(skill) for skill in skills if skill.strip()}
    expanded = set(normalized)
    for skill in normalized:
        expanded.update(SKILL_ALIASES.get(skill, set()))
    return expanded


def skill_matches(required_skill: str, expanded_user_skills: set[str]) -> bool:
    normalized = normalize(required_skill)
    expanded_required = {normalized, *SKILL_ALIASES.get(normalized, set())}
    return bool(expanded_required & expanded_user_skills)


def analyze_skill_gap(
    jobs: list[Job],
    user_skills: list[str],
    match_results: list[JobMatchResult] | None = None,
) -> tuple[list[str], list[str], list[JobSkillGap], bool, str, str]:
    top_jobs = jobs[:3]
    match_by_job_id = {match.job.id: match for match in (match_results or [])}
    expanded_user_skills = expand_skills(user_skills)
    per_job_gaps: list[JobSkillGap] = []
    missing_by_job: list[set[str]] = []
    missing_counter: Counter[str] = Counter()

    for job in top_jobs:
        missing: list[str] = []
        matched: list[str] = []
        for skill in job.required_skills:
            if skill_matches(skill, expanded_user_skills):
                matched.append(skill)
            else:
                missing.append(skill)
                missing_counter[skill] += 1

        missing_by_job.append(set(missing))
        per_job_gaps.append(
            JobSkillGap(
                job_id=job.id,
                title=job.title,
                company=job.company,
                missing_skills=missing,
                matched_skills=matched,
                evaluation_description=build_gap_evaluation(
                    job,
                    missing,
                    matched,
                    match_by_job_id.get(job.id),
                ),
            )
        )

    if not missing_by_job:
        return [], [], [], False, "none", "没有可分析的 Top3 岗位。"

    common_missing = sorted(set.intersection(*missing_by_job)) if missing_by_job else []
    priority = [
        skill
        for skill, _ in sorted(
            missing_counter.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
    ]
    total_required = sum(len(job.required_skills) for job in top_jobs)
    total_missing = sum(len(gap.missing_skills) for gap in per_job_gaps)
    missing_ratio = total_missing / total_required if total_required else 0
    has_gap = total_missing > 0

    if not has_gap:
        severity = "none"
    elif missing_ratio <= 0.25 and len(common_missing) == 0:
        severity = "mild"
    elif missing_ratio <= 0.5:
        severity = "moderate"
    else:
        severity = "severe"

    summary = (
        f"Top3 岗位共要求 {total_required} 个技能，缺失 {total_missing} 个；"
        f"共同缺失 {len(common_missing)} 个，gap 严重程度为 {severity}。"
    )
    return common_missing, priority, per_job_gaps, has_gap, severity, summary


def build_gap_evaluation(
    job: Job,
    missing_skills: list[str],
    matched_skills: list[str],
    match: JobMatchResult | None,
) -> str:
    required_count = len(job.required_skills)
    matched_count = len(matched_skills)
    coverage_text = (
        f"硬技能覆盖 {matched_count}/{required_count}"
        if required_count
        else "岗位没有明确列出必备技能"
    )
    gap_text = (
        f"主要差距在 {', '.join(missing_skills[:3])}。"
        if missing_skills
        else "必备技能没有明显缺口。"
    )

    if not match:
        return f"{coverage_text}，{gap_text} 该评估仅基于岗位技能字段，未包含语境评估分。"

    if match.llm_score >= 95:
        context_text = "综合语境评估也认为项目证据和岗位语境高度匹配。"
    elif match.llm_score >= 80:
        context_text = "语境评估没有满分通常不是因为缺技能，而是项目证据、行业场景或岗位职责表达还可以更贴近。"
    else:
        context_text = "语境评估偏保守，通常表示简历里的项目证据和岗位实际职责之间还需要更清楚的桥接。"

    return (
        f"{coverage_text}，{gap_text} "
        f"规则分 {match.rule_score:.0f}，语境评估分 {match.llm_score:.0f}。{context_text}"
    )


def fallback_next_step_plan(
    common_missing_skills: list[str],
    priority_skills: list[str],
    target_direction: str,
    gap_severity: str,
    output_language: str,
) -> list[NextStepPlanWeek]:
    focus_skills = common_missing_skills or priority_skills[:5]
    skill_text = "、".join(focus_skills) if focus_skills else "岗位核心技能"

    if output_language == "Chinese":
        if gap_severity == "none":
            return [
                NextStepPlanWeek(
                    week=1,
                    plan_type="application_prep",
                    focus="投递材料准备",
                    goals=["确认 Top3 岗位关键词", "对齐简历项目表达", "准备一版定向简历"],
                    tasks=[
                        "把 Top3 岗位的 required skills 对照到简历技能和项目经历",
                        "调整项目 bullet，让技能证据更直接",
                        "准备 1 段面向目标岗位的自我介绍",
                    ],
                    deliverable="一版定向简历和岗位关键词对照表",
                ),
                NextStepPlanWeek(
                    week=2,
                    plan_type="application_prep",
                    focus="面试证据准备",
                    goals=["准备项目讲述", "准备技术追问答案", "整理可展示材料"],
                    tasks=[
                        "为 2-3 个核心项目准备 STAR 讲述",
                        "整理每个项目的技术选择、难点和结果",
                        "准备 GitHub/作品集/项目截图链接",
                    ],
                    deliverable="项目面试讲述稿和作品集清单",
                ),
                NextStepPlanWeek(
                    week=3,
                    plan_type="application_prep",
                    focus="投递和复盘",
                    goals=["开始定向投递", "记录反馈", "迭代简历"],
                    tasks=[
                        "优先投递匹配度最高的岗位",
                        "记录每次投递岗位、简历版本和反馈",
                        "根据反馈更新简历和岗位筛选条件",
                    ],
                    deliverable="投递追踪表和下一版简历修改清单",
                ),
            ]

        plan_type = "improvement" if gap_severity == "mild" else "learning"
        return [
            NextStepPlanWeek(
                week=1,
                plan_type=plan_type,
                focus=f"补齐基础概念：{skill_text}",
                goals=["理解 Top3 岗位共同要求", "完成核心技能基础练习", "整理技能差距笔记"],
                tasks=[
                    "把每个缺失技能拆成概念、常用场景和岗位要求",
                    "每天完成一个小练习并记录问题",
                    "把已有项目中能补充这些技能的位置标出来",
                ],
                deliverable="一页技能差距清单和学习笔记",
            ),
            NextStepPlanWeek(
                week=2,
                plan_type=plan_type,
                focus=f"做一个面向 {target_direction or '目标岗位'} 的小功能",
                goals=["用缺失技能完成可运行功能", "形成能写进简历的项目证据"],
                tasks=[
                    "选择一个已有项目，加入 1-2 个缺失技能相关功能",
                    "补充 README、接口说明或截图",
                    "记录实现过程中的问题和解决方式",
                ],
                deliverable="一个可运行的小功能或项目分支",
            ),
            NextStepPlanWeek(
                week=3,
                plan_type="application_prep",
                focus="岗位匹配强化和简历更新",
                goals=["把新学习内容转化为简历表达", "重新生成岗位推荐并检查分数变化"],
                tasks=[
                    "把新增功能改写成 2-3 条项目 bullet",
                    "更新 Profile skills 和项目 technologies",
                    "重新运行职业方向、简历生成和岗位推荐",
                ],
                deliverable="更新后的项目描述和新版简历",
            ),
        ]

    if gap_severity == "none":
        return [
            NextStepPlanWeek(
                week=1,
                plan_type="application_prep",
                focus="Prepare targeted application materials",
                goals=["Map Top3 job keywords to resume evidence", "Create a targeted resume version"],
                tasks=["Align required skills to resume bullets", "Prepare a short role-specific introduction"],
                deliverable="Targeted resume and keyword mapping",
            ),
            NextStepPlanWeek(
                week=2,
                plan_type="application_prep",
                focus="Prepare interview evidence",
                goals=["Prepare project stories", "Collect portfolio evidence"],
                tasks=["Write STAR stories for 2-3 projects", "Prepare technical decision explanations"],
                deliverable="Project interview notes and portfolio checklist",
            ),
            NextStepPlanWeek(
                week=3,
                plan_type="application_prep",
                focus="Apply and iterate",
                goals=["Start targeted applications", "Track feedback"],
                tasks=["Apply to highest-match jobs", "Record feedback by resume version"],
                deliverable="Application tracker and resume iteration list",
            ),
        ]

    plan_type = "improvement" if gap_severity == "mild" else "learning"
    return [
        NextStepPlanWeek(
            week=1,
            plan_type=plan_type,
            focus=f"Build fundamentals: {', '.join(focus_skills) or 'core job skills'}",
            goals=["Understand common Top3 job requirements", "Complete basic exercises"],
            tasks=["Break each missing skill into concepts and use cases", "Do one small exercise per day"],
            deliverable="Skill gap notes and practice summary",
        ),
        NextStepPlanWeek(
            week=2,
            plan_type=plan_type,
            focus=f"Build a small {target_direction or 'target role'} feature",
            goals=["Use missing skills in a runnable feature", "Create resume evidence"],
            tasks=["Add one feature to an existing project", "Document setup, API, or screenshots"],
            deliverable="Runnable feature or project branch",
        ),
        NextStepPlanWeek(
            week=3,
            plan_type="application_prep",
            focus="Update resume and rerun matching",
            goals=["Turn learning into resume bullets", "Check match score changes"],
            tasks=["Rewrite project bullets", "Update profile skills", "Rerun job matching"],
            deliverable="Updated project description and resume version",
        ),
    ]


def format_jobs(jobs: list[Job]) -> str:
    return "\n".join(
        f"- {job.title} | {job.company} | {job.location} | {job.level} | {job.role_family} | required: {', '.join(job.required_skills)} | nice: {', '.join(job.nice_to_have_skills)}"
        for job in jobs[:3]
    )


def plan_matches_gap_severity(
    plan: list[NextStepPlanWeek],
    gap_severity: str,
) -> bool:
    plan_types = {week.plan_type for week in plan}
    if gap_severity == "none":
        return plan_types <= {"application_prep"}
    if gap_severity == "mild":
        return plan_types <= {"improvement", "application_prep"}
    return bool(plan_types & {"learning", "improvement"})


def generate_next_step_plan(
    profile: Profile,
    projects: list[Project],
    jobs: list[Job],
    user_skills: list[str],
    common_missing_skills: list[str],
    priority_skills: list[str],
    target_direction: str,
    gap_severity: str,
    gap_summary: str,
) -> list[NextStepPlanWeek]:
    output_language = detect_output_language(
        profile.name,
        profile.headline,
        profile.summary,
        user_skills,
        common_missing_skills,
        [job.model_dump() for job in jobs[:3]],
    )
    try:
        draft = invoke_model_with_logging(
            "next_step_plan",
            NEXT_STEP_PLAN_MODEL,
            lambda: (
                NEXT_STEP_PLAN_PROMPT
                | ChatOpenAI(model=NEXT_STEP_PLAN_MODEL).with_structured_output(
                    NextStepPlanDraft,
                    method="json_schema",
                )
            ).invoke(
                {
                    "target_direction": target_direction,
                    "user_skills": ", ".join(user_skills),
                    "common_missing_skills": ", ".join(common_missing_skills),
                    "priority_skills": ", ".join(priority_skills),
                    "gap_severity": gap_severity,
                    "gap_summary": gap_summary,
                    "jobs_text": format_jobs(jobs),
                    "language_instruction": language_instruction(output_language),
                }
            ),
        )
        if len(draft.next_step_plan) == 3 and plan_matches_gap_severity(
            draft.next_step_plan,
            gap_severity,
        ):
            return draft.next_step_plan
    except Exception:
        log_model_fallback("next_step_plan", NEXT_STEP_PLAN_MODEL, "template_next_step_plan")

    return fallback_next_step_plan(
        common_missing_skills,
        priority_skills,
        target_direction,
        gap_severity,
        output_language,
    )


def run_skill_gap_analysis(
    request: SkillGapAnalysisRequest,
    profile: Profile,
    projects: list[Project],
) -> SkillGapAnalysisResponse:
    combined_user_skills = list(request.user_skills)
    combined_user_skills.extend(profile.skills)
    for project in projects:
        combined_user_skills.extend(project.technologies)

    deduped_user_skills = list(dict.fromkeys(skill for skill in combined_user_skills if skill.strip()))
    (
        common_missing,
        priority_skills,
        per_job_gaps,
        has_gap,
        gap_severity,
        gap_summary,
    ) = analyze_skill_gap(
        request.jobs[:3],
        deduped_user_skills,
        request.top_matches[:3],
    )
    next_step_plan = generate_next_step_plan(
        profile,
        projects,
        request.jobs[:3],
        deduped_user_skills,
        common_missing,
        priority_skills,
        request.target_direction,
        gap_severity,
        gap_summary,
    )
    return SkillGapAnalysisResponse(
        has_gap=has_gap,
        gap_severity=gap_severity,
        gap_summary=gap_summary,
        common_missing_skills=common_missing,
        priority_skills=priority_skills,
        per_job_gaps=per_job_gaps,
        next_step_plan=next_step_plan,
        learning_plan=next_step_plan,
    )
