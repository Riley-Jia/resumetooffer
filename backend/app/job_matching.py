import hashlib
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.language import detect_output_language, language_instruction
from app.model_config import JOB_MATCHING_MODEL
from app.model_logging import invoke_model_with_logging, log_model_fallback
from app.schemas import Job, JobMatchRequest, JobMatchResponse, JobMatchResult, Profile, Project


logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

CHROMA_COLLECTION_NAME = "job_matching_persistent"
CHROMA_PERSIST_DIR = Path(os.getenv("JOB_CHROMA_PERSIST_DIR", "backend/data/chroma_jobs"))
BM25_INDEX_CACHE: dict[str, "BM25Index"] = {}

DEFAULT_LEVELS = ["Intern", "Graduate", "Junior", "实习", "校招", "初级", "应届", "1年以内"]
DEFAULT_ROLE_FAMILIES = ["Backend", "AI Application", "Graduate Software Engineer", "Software Engineer"]
ROLE_ALIASES = {
    "Software Engineer": {"Graduate Software Engineer", "Backend", "Full Stack", "Frontend"},
    "Backend Developer": {"Backend"},
    "AI Application Developer": {"AI Application"},
    "Data Analyst": {"Data Analyst"},
    "Full Stack Developer": {"Full Stack"},
    "Frontend Developer": {"Frontend"},
    "Product Manager": {"Product Manager"},
    "Technical Product Manager": {"Product Manager"},
}
SKILL_ALIASES = {
    "fastapi": {"backend api", "python api", "rest api"},
    "spring boot": {"java backend", "microservice", "rest api"},
    "chromadb": {"vector database", "vector search", "rag"},
    "langchain": {"llm application", "agent", "rag"},
    "github actions": {"ci/cd", "automation"},
    "postgresql": {"sql", "database"},
    "mysql": {"sql", "database"},
    "react": {"frontend", "web ui"},
    "tableau": {"bi", "dashboard", "visualization"},
    "power bi": {"bi", "dashboard", "visualization"},
}


class LLMJobJudgement(BaseModel):
    job_id: str
    llm_score: int = Field(ge=0, le=100)
    match_reason: str
    missing_skills: list[str] = Field(default_factory=list)


class LLMRerankOutput(BaseModel):
    judgements: list[LLMJobJudgement] = Field(default_factory=list)


LLM_RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a job matching reranker. The deterministic rule score is already computed.
Only judge the shortlisted jobs using evidence from the profile, projects, resume,
and job requirements. Do not invent user skills or experience.

Give each job an llm_score from 0 to 100. Use the score to capture implicit but
reasonable relationships, for example:
- FastAPI means backend API experience
- Spring Boot means Java backend ecosystem
- ChromaDB means vector search / RAG experience
- GitHub Actions means CI/CD

Return concise match_reason and missing_skills for each job.
Match the user's language. {language_instruction}
""",
        ),
        (
            "human",
            """
User profile:
{profile_text}

Projects:
{projects_text}

Resume:
{resume_text}

Shortlisted jobs:
{jobs_text}
""",
        ),
    ]
)


@dataclass(frozen=True)
class RetrievalHit:
    job_id: str
    score: float
    source: str


@dataclass
class BM25Index:
    corpus_key: str
    job_ids: list[str]
    documents: list[list[str]]
    doc_freq: Counter[str]
    average_length: float
    document_lengths: dict[str, int]


def normalize(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ").strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    words = re.findall(r"[a-z0-9+#.]+|[\u4e00-\u9fff]", normalized)
    phrase_tokens = re.findall(r"[a-z0-9+#.]+(?:\s+[a-z0-9+#.]+)+", normalized)
    return words + phrase_tokens


def build_job_document(job: Job) -> str:
    sections = [
        ("title", job.title),
        ("company", job.company),
        ("location", job.location),
        ("level", job.level),
        ("role_family", job.role_family),
        ("required_skills", ", ".join(job.required_skills)),
        ("nice_to_have_skills", ", ".join(job.nice_to_have_skills)),
        ("description", job.description),
    ]
    return "\n".join(f"{label}: {value}" for label, value in sections if value)


def job_document(job: Job) -> str:
    return build_job_document(job)


def user_document(
    profile: Profile,
    projects: list[Project],
    resume: dict[str, Any] | None,
    target_direction: str,
) -> str:
    project_text = " ".join(
        " ".join(
            [
                project.category,
                project.title,
                project.role,
                project.description,
                " ".join(project.technologies),
                " ".join(project.highlights),
            ]
        )
        for project in projects
    )
    resume_text = ""
    if resume:
        resume_text = " ".join(
            [
                str(resume.get("target_direction", "")),
                str(resume.get("introduction", "")),
                " ".join(resume.get("skills", []) or []),
                str(resume.get("projects", "")),
            ]
        )

    return " ".join(
        [
            target_direction,
            profile.headline,
            profile.summary,
            profile.location,
            " ".join(profile.skills),
            " ".join(profile.education),
            project_text,
            resume_text,
        ]
    )


def user_skills(profile: Profile, projects: list[Project], resume: dict[str, Any] | None) -> set[str]:
    skills = {normalize(skill) for skill in profile.skills if skill.strip()}
    for project in projects:
        skills.update(normalize(skill) for skill in project.technologies if skill.strip())

    if resume:
        skills.update(normalize(skill) for skill in resume.get("skills", []) or [] if skill.strip())

    expanded = set(skills)
    for skill in skills:
        expanded.update(SKILL_ALIASES.get(skill, set()))
    return expanded


def metadata_filter_jobs(
    jobs: list[Job],
    request: JobMatchRequest,
    profile: Profile,
) -> tuple[list[Job], dict[str, list[str] | str]]:
    locations = request.locations or ([profile.location] if profile.location else [])
    levels = request.levels or DEFAULT_LEVELS
    role_families = request.role_families or role_families_for_direction(request.target_direction)
    status = request.status or "active"

    def matches_location(job: Job) -> bool:
        if not locations:
            return True
        wanted = {normalize(location) for location in locations}
        actual = normalize(job.location)
        return actual in wanted or "remote" in wanted or "远程" in wanted

    filtered = [
        job
        for job in jobs
        if normalize(job.status) == normalize(status)
        and matches_location(job)
        and any(normalize(level) in normalize(job.level) or normalize(job.level) in normalize(level) for level in levels)
        and any(job.role_family == role_family for role_family in role_families)
    ]

    if not filtered:
        filtered = [
            job
            for job in jobs
            if normalize(job.status) == normalize(status)
            and any(normalize(level) in normalize(job.level) or normalize(job.level) in normalize(level) for level in levels)
            and any(job.role_family == role_family for role_family in role_families)
        ]

    return filtered, {
        "locations": locations,
        "levels": levels,
        "role_families": role_families,
        "status": status,
    }


def role_families_for_direction(target_direction: str) -> list[str]:
    if not target_direction:
        return DEFAULT_ROLE_FAMILIES

    target = normalize(target_direction)
    families: set[str] = set()
    for alias, mapped in ROLE_ALIASES.items():
        alias_text = normalize(alias)
        if target in alias_text or alias_text in target:
            families.update(mapped)

    for family in [
        "Backend",
        "AI Application",
        "Graduate Software Engineer",
        "Data Analyst",
        "Full Stack",
        "Frontend",
        "Product Manager",
    ]:
        if normalize(family) in target or target in normalize(family):
            families.add(family)

    return sorted(families) if families else DEFAULT_ROLE_FAMILIES


def jobs_corpus_key(jobs: list[Job]) -> str:
    fingerprint = "|".join(
        f"{job.id}:{hashlib.md5(job_document(job).encode('utf-8'), usedforsecurity=False).hexdigest()}"
        for job in sorted(jobs, key=lambda item: item.id)
    )
    return hashlib.md5(fingerprint.encode("utf-8"), usedforsecurity=False).hexdigest()


def build_bm25_index(jobs: list[Job]) -> BM25Index:
    corpus_key = jobs_corpus_key(jobs)
    documents = [tokenize(job_document(job)) for job in jobs]
    doc_freq: Counter[str] = Counter()
    for document in documents:
        doc_freq.update(set(document))

    average_length = sum(len(document) for document in documents) / len(documents) if documents else 0.0
    return BM25Index(
        corpus_key=corpus_key,
        job_ids=[job.id for job in jobs],
        documents=documents,
        doc_freq=doc_freq,
        average_length=average_length,
        document_lengths={job.id: len(document) for job, document in zip(jobs, documents, strict=True)},
    )


def get_bm25_index(jobs: list[Job]) -> BM25Index:
    corpus_key = jobs_corpus_key(jobs)
    cached = BM25_INDEX_CACHE.get(corpus_key)
    if cached:
        return cached

    index = build_bm25_index(jobs)
    BM25_INDEX_CACHE.clear()
    BM25_INDEX_CACHE[corpus_key] = index
    return index


def normalize_scores(scored: list[tuple[float, str]]) -> list[RetrievalHit]:
    if not scored:
        return []
    positive = [(score, job_id) for score, job_id in scored if score > 0]
    if not positive:
        return []
    max_score = max(score for score, _ in positive)
    if max_score <= 0:
        return []
    return [
        RetrievalHit(job_id=job_id, score=round(score / max_score, 4), source="")
        for score, job_id in positive
    ]


def bm25_retrieve(query: str, jobs: list[Job], top_n: int = 50) -> list[RetrievalHit]:
    index = get_bm25_index(jobs)
    query_terms = tokenize(query)
    if not index.documents or not query_terms or index.average_length == 0:
        return []

    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, str]] = []
    for job_id, document in zip(index.job_ids, index.documents, strict=True):
        term_counts = Counter(document)
        score = 0.0
        for term in query_terms:
            if term not in term_counts:
                continue
            idf = math.log(1 + (len(index.documents) - index.doc_freq[term] + 0.5) / (index.doc_freq[term] + 0.5))
            tf = term_counts[term]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(document) / index.average_length))
        scored.append((score, job_id))

    hits = normalize_scores(sorted(scored, reverse=True)[:top_n])
    return [RetrievalHit(job_id=hit.job_id, score=hit.score, source="bm25") for hit in hits]


def hash_embedding(text: str, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
        index = int(digest[:8], 16) % dimensions
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def local_vector_retrieve(query: str, jobs: list[Job], top_n: int = 50) -> list[RetrievalHit]:
    query_vector = hash_embedding(query)
    scored = [
        (cosine_similarity(query_vector, hash_embedding(job_document(job))), job.id)
        for job in jobs
    ]
    hits = normalize_scores(sorted(scored, reverse=True)[:top_n])
    return [RetrievalHit(job_id=hit.job_id, score=hit.score, source="local_vector") for hit in hits]


class HashEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [hash_embedding(item) for item in input]


def job_document_hash(job: Job) -> str:
    return hashlib.md5(job_document(job).encode("utf-8"), usedforsecurity=False).hexdigest()


def ensure_chroma_job_index(jobs: list[Job]) -> Any:
    import chromadb
    from chromadb.config import Settings

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        CHROMA_COLLECTION_NAME,
        embedding_function=HashEmbeddingFunction(),
    )
    ids = [job.id for job in jobs]
    if ids:
        collection.upsert(
            ids=ids,
            documents=[job_document(job) for job in jobs],
            metadatas=[
                {
                    "job_id": job.id,
                    "document_hash": job_document_hash(job),
                    "role_family": job.role_family,
                    "location": job.location,
                    "level": job.level,
                    "status": job.status,
                }
                for job in jobs
            ],
        )
    return collection


def chroma_retrieve(query: str, jobs: list[Job], top_n: int = 50) -> list[RetrievalHit]:
    try:
        collection = ensure_chroma_job_index(jobs)
        allowed_ids = {job.id for job in jobs}
        collection_count = collection.count()
        if collection_count == 0:
            return []
        query_count = min(max(top_n * 4, top_n), collection_count)
        result = collection.query(
            query_texts=[query],
            n_results=query_count,
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        scored: list[tuple[float, str]] = []
        for job_id, distance in zip(ids, distances, strict=False):
            if job_id not in allowed_ids:
                continue
            similarity = 1 / (1 + float(distance))
            scored.append((similarity, job_id))
            if len(scored) >= top_n:
                break
        hits = normalize_scores(scored)
        return [RetrievalHit(job_id=hit.job_id, score=hit.score, source="chroma") for hit in hits]
    except Exception:
        return local_vector_retrieve(query, jobs, top_n)


def merge_candidates(
    jobs_by_id: dict[str, Job],
    bm25_hits: list[RetrievalHit],
    vector_hits: list[RetrievalHit],
) -> dict[str, dict[str, float]]:
    candidates: dict[str, dict[str, float]] = defaultdict(dict)
    for hit in bm25_hits:
        if hit.job_id in jobs_by_id:
            candidates[hit.job_id][hit.source] = hit.score
    for hit in vector_hits:
        if hit.job_id in jobs_by_id:
            candidates[hit.job_id][hit.source] = hit.score

    for source_scores in candidates.values():
        bm25_score = source_scores.get("bm25", 0.0)
        vector_score = max(source_scores.get("chroma", 0.0), source_scores.get("local_vector", 0.0))
        source_scores["fusion"] = round(bm25_score * 0.45 + vector_score * 0.55, 4)
    return candidates


def level_score(job: Job, wanted_levels: list[str]) -> float:
    if not wanted_levels:
        return 1.0
    job_level = normalize(job.level)
    return 1.0 if any(normalize(level) in job_level or job_level in normalize(level) for level in wanted_levels) else 0.0


def location_score(job: Job, wanted_locations: list[str]) -> float:
    if not wanted_locations:
        return 0.8
    wanted = {normalize(location) for location in wanted_locations}
    actual = normalize(job.location)
    if actual in wanted:
        return 1.0
    if "remote" in wanted or "远程" in wanted:
        return 0.8
    return 0.0


def role_family_score(job: Job, wanted_families: list[str]) -> float:
    return 1.0 if not wanted_families or job.role_family in wanted_families else 0.0


def skill_scores(job: Job, skills: set[str]) -> tuple[float, list[str], list[str]]:
    required = [normalize(skill) for skill in job.required_skills if skill.strip()]
    if not required:
        return 1.0, [], []

    matched: list[str] = []
    missing: list[str] = []
    for original, normalized in zip(job.required_skills, required, strict=True):
        expanded = {normalized, *SKILL_ALIASES.get(normalized, set())}
        if expanded & skills:
            matched.append(original)
        else:
            missing.append(original)

    return len(matched) / len(required), matched, missing


def rule_rerank(
    candidates: dict[str, dict[str, float]],
    jobs_by_id: dict[str, Job],
    request: JobMatchRequest,
    profile: Profile,
    projects: list[Project],
    resume: dict[str, Any] | None,
) -> list[JobMatchResult]:
    skills = user_skills(profile, projects, resume)
    wanted_locations = request.locations or ([profile.location] if profile.location else [])
    wanted_levels = request.levels or DEFAULT_LEVELS
    wanted_families = request.role_families or role_families_for_direction(request.target_direction)

    results: list[JobMatchResult] = []
    for job_id, source_scores in candidates.items():
        job = jobs_by_id[job_id]
        coverage, matched, missing = skill_scores(job, skills)
        location = location_score(job, wanted_locations)
        level = level_score(job, wanted_levels)
        role = role_family_score(job, wanted_families)
        rule = coverage * 50 + location * 20 + level * 15 + role * 15
        results.append(
            JobMatchResult(
                job=job,
                final_score=round(rule, 2),
                rule_score=round(rule, 2),
                llm_score=round(rule, 2),
                retrieval_fusion_score=round(source_scores.get("fusion", 0.0) * 100, 2),
                skill_coverage=round(coverage, 2),
                location_score=round(location * 20, 2),
                level_score=round(level * 15, 2),
                role_family_score=round(role * 15, 2),
                match_reason=rule_reason(job, coverage, matched, missing, location, role, source_scores),
                missing_skills=missing,
                matched_skills=matched,
                retrieval_sources=sorted(source for source in source_scores if source != "fusion"),
                retrieval_source_scores={
                    source: round(score * 100, 2)
                    for source, score in sorted(source_scores.items())
                },
            )
        )

    return sorted(results, key=lambda item: item.rule_score, reverse=True)


def rule_reason(
    job: Job,
    coverage: float,
    matched: list[str],
    missing: list[str],
    location: float,
    role: float,
    source_scores: dict[str, float],
) -> str:
    parts = [
        f"技能覆盖率 {round(coverage * 100)}%",
        f"匹配技能：{', '.join(matched) if matched else '暂无'}",
    ]
    if source_scores:
        source_text = "，".join(
            f"{source} {round(score * 100)}"
            for source, score in sorted(source_scores.items())
        )
        parts.append(f"检索分：{source_text}")
    if missing:
        parts.append(f"缺失技能：{', '.join(missing)}")
    if location == 0:
        parts.append(f"地点不匹配：岗位在{job.location}")
    if role == 0:
        parts.append(f"方向不完全匹配：{job.role_family}")
    return "；".join(parts)


def format_projects(projects: list[Project]) -> str:
    return "\n".join(
        f"- {project.category} | {project.title} | {project.role} | {project.description} | 技术栈：{', '.join(project.technologies)} | 要点：{'; '.join(project.highlights)}"
        for project in projects
    )


def format_resume(resume: dict[str, Any] | None) -> str:
    if not resume:
        return ""
    return " ".join(
        [
            str(resume.get("target_direction", "")),
            str(resume.get("introduction", "")),
            " ".join(resume.get("skills", []) or []),
            str(resume.get("projects", "")),
        ]
    )


def format_jobs_for_llm(candidates: list[JobMatchResult]) -> str:
    return "\n".join(
        "\n".join(
            [
                f"job_id: {item.job.id}",
                f"title: {item.job.title}",
                f"company: {item.job.company}",
                f"location: {item.job.location}",
                f"level: {item.job.level}",
                f"role_family: {item.job.role_family}",
                f"required_skills: {', '.join(item.job.required_skills)}",
                f"nice_to_have_skills: {', '.join(item.job.nice_to_have_skills)}",
                f"description: {item.job.description}",
                f"rule_score: {item.rule_score}",
                f"matched_skills: {', '.join(item.matched_skills)}",
                f"missing_skills: {', '.join(item.missing_skills)}",
            ]
        )
        for item in candidates
    )


def apply_llm_rerank(
    candidates: list[JobMatchResult],
    profile: Profile,
    projects: list[Project],
    resume: dict[str, Any] | None,
) -> list[JobMatchResult]:
    if not candidates:
        return candidates

    output_language = detect_output_language(
        profile.name,
        profile.headline,
        profile.summary,
        profile.skills,
        [project.model_dump() for project in projects],
        resume or {},
    )
    try:
        output = invoke_model_with_logging(
            "job_matching",
            JOB_MATCHING_MODEL,
            lambda: (
                LLM_RERANK_PROMPT
                | ChatOpenAI(model=JOB_MATCHING_MODEL).with_structured_output(
                    LLMRerankOutput,
                    method="json_schema",
                )
            ).invoke(
                {
                    "profile_text": user_document(profile, projects, resume, ""),
                    "projects_text": format_projects(projects),
                    "resume_text": format_resume(resume),
                    "jobs_text": format_jobs_for_llm(candidates),
                    "language_instruction": language_instruction(output_language),
                }
            ),
        )
    except Exception:
        log_model_fallback("job_matching", JOB_MATCHING_MODEL, "rule_only_job_matching")
        return candidates

    judgements = {item.job_id: item for item in output.judgements}
    reranked: list[JobMatchResult] = []
    for candidate in candidates:
        judgement = judgements.get(candidate.job.id)
        if judgement:
            candidate.llm_score = judgement.llm_score
            candidate.match_reason = judgement.match_reason or candidate.match_reason
            candidate.missing_skills = judgement.missing_skills or candidate.missing_skills
        candidate.final_score = round(candidate.rule_score * 0.7 + candidate.llm_score * 0.3, 2)
        reranked.append(candidate)

    return sorted(reranked, key=lambda item: item.final_score, reverse=True)


def match_jobs(
    profile: Profile,
    projects: list[Project],
    resume: dict[str, Any] | None,
    jobs: list[Job],
    request: JobMatchRequest,
) -> JobMatchResponse:
    if jobs:
        try:
            ensure_chroma_job_index(jobs)
        except Exception:
            pass

    filtered_jobs, metadata = metadata_filter_jobs(jobs, request, profile)
    query = user_document(profile, projects, resume, request.target_direction)
    bm25_hits = bm25_retrieve(query, filtered_jobs, 50)
    vector_hits = chroma_retrieve(query, filtered_jobs, 50)
    jobs_by_id = {job.id: job for job in filtered_jobs}
    candidates = merge_candidates(jobs_by_id, bm25_hits, vector_hits)

    if not candidates:
        candidates = {job.id: {"metadata": 1.0, "fusion": 1.0} for job in filtered_jobs[:50]}

    rule_ranked = rule_rerank(candidates, jobs_by_id, request, profile, projects, resume)
    llm_count = max(0, min(request.llm_candidate_count, len(rule_ranked)))
    llm_ranked = apply_llm_rerank(rule_ranked[:llm_count], profile, projects, resume)
    remaining = rule_ranked[llm_count:]
    final = sorted([*llm_ranked, *remaining], key=lambda item: item.final_score, reverse=True)

    top_k = max(1, min(request.top_k, 25))
    return JobMatchResponse(
        matches=final[:top_k],
        metadata_filter=metadata,
        candidate_counts={
            "metadata_filtered": len(filtered_jobs),
            "bm25_top": len(bm25_hits),
            "chroma_top": len([hit for hit in vector_hits if hit.source == "chroma"]),
            "local_vector_top": len([hit for hit in vector_hits if hit.source == "local_vector"]),
            "merged_candidates": len(candidates),
            "llm_reranked": llm_count,
        },
    )
