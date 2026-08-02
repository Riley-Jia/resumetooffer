import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


logger = logging.getLogger("resume_to_offer.telemetry")
logger.setLevel(logging.INFO)
current_agent_run_id: ContextVar[str] = ContextVar("current_agent_run_id", default="")


@contextmanager
def telemetry_context(agent_run_id: str = ""):
    token = current_agent_run_id.set(agent_run_id)
    try:
        yield
    finally:
        current_agent_run_id.reset(token)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_database_event(payload: dict[str, Any]) -> None:
    if os.getenv("TELEMETRY_DB_ENABLED") != "1":
        return

    try:
        from app.database import SessionLocal
        from app.models import (
            AgentRunEventModel,
            AgentStepEventModel,
            ModelCallEventModel,
            PartialRerunEventModel,
            RetrievalRankingEventModel,
            TelemetryEventModel,
        )

        event_type = str(payload["event_type"])
        agent_run_id = str(payload.get("agent_run_id") or current_agent_run_id.get())
        stored_payload = _json_safe({**payload, "agent_run_id": agent_run_id})
        with SessionLocal() as db:
            db.add(
                TelemetryEventModel(
                    event_type=event_type,
                    agent_run_id=agent_run_id,
                    payload=stored_payload,
                )
            )
            if event_type == "agent_run_summary":
                db.add(
                    AgentRunEventModel(
                        agent_run_id=agent_run_id,
                        profile_id=int(payload.get("profile_id") or 1),
                        target_direction=str(payload.get("target_direction") or ""),
                        selected_tools=payload.get("selected_tools") or [],
                        step_count=int(payload.get("step_count") or 0),
                        reused_step_count=int(payload.get("reused_step_count") or 0),
                        rerun_step_count=int(payload.get("rerun_step_count") or 0),
                        is_feedback_rerun=bool(payload.get("is_feedback_rerun")),
                        latest_resume_id=str(payload.get("latest_resume_id") or ""),
                        latest_job_match_count=int(payload.get("latest_job_match_count") or 0),
                        gap_severity=str(payload.get("gap_severity") or ""),
                    )
                )
            elif event_type == "task_step_trace":
                db.add(
                    AgentStepEventModel(
                        agent_run_id=agent_run_id,
                        step_id=str(payload.get("step_id") or ""),
                        tool_name=str(payload.get("tool_name") or ""),
                        depends_on=payload.get("depends_on") or [],
                        status=str(payload.get("status") or ""),
                        rerun_policy=str(payload.get("rerun_policy") or ""),
                        input_refs=payload.get("input_refs") or [],
                        output_refs=payload.get("output_refs") or [],
                        is_reused=bool(payload.get("is_reused")),
                        is_rerun=bool(payload.get("is_rerun")),
                    )
                )
            elif event_type == "partial_rerun_event":
                db.add(
                    PartialRerunEventModel(
                        agent_run_id=agent_run_id,
                        feedback_text=str(payload.get("feedback_text") or ""),
                        changed_preferences=payload.get("changed_preferences") or {},
                        reused_steps=payload.get("reused_steps") or [],
                        rerun_steps=payload.get("rerun_steps") or [],
                        saved_step_count=int(payload.get("saved_step_count") or 0),
                        reuse_rate=payload.get("reuse_rate") or 0,
                    )
                )
            elif event_type == "retrieval_ranking_trace":
                db.add(
                    RetrievalRankingEventModel(
                        agent_run_id=agent_run_id,
                        target_direction=str(payload.get("target_direction") or ""),
                        top_k=int(payload.get("top_k") or 10),
                        metadata_candidate_count=int(payload.get("metadata_candidate_count") or 0),
                        bm25_candidate_count=int(payload.get("bm25_candidate_count") or 0),
                        vector_candidate_count=int(payload.get("vector_candidate_count") or 0),
                        merged_candidate_count=int(payload.get("merged_candidate_count") or 0),
                        hybrid_overlap_count=int(payload.get("hybrid_overlap_count") or 0),
                        llm_rerank_candidate_count=int(payload.get("llm_rerank_candidate_count") or 0),
                        top_matches=payload.get("top_matches") or [],
                    )
                )
            elif event_type.startswith("model_call_"):
                db.add(
                    ModelCallEventModel(
                        agent_run_id=agent_run_id,
                        task=str(payload.get("task") or ""),
                        model=str(payload.get("model") or ""),
                        event_type=event_type,
                        success=True
                        if event_type == "model_call_success"
                        else False
                        if event_type in {"model_call_failed", "model_call_fallback"}
                        else None,
                        elapsed_ms=payload.get("elapsed_ms"),
                        fallback=str(payload.get("fallback") or ""),
                        error_type=str(payload.get("error_type") or ""),
                        error_message=str(payload.get("error") or ""),
                        prompt_tokens=payload.get("prompt_tokens"),
                        completion_tokens=payload.get("completion_tokens"),
                        estimated_cost=payload.get("estimated_cost"),
                    )
                )
            db.commit()
    except Exception as error:
        logger.debug("telemetry_db_write_failed error=%s", error)


def log_event(event_type: str, **fields: Any) -> None:
    payload = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_run_id": fields.pop("agent_run_id", "") or current_agent_run_id.get(),
        **fields,
    }
    logger.info(
        "telemetry_event %s",
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
    )
    _write_database_event(payload)
