import logging
import time
from collections.abc import Callable
from collections import Counter
from typing import TypeVar

from app.telemetry import log_event


logger = logging.getLogger("resume_to_offer.model")
logger.setLevel(logging.INFO)

T = TypeVar("T")
_MODEL_METRICS: Counter[str] = Counter()


def reset_model_metrics() -> None:
    _MODEL_METRICS.clear()


def get_model_metrics() -> dict[str, int | float]:
    starts = _MODEL_METRICS["starts"]
    fallbacks = _MODEL_METRICS["fallbacks"]
    return {
        "starts": starts,
        "successes": _MODEL_METRICS["successes"],
        "failures": _MODEL_METRICS["failures"],
        "fallbacks": fallbacks,
        "fallback_rate": fallbacks / starts if starts else 0.0,
    }


def invoke_model_with_logging(
    task: str,
    model: str,
    invoke: Callable[[], T],
) -> T:
    started_at = time.perf_counter()
    _MODEL_METRICS["starts"] += 1
    _MODEL_METRICS[f"task.{task}.starts"] += 1
    logger.info("model_call_start task=%s model=%s", task, model)
    log_event("model_call_start", task=task, model=model)

    try:
        result = invoke()
    except Exception as error:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        _MODEL_METRICS["failures"] += 1
        _MODEL_METRICS[f"task.{task}.failures"] += 1
        logger.error(
            "model_call_failed task=%s model=%s elapsed_ms=%s error_type=%s error=%s",
            task,
            model,
            elapsed_ms,
            type(error).__name__,
            str(error),
        )
        log_event(
            "model_call_failed",
            task=task,
            model=model,
            elapsed_ms=elapsed_ms,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    _MODEL_METRICS["successes"] += 1
    _MODEL_METRICS[f"task.{task}.successes"] += 1
    logger.info(
        "model_call_success task=%s model=%s elapsed_ms=%s",
        task,
        model,
        elapsed_ms,
    )
    log_event("model_call_success", task=task, model=model, elapsed_ms=elapsed_ms)
    return result


def log_model_fallback(task: str, model: str, fallback: str) -> None:
    _MODEL_METRICS["fallbacks"] += 1
    _MODEL_METRICS[f"task.{task}.fallbacks"] += 1
    logger.warning(
        "model_call_fallback task=%s model=%s fallback=%s",
        task,
        model,
        fallback,
    )
    log_event("model_call_fallback", task=task, model=model, fallback=fallback)
