import logging
import time
from collections.abc import Callable
from typing import TypeVar


logger = logging.getLogger("resume_to_offer.model")
logger.setLevel(logging.INFO)

T = TypeVar("T")


def invoke_model_with_logging(
    task: str,
    model: str,
    invoke: Callable[[], T],
) -> T:
    started_at = time.perf_counter()
    logger.info("model_call_start task=%s model=%s", task, model)

    try:
        result = invoke()
    except Exception as error:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        logger.error(
            "model_call_failed task=%s model=%s elapsed_ms=%s error_type=%s error=%s",
            task,
            model,
            elapsed_ms,
            type(error).__name__,
            str(error),
        )
        raise

    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    logger.info(
        "model_call_success task=%s model=%s elapsed_ms=%s",
        task,
        model,
        elapsed_ms,
    )
    return result


def log_model_fallback(task: str, model: str, fallback: str) -> None:
    logger.warning(
        "model_call_fallback task=%s model=%s fallback=%s",
        task,
        model,
        fallback,
    )
