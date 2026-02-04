from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, Optional

from loguru import logger


_executor = ThreadPoolExecutor(max_workers=2)
_jobs: Dict[str, Future] = {}
_results: Dict[str, Any] = {}
_errors: Dict[str, str] = {}


def submit(job_id: str, fn: Callable[[], Any]) -> None:
    if job_id in _jobs and not _jobs[job_id].done():
        return

    # reset previous state
    _results.pop(job_id, None)
    _errors.pop(job_id, None)

    def _wrap():
        try:
            res = fn()
            _results[job_id] = res
            return res
        except Exception:
            logger.exception(f"Background job failed: {job_id}")
            _errors[job_id] = "failed"
            raise

    _jobs[job_id] = _executor.submit(_wrap)


def status(job_id: str) -> str:
    f = _jobs.get(job_id)
    if f is None:
        return "unknown"
    if f.running():
        return "running"
    if f.done():
        return "done" if f.exception() is None else "error"
    return "queued"


def get_result(job_id: str) -> Optional[Any]:
    return _results.get(job_id)


def get_error(job_id: str) -> Optional[str]:
    # If Future has exception, prefer a stringified version
    f = _jobs.get(job_id)
    if f is not None and f.done() and f.exception() is not None:
        return str(f.exception())
    return _errors.get(job_id)
