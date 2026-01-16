from __future__ import annotations

from pathlib import Path

from app.core.config import settings


def project_dir(project_id: int) -> Path:
    d = settings.data_dir / "projects" / f"p{project_id:06d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def documents_dir(project_id: int) -> Path:
    d = project_dir(project_id) / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def indexes_dir(project_id: int) -> Path:
    d = project_dir(project_id) / "indexes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(project_id: int) -> Path:
    d = project_dir(project_id) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d
