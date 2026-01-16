from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import db_session
from app.db.models import Run
from app.services.storage import runs_dir

router = APIRouter()


class RunOut(BaseModel):
    id: int
    project_id: int
    pipeline_id: int
    dataset_id: Optional[int]
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    metrics: Dict[str, Any] = {}
    notes: str = ""


@router.get("", response_model=List[RunOut])
def list_runs(project_id: int, session: Session = Depends(db_session)):
    items = session.exec(select(Run).where(Run.project_id == project_id).order_by(Run.id.desc())).all()
    out: List[RunOut] = []
    for r in items:
        try:
            metrics = json.loads(r.metrics_json or "{}")
        except Exception:
            metrics = {}
        out.append(
            RunOut(
                id=r.id,
                project_id=r.project_id,
                pipeline_id=r.pipeline_id,
                dataset_id=r.dataset_id,
                status=r.status,
                created_at=r.created_at.isoformat(),
                started_at=r.started_at.isoformat() if r.started_at else None,
                finished_at=r.finished_at.isoformat() if r.finished_at else None,
                metrics=metrics,
                notes=r.notes or "",
            )
        )
    return out


@router.get("/{run_id}")
def get_run(run_id: int, session: Session = Depends(db_session)):
    r = session.get(Run, run_id)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        metrics = json.loads(r.metrics_json or "{}")
    except Exception:
        metrics = {}

    out_dir = runs_dir(r.project_id) / f"run_{r.id:06d}"
    report_path = out_dir / "report.json"
    report: Optional[Dict[str, Any]] = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = None

    return {
        "id": r.id,
        "project_id": r.project_id,
        "pipeline_id": r.pipeline_id,
        "dataset_id": r.dataset_id,
        "status": r.status,
        "created_at": r.created_at.isoformat(),
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "notes": r.notes or "",
        "metrics": metrics,
        "report": report,
        "artifacts_dir": str(out_dir) if out_dir.exists() else None,
    }
