from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.api.deps import db_session
from app.db.models import Dataset, Pipeline, Run
from app.db.session import engine
from app.rag.eval import evaluate_dataset, score_one
from app.rag.pipeline import PipelineConfig
from app.services.background import submit, status as job_status
from app.services.storage import indexes_dir, runs_dir

router = APIRouter()


class EvalRunRequest(BaseModel):
    project_id: int
    pipeline_id: int
    dataset_id: int
    notes: str = ""
    async_run: bool = Field(default=True, description="Run in background thread")


class QuickScoreRequest(BaseModel):
    project_id: int
    pipeline_id: int
    question: str


def _run_eval_job(run_id: int) -> None:
    """Background evaluation job.

    Uses a dedicated DB session (sqlite check_same_thread disabled).
    """
    from sqlmodel import Session as _Session

    with _Session(engine) as session:
        run = session.get(Run, run_id)
        if not run:
            return

        run.status = "running"
        run.started_at = datetime.utcnow()
        session.add(run)
        session.commit()

        try:
            pipe = session.get(Pipeline, run.pipeline_id)
            if not pipe:
                raise RuntimeError("Pipeline not found")

            cfg = PipelineConfig(**json.loads(pipe.config_json or "{}"))

            ds = session.get(Dataset, run.dataset_id) if run.dataset_id else None
            if not ds:
                raise RuntimeError("Dataset not found")

            dataset_rows = json.loads(ds.data_json or "[]")

            idx_dir = indexes_dir(run.project_id) / f"pipeline_{run.pipeline_id}"
            if not idx_dir.exists():
                raise RuntimeError("Index not built for this pipeline")

            report = evaluate_dataset(index_dir=str(idx_dir), dataset=dataset_rows, cfg=cfg)

            # Persist artifacts
            out_dir = runs_dir(run.project_id) / f"run_{run.id:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "pipeline.json").write_text(
                json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (out_dir / "dataset_meta.json").write_text(
                json.dumps(
                    {
                        "dataset_id": ds.id,
                        "name": ds.name,
                        "description": ds.description,
                        "count": len(dataset_rows),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            run.metrics_json = json.dumps(report.get("metrics", {}), ensure_ascii=False)
            run.status = "done"
            run.finished_at = datetime.utcnow()
            session.add(run)
            session.commit()

        except Exception as e:
            run.status = "error"
            run.metrics_json = json.dumps({"error": str(e)}, ensure_ascii=False)
            run.finished_at = datetime.utcnow()
            session.add(run)
            session.commit()


@router.post("/run")
def run_eval(payload: EvalRunRequest, session: Session = Depends(db_session)):
    pipe = session.get(Pipeline, payload.pipeline_id)
    if not pipe or pipe.project_id != payload.project_id:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    ds = session.get(Dataset, payload.dataset_id)
    if not ds or ds.project_id != payload.project_id:
        raise HTTPException(status_code=404, detail="Dataset not found")

    idx_dir = indexes_dir(payload.project_id) / f"pipeline_{payload.pipeline_id}"
    if not idx_dir.exists():
        raise HTTPException(status_code=400, detail="Index not built for this pipeline")

    run = Run(
        project_id=payload.project_id,
        pipeline_id=payload.pipeline_id,
        dataset_id=payload.dataset_id,
        status="queued",
        notes=payload.notes or "",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    job_id = f"run_{run.id}"
    if payload.async_run:
        submit(job_id=job_id, fn=lambda: _run_eval_job(run.id))
        return {"run_id": run.id, "status": "queued", "job_id": job_id}

    _run_eval_job(run.id)
    session.refresh(run)
    return {"run_id": run.id, "status": run.status, "job_id": job_id}


@router.get("/job/{run_id}")
def eval_job_status(run_id: int):
    return {"run_id": run_id, "job_status": job_status(f"run_{run_id}")}


@router.post("/score_one")
def quick_score(payload: QuickScoreRequest, session: Session = Depends(db_session)):
    pipe = session.get(Pipeline, payload.pipeline_id)
    if not pipe or pipe.project_id != payload.project_id:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    cfg = PipelineConfig(**json.loads(pipe.config_json or "{}"))
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    idx_dir = indexes_dir(payload.project_id) / f"pipeline_{payload.pipeline_id}"
    if not idx_dir.exists():
        raise HTTPException(status_code=400, detail="Index not built for this pipeline")

    metrics, sample = score_one(index_dir=str(idx_dir), question=q, cfg=cfg)
    return {"metrics": metrics, "sample": sample}
