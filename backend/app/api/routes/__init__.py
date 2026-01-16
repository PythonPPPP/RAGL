from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.projects import router as projects_router
from app.api.routes.documents import router as documents_router
from app.api.routes.pipelines import router as pipelines_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.chat import router as chat_router
from app.api.routes.eval import router as eval_router
from app.api.routes.runs import router as runs_router
from app.api.routes.status import router as status_router
from app.api.routes.demo import router as demo_router
from app.api.routes.share import router as share_router
from app.api.routes.playground import router as playground_router

router = APIRouter()
router.include_router(status_router, tags=["status"]) 
router.include_router(projects_router, prefix="/projects", tags=["projects"])
router.include_router(documents_router, prefix="/documents", tags=["documents"])
router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
router.include_router(pipelines_router, prefix="/pipelines", tags=["pipelines"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(eval_router, prefix="/eval", tags=["eval"])
router.include_router(runs_router, prefix="/runs", tags=["runs"])
router.include_router(demo_router, prefix="/demo", tags=["demo"])
router.include_router(share_router, prefix="/share", tags=["share"])
router.include_router(playground_router, prefix="/playground", tags=["playground"])
