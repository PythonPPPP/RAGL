from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.db.session import init_db
from app.api.routes import router


def create_app() -> FastAPI:
    # Make HF downloads more reliable/consistent across runs.
    # This helps avoid transient timeouts when downloading small models.
    try:
        data_dir = Path(settings.data_dir)
        hf_home = data_dir / "hf_home"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", "60")
    except Exception:
        # best-effort only
        pass

    setup_logging()
    init_db()

    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=settings.api_prefix)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
