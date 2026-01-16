from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
def status():
    return {
        "engine": "local",
        "ready": True,
        "message": "Local Engine Ready",
    }
