from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_ARCH_", env_file=".env", extra="ignore")

    app_name: str = "RAG-Architect"
    api_prefix: str = "/api"


    data_dir: Path = Path(__file__).resolve().parents[2] / "data"


    cors_origins: str = "http://localhost:5173"


    default_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    default_rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    default_llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"


settings = Settings()
