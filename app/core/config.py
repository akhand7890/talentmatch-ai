"""
Configuration module for TalentMatch AI Agent.
Loads environment variables using Pydantic Settings.
"""
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "TalentMatch AI - Job Matching & Career Co-Pilot Agent"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/talent_match.db"

    # LLM Settings
    LLM_PROVIDER: Literal["local_mock", "openai", "groq", "gemini"] = "local_mock"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Embeddings & Search
    EMBEDDING_PROVIDER: Literal["local_dense", "faiss", "openai"] = "local_dense"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5

    # Agent Context & Memory
    MAX_CHAT_TURNS_MEMORY: int = 10
    AUTO_EXTRACT_CANDIDATE_MEMORY: bool = True

    # Vector store index file path
    VECTOR_INDEX_DIR: str = str(BASE_DIR / "data" / "index")


settings = Settings()
