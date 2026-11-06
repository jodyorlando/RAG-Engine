from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Engine"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database & Vector Store
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    EMBEDDING_DIMENSION: int = 1536  # text-embedding-3-small dimension

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # OpenAI & AI Services
    OPENAI_API_KEY: str = "mock-key-for-dev"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v.startswith("postgres://"):
                return v.replace("postgres://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
                return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()