from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PositiveDays = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Settings(BaseSettings):
    app_name: str = "chat_reply_agent"
    database_url: str = "sqlite:///./chat_reply_agent.db"
    upload_dir: str = "chat_reply_agent/uploads"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model_name: str = "gpt-4.1-mini"
    openai_vision_model_name: str = "gpt-4.1-mini"
    openai_vision_fallback_model_name: str = ""
    openai_strong_model_name: str = ""
    openai_review_model_name: str = ""
    openai_memory_model_name: str = ""
    model_timeout_seconds: float = Field(default=90, gt=0, le=300)
    vision_timeout_seconds: float = Field(default=180, gt=0, le=300)
    quality_timeout_seconds: float = Field(default=90, ge=5, le=300)
    quality_queue_timeout_seconds: float = Field(default=30, ge=1, le=120)
    quality_total_timeout_seconds: float = Field(default=240, ge=5, le=600)
    quality_thinking_budget: int = Field(default=2048, ge=256, le=32768)
    quality_quick_enabled: bool = True
    quality_quick_thinking_budget: int = Field(default=512, ge=256, le=32768)
    quality_quick_max_completion_tokens: int = Field(default=3072, ge=1024, le=65536)
    quality_max_completion_tokens: int = Field(default=6144, ge=1024, le=65536)
    quality_repair_enabled: bool = True
    quality_repair_threshold: int = Field(default=80, ge=1, le=100)
    model_structured_output: Literal["auto", "json_schema", "json_object", "prompt"] = "auto"
    weather_tool_enabled: bool = True
    weather_tool_timeout_seconds: float = Field(default=18, ge=3, le=60)
    weather_cache_seconds: int = Field(default=300, ge=0, le=900)

    memory_decay_half_life_days: PositiveDays = 30.0
    memory_decay_half_life_by_scenario: dict[str, PositiveDays] = Field(
        default_factory=dict,
    )
    memory_similarity_backend: Literal["semantic", "lexical"] = "semantic"
    memory_embedding_provider: Literal["local", "dashscope", "openai_compatible"] = "local"
    memory_embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        min_length=1,
    )
    # Reuse is opt-in and restricted to the exact same API base URL.
    memory_embedding_base_url: str = ""
    memory_embedding_api_key: str | None = Field(default=None, repr=False, exclude=True)
    memory_embedding_reuse_chat_credentials: bool = False
    memory_embedding_fallback: Literal["lexical", "error"] = "lexical"
    memory_embedding_timeout_seconds: float = Field(default=20, gt=0, le=120)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
