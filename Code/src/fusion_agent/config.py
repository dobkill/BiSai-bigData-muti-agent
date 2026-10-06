from pathlib import Path
import tomllib
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Paths(BaseModel):
    data_dir: Path = Path("../Datas/数据")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")
    review_dir: Path | None = None


class APISettings(BaseModel):
    base_url: str = "https://api.example.com/v1"
    model: str = ""
    api_key: str = Field(default="", repr=False)
    api_key_env: str = "LLM_API_KEY"
    json_mode: bool = True


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = ""
    keep_alive: str = "5m"
    num_ctx: int = Field(default=8192, ge=1024)
    think: bool | None = None
    num_predict: int = Field(default=2048, ge=128)


class LLMSettings(BaseModel):
    enabled: bool = False
    provider: str = "ollama"
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout_seconds: float = Field(default=180, gt=0)
    retries: int = Field(default=2, ge=0, le=10)
    concurrency: int = Field(default=4, ge=1, le=32)
    max_calls: int = Field(default=200, ge=1)
    cache: bool = True
    fail_on_error: bool = False
    api: APISettings = Field(default_factory=APISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)

    @model_validator(mode="after")
    def provider_known(self):
        if self.provider not in {"api", "ollama"}:
            raise ValueError("llm.provider 必须为 api 或 ollama")
        return self


class ExtractionSettings(BaseModel):
    mode: str = "hybrid"

    @model_validator(mode="after")
    def mode_known(self):
        if self.mode not in {"rules", "llm", "hybrid"}:
            raise ValueError("extraction.mode 必须为 rules、llm 或 hybrid")
        return self


class AlignmentSettings(BaseModel):
    mode: Literal["one_to_one", "legacy"] = "one_to_one"
    minimum_pair_score: float = Field(default=22, ge=0)
    ambiguity_margin: float = Field(default=3, ge=0)
    llm_review: bool = True
    max_llm_reviews: int = Field(default=8, ge=0, le=50)
    candidate_gap_hours: float = Field(default=3, ge=0, le=24)
    overlap_slack_minutes: float = Field(default=15, ge=0, le=60)


class VerificationSettings(BaseModel):
    required_fields: list[str] = ["country", "platform", "direction"]
    time_tolerance_minutes: float = Field(default=5, ge=0)
    min_radar_points: int = Field(default=2, ge=1)
    geo_boundary_margin_degrees: float = Field(default=0.3, ge=0, le=2)
    max_air_track_jump_kmh: float = Field(default=2500, gt=0)

    @model_validator(mode="after")
    def fields_known(self):
        allowed = {"country", "platform", "direction", "region", "quantity", "task", "start", "end"}
        if not self.required_fields or not set(self.required_fields) <= allowed:
            raise ValueError("verification.required_fields 含未知字段或为空")
        return self


class ReportSettings(BaseModel):
    lookback_days: int = Field(default=7, ge=1)
    node_dates: list[str] = []


class ReviewExportSettings(BaseModel):
    enabled: bool = True


class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    review_export: ReviewExportSettings = Field(default_factory=ReviewExportSettings)


def load_settings(path: Path) -> Settings:
    path = path.resolve()
    with path.open("rb") as f:
        settings = Settings.model_validate(tomllib.load(f))
    for name in ("data_dir", "output_dir", "cache_dir"):
        value = getattr(settings.paths, name)
        setattr(settings.paths, name, (path.parent / value).resolve())
    settings.paths.review_dir = ((path.parent / settings.paths.review_dir).resolve()
                                 if settings.paths.review_dir is not None else settings.paths.output_dir / "review")
    return settings

