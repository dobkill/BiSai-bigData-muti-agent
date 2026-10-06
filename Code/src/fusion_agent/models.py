from typing import Any, Literal

from pydantic import BaseModel, Field

Source = Literal["open_source", "opinion", "radar"]
FACT_FIELDS = ("country", "platform", "direction", "region", "quantity", "task", "start", "end")


class Document(BaseModel):
    id: str
    date: str
    source: Source
    path: str
    text: str
    title: str = ""


class RadarBatch(BaseModel):
    id: str
    date: str
    path: str
    sheet: str
    batch: str
    rows: list[dict[str, Any]]
    row_numbers: list[int]


class Observation(BaseModel):
    id: str
    date: str
    source: Source
    path: str
    # 保留同一来源内部的所有显式不同值，禁止多数票覆盖矛盾。
    facts: dict[str, list[Any]] = Field(default_factory=dict)
    quotes: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PairRelation(BaseModel):
    left_id: str
    right_id: str
    status: Literal["confirmed", "candidate", "conflicted", "unrelated"]
    fields: dict[str, str] = Field(default_factory=dict)
    time_relation: str = "unknown"
    gap_minutes: float | None = None
    discriminating: bool = False
    basis: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    id: str
    date: str
    observation_ids: list[str]
    scores: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    relation_status: Literal["confirmed", "candidate", "conflicted", "unmatched"] | None = None
    relations: list[PairRelation] = Field(default_factory=list)
    basis: list[str] = Field(default_factory=list)
    assignment: dict[str, Any] = Field(default_factory=dict)


class FieldVerdict(BaseModel):
    status: Literal["supported", "conflict", "insufficient", "absent"]
    value: Any = None
    support_sources: list[Source] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    values_by_source: dict[str, list[Any]] = Field(default_factory=dict)


class Event(BaseModel):
    id: str
    date: str
    accepted: bool
    facts: dict[str, Any] = Field(default_factory=dict)
    verdicts: dict[str, FieldVerdict] = Field(default_factory=dict)
    evidence_ids: list[str]
    reasons: list[str] = Field(default_factory=list)
    quality: float = 0.0
    platform_verdicts: dict[str, dict[str, FieldVerdict]] = Field(default_factory=dict)
    classification: Literal["trusted", "suspicious", "unpaired"] | None = None
    verification_reasons: list[str] = Field(default_factory=list)


class DayResult(BaseModel):
    date: str
    observations: list[Observation]
    candidates: list[Candidate]
    events: list[Event]
    trace: list[dict[str, Any]]
    unassigned_observation_ids: list[str] = Field(default_factory=list)


class ExtractedClaim(BaseModel):
    value: str | int | None = None
    quote: str = ""


class LLMExtraction(BaseModel):
    # 使用列表表示多个显式值，不让模型静默解决原文矛盾。
    country: list[ExtractedClaim] = Field(default_factory=list)
    platform: list[ExtractedClaim] = Field(default_factory=list)
    region: list[ExtractedClaim] = Field(default_factory=list)
    quantity: list[ExtractedClaim] = Field(default_factory=list)
    task: list[ExtractedClaim] = Field(default_factory=list)
    start: list[ExtractedClaim] = Field(default_factory=list)
    end: list[ExtractedClaim] = Field(default_factory=list)

