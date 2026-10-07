from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuotedFact(StrictModel):
    value: str | None
    quote: str


class Platform(StrictModel):
    name: str
    country: str | None
    quantity: int | None
    quote: str
    quantity_quote: str


class Extracted(StrictModel):
    source_id: str
    platforms: list[Platform] = Field(min_length=1)
    location: QuotedFact
    mission: QuotedFact
    start_time: str | None
    end_time: str | None
    time_quote: str
    movement: Literal["进入", "离开", "活动", "未说明"]
    movement_quote: str


class ExtractionBatch(StrictModel):
    items: list[Extracted]


class Edge(StrictModel):
    candidate_id: str
    platform_indices: list[int]
    relation: Literal["same", "uncertain", "different"]
    confidence: float = Field(ge=0, le=1)
    reason: str


class Alignment(StrictModel):
    opinion_edges: list[Edge]
    radar_edges: list[Edge]
    ambiguity: str


class FactDecision(StrictModel):
    claim: str
    supporting_sources: list[str]
    conflicting_sources: list[str]
    status: Literal["supported", "conflict", "insufficient"]
    reason: str


class PlatformVerification(StrictModel):
    platform_index: int
    platform: FactDecision
    country: FactDecision
    quantity: FactDecision


class Verification(StrictModel):
    same_event: bool
    platforms: list[PlatformVerification]
    location: FactDecision
    time: FactDecision
    mission: FactDecision
    movement: FactDecision
    unresolved_issues: list[str]
    explanation: str
