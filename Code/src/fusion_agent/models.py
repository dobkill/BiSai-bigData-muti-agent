from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Source = Literal["open_source", "opinion", "radar"]
FACT_FIELDS = ("country", "platform", "direction", "region", "quantity", "task", "start", "end")
FACT_DEFINITIONS = {
    "country": "当前活动主体、平台或执行方所属国家；主体国籍就是事实，不是背景。不是活动发生地国家，也不是媒体或评论者国别。",
    "platform": "当前活动主体的型号、舰级或具体机舰名，区分类型与个体身份；主体身份也是事实。",
    "direction": "赛题海区分类：东海、南海、黄海或其他；由活动海域/坐标规范化得到，不是罗盘航向。",
    "region": "当前主体实际活动地点或海域；组织名称、媒体国籍、纯背景地点不属于活动地点。",
    "quantity": "明确陈述的该主体架数/艘数；不能将编队总数复制给每个主体。",
    "task": "当前主体明确执行的任务；能力、影响、可能性分析不等于已经执行的任务。",
    "start": "正文活动开始时刻；不是发布时间，不能从只有日期的原文补时分。",
    "end": "正文活动结束时刻；不是发布时间，窗口重叠不能证明结束边界相同。",
}


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
    classification: Literal["trusted", "untrusted"] | None = None
    verification_reasons: list[str] = Field(default_factory=list)
    semantic_review: dict[str, Any] = Field(default_factory=dict)

    @field_validator("classification", mode="before")
    @classmethod
    def migrate_classification(cls, value):
        # Historical snapshots remain readable; new results use two classes.
        return "untrusted" if value in ("suspicious", "unpaired") else value


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

