"""Model contracts and evidence checks; string lookup never chooses semantic roles."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .domain import (COUNTRY_ALIASES, PLATFORM_ALIASES, REGION_ALIASES,
                     REGION_DIRECTION, TASK_ALIASES, TASKS, canonical_platform, clean)
from .models import ExtractedClaim, FACT_FIELDS, LLMExtraction
from .utils import unique


def anchored_quote(quote, text):
    """Restore presentation whitespace only; never rewrite words or punctuation."""
    if not isinstance(quote, str) or not quote.strip():
        return None
    if quote in text:
        return quote
    import re
    characters = [c for c in quote if not c.isspace()]
    match = re.search(r'\s*'.join(re.escape(c) for c in characters), text)
    return text[match.start():match.end()] if match else None


class ActivityExtraction(LLMExtraction):
    activity_id: str
    platform_kind: Literal["air", "ship", "unknown"]
    individual: list[ExtractedClaim] = Field(default_factory=list)


class ExcludedMention(BaseModel):
    quote: str
    role: Literal["background", "organization", "commentary", "publication_time"]
    reason: str


class SemanticExtraction(BaseModel):
    activities: list[ActivityExtraction]
    excluded_mentions: list[ExcludedMention] = Field(default_factory=list)


def source_quote_options(text):
    """Offer verbatim source spans; the model still chooses their semantic role."""
    import re
    spans = [text] + [s.strip() for s in re.split(r"[\n。！？]", text) if s.strip()]
    spans += [s.strip() for s in re.split(r"[\n。！？，,；;]", text) if s.strip()]
    return unique(spans)


def association_quote_options(text):
    """Use full body sentences for activity citations, avoiding copied summaries."""
    import re
    body = [line for line in text.splitlines() if line.startswith("正文：") or line.startswith("正文:")]
    return unique(s.strip() for line in body for s in re.split(r"[。！？]", line) if s.strip()) if body else [text]


def extraction_schema(text=None):
    """Require explicit empty arrays in generation, retaining sparse snapshot compatibility."""
    from copy import deepcopy
    schema = deepcopy(SemanticExtraction.model_json_schema())
    definitions = schema["$defs"]
    activity = definitions["ActivityExtraction"]
    activity["required"] = list(activity["properties"])
    claim = definitions["ExtractedClaim"]
    claim["required"] = ["value", "quote"]
    claim["properties"]["value"] = {"anyOf": [{"type": "string", "minLength": 1}, {"type": "integer"}]}
    claim["properties"]["quote"] = {"type": "string", "minLength": 1,
                                       "description": "原文连续片段，不改写、不拼接"}
    definitions["ActivityTimeClaim"] = {"type": "object", "required": ["value", "quote"],
        "properties": {"value": {"type": "string", "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$"},
                       "quote": claim["properties"]["quote"]}}
    definitions["ActivityQuantityClaim"] = {"type": "object", "required": ["value", "quote"],
        "properties": {"value": {"type": "integer", "minimum": 1}, "quote": claim["properties"]["quote"]}}
    for field in ("start", "end"):
        activity["properties"][field]["items"] = {"$ref": "#/$defs/ActivityTimeClaim"}
    activity["properties"]["quantity"]["items"] = {"$ref": "#/$defs/ActivityQuantityClaim"}
    if text:
        quotes = source_quote_options(text)
        claim["properties"]["quote"]["enum"] = quotes
        definitions["ExcludedMention"]["properties"]["quote"]["enum"] = quotes
        # Time citations must actually contain a clock. This restricts copying,
        # not the model's choice of subject, activity, or boundary.
        import re
        time_quotes = [q for q in quotes if re.search(r"\d{1,2}(?:时|:|：)\d{1,2}", q)]
        definitions["ActivityTimeClaim"]["properties"]["quote"] = {
            "type": "string", "enum": time_quotes or quotes}
        if not time_quotes:
            for field in ("start", "end"):
                activity["properties"][field]["maxItems"] = 0
    return schema


class LinkDecision(BaseModel):
    opinion_id: str
    decision: Literal["same_event", "different_event", "uncertain"]
    news_quote: str
    opinion_quote: str
    reason: str


class LinkReview(BaseModel):
    decisions: list[LinkDecision]
    radar_decisions: list["RadarLinkDecision"] = Field(default_factory=list)


class RadarLinkDecision(BaseModel):
    observation_id: str
    decision: Literal["same_event", "different_event", "uncertain"]
    news_quote: str
    row_numbers: list[int]
    reason: str


class SemanticField(BaseModel):
    field: Literal["country", "platform", "direction", "region", "quantity", "task", "start", "end"]
    status: Literal["supported", "conflict", "insufficient", "absent"]
    value: str | int | None = None
    claim_ids: list[str]
    reason: str


class ClaimExclusion(BaseModel):
    claim_id: str
    role: Literal["background", "organization", "commentary", "other_activity", "other_entity"]
    reason: str


class AssociationEvidence(BaseModel):
    observation_id: str
    # Text: verbatim quote. Radar: original row numbers, not generated prose.
    quote: str = ""
    row_numbers: list[int] = Field(default_factory=list)


class SemanticReview(BaseModel):
    association: Literal["same_activity", "different_activity", "uncertain"]
    association_evidence: list[AssociationEvidence]
    reason: str
    fields: list[SemanticField]
    excluded_claims: list[ClaimExclusion] = Field(default_factory=list)


def normalized_claim(field, value, quote, day, platform_aliases=None):
    """Check the *selected value*, never extract every keyword from its quote.

    Role/subject/activity assignment belongs to the model. This function only
    checks citation/value anchoring and normalizes explicit spellings/numbers.
    """
    if value is None or not isinstance(quote, str) or not quote.strip():
        return None
    if field == "quantity":
        from .agents.extraction import quantities
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value if value in quantities(quote) else None
    if field in {"start", "end"}:
        from .agents.extraction import text_times
        try:
            normalized = datetime.fromisoformat(str(value)).isoformat(timespec="seconds")
        except (ValueError, TypeError):
            return None
        # A boundary is validated as a boundary; overlap cannot invent an end.
        if normalized in quote:
            return normalized
        boundaries = text_times(quote, day)[0 if field == "start" else 1]
        if normalized in boundaries:
            return normalized
        if boundaries:
            return None
        import re
        moment = datetime.fromisoformat(normalized)
        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", quote)
        if date_match and datetime(*(int(v) for v in date_match.groups())).date() != moment.date():
            return None
        if not date_match and moment.date().isoformat() != day:
            return None
        token = rf"(?<!\d)0?{moment.hour}(?:时|:|：)0?{moment.minute}(?:分)?(?!\d)"
        if re.search(token, quote) and moment.second == 0:
            return normalized
        return normalized if normalized in quote else None
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if field == "country":
        candidates = [c for c, aliases in COUNTRY_ALIASES.items()
                      if clean(value) in {clean(c), *(clean(a) for a in aliases)}]
        canonical = candidates[0] if candidates else value
        variants = COUNTRY_ALIASES.get(canonical, [canonical])
    elif field == "region":
        region_names = {**{r: r for r in REGION_DIRECTION}, **REGION_ALIASES}
        canonical = next((c for a, c in region_names.items() if clean(a) == clean(value)), value)
        variants = [canonical, value] + [a for a, c in REGION_ALIASES.items() if c == canonical]
        if canonical == "菲律宾海":
            import re
            return canonical if re.search(r"菲律宾海(?!军)", quote) else None
    elif field == "task":
        task_names = {**{t: t for t in TASKS}, **TASK_ALIASES}
        canonical = next((c for a, c in task_names.items() if clean(a) == clean(value)), value)
        variants = [canonical, value] + [a for a, c in TASK_ALIASES.items() if c == canonical]
    elif field == "platform":
        canonical = canonical_platform(value, quote, platform_aliases)
        variants = [value, canonical] + {**PLATFORM_ALIASES, **(platform_aliases or {})}.get(canonical, [])
    else:
        canonical, variants = value, [value]
    return canonical if any(clean(v) in clean(quote) for v in variants if v) else None


def activity_observation(parent, activity, document, aliases):
    from .models import Observation
    from .utils import stable_id
    facts, quotes, rejected, normalizations = {}, {}, [], []
    for field in (*[f for f in FACT_FIELDS if f != "direction"], "individual"):
        for claim in getattr(activity, field):
            if claim.quote not in document.text:
                rejected.append(f"llm_unanchored_claim:{field}")
                continue
            value = normalized_claim(field, claim.value, claim.quote, document.date, aliases)
            if value is None:
                rejected.append(f"llm_unvalidated_claim:{field} value={claim.value!r} quote={claim.quote!r}："
                                "此引用不能证明该字段值；须选择实际包含该值或明确别名/时刻的原文片段")
                continue
            if field == "platform" and str(claim.value) != value:
                normalizations.append({"field": field, "model_value": claim.value,
                                       "canonical_value": value, "quote": claim.quote,
                                       "basis": "selected_entity_and_cited_spelling"})
            if field == "individual" and not str(value).endswith("号"):
                import re
                cue = re.search(r"(?:舷号|尾号|注册号|目标编号|舰艇编号)[：:\s]*[\"“]?" + re.escape(str(value)), document.text)
                if not cue and clean(str(value)) in {clean(str(p)) for p in parent.facts.get("platform", [])}:
                    rejected.append(f"llm_platform_used_as_individual:{value}（型号/舰级应放platform，无真实身份时individual=[]）")
                    continue
            facts.setdefault(field, []).append(value)
            quotes.setdefault(field, []).append(claim.quote)
    from .radar_units import normalized_identifier
    individuals = [normalized_identifier(v) for v in facts.pop("individual", [])]
    facts = {f: unique(vs) for f, vs in facts.items()}
    facts["direction"] = unique(REGION_DIRECTION[r] for r in facts.get("region", []) if r in REGION_DIRECTION)
    quotes["direction"] = quotes.get("region", [])
    metadata = {"extractor": "llm_semantic", "activity_id": activity.activity_id,
                "parent_observation_id": parent.id, "individual_ids": individuals,
                "platform_kinds": [] if activity.platform_kind == "unknown" else [activity.platform_kind],
                "event_date": next(iter(facts.get("start", [])), document.date)[:10],
                "time_role": "reported_activity_window", "segmentation_basis": "模型按活动及主体归属，并逐项检查原文引用"}
    if normalizations:
        metadata["entity_normalizations"] = normalizations
    return Observation(id=stable_id(parent.id + "|semantic-activity|" + activity.activity_id),
                       date=parent.date, source=parent.source, path=parent.path,
                       facts={f: vs for f, vs in facts.items() if vs}, quotes=quotes,
                       metadata=metadata, warnings=unique(rejected))
