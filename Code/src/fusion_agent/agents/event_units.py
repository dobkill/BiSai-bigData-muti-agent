"""将可明确归属的联合活动拆成平台子事件；原文及父事件事实保留供审计。"""
from copy import deepcopy
from datetime import datetime
import re

from ..domain import MODEL_RE, PLATFORM_ALIASES, canonical_platform, regions, REGION_DIRECTION, tasks, countries, platform_kinds
from ..models import Document, Observation
from ..relations import platform_key
from ..utils import stable_id, unique
from .extraction import quantities, text_times


class EventUnitAgent:
    name = "event_units"

    def canonicalize(self, obs: Observation, text: str = "") -> Observation:
        result = obs.model_copy(deep=True)
        if result.metadata.get("extractor") == "llm_semantic":
            # Also normalize older saved model observations, keeping every
            # semantic activity and the parent document in the same namespace.
            containers = [result.model_dump()] + result.metadata.get("semantic_activities", [])
            for container in containers:
                quote = "\n".join(container.get("quotes", {}).get("platform", []))
                values = container["facts"].get("platform", [])
                canonical_values = [canonical_platform(value, quote) for value in values]
                for before, after in zip(values, canonical_values):
                    if before != after:
                        container["metadata"].setdefault("entity_normalizations", []).append(
                            {"field": "platform", "model_value": before, "canonical_value": after,
                             "quote": quote, "basis": "selected_entity_and_cited_spelling"})
                if values:
                    container["facts"]["platform"] = unique(canonical_values)
            result.facts = containers[0]["facts"]
            result.metadata["entity_normalizations"] = containers[0]["metadata"].get("entity_normalizations", [])
            # Semantic roles/boundaries are already cited. Never reparse the
            # whole document and overwrite them with background keywords.
            result.metadata["time_role"] = "reported_activity_window"
            result.metadata["event_date"] = next(iter(result.facts.get("start", [])), result.date)[:10]
            result.metadata["time_precision"] = ("interval" if result.facts.get("end") else
                                                 "point" if result.facts.get("start") else "date_only")
            return result
        # 具体舰级后的括号舷号/项目代号是身份补充，不是另一个平台。
        auxiliary = []
        individuals = []
        for match in re.finditer(r"[（(]([^）)]+)[）)]", text):
            prefix = text[max(0, match.start() - 40):match.start()]
            token = match.group(1)
            if re.search(r"级[\u4e00-\u9fff]{0,14}(?:舰|艇|船)\s*$", prefix) and MODEL_RE.fullmatch(token):
                auxiliary.append(platform_key(token))
                if re.fullmatch(r"(?:DDG|FFH|CVN|CG)-?\d+", token, re.I):
                    individuals.append(platform_key(token))
        before = result.facts.get("platform", [])
        normalized = []
        seen = set()
        for value in before:
            key = platform_key(value)
            if key in auxiliary or key in seen:
                continue
            normalized.append(value)
            seen.add(key)
        result.facts["platform"] = normalized
        result.metadata["individual_ids"] = unique(result.metadata.get("individual_ids", []) + individuals
                                                  + [platform_key(v) for v in normalized if v.endswith("号")])
        if auxiliary:
            result.metadata["platform_designations"] = auxiliary
        result.warnings = [w for w in result.warnings if w != "multiple_platforms_unresolved"]
        if len([v for v in normalized if not v.startswith("generic:")]) > 1:
            result.warnings.append("multiple_platforms_unresolved")
        date = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        result.metadata["event_date"] = datetime(*(int(v) for v in date.groups())).date().isoformat() if date else obs.date
        result.metadata["time_role"] = "observation_window" if obs.source == "radar" else "reported_activity_window"
        if text:
            body = text.split("正文：", 1)[-1].split("主题词：", 1)[0]
            body_date = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", body)
            event_date = datetime(*(int(v) for v in body_date.groups())).date().isoformat() if body_date else obs.date
            result.metadata["event_date"] = event_date
            start, end = text_times(body, event_date)
            for field, values in (("start", start), ("end", end)):
                if values:
                    result.facts[field] = values
                else:
                    result.facts.pop(field, None)
            if not start and not end:
                point = re.search(r"(?:于|在|日[，,]?\s*)(\d{1,2})[时:：](\d{1,2})分?(?!\s*(?:至|到))", body)
                if point and int(point[1]) < 24 and int(point[2]) < 60:
                    result.facts["start"] = [f"{event_date}T{int(point[1]):02d}:{int(point[2]):02d}:00"]
        result.metadata["time_precision"] = ("interval" if result.facts.get("start") and result.facts.get("end") else
                                             "point" if result.facts.get("start") else "date_only")
        return result

    def mentions(self, text: str, name: str):
        hits = [m for m in MODEL_RE.finditer(text) if platform_key(m.group().removesuffix("型")) == platform_key(name)]
        variants = PLATFORM_ALIASES.get(name, [name])
        for variant in variants:
            hits.extend(re.finditer(re.escape(variant), text))
        return sorted(hits, key=lambda m: m.start())

    def split(self, obs: Observation, text: str) -> list[Observation]:
        if obs.metadata.get("extractor") == "llm_semantic":
            return [Observation.model_validate(u) for u in obs.metadata["semantic_activities"]]
        names = [v for v in obs.facts.get("platform", []) if not v.startswith("generic:")]
        if len(names) < 2:
            return [obs]
        body = text.split("正文：", 1)[-1].split("主题词：", 1)[0].strip()
        sentences = [s.strip() for s in re.split(r"[。；;\n]", body) if s.strip()]
        mentions = {name: self.mentions(body, name) for name in names}
        # 正文必须明确提到所有目标；分句能明确归属时可处理不同国家/方向。
        if not all(mentions.values()):
            return [obs]
        exclusive_by_name = {name: [s for s in sentences if self.mentions(s, name)
                                   and not any(self.mentions(s, other) for other in names if other != name)] for name in names}
        if not all(exclusive_by_name.values()) and (len(obs.facts.get("country", [])) > 1 or len(obs.facts.get("direction", [])) > 1):
            return [obs]
        units = []
        for name in names:
            exclusive = exclusive_by_name[name]
            context = "。".join(exclusive) if exclusive else body
            # 多个地点或多个时段共存且无法逐目标分配时不自动拆分。
            if not exclusive and (len(obs.facts.get("start", [])) > 1 or len(obs.facts.get("end", [])) > 1
                                  or len([r for r in obs.facts.get("region", []) if r not in REGION_DIRECTION.values()]) > 1):
                return [obs]
            unit = obs.model_copy(deep=True)
            unit.id = stable_id(obs.id + "|unit|" + platform_key(name))
            unit.facts["platform"] = [name]
            unit.metadata.update(parent_observation_id=obs.id, unit_platform=name,
                                 parent_facts=deepcopy(obs.facts), segment_text=context,
                                 shared_context=not bool(exclusive), unassigned_fields={})
            unit.warnings = [w for w in unit.warnings if w != "multiple_platforms_unresolved"]
            if exclusive:
                rs = regions(context)
                start, end = text_times(context, obs.metadata["event_date"])
                for field, values in {"country": countries(context), "region": rs,
                                      "direction": unique([REGION_DIRECTION[r] for r in rs]),
                                      "start": start, "end": end, "task": tasks(context)}.items():
                    if values:
                        unit.facts[field] = values
                    else:
                        unit.facts.pop(field, None)
                unit.metadata["platform_kinds"] = platform_kinds(context)
                unit.warnings = [w for w in unit.warnings if w not in {"invalid_time_interval", "text_date_mismatch"}]
            # 数量必须紧邻该目标的明确“架/艘”表达，不把编队总数复制给每个目标。
            qs = []
            for hit in self.mentions(context, name):
                prefix = context[max(0, hit.start() - 16):hit.start()]
                quantity = re.search(r"(\d+|[一二两三四五六七八九十])\s*(架|艘)\s*$", prefix)
                if quantity:
                    qs.extend(quantities(quantity.group()))
            if qs:
                unit.facts["quantity"] = unique(qs)
            else:
                if unit.facts.get("quantity"):
                    unit.metadata["unassigned_fields"]["quantity"] = unit.facts["quantity"]
                unit.facts.pop("quantity", None)
            if not exclusive and len(unit.facts.get("task", [])) > 1:
                unit.metadata["unassigned_fields"]["task"] = unit.facts.pop("task")
            unit.metadata["segmentation_basis"] = "按正文明确列出的平台拆分；共同活动上下文保留，数量仅按该平台前的架/艘表达归属"
            units.append(unit)
        return units

    def run(self, docs: list[Document], observations: list[Observation], keep_documents: bool = False) -> list[Observation]:
        texts = {d.id: d.text for d in docs}
        result = []
        for obs in observations:
            text = texts.get(obs.id, "")
            canonical = self.canonicalize(obs, text)
            units = self.split(canonical, text) if text else [canonical]
            if keep_documents:
                if len(units) > 1 or canonical.metadata.get("extractor") == "llm_semantic":
                    canonical.metadata["platform_units"] = [u.model_dump(mode="json") for u in units]
                    canonical.warnings = [w for w in canonical.warnings if w != "multiple_platforms_unresolved"]
                result.append(canonical)
            else:
                result.extend(units)
        return result
