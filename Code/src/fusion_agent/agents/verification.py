from datetime import datetime
from itertools import combinations
from typing import Any

from ..config import AlignmentSettings, VerificationSettings
from ..domain import REGIONS, region_compatible
from ..models import Candidate, Event, FACT_FIELDS, FieldVerdict, Observation
from ..utils import unique
from ..relations import RelationMatcher, event_window, platform_key, time_relation

METHOD = "three_source_semantic_verification_v4"


class VerificationAgent:
    name = "verification"

    def __init__(self, settings: VerificationSettings, alignment: AlignmentSettings | None = None, llm=None):
        self.settings = settings
        self.alignment = alignment or AlignmentSettings()
        self.llm = llm
        self.audit = {}

    def association_reasons(self, candidate: Candidate, members: list[Observation]) -> list[str]:
        if candidate.relation_status is not None and candidate.relation_status != "confirmed":
            return ["association_not_confirmed:" + candidate.relation_status]
        relations = candidate.relations
        if not relations:
            matcher = RelationMatcher(self.alignment)
            relations = [matcher.compare(a, b) for a, b in combinations(members, 2) if a.source != b.source]
        # All records must belong to the same linked event. A third source merely
        # present in the package cannot turn a two-source event into a trusted one.
        linked = {members[0].id} if members else set()
        edges = [r for r in relations if r.status in {"confirmed", "candidate"} and not r.conflicts
                 and r.time_relation not in {"nearby_hours", "far", "different_date"}]
        while True:
            expanded = linked | {oid for r in edges if linked & {r.left_id, r.right_id}
                                 for oid in (r.left_id, r.right_id)}
            if expanded == linked:
                break
            linked = expanded
        if linked != {o.id for o in members} or any(r.status == "unrelated" or r.conflicts for r in relations):
            return ["three_source_association_incomplete"]
        if candidate.relation_status is None and not any(r.status == "confirmed" for r in relations):
            return ["association_not_confirmed:candidate"]
        return []

    def compatible(self, field: str, a: Any, b: Any) -> bool:
        if field in {"start", "end"}:
            return abs((datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds()) <= self.settings.time_tolerance_minutes * 60
        if field == "region":
            return region_compatible(a, b)
        if field == "platform":
            return platform_key(a) == platform_key(b)
        return a == b

    def time_field(self, name: str, observations: list[Observation]) -> FieldVerdict:
        values = {}
        ids = []
        for obs in observations:
            if obs.facts.get(name):
                values[obs.source] = unique(values.get(obs.source, []) + obs.facts[name])
                ids.append(obs.id)
        if not values:
            return FieldVerdict(status="absent")
        eligible = {o.source for o in observations if o.facts.get(name) and "insufficient_radar_points" not in o.warnings}
        timed = [o for o in observations if event_window(o) is not None]
        hours_candidate = False
        for i, left in enumerate(timed):
            for right in timed[i+1:]:
                state, _ = time_relation(left, right, self.alignment)
                if state in {"far", "different_date"}:
                    return FieldVerdict(status="conflict", values_by_source=values, evidence_ids=ids)
                hours_candidate |= state == "nearby_hours"
        if hours_candidate:
            # 数小时差距无法确认边界对应，也不能据此断言材料互相矛盾。
            return FieldVerdict(status="insufficient", values_by_source=values, evidence_ids=ids)
        # 覆盖区间相容不代表起止边界相同；只写获得两类来源支持的真实边界。
        for chosen in sorted(unique([v for vs in values.values() for v in vs])):
            support = [s for s, vs in values.items() if s in eligible and all(self.compatible(name, chosen, value) for value in vs)]
            if len(support) >= 2:
                return FieldVerdict(status="supported", value=chosen, support_sources=support,
                                    evidence_ids=ids, values_by_source=values)
        return FieldVerdict(status="insufficient", values_by_source=values, evidence_ids=ids)

    def field(self, name: str, observations: list[Observation]) -> FieldVerdict:
        values = {}
        ids = []
        for obs in observations:
            vs = obs.facts.get(name, [])
            if name == "platform":
                vs = [x for x in vs if not str(x).startswith("generic:")]
            if vs:
                values[obs.source] = unique(values.get(obs.source, []) + vs)
                ids.append(obs.id)
        if not values:
            return FieldVerdict(status="absent")
        all_values = unique([v for vs in values.values() for v in vs])
        eligible = {o.source for o in observations if o.facts.get(name) and "insufficient_radar_points" not in o.warnings}
        # 任意来源显式矛盾都阻止接纳，即使其余两源完全一致。
        if any(not self.compatible(name, a, b) for a in all_values for b in all_values):
            return FieldVerdict(status="conflict", values_by_source=values, evidence_ids=ids)
        support = [s for s in values if s in eligible]
        if len(support) < 2:
            return FieldVerdict(status="insufficient", values_by_source=values, evidence_ids=ids)
        if name == "region":
            # 广义方向不能证明某个子海域。具体子海域需至少两类来源明确支持。
            specific = [v for v in all_values if v not in REGIONS]
            chosen = specific[0] if specific else all_values[0]
            exact_sources = [s for s, vs in values.items() if s in eligible and chosen in vs]
            if specific and len(exact_sources) < 2:
                broad = [v for v in all_values if v in REGIONS]
                if broad:
                    chosen = broad[0]
                else:
                    return FieldVerdict(status="insufficient", values_by_source=values, evidence_ids=ids)
            support = [s for s, vs in values.items() if s in eligible and any(self.compatible(name, chosen, v) for v in vs)]
        else:
            # 时间在容差内支持，但输出真实证据中的时间，不求均值或偏爱雷达。
            chosen = min(all_values) if name in {"start", "end"} else all_values[0]
        return FieldVerdict(status="supported", value=chosen, support_sources=support,
                            evidence_ids=ids, values_by_source=values)

    def joint_fields(self, members: list[Observation], news: Observation):
        """Verify each explicitly scoped platform, without treating a joint
        report as contradictory names or copying a formation's total quantity.
        """
        per_platform = {}
        for name in news.facts.get("platform", []):
            scoped = []
            for obs in members:
                if platform_key(name) not in {platform_key(p) for p in obs.facts.get("platform", [])}:
                    continue
                units = [Observation.model_validate(u) for u in obs.metadata.get("platform_units", [])]
                unit = next((u for u in units if platform_key(name) in
                             {platform_key(p) for p in u.facts.get("platform", [])}), None)
                if unit is None:
                    unit = obs.model_copy(deep=True)
                else:
                    unit = unit.model_copy(deep=True)
                # A model may put several platforms in one shared activity.
                # The other participant is not a conflicting platform name,
                # and a shared quantity cannot be assigned to either one.
                if len(unit.facts.get("platform", [])) > 1:
                    unit.facts.pop("quantity", None)
                unit.facts["platform"] = [name]
                unit.id = obs.id  # verdict evidence IDs always refer to public source records
                scoped.append(unit)
            per_platform[name] = {f: self.time_field(f, scoped) if f in {"start", "end"}
                                  else self.field(f, scoped) for f in FACT_FIELDS}
        aggregate = {}
        for field in FACT_FIELDS:
            verdicts = [vs[field] for vs in per_platform.values()]
            values = {s: unique(v for o in members if o.source == s for v in o.facts.get(field, []))
                      for s in ("open_source", "opinion", "radar")}
            values = {s: vs for s, vs in values.items() if vs}
            evidence = unique(i for v in verdicts for i in v.evidence_ids)
            support = unique(s for v in verdicts for s in v.support_sources)
            if any(v.status == "conflict" for v in verdicts):
                status, value = "conflict", None
            elif verdicts and all(v.status == "supported" for v in verdicts):
                # A source covering just one platform cannot support the
                # entire combined value; its partial support stays below.
                support = sorted(set.intersection(*(set(v.support_sources) for v in verdicts)))
                if field == "platform":
                    status = "supported" if len(support) >= 2 else "insufficient"
                    value = "；".join(str(v.value) for v in verdicts) if status == "supported" else None
                elif field == "quantity":
                    # Per-platform counts are available in platform_verdicts;
                    # never output "1 ship" as a two-platform group's total.
                    status, value = "insufficient", None
                elif len(support) >= 2 and all(v.value == verdicts[0].value for v in verdicts):
                    status, value = "supported", verdicts[0].value
                else:
                    status, value = "insufficient", None
            else:
                status = "absent" if all(v.status == "absent" for v in verdicts) else "insufficient"
                value = None
            aggregate[field] = FieldVerdict(status=status, value=value, support_sources=support,
                                            evidence_ids=([o.id for o in members if o.source in support]
                                                          if status == "supported" else evidence), values_by_source=values)
        return aggregate, per_platform

    def run(self, candidates: list[Candidate], observations: list[Observation], docs=None) -> list[Event]:
        index = {o.id: o for o in observations}
        documents = {d.id: d.text for d in (docs or [])}
        self.audit = {"decision_engine": "llm_semantic" if self.llm else "offline_rules",
                      "llm_verifications": 0, "llm_verification_fallbacks": 0}
        events = []
        blocking_warnings = {"country_role_uncertain", "invalid_time_interval", "radar_date_mismatch", "text_date_mismatch",
                             "radar_cross_direction_track", "radar_implausible_track_jump",
                             "radar_individual_identity_conflict", "multiple_platforms_unresolved"}
        for candidate in candidates:
            members = [index[i] for i in candidate.observation_ids]
            reasons = list(candidate.warnings)
            semantic_review = {"method": "offline_rules"}
            semantic_verdicts = None
            semantic_platforms = {}
            if self.llm:
                from .semantic_verification import SemanticVerifier
                try:
                    semantic_verdicts, semantic_platforms, semantic_review = SemanticVerifier(self.llm).review(
                        candidate, members, documents, self)
                    self.audit["llm_verifications"] += 1
                    if semantic_review["response"]["association"] != "same_activity":
                        reasons.append("semantic_association_not_confirmed:" + semantic_review["response"]["association"])
                        candidate.relation_status = "candidate"
                    else:
                        # Keep rule comparisons as recall/audit signals, with
                        # the actual semantic association separately visible.
                        candidate.assignment["rule_relation_signals"] = [r.model_dump(mode="json") for r in candidate.relations]
                        candidate.assignment["semantic_association"] = semantic_review["response"]["association"]
                        candidate.relation_status = "conflicted" if any(v.status == "conflict" for v in semantic_verdicts.values()) else "confirmed"
                except Exception as exc:
                    if self.llm.settings.fail_on_error:
                        raise
                    self.audit["llm_verification_fallbacks"] += 1
                    reasons.append("semantic_verification_failed")
                    semantic_review = {"method": "llm_semantic_three_source_v1", "status": "failed", "error": str(exc)}
            else:
                reasons.extend(self.association_reasons(candidate, members))
            sources = {o.source for o in members}
            if sources != {"open_source", "opinion", "radar"}:
                reasons.append("three_sources_not_linked")
            if len(sources) < 2:
                reasons.append("fewer_than_two_source_types")
            for obs in members:
                reasons.extend(w for w in obs.warnings if w in blocking_warnings)
                if self.llm and any(w.startswith("llm_") for w in obs.warnings):
                    reasons.append("semantic_extraction_invalid:" + obs.id)
            kinds = unique([k for o in members for k in o.metadata.get("platform_kinds", [])])
            if len(kinds) > 1:
                reasons.append("platform_kind_conflict")
            news = next((o for o in members if o.source == "open_source"), None)
            platform_verdicts = {}
            if semantic_verdicts is not None:
                verdicts, platform_verdicts = semantic_verdicts, semantic_platforms
            elif candidate.assignment and news and news.metadata.get("platform_units"):
                verdicts, platform_verdicts = self.joint_fields(members, news)
            else:
                verdicts = {name: (self.time_field(name, members) if candidate.relation_status is not None
                                  and name in {"start", "end"} else self.field(name, members)) for name in FACT_FIELDS}
            for field, verdict in verdicts.items():
                if verdict.status == "conflict":
                    reasons.append("fact_conflict:" + field)
                elif field in self.settings.required_fields and verdict.status != "supported":
                    reasons.append("fact_not_verified:" + field)
            facts = {field: v.value for field, v in verdicts.items() if v.status == "supported"}
            if facts.get("start") and facts.get("end") and facts["end"] < facts["start"]:
                reasons.append("invalid_time_interval")
            reasons = unique(reasons)
            classification = "untrusted" if reasons else "trusted"
            report_reasons = list(reasons)
            if facts.get("direction") not in {"东海", "南海", "黄海"}:
                report_reasons.append("outside_core_or_unknown_direction")
            quality = sum(len(v.support_sources) for v in verdicts.values() if v.status == "supported") / (3*len(FACT_FIELDS))
            events.append(Event(id=candidate.id, date=candidate.date, accepted=classification == "trusted" and not report_reasons,
                                facts=facts, verdicts=verdicts, evidence_ids=candidate.observation_ids,
                                reasons=unique(report_reasons), quality=round(quality, 3), platform_verdicts=platform_verdicts,
                                classification=classification, verification_reasons=reasons,
                                semantic_review=semantic_review))
        return events

