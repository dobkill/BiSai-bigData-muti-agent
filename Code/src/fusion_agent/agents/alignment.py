from collections import Counter
from itertools import combinations, product

from ..config import AlignmentSettings
from ..models import Candidate, Observation
from ..relations import FIELD_LABELS, RelationMatcher, specific_platforms
from ..utils import stable_id, unique


class AlignmentAgent:
    """保留候选证据；按配置做整体一一归属或历史多组合关联。"""
    name = "alignment"

    def __init__(self, settings: AlignmentSettings, llm=None):
        self.settings = settings
        self.matcher = RelationMatcher(settings)
        self.llm = llm
        self.unassigned_observation_ids = []
        self.audit = {}

    def run(self, observations: list[Observation], docs=None) -> list[Candidate]:
        if self.settings.mode == "one_to_one":
            from .one_to_one import OneToOneAlignmentAgent
            agent = OneToOneAlignmentAgent(self.settings, self.llm)
            result = agent.run(observations, docs or [])
            self.unassigned_observation_ids = agent.unassigned_observation_ids
            self.audit = agent.audit
            return result
        self.unassigned_observation_ids = []
        self.audit = {"method": "constraints_intervals_v2"}
        return self._run_legacy(observations)

    def _run_legacy(self, observations: list[Observation]) -> list[Candidate]:
        index = {o.id: o for o in observations}
        by_source = {s: sorted((o for o in observations if o.source == s), key=lambda o: o.id)
                     for s in ("open_source", "opinion", "radar")}
        comparisons = {}

        def compare(a, b):
            key = tuple(sorted((a.id, b.id)))
            if key not in comparisons:
                comparisons[key] = self.matcher.compare(a, b)
            return comparisons[key]

        def identity_ready(ids):
            members = [index[i] for i in ids]
            if len(members) < 2 or any("multiple_platforms_unresolved" in o.warnings for o in members):
                return False
            relations = [compare(index[a], index[b]) for a, b in combinations(sorted(ids), 2)]
            if any(r.status == "unrelated" or r.time_relation in {"nearby_hours", "far", "different_date"}
                   or any(r.fields.get(f) == "conflict" for f in ("country", "platform", "direction", "individual", "region"))
                   for r in relations):
                return False
            if not any(r.discriminating for r in relations):
                return False
            identities = [specific_platforms(o) for o in members if specific_platforms(o)]
            if not identities or not set.intersection(*identities):
                return False
            # 第三源可缺时段；缺身份的材料必须自身有明确时空与上下文对应。
            for obs in members:
                if not specific_platforms(obs) and not any(
                    obs.id in (r.left_id, r.right_id) and r.time_relation in {"overlap", "adjacent"}
                    and r.fields.get("country") == r.fields.get("direction") == "equal" for r in relations):
                    return False
            return True

        groups = []
        used = set()
        for radar in by_source["radar"]:
            news = [o for o in by_source["open_source"] if compare(radar, o).status != "unrelated"]
            opinions = [o for o in by_source["opinion"] if compare(radar, o).status != "unrelated"]
            current = []
            linked = set()
            for opened, opinion in product(news, opinions):
                if compare(opened, opinion).status != "unrelated":
                    current.append({radar.id, opened.id, opinion.id})
                    linked.update((opened.id, opinion.id))
            # 没有形成完整三角关系的材料保留两源组，不伪造第三源对应关系。
            current.extend({radar.id, o.id} for o in news + opinions if o.id not in linked)
            if not current:
                current = [{radar.id}]
            # 组内身份相容且有区分证据时合并多篇报道，不强制一来源只取一篇。
            merged = []
            for members in current:
                for group in merged:
                    union = group | members
                    if identity_ready(union):
                        group.update(members)
                        break
                else:
                    merged.append(set(members))
            groups.extend(merged)
            used.update(i for group in merged for i in group)
        # 已进入雷达组不代表所有关系都已找到。检查全部新闻↔舆论边，
        # 只跳过已经在同一组内保留的关系，避免漏配及重复添加三源组的子集。
        represented_text_pairs = {
            frozenset((a, b)) for group in groups for a, b in combinations(group, 2)
            if {index[a].source, index[b].source} == {"open_source", "opinion"}
        }
        for opened, opinion in product(by_source["open_source"], by_source["opinion"]):
            pair = frozenset((opened.id, opinion.id))
            if pair not in represented_text_pairs and compare(opened, opinion).status != "unrelated":
                groups.append({opened.id, opinion.id})
                used.update((opened.id, opinion.id))
        groups.extend({o.id} for o in observations if o.id not in used)
        # 稳定去重；多个合理候选保留，不使用排序第一名决定归属。
        distinct = {tuple(sorted(group)): group for group in groups}
        occurrences = Counter(i for ids in distinct for i in ids)
        ready = {ids: identity_ready(ids) for ids in distinct}
        # 只有具备同级区分证据的组合才会相互阻止确认；宽窗口疑似组不会
        # 抵消明确时段对应。低证据组合仍完整保留供人工查看。
        competing = Counter(i for ids in distinct if ready[ids] for i in ids)
        candidates = []
        for ids in sorted(distinct):
            members = [index[i] for i in ids]
            relations = [compare(index[a], index[b]) for a, b in combinations(ids, 2)]
            sources = {o.source for o in members}
            warnings = ["missing_source:" + source for source in by_source if source not in sources]
            if len(ids) == 1:
                status = "unmatched"
                warnings.append("unlinked_observation")
                basis = ["未找到具备足够身份及时空线索的其他来源"]
                # 将同型号但被明确差异阻断的原因写入阅读说明，避免
                # “未找到”被误读成原始数据完全没有相似的另一源材料。
                source_names = {"open_source": "新闻", "opinion": "舆论", "radar": "雷达"}
                current_obs = members[0]
                for other in observations:
                    if other.source == current_obs.source or not specific_platforms(current_obs) & specific_platforms(other):
                        continue
                    relation = compare(current_obs, other)
                    differences = [FIELD_LABELS[f] + "不同（" + "、".join(map(str, current_obs.facts.get(f, [])))
                                   + " / " + "、".join(map(str, other.facts.get(f, []))) + "）"
                                   for f in ("country", "direction", "region") if relation.fields.get(f) == "conflict"]
                    if relation.status == "unrelated" and differences:
                        basis = unique(basis + ["同型号" + source_names[other.source] + "：" + "；".join(differences)])
                basis = basis[:4]
            else:
                status = ("conflicted" if any(r.status == "conflicted" for r in relations) else
                          "confirmed" if ready[ids] else "candidate")
                basis = unique(line for r in relations for line in r.basis)
                # 同等证据的其他组合会阻止自动确认；弱候选只标记其自身歧义。
                ambiguous = any((competing if ready[ids] else occurrences)[i] > 1 for i in ids)
                if ambiguous:
                    warnings.append("multiple_compatible_hypotheses")
                    basis.append("存在其他合理组合，未强选唯一对应")
                    if status == "confirmed":
                        status = "candidate"
                elif status == "confirmed" and any(r.time_relation == "date_only" for r in relations):
                    basis.append("部分来源缺少时段；另两源有明确对应且完整组合归属唯一")
                if any("multiple_platforms_unresolved" in o.warnings for o in members):
                    warnings.append("multiple_platforms_unresolved")
                    if status == "confirmed":
                        status = "candidate"
            candidates.append(Candidate(id=stable_id("association_v2|" + "|".join(ids)), date=members[0].date,
                                        observation_ids=list(ids), relation_status=status, relations=relations,
                                        basis=basis, warnings=warnings))
        return candidates
