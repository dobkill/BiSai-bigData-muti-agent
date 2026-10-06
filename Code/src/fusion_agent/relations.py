"""不使用加权分数：比较身份粒度、活动区间和区域，保存每项依据。"""
from datetime import datetime
import re

from .config import AlignmentSettings
from .domain import REGIONS, region_compatible
from .models import Observation, PairRelation

RELATION_NAMES = {"confirmed": "可确认关联", "candidate": "相关候选",
                  "conflicted": "疑似相关但有冲突", "unmatched": "未找到关联", "unrelated": "不构成候选"}
FIELD_LABELS = {"country": "国家", "platform": "平台", "direction": "方向", "region": "区域",
                "quantity": "数量", "task": "任务", "time": "活动时间", "individual": "目标编号"}
STATE_NAMES = {"equal": "一致", "compatible": "兼容", "unknown": "未知", "conflict": "冲突"}


def platform_key(value: str) -> str:
    return re.sub(r"[\s\-\u3000\"'“”‘’]", "", str(value)).upper()


def specific_platforms(obs: Observation) -> set[str]:
    return {platform_key(v) for v in obs.facts.get("platform", []) if not str(v).startswith("generic:")}


def event_window(obs: Observation):
    starts, ends = obs.facts.get("start", []), obs.facts.get("end", [])
    if len(starts) == 1 and len(ends) == 1:
        begin, finish = datetime.fromisoformat(starts[0]), datetime.fromisoformat(ends[0])
        return (begin, finish) if finish >= begin else None
    if len(starts) == 1 and not ends:
        point = datetime.fromisoformat(starts[0])
        return point, point
    return None


def time_relation(a: Observation, b: Observation, settings: AlignmentSettings) -> tuple[str, float | None]:
    aw, bw = event_window(a), event_window(b)
    if aw is None or bw is None:
        ad = a.metadata.get("event_date", a.date)
        bd = b.metadata.get("event_date", b.date)
        return ("date_only", None) if ad == bd else ("different_date", None)
    gap = max(0.0, (max(aw[0], bw[0]) - min(aw[1], bw[1])).total_seconds() / 60)
    if gap == 0:
        return "overlap", 0.0
    if gap <= settings.overlap_slack_minutes:
        return "adjacent", round(gap, 2)
    if gap <= settings.candidate_gap_hours * 60:
        return "nearby_hours", round(gap, 2)
    return "far", round(gap, 2)


def ordinary_relation(av, bv) -> str:
    if not av or not bv:
        return "unknown"
    return "equal" if set(av) == set(bv) and len(set(av)) == 1 else (
        "compatible" if set(av) & set(bv) else "conflict")


class RelationMatcher:
    def __init__(self, settings: AlignmentSettings):
        self.settings = settings

    def compare(self, a: Observation, b: Observation) -> PairRelation:
        fields = {name: ordinary_relation(a.facts.get(name, []), b.facts.get(name, []))
                  for name in ("country", "direction", "quantity", "task")}
        ap, bp = specific_platforms(a), specific_platforms(b)
        if ap and bp:
            if ap == bp and len(ap) == 1:
                fields["platform"] = "equal"
            elif ap & bp:
                fields["platform"] = "compatible"
            else:
                # 舰名与舰级/型号属于不同身份粒度，没有父子映射时保持未知。
                an = all("号" in v for v in ap)
                bn = all("号" in v for v in bp)
                fields["platform"] = "unknown" if an != bn else "conflict"
        else:
            ag = a.facts.get("platform", [])
            bg = b.facts.get("platform", [])
            fields["platform"] = "compatible" if ag and bg else "unknown"
        ak, bk = set(a.metadata.get("platform_kinds", [])), set(b.metadata.get("platform_kinds", []))
        if ak and bk and not ak & bk:
            fields["platform"] = "conflict"
        ai = set(a.metadata.get("individual_ids", []))
        bi = set(b.metadata.get("individual_ids", []))
        def identifier_kind(value):
            return "hull" if re.fullmatch(r"(?:DDG|FFH|CVN|CG)\d+", value) else "name"
        id_conflict = any({v for v in ai if identifier_kind(v) == kind}
                          and {v for v in bi if identifier_kind(v) == kind}
                          and not ({v for v in ai if identifier_kind(v) == kind} & {v for v in bi if identifier_kind(v) == kind})
                          for kind in ("hull", "name"))
        fields["individual"] = "conflict" if id_conflict else "equal" if ai & bi else "unknown"
        if fields["quantity"] == "compatible":
            fields["quantity"] = "conflict"
        ar, br = a.facts.get("region", []), b.facts.get("region", [])
        if not ar or not br:
            fields["region"] = "unknown"
        elif all(region_compatible(x, y) for x in ar for y in br):
            fields["region"] = "equal" if set(ar) == set(br) else "compatible"
        else:
            fields["region"] = "conflict"
        exact_region = bool((set(ar) & set(br)) - set(REGIONS))
        timing, gap = time_relation(a, b, self.settings)
        fields["time"] = ("compatible" if timing in {"overlap", "adjacent"} else
                          "conflict" if timing in {"far", "different_date"} else "unknown")
        identity = fields["platform"] == "equal" or fields["individual"] == "equal"
        precise = timing in {"overlap", "adjacent"}
        country = fields["country"] == "equal"
        direction = fields["direction"] == "equal"
        kind_ok = not ak or not bk or bool(ak & bk)
        # 同型号、同国家、同方向可以召回；泛称/身份缺失必须有时空区分线索。
        plausible = kind_ok and ((identity and (country or direction or exact_region or precise)) or
                                 (bool(ap & bp) and country and direction) or
                                 (country and direction and (precise or exact_region)))
        hard_conflicts = [f for f in ("country", "platform", "direction", "individual") if fields[f] == "conflict"]
        if hard_conflicts:
            # 明确身份冲突只在其他身份与精确时空仍强烈提示相关时保留疑似组。
            plausible = kind_ok and precise and (
                (identity and (country or direction or exact_region)) or
                (country and direction and exact_region))
        if timing in {"far", "different_date"}:
            plausible = False
        conflicts = [f for f, state in fields.items() if state == "conflict"]
        # 同源或多平台内部不同值不能被集合交集掩盖。
        if any(len(set(o.facts.get(f, []))) > 1 for o in (a, b) for f in ("country", "direction")):
            conflicts.append("country/direction_internal")
        # 同一舰名、同一天、同一区域仍可能有多次活动，不能替代时段对应。
        discriminating = precise
        blocked = any("multiple_platforms_unresolved" in o.warnings for o in (a, b))
        if not plausible:
            status = "unrelated"
        elif conflicts:
            status = "conflicted"
        elif identity and discriminating and not blocked:
            status = "confirmed"
        else:
            status = "candidate"
        basis = [f"{FIELD_LABELS[f]}{STATE_NAMES[fields[f]]}" for f in ("country", "platform", "direction")]
        time_text = {"overlap": "活动区间重叠或包含", "adjacent": f"活动区间接壤（间隔 {gap:g} 分钟）" if gap is not None else "时间接壤",
                     "nearby_hours": f"活动区间相隔 {gap / 60:.2f} 小时，仅保留候选" if gap is not None else "数小时候选",
                     "date_only": "至少一方仅有日期，不能证明时段对应", "different_date": "事件日期不对应",
                     "far": "活动区间间隔超过候选窗口"}
        basis.append(time_text.get(timing, "活动时段未知"))
        if exact_region:
            basis.append("具体区域相同")
        elif fields["region"] == "compatible":
            basis.append("区域为包含关系")
        for field in conflicts:
            basis.append(FIELD_LABELS.get(field, field) + "有明确差异")
        if blocked:
            basis.append("多平台归属尚未消解")
        return PairRelation(left_id=a.id, right_id=b.id, status=status, fields=fields,
                            time_relation=timing, gap_minutes=gap, discriminating=discriminating,
                            basis=basis, conflicts=conflicts)
