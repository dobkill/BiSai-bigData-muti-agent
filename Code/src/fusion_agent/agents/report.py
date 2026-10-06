from collections import Counter

from ..models import DayResult


FIELD_NAMES = {"country": "国家地区", "platform": "平台", "direction": "方向", "region": "区域",
               "quantity": "数量", "task": "任务", "start": "开始时间", "end": "结束时间"}


class ReportAgent:
    name = "report"

    def run(self, day: DayResult, temporal: dict) -> tuple[dict, str]:
        events = sorted((e for e in day.events if e.accepted),
                        key=lambda e: (e.facts["direction"], e.facts.get("country", ""), e.facts.get("start", ""), e.id))
        reasons = Counter(r for e in day.events if not e.accepted for r in e.reasons)
        obs = {o.id: o for o in day.observations}
        decisions = []
        for direction in ("东海", "南海", "黄海"):
            current = [e for e in events if e.facts["direction"] == direction]
            decisions.append({"direction": direction, "verified_count": len(current),
                              "assessment": f"当日核验通过 {len(current)} 项事件。" if current else "当日没有满足核验条件的事件，不能据此推断该方向无活动。",
                              "follow_up": "优先补核时空关联和数量、任务冲突；继续观察已核验平台类型的后续公开证据。",
                              "event_ids": [e.id for e in current]})
        output = {"date": day.date, "format": "internal_v1_not_official_submission", "verified_events": [e.model_dump() for e in events],
                  "temporal_analysis": temporal, "decisions": decisions,
                  "audit_summary": {"candidates": len(day.candidates), "accepted": len(events),
                                    "event_classification": dict(Counter(e.classification for e in day.events)),
                                    "rejected": len(day.events)-len(events), "rejection_reasons": dict(reasons)},
                  "limitations": ["没有标准答案，接纳率不是评测准确率。", "航空坐标方向为粗粒度启发式，不能确定精确子海域。",
                                   "本格式为内部日报，待官方提交模板发布后适配。"]}
        lines = [f"# {day.date} 多源核验日报", "", f"当日通过核验的核心海域事件：**{len(events)}** 项。", "",
                 "本日报只陈述新闻、舆论、雷达三类来源完成同一事件关联、关键事实至少两类来源相互支持、"
                 "第三类来源未给出明确冲突值的核心海域可信事件。雷达作为时空、数量和轨迹核验依据之一，"
                 "不作为无条件优先真值；未核验字段不补全。", ""]
        for decision in decisions:
            direction = decision["direction"]
            lines += ["## " + direction, "", decision["assessment"], ""]
            for event in [e for e in events if e.facts["direction"] == direction]:
                f = event.facts
                desc = "、".join(f"{FIELD_NAMES[k]}：{v}" for k,v in f.items() if k != "direction")
                lines += [f"- {desc}。事件 `{event.id}`。"]
            lines += ["", "研判与后续核验：" + decision["follow_up"], ""]
        lines += ["## 时序观察", "", f"窗口：{temporal['window_start']} 至 {day.date}，已有 {len(temporal['available_days'])} 天结果，"
                  f"核验事件 {temporal['verified_event_count']} 项。", ""]
        if temporal["missing_days"]:
            lines += ["窗口内缺失日期：" + "、".join(temporal["missing_days"]) + "；缺失日期不按零活动计。", ""]
        for item in temporal["recurring_platform_types"]:
            lines += [f"- {item['country']} {item['platform']} 在 {len(item['observed_dates'])} 个日期出现，方向：{'、'.join(item['directions'])}。"
                      "这些记录仅说明同型号重复出现，尚不能确认同一具体平台连续部署。"]
        lines += ["", "## 证据索引", ""]
        for event in events:
            lines += [f"- `{event.id}`"]
            for oid in event.evidence_ids:
                o = obs[oid]
                suffix = f"，{o.metadata['sheet']} 工作表，行 {o.metadata['row_numbers']}" if o.source == "radar" else ""
                lines += [f"  - {o.source}：`{o.path}`{suffix}，证据 `{oid}`。"]
        lines += ["", "## 核验统计与范围", "", f"候选 {len(day.candidates)} 项，接纳 {len(events)} 项，"
                  f"排除 {len(day.events)-len(events)} 项。排除数量包含核心海域以外事件和未关联观测。", "",
                  "没有标准答案，当前不能报告准确率或比赛分数。航空方向来自粗粒度坐标包围框。日报格式仍需与官方提交模板对齐。", ""]
        return output, "\n".join(lines)

