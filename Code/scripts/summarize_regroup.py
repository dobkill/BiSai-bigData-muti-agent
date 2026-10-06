"""汇总新版单日分组，可核对全部阅读副本；不调用模型、不重算关联。"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
from urllib.parse import quote

import openpyxl

from fusion_agent.models import DayResult
from fusion_agent.config import AlignmentSettings
from fusion_agent.relations import RelationMatcher
from fusion_agent.review_export import ASSOCIATION_STATUSES, ReviewExporter, json_value
from fusion_agent.utils import write_json, write_text


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(day, index, review_dir, data_dir):
    exporter = ReviewExporter(data_dir, review_dir)
    _, radar = exporter.preflight(day)
    observations = {o.id: o for o in day.observations}
    candidates = {c.id: c for c in day.candidates}
    events = {e.id: e for e in day.events}
    require({g["id"] for g in index["groups"]} == set(candidates), "索引与候选不一致")
    covered = set()
    original_texts = set()
    original_radar_rows = set()
    text_copies = radar_workbooks = 0
    for group in index["groups"]:
        folder = exporter.target(review_dir / group["path"])
        info = json.loads((folder / "事件信息.json").read_text(encoding="utf-8"))
        candidate, event = candidates[group["id"]], events[group["id"]]
        require(set(group["observation_ids"]) == set(candidate.observation_ids) == set(event.evidence_ids),
                f"证据映射不一致：{group['id']}")
        require(info["candidate"] == candidate.model_dump(mode="json"), "资料包中的关联结果不一致")
        require(info["event"] == event.model_dump(mode="json"), "资料包中的核验结果不一致")
        covered.update(candidate.observation_ids)
        readme = (folder / "README.md").read_text(encoding="utf-8")
        require("分组依据" in readme and "匹配分数" not in readme and len(readme.splitlines()) < 80,
                f"分组说明缺失或过长：{group['id']}")
        for copied in info["copied_texts"]:
            original = exporter.source(copied["original_path"]).read_bytes()
            data = (folder / copied["file"]).read_bytes()
            require(data == original and hashlib.sha256(data).hexdigest() == copied["sha256"],
                    f"原文副本不一致：{group['id']}/{copied['file']}")
            original_texts.add(copied["original_path"])
            text_copies += 1
        expected = defaultdict(list)
        for obs_id in candidate.observation_ids:
            obs = observations[obs_id]
            if obs.source != "radar":
                continue
            record = radar[obs.id]
            expected[record["sheet"]].extend([[number, *values] for number, values in record["rows"]])
            original_radar_rows.update((obs.path, record["sheet"], number) for number, _ in record["rows"])
        if expected:
            wb = openpyxl.load_workbook(folder / "雷达.xlsx", read_only=True, data_only=True)
            try:
                require(set(wb.sheetnames) == set(expected) | {"来源说明"}, "雷达工作表不一致")
                for name, rows in expected.items():
                    copied_rows = list(wb[name].values)
                    require(copied_rows[0][0] == "原始Excel行号", "雷达缺少原始行号")
                    require(json_value([list(row) for row in copied_rows[1:]]) == json_value(rows),
                            f"雷达行或顺序不一致：{group['id']}/{name}")
            finally:
                wb.close()
            radar_workbooks += 1
        if candidate.relation_status == "confirmed":
            require(any(r.discriminating and r.time_relation in {"overlap", "adjacent"} for r in candidate.relations),
                    "确认关联缺少明确时段对应")
            require(all(r.time_relation not in {"nearby_hours", "far", "different_date"}
                        and not any(r.fields.get(f) == "conflict" for f in
                                    ("country", "platform", "direction", "individual", "region"))
                        for r in candidate.relations), "确认关联存在身份或时空冲突")
        if event.accepted:
            require(candidate.relation_status == "confirmed" and not event.reasons, "接纳未经确认的关联")
            require({observations[i].source for i in event.evidence_ids} == {"open_source", "opinion", "radar"},
                    "接纳事件三源不齐")
            require(all(event.verdicts[f].status == "supported" and len(event.verdicts[f].support_sources) >= 2
                        for f in event.facts), "日报字段缺少跨源支持")
    require(covered == set(observations), "存在未导出的观测")
    # 覆盖全部材料不等于覆盖全部关系。逐一检查符合条件的跨源边，
    # 防止已经进入其他分组的材料在新闻↔舆论补配时被跳过。
    represented = {frozenset((a, b)) for candidate in day.candidates
                   for a, b in combinations(candidate.observation_ids, 2)}
    matcher = RelationMatcher(AlignmentSettings(**index.get("association_rules", {})))
    eligible_pairs = 0
    for left, right in combinations(day.observations, 2):
        if left.source != right.source and matcher.compare(left, right).status != "unrelated":
            eligible_pairs += 1
            require(frozenset((left.id, right.id)) in represented,
                    f"漏掉符合条件的跨源关系：{left.path} ↔ {right.path}")
    expected_texts = {o.path for o in observations.values() if o.source != "radar"}
    expected_rows = {(o.path, o.metadata["sheet"], n) for o in observations.values() if o.source == "radar"
                     for n in o.metadata["row_numbers"]}
    require(original_texts == expected_texts and original_radar_rows == expected_rows, "原始材料覆盖不完整")
    with (review_dir / day.date / "索引.csv").open(encoding="utf-8-sig", newline="") as f:
        require(len(list(csv.DictReader(f))) == len(day.candidates), "CSV 索引行数不一致")
    return {"date": day.date, "all_group_mappings": "passed", "all_observations_covered": len(covered),
            "text_copies_byte_identical": text_copies, "unique_original_text_files": len(original_texts),
            "radar_workbooks_checked": radar_workbooks, "unique_original_radar_rows": len(original_radar_rows),
            "confirmed_association_constraints": "passed", "accepted_event_invariants": "passed",
            "concise_group_readmes": "passed", "all_eligible_cross_source_pairs": "passed",
            "eligible_cross_source_pairs_checked": eligible_pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2025-05-10")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/constraints_trial")
    parser.add_argument("--validate", action="store_true", help="逐字节/逐行核对全部分组副本")
    parser.add_argument("--compare-to", type=Path, help="修复前快照目录，内含 manifest.json")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("association_method") == "constraints_intervals_v2", "请指定新版关联输出目录")
    day = DayResult.model_validate_json((output / "daily" / f"{args.date}.json").read_text(encoding="utf-8"))
    review = Path(manifest["review_export"]["output_dir"])
    index = json.loads((review / args.date / "索引.json").read_text(encoding="utf-8"))
    counter = Counter(g["status"] for g in index["groups"])
    counts = {status: counter[status] for status in ASSOCIATION_STATUSES}
    sources = {s: len({o.path for o in day.observations if o.source == s})
               for s in ("open_source", "opinion", "radar")}
    radar_rows = sum(len(o.metadata["row_numbers"]) for o in day.observations if o.source == "radar")
    accepted = sum(e.accepted for e in day.events)
    summary = {"date": day.date, "association_method": manifest["association_method"], "groups": len(day.candidates),
               "review_counts": counts, "accepted_events": accepted, "original_files": sources,
               "radar_rows": radar_rows, "event_units": len(day.observations),
               "llm_requests": manifest["llm_requests"], "llm_cache_hits": manifest["llm_cache_hits"],
               "llm_fallbacks": manifest["llm_fallbacks"]}
    if args.validate:
        validation = validate(day, index, review, Path(manifest["review_export"]["data_dir"]))
        write_json(output / "export_validation.json", validation)
        summary["validation"] = validation
    def link(path):
        return quote(Path(os.path.relpath(path, output)).as_posix())
    lines = [f"# {day.date} 重分组效果", "",
             f"范围：{sources['open_source']} 篇新闻、{sources['opinion']} 篇舆论、{radar_rows} 条雷达原始记录。",
             f"共 {len(day.candidates)} 组，{accepted} 组通过字段核验进入日报。", "",
             "| 阅读分类 | 组数 |", "| --- | --- |"]
    lines += [f"| {name} | {number} |" for name, number in counts.items()]
    if args.compare_to:
        previous = json.loads((args.compare_to.resolve() / "manifest.json").read_text(encoding="utf-8"))
        require(previous["processed_dates"] == manifest["processed_dates"], "对照结果处理日期不同")
        before = previous["review_export"]["counts"]
        summary["before_fix"] = {"groups": previous["total_candidates"], "review_counts": before,
                                 "accepted_events": previous["accepted_events"]}
        lines += ["", "本次修复：已进入其他组的新闻仍参加新闻—舆论补配；明确识别菲律宾海，阻止不同海域报道误合并。",
                  "全部满足条件的跨源关系都检查是否已保存；已有完整组中的关系不会重复生成两源子集。", "",
                  "| 分类 | 修复前 | 修复后 |", "| --- | --- | --- |"]
        lines += [f"| {name} | {before.get(name, 0)} | {number} |" for name, number in counts.items()]
        lines += [f"| 日报接纳 | {previous['accepted_events']} | {accepted} |"]
        old_path = args.compare_to.resolve() / "daily" / f"{day.date}.json"
        if old_path.exists():
            old_day = DayResult.model_validate_json(old_path.read_text(encoding="utf-8"))
            old_unmatched = {i for c in old_day.candidates if c.relation_status == "unmatched" for i in c.observation_ids}
            recovered = [g for g in index["groups"] if g["relation_status"] != "unmatched"
                         and old_unmatched.intersection(g["observation_ids"])]
            summary["recovered_unmatched_groups"] = [{"id": g["id"], "path": g["path"], "status": g["status"]}
                                                      for g in recovered]
            if recovered:
                lines += ["", "原来未关联、修复后已恢复的关系：", ""]
                for group in recovered:
                    lines.append(f"- [{group['country']} / {group['platform']}：{group['status']}]"
                                 f"({link(review / group['path'] / 'README.md')})。")
    lines += ["", "当前规则：身份硬校验、活动区间相容、整组一致性及同等证据候选唯一性。缺失不当作冲突；",
              "数小时差距只保留候选，多平台正文可明确归属时拆分。数量、任务差异保留供核验。", "",
              "分组增加是因为保留多个可能对应、拆开多平台报道；分组数与接纳数不能代表准确率。", "",
              f"本次启用 Ollama `{manifest['model_name']}`，复用 {manifest['llm_cache_hits']} 条已有抽取缓存，",
              f"新增 HTTP 请求 {manifest['llm_requests']} 次，规则回退 {manifest['llm_fallbacks']} 次。", "",
              f"阅读入口：[总 README]({link(review / 'README.md')})、"
              f"[当天索引]({link(review / day.date / '索引.csv')})。", "", "建议先看以下实际分组：", ""]
    for status in counts:
        examples = [g for g in index["groups"] if g["status"] == status]
        if not examples:
            continue
        example = max(examples, key=lambda g: (g["full_sources"], g["accepted"], len(g["observation_ids"])))
        lines.append(f"- [{status}：{example['country']} / {example['platform']}]("
                     f"{link(review / example['path'] / 'README.md')})。")
    split = next((g for g in index["groups"] if any(
        o.metadata.get("unit_platform") and o.id in g["observation_ids"] for o in day.observations)), None)
    if split:
        lines.append(f"- [多平台子事件示例]({link(review / split['path'] / 'README.md')})。")
    if args.validate:
        lines += ["", "已核对全部当前分组：原文副本逐字节一致，雷达子表逐行一致；覆盖当天全部材料。"]
    write_text(output / "重分组说明.md", "\n".join(lines) + "\n")
    write_json(output / "regroup_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
