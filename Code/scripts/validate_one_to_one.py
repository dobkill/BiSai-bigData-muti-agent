"""Validate main-group uniqueness and exported original text/radar fidelity."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "src"))
from fusion_agent.models import DayResult


def validate_day(path, review_root, data_root, reference=None):
    day = DayResult.model_validate_json(path.read_text(encoding="utf-8"))
    observations = {o.id: o for o in day.observations}
    seen, radar_rows, text_files = set(), set(), set()
    news_count = sum(o.source == "open_source" for o in day.observations)
    assert len(day.candidates) == news_count, "main group count must equal source news count"
    index = json.loads((review_root / day.date / "索引.json").read_text(encoding="utf-8"))
    exported = {g["id"]: g for g in index["groups"]}
    assert set(exported) == {c.id for c in day.candidates}
    workbook_cache = {}
    matched_reference = 0
    def validate_package(members, folder):
        metadata = json.loads((folder / "事件信息.json").read_text(encoding="utf-8"))
        assert {o["id"] for o in metadata["observations"]} == {o.id for o in members}
        for copied in metadata["copied_texts"]:
            original = data_root / copied["original_path"]
            assert (folder / copied["file"]).read_bytes() == original.read_bytes(), "text copy differs"
            text_files.add(copied["original_path"])
        expected_rows = {}
        for obs in members:
            if obs.source != "radar":
                continue
            source = data_root / obs.path
            if source not in workbook_cache:
                workbook = openpyxl.load_workbook(source, read_only=True, data_only=True)
                try:
                    workbook_cache[source] = {s.title: list(s.values) for s in workbook}
                finally:
                    workbook.close()
            sheet = obs.metadata["sheet"]
            for number in obs.metadata["row_numbers"]:
                key = (obs.path, sheet, number)
                assert key not in radar_rows, "a radar row is occupied twice"
                radar_rows.add(key)
                expected_rows[(sheet, number)] = list(workbook_cache[source][sheet][number - 1])
        if expected_rows:
            workbook = openpyxl.load_workbook(folder / "雷达.xlsx", read_only=True, data_only=True)
            try:
                actual_rows = {}
                for sheet in workbook:
                    if sheet.title == "来源说明":
                        continue
                    for row in list(sheet.values)[1:]:
                        key = (sheet.title, row[0])
                        assert key not in actual_rows, "duplicate row in exported workbook"
                        actual_rows[key] = list(row[1:])
                assert actual_rows == expected_rows, "radar export differs from original rows"
            finally:
                workbook.close()
    for candidate in day.candidates:
        ids = set(candidate.observation_ids)
        assert len(ids) == len(candidate.observation_ids) and not seen & ids, "a source is occupied twice"
        seen.update(ids)
        members = [observations[i] for i in candidate.observation_ids]
        sources = Counter(o.source for o in members)
        assert sources["open_source"] == 1 and sources["opinion"] <= 1
        validate_package(members, review_root / exported[candidate.id]["path"])
        if reference is not None:
            number = candidate.assignment["news_number"]
            assert reference[number] == candidate.assignment["opinion_number"], "differs from proposed manual correspondence"
            matched_reference += 1
    unassigned = set(day.unassigned_observation_ids)
    assert len(unassigned) == len(day.unassigned_observation_ids) and not seen & unassigned
    assert seen | unassigned == set(observations), "a source is missing from groups and unassigned list"
    assert {v["id"] for v in index.get("unassigned_observations", [])} == unassigned
    assigned_rows = len(radar_rows)
    copied_unassigned = set()
    for record in index.get("unassigned_records", []):
        ids = record["observation_ids"]
        assert len(ids) == 1 and ids[0] in unassigned and ids[0] not in copied_unassigned
        copied_unassigned.update(ids)
        assert record["status"] == "未配对记录"
        validate_package([observations[ids[0]]], review_root / record["path"])
    assert copied_unassigned == unassigned, "an unassigned source has no readable original copy"
    assert index["counts"] == {s: sum(g["status"] == s for g in index["groups"])
                               for s in ("可信事件", "可疑事件", "未配对记录")}
    assert len(text_files) == sum(o.source != "radar" for o in day.observations)
    assert len(radar_rows) == sum(len(o.metadata["row_numbers"]) for o in day.observations if o.source == "radar")
    return {"date": day.date, "main_groups": len(day.candidates),
            "paired_opinions": sum(c.assignment["opinion_id"] is not None for c in day.candidates),
            "assigned_radar_batches": sum(observations[i].source == "radar" for i in seen),
            "assigned_radar_rows": assigned_rows, "unassigned_observations": len(unassigned),
            "exported_radar_rows": len(radar_rows), "unpaired_record_count": index.get("unpaired_record_count", 0),
            "exported_original_texts": len(text_files), "accepted": sum(e.accepted for e in day.events),
            "classification": index["counts"], "manual_proposed_pairs_checked": matched_reference,
            "scope": "structure, unique ownership and original-copy fidelity; not official accuracy"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "Code/outputs/one_to_one_trial")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "Datas_分组_新版")
    parser.add_argument("--date", action="append")
    parser.add_argument("--reference", type=Path, help="Optional manually proposed 2025-05-01 CSV; not a truth label")
    args = parser.parse_args()
    reference = None
    if args.reference:
        with args.reference.open(encoding="utf-8-sig", newline="") as file:
            reference = {r["新闻编号"]: r["舆论编号"] for r in csv.DictReader(file)}
    paths = ([args.analysis_dir / "daily" / (d + ".json") for d in args.date] if args.date else
             sorted((args.analysis_dir / "daily").glob("*.json")))
    results = [validate_day(p, args.review_dir, ROOT / "Datas/数据", reference if p.stem == "2025-05-01" else None)
               for p in paths]
    target = args.analysis_dir / "structure_validation.json"
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
