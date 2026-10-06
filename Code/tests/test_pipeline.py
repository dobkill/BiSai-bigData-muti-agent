import json
import shutil
import sys

import openpyxl
import pytest

from fusion_agent.cli import export_reports
from fusion_agent import cli
from fusion_agent.config import Paths, ReportSettings, Settings
from fusion_agent.pipeline import AgentPipeline


def dataset(root):
    folder = root / "2025.5.10"
    for name in ("开源军情", "网上舆论"):
        (folder / name).mkdir(parents=True)
        text = "标题：中国1艘054A型护卫舰在南海执行战备巡逻\n正文：2025年5月10日，中国1艘054A型护卫舰在南海从10时00分至12时00分执行战备巡逻。"
        (folder / name / "001.txt").write_text(text, encoding="utf-8")
    wb = openpyxl.Workbook()
    air = wb.active
    air.title = "飞机"
    air.append(["批号", "国家地区", "目标类型", "位置时间", "经度", "纬度", "目标数量", "任务"])
    ship = wb.create_sheet("舰船")
    ship.append(["批号", "国家地区", "机舰名", "活动区域", "首点时间", "结束时间", "数量", "任务"])
    for _ in range(2):
        ship.append(["SHIP-1", "中国", "054A型护卫舰", "南海", "2025.05.10 10:00:00", "2025.05.10 12:00:00", 1, "战备巡逻"])
    wb.save(folder / "雷达.xlsx")
    wb.close()


def test_real_file_pipeline_and_cache_invalidation(tmp_path):
    root = tmp_path / "data"
    dataset(root)
    settings = Settings(paths=Paths(data_dir=root, output_dir=tmp_path/"out", cache_dir=tmp_path/"cache"),
                        report=ReportSettings(lookback_days=1))
    pipeline = AgentPipeline(settings)
    try:
        first = pipeline.run(["2025-05-10"])
        assert first["accepted_events"] == 1
        assert first["review_export"]["counts"]["可信事件"] == 1
        assert (settings.paths.output_dir/"review"/"manifest.json").is_file()
        report = settings.paths.output_dir / "reports" / "2025-05-10.json"
        assert json.loads(report.read_text(encoding="utf-8"))["verified_events"][0]["facts"]["quantity"] == 1
        graph = json.loads((settings.paths.output_dir / "knowledge_graph.json").read_text(encoding="utf-8"))
        assert len([n for n in graph["nodes"] if n["type"] == "evidence"]) == 3
        exported = tmp_path/"reports.jsonl"
        assert export_reports(settings, exported, None)["reports"] == 1
        assert len(exported.read_text(encoding="utf-8").splitlines()) == 1
        old_key = pipeline.cache_key("2025-05-10")
        path = root/"2025.5.10"/"网上舆论"/"001.txt"
        path.write_text(path.read_text(encoding="utf-8").replace("1艘", "2艘"), encoding="utf-8")
        assert pipeline.cache_key("2025-05-10") != old_key
        assert pipeline.run(["2025-05-10"])["accepted_events"] == 0
    finally:
        pipeline.close()


def test_cli_single_day_trial_isolates_output_and_review(tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    dataset(root)
    shutil.copytree(root / "2025.5.10", root / "2025.5.9")
    default_out = tmp_path / "baseline"
    default_review = tmp_path / "baseline_review"
    trial = tmp_path / "trial"
    settings = Settings(paths=Paths(data_dir=root, output_dir=default_out, cache_dir=tmp_path / "cache",
                                   review_dir=default_review), report=ReportSettings(lookback_days=7))
    monkeypatch.setattr(cli, "load_settings", lambda path: settings)
    monkeypatch.setattr(sys, "argv", ["fusion_agent", "run", "--date", "2025-05-10", "--single-day",
                                    "--offline", "--output-dir", str(trial)])
    cli.main()
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["processed_dates"] == ["2025-05-10"]
    assert manifest["lookback_days"] == 1
    assert (trial / "daily" / "2025-05-10.json").is_file()
    assert (trial / "review" / "manifest.json").is_file()
    assert not default_out.exists()
    assert not default_review.exists()


def test_failed_later_day_preserves_exports_and_resumes_from_day_cache(tmp_path, monkeypatch):
    root = tmp_path / "data"
    dataset(root)
    shutil.copytree(root / "2025.5.10", root / "2025.5.11")
    settings = Settings(paths=Paths(data_dir=root, output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"),
                        report=ReportSettings(lookback_days=2))
    pipeline = AgentPipeline(settings)
    original_process = pipeline.process_day
    original_extract = pipeline.extraction.run
    extractions = []

    def extract(docs, batches):
        extractions.append(docs[0].date)
        return original_extract(docs, batches)

    def interrupt_second(day, force=False):
        if day == "2025-05-11":
            raise RuntimeError("test service interruption")
        return original_process(day, force)

    monkeypatch.setattr(pipeline.extraction, "run", extract)
    monkeypatch.setattr(pipeline, "process_day", interrupt_second)
    try:
        with pytest.raises(RuntimeError, match="service interruption"):
            pipeline.run(["2025-05-10", "2025-05-11"])
        progress = json.loads((settings.paths.output_dir / "progress.json").read_text(encoding="utf-8"))
        assert progress["status"] == "failed"
        assert progress["completed_dates"] == ["2025-05-10"]
        assert progress["current_date"] == "2025-05-11"
        assert (settings.paths.output_dir / "daily/2025-05-10.json").is_file()
        assert (settings.paths.output_dir / "reports/2025-05-10.md").is_file()
        assert (settings.paths.output_dir / "review/2025-05-10/索引.json").is_file()
        assert not (settings.paths.output_dir / "manifest.json").exists()

        monkeypatch.setattr(pipeline, "process_day", original_process)
        manifest = pipeline.run(["2025-05-10", "2025-05-11"])
        assert extractions == ["2025-05-10", "2025-05-11"]
        assert manifest["total_candidates"] == 2
        assert manifest["review_export"]["updated_dates"] == ["2025-05-10", "2025-05-11"]
        progress = json.loads((settings.paths.output_dir / "progress.json").read_text(encoding="utf-8"))
        assert progress["status"] == "complete" and progress["completed_days"] == 2
    finally:
        pipeline.close()

