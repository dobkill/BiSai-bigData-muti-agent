import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys

import httpx

from .config import load_settings
from .ingestion import IngestionAgent
from .llm import LLMClient
from .pipeline import AgentPipeline
from .review_export import export_saved_results
from .runtime import add_llm_arguments, apply_llm_arguments, estimate_request_budget
from .utils import parse_date, write_json, write_text


def default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "config.toml"


def inspect_data(settings, out: Path | None) -> dict:
    agent = IngestionAgent(settings.paths.data_dir)
    dates = agent.dates()
    counts = Counter()
    incomplete = []
    for day, path in dates.items():
        opened = len(list((path / "开源军情").glob("*.txt")))
        opinion = len(list((path / "网上舆论").glob("*.txt")))
        workbooks = len([p for p in path.glob("*.xlsx") if not p.name.startswith("~$")])
        counts.update({"open_source_texts": opened, "opinion_texts": opinion, "radar_workbooks": workbooks})
        if (opened, opinion, workbooks) != (50, 50, 1):
            incomplete.append({"date": day, "open_source": opened, "opinion": opinion, "workbooks": workbooks})
    result = {"data_dir": str(agent.root), "days": len(dates), "start": min(dates), "end": max(dates),
              **dict(counts), "nonstandard_days": incomplete,
              "report_nodes_available": {d: d in dates for d in settings.report.node_dates}}
    if out:
        write_json(out, result)
    return result


def doctor(settings, check_llm: bool) -> dict:
    result = {"python": sys.executable, "python_version": sys.version.split()[0],
              "data_dir_exists": settings.paths.data_dir.is_dir(), "llm_enabled": settings.llm.enabled,
              "provider": settings.llm.provider}
    if check_llm:
        client = LLMClient(settings.llm, settings.paths.cache_dir)
        try:
            client.validate()
            client.prepare_concurrency()
            # 最小结构化调用同时验证模型可访问性与 JSON 返回。
            response = client.complete_json("只输出 JSON 对象：{\"ok\":true}", "连接测试", {
                "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
            result["llm_check"] = response.get("ok") is True
            result["llm_concurrency"] = client.concurrency_snapshot()
        finally:
            client.close()
    return result


def export_reports(settings, target: Path, dates: list[str] | None) -> dict:
    reports = settings.paths.output_dir / "reports"
    manifest_path = settings.paths.output_dir / "manifest.json"
    if dates is None:
        if not manifest_path.exists():
            raise FileNotFoundError("请先 run 生成日报")
        dates = json.loads(manifest_path.read_text(encoding="utf-8"))["target_dates"]
    records = []
    for day in sorted(set(dates)):
        path = reports / (day + ".json")
        if not path.exists():
            raise FileNotFoundError(f"日报不存在：{path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["report_markdown"] = (reports / (day + ".md")).read_text(encoding="utf-8")
        records.append(record)
    write_text(target, "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return {"path": str(target.resolve()), "reports": len(records), "format": "internal_jsonl_not_official_submission"}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="多源事实核验与日报 Agent")
    parser.add_argument("--config", type=Path, default=default_config(), help="统一 TOML 配置文件")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="检查数据目录及指定节点覆盖情况")
    inspect.add_argument("--out", type=Path)
    check = sub.add_parser("doctor", help="检查环境；可选模型连接测试")
    check.add_argument("--check-llm", action="store_true")
    add_llm_arguments(check)
    run = sub.add_parser("run", help="执行 Agent 工作流并生成日报")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--date", action="append", help="可重复指定日报日期")
    selection.add_argument("--nodes", action="store_true", help="生成配置中的研判节点日报")
    selection.add_argument("--all", action="store_true", help="处理全部日期并生成全部日报")
    run.add_argument("--use-llm", action="store_true", help="启用已配置模型")
    run.add_argument("--offline", action="store_true", help="强制使用无需模型的规则基线")
    run.add_argument("--force", action="store_true", help="重算日缓存；模型请求缓存仍复用")
    run.add_argument("--single-day", action="store_true", help="每个指定日期只处理当天，不加载历史窗口")
    run.add_argument("--output-dir", type=Path, help="本次结果目录；相对路径按当前工作目录解析，阅读资料默认存到其 review 子目录")
    run.add_argument("--review-dir", type=Path, help="本次人工阅读目录；相对路径按当前工作目录解析")
    run.add_argument("--strict-llm", action="store_true", help="模型失败立即终止，避免把规则回退当成模型试跑")
    run.add_argument("--max-llm-calls", type=int, help="本次模型实际 HTTP 请求上限（含重试）；全量运行可覆盖配置中的单日预算")
    add_llm_arguments(run)
    export = sub.add_parser("export", help="导出内部 JSONL；官方格式待补充")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--date", action="append")
    review = sub.add_parser("export-review", help="将三源关联结果导出为逐事件人工阅读文件夹，不调用模型")
    review_selection = review.add_mutually_exclusive_group()
    review_selection.add_argument("--date", action="append", help="可重复指定需要导出的日期")
    review_selection.add_argument("--all", action="store_true", help="导出全部已保存的日关联结果（默认）")
    review_selection.add_argument("--nodes", action="store_true", help="只导出配置中的研判节点")
    review.add_argument("--out", type=Path, help="覆盖配置中的人工阅读目录")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = load_settings(args.config)
        if args.command in {"run", "doctor"}:
            apply_llm_arguments(settings, args)
        if args.command == "inspect":
            result = inspect_data(settings, args.out)
        elif args.command == "doctor":
            result = doctor(settings, args.check_llm)
        elif args.command == "export":
            result = export_reports(settings, args.out, [parse_date(d) for d in args.date] if args.date else None)
        elif args.command == "export-review":
            dates = settings.report.node_dates if args.nodes else [parse_date(d) for d in args.date] if args.date else None
            result = export_saved_results(settings.paths.data_dir, settings.paths.output_dir / "daily",
                                          args.out or settings.paths.review_dir, dates)
        else:
            if args.use_llm and args.offline:
                raise ValueError("--use-llm 与 --offline 不能同时使用")
            if args.use_llm:
                settings.llm.enabled = True
                if settings.extraction.mode == "rules":
                    settings.extraction.mode = "hybrid"
            if args.offline:
                settings.llm.enabled = False
            if args.single_day:
                settings.report.lookback_days = 1
            if args.output_dir:
                settings.paths.output_dir = args.output_dir.resolve()
                settings.paths.review_dir = settings.paths.output_dir / "review"
            if args.review_dir:
                settings.paths.review_dir = args.review_dir.resolve()
            if args.strict_llm:
                if not settings.llm.enabled or settings.extraction.mode == "rules":
                    raise ValueError("--strict-llm 需要启用模型抽取，请搭配 --use-llm")
                settings.llm.fail_on_error = True
            if args.max_llm_calls is not None:
                if args.max_llm_calls < 1:
                    raise ValueError("--max-llm-calls 必须大于零")
                settings.llm.max_calls = args.max_llm_calls
            available = IngestionAgent(settings.paths.data_dir).dates()
            targets = list(available) if args.all else settings.report.node_dates if args.nodes else [parse_date(d) for d in args.date]
            if args.all and settings.llm.enabled and args.max_llm_calls is None:
                settings.llm.max_calls = estimate_request_budget(settings, targets)
                logging.info("全量模型请求预算（含重试）自动设为 %d；可用 --max-llm-calls 显式限制", settings.llm.max_calls)
            pipeline = AgentPipeline(settings)
            try:
                result = pipeline.run(targets, force=args.force)
            finally:
                pipeline.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1)

