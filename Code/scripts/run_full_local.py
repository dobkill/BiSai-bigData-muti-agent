"""Run every dataset date with the selected backend, then validate source copies.

Re-running this command resumes with successful day/model caches. No rule fallback.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE / "src"))

from fusion_agent.cli import inspect_data
from fusion_agent.config import load_settings
from fusion_agent.ingestion import IngestionAgent
from fusion_agent.pipeline import AgentPipeline
from fusion_agent.runtime import add_llm_arguments, apply_llm_arguments, estimate_request_budget
from fusion_agent.utils import write_json
from validate_one_to_one import validate_day

logger = logging.getLogger(__name__)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE / "config.toml")
    parser.add_argument("--output-dir", type=Path, default=CODE / "outputs/one_to_one_full")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "Datas_分组_新版")
    parser.add_argument("--max-llm-calls", type=int, help="显式请求预算；默认按全量文件数及重试自动计算")
    parser.add_argument("--force", action="store_true", help="重算日结果，仍复用模型请求缓存")
    add_llm_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(args.config)
    apply_llm_arguments(settings, args)
    settings.paths.output_dir = args.output_dir.resolve()
    settings.paths.review_dir = args.review_dir.resolve()
    settings.llm.enabled = True
    settings.llm.fail_on_error = True
    settings.extraction.mode = "hybrid"
    settings.alignment.mode = "one_to_one"
    settings.review_export.enabled = True
    inventory = inspect_data(settings, None)
    if inventory["nonstandard_days"]:
        raise ValueError(f"全量数据中存在非标准日期：{inventory['nonstandard_days']}")
    text_count = inventory["open_source_texts"] + inventory["opinion_texts"]
    # Allow every planned extraction/review its configured retry count.
    targets = sorted(IngestionAgent(settings.paths.data_dir).dates())
    if args.max_llm_calls is not None and args.max_llm_calls < 1:
        raise ValueError("--max-llm-calls 必须大于零")
    settings.llm.max_calls = args.max_llm_calls if args.max_llm_calls is not None else estimate_request_budget(settings, targets)
    job = {"status": "running", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
           "model": settings.llm.endpoint.model, "provider": settings.llm.provider,
           "strict_llm": True, "inventory": inventory,
           "max_llm_calls": settings.llm.max_calls, "output_dir": str(settings.paths.output_dir),
           "review_dir": str(settings.paths.review_dir)}
    job_path = settings.paths.output_dir / "full_run.json"
    write_json(job_path, job)
    # Save the effective settings for reproducibility, without API credentials.
    effective = settings.model_dump(mode="json")
    effective["llm"]["api"].pop("api_key", None)
    effective["llm"]["llama_cpp"].pop("api_key", None)
    write_json(settings.paths.output_dir / "effective_settings.json", effective)
    logger.info("全量启动：%d 天，%d 篇文本，模型 %s，请求上限 %d", inventory["days"], text_count,
                settings.llm.endpoint.model, settings.llm.max_calls)
    start = time.perf_counter()
    pipeline = None
    try:
        pipeline = AgentPipeline(settings)
        manifest = pipeline.run(targets, force=args.force)
        job["status"] = "validating"
        write_json(job_path, job)
        validations = []
        for day in targets:
            validations.append(validate_day(settings.paths.output_dir / "daily" / (day + ".json"),
                                             settings.paths.review_dir, settings.paths.data_dir))
            logger.info("结构及原始材料校验 %d/%d：%s", len(validations), len(targets), day)
        write_json(settings.paths.output_dir / "structure_validation.json", validations)
        job.update(status="complete", validated_days=len(validations), total_groups=manifest["total_candidates"],
                   llm_requests=manifest["llm_requests"], llm_cache_hits=manifest["llm_cache_hits"],
                   llm_fallbacks=manifest["llm_fallbacks"],
                   classification=manifest["review_export"]["counts"],
                   accepted_events=manifest["accepted_events"])
    except (Exception, KeyboardInterrupt) as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logger.exception("全量运行未完成；已成功的日期和模型响应缓存可在重跑时复用")
        raise
    finally:
        if pipeline:
            job["llm_concurrency"] = pipeline.llm.concurrency_snapshot() if pipeline.llm else None
            pipeline.close()
        job.update(updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.perf_counter() - start, 2))
        write_json(job_path, job)
    print(json.dumps(job, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
